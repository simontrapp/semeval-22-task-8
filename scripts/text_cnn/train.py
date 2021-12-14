import os.path

import pandas
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from preprocess import load_data
from cnn import TextCNN
from data_set import SentenceDataset
from torch.utils.data import DataLoader
import math
import time
from sklearn import metrics
import numpy as np
import logging as log


train_batch_size = 8  # after how many samples to update weights (default = 1)
epochs = 1
warmup_steps = 50
evaluation_steps = 300
output_path = os.path.join("..", "..", "models", "text_cnn")  # where the BERT model is saved
if not os.path.exists(output_path):
    os.makedirs(output_path)
log_base_path = os.path.join("..","..","logs")
if not os.path.exists(log_base_path):
    os.makedirs(log_base_path)
log.basicConfig(filename=os.path.join(log_base_path, "train.py.log"), encoding='utf-8', level=log.DEBUG)

# STEP 1: Split data in training and evaluation set
training_sentences_1, training_sentences_2, training_scores, training_ids, \
validation_sentences_1, validation_sentences_2, validation_scores, validation_ids, \
test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids \
    = load_data()

log.info(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences_1)}\n"
      f"# evaluation sentence pairs: {len(validation_sentences_1)}\n"
      f"# test sentence pairs: {len(test_sentences_1)}")

# STEP 2: Train BERT
network = TextCNN()


def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    # print(data.shape)
    max = np.max([x.shape[1] for x in data])
    padded = [np.concatenate((x, np.zeros((1, max - x.shape[1], x.shape[2]))), axis=1) for x in data]
    padded = torch.Tensor(np.array(padded))
    target = np.array([item[1].numpy() for item in batch])
    target = torch.Tensor(target)
    return [padded, target]


bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
train_dataset_1 = SentenceDataset(training_sentences_1, training_scores, bert)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=train_batch_size, collate_fn=my_collate)
train_dataset_2 = SentenceDataset(training_sentences_2, training_scores, bert)
train_dataloader_2 = DataLoader(train_dataset_2, batch_size=train_batch_size, collate_fn=my_collate)

validation_dataset_1 = SentenceDataset(validation_sentences_1, validation_scores, bert)
validation_dataloader_1 = DataLoader(validation_dataset_1, batch_size=train_batch_size, collate_fn=my_collate)
validation_dataset_2 = SentenceDataset(validation_sentences_2, validation_scores, bert)
validation_dataloader_2 = DataLoader(validation_dataset_2, batch_size=train_batch_size, collate_fn=my_collate)

test_dataset_1 = SentenceDataset(test_sentences_1, test_scores_normalized, bert)
test_dataloader_1 = DataLoader(test_dataset_1, batch_size=train_batch_size, collate_fn=my_collate)
test_dataset_2 = SentenceDataset(test_sentences_2, test_scores_normalized, bert)
test_dataloader_2 = DataLoader(test_dataset_2, batch_size=train_batch_size, collate_fn=my_collate)

# bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

log.info("Starting BERT training...")
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=5e-2)


def train(model, loss_fn, optimizer, device, result_path=None):
    size = len(train_dataloader_1.dataset)
    num_batches = math.ceil(size / train_dataloader_1.batch_size)
    model.train()
    sentence_dl_iter_1 = iter(train_dataloader_1)
    sentence_dl_iter_2 = iter(train_dataloader_2)
    loss_sum = 0
    i = 0
    for batch_index in range(num_batches):
        X_1, y = next(sentence_dl_iter_1)
        X_2, y = next(sentence_dl_iter_2)
        X_1, X_2, y = X_1.to(device), X_2.to(device), y.to(device)
        pred = model([X_1, X_2])
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        i += 1
        if batch_index % 20 == 0 or batch_index + 1 == num_batches:
            log.info(f"loss: {loss_sum / i:>7f}  [{batch_index + 1:>5d}/{num_batches:>5d}]")
            loss_sum = 0
            i = 0


def validate(dl_1, dl_2, model, device, result_path=None, save_predictions=False, ids_path=None):
    log.info("validate")
    model.eval()
    size = len(dl_1.dataset)
    num_batches = math.ceil(size / dl_1.batch_size)
    model.train()
    sentence_dl_iter_1 = iter(dl_1)
    sentence_dl_iter_2 = iter(dl_2)
    p = None
    l = None
    for batch_index in range(num_batches):
        X_1, y = next(sentence_dl_iter_1)
        X_2, y = next(sentence_dl_iter_2)
        X_1, X_2, y = X_1.to(device), X_2.to(device), y.to(device)
        pred = model([X_1, X_2])
        if p is None:
            p = pred
            l = y
        else:
            p = torch.cat((p, pred), dim=0)
            l = torch.cat((l, y), dim=0)

    p = torch.argmax(p, dim=1).detach()
    l = torch.argmax(l, dim=1).detach()
    log.info(metrics.classification_report(l, p))
    # log.info(f"accuracy: {accuracy}")
    log.info(f"\npredicted labels {torch.unique(p, return_counts=True)}")
    log.info(f"actual labels {torch.unique(l, return_counts=True)}")

    if save_predictions:
        ids = pandas.read_csv(ids_path, index_col=False)['pair_id']
        pandas.DataFrame({'pair_id': ids, 'predictions': p}).to_csv(result_path, index=False)


def train_one_epoch(model, optimizer, loss_fn, epoch_index, result_path,
                    device):
    start = time.time()
    log.info(f"Epoch {epoch_index + 1}\n-------------------------------")
    train(model, loss_fn, optimizer, device)
    log.info(f"Validation {epoch_index + 1}\n-------------------------------")
    validate(validation_dataloader_1, validation_dataloader_2, model, device)
    end = time.time()
    log.info(f"Epoch{epoch_index + 1} duration was {end - start}\n\n")


for t in range(epochs):
    train_one_epoch(model=network, optimizer=optimizer, loss_fn=loss, epoch_index=t,
                    result_path="", device='cpu')

log.info("\n Test network:")
validate(test_dataloader_1, test_dataloader_2, network, "cpu", result_path=os.path.join(output_path, "predictions.csv"),
         save_predictions=True,
         ids_path=os.path.join("..", "..", "data", "embeddings", "test_ids.csv"))
log.info("Save the model:")
torch.save(network.state_dict(), os.path.join(output_path, "network"))
torch.save(optimizer.state_dict(), os.path.join(output_path, "optimizer"))
# for loading:
# model = TextCNN()
# model.load_state_dict(torch.load(os.path.join(output_path,"network")))
# model.eval()
# optimizer = torch.optim.Adam(network.parameters(), lr=5e-2)
# optimizer.load_state_dict(torch.load(os.path.join(output_path,"optimizer")))

log.info("Done!")
