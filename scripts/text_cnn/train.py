import torch
from torch import nn
from sentence_transformers import SentenceTransformer, SentencesDataset, \
    InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from util import preprocess_data, MaxOverTimePooling, load_data
from cnn import TextCNN
from data_set import SentenceDataset
from torch.utils.data import DataLoader
import math
import time
import os
from sklearn import metrics
import numpy as np

evaluation_ratio = 0.2  # ~20% of pairs for evaluation
create_test_set = True
test_ratio = 0.2  # ~20% of pairs for testing if desired

train_batch_size = 8  # after how many samples to update weights (default = 1)
epochs = 5
warmup_steps = 50
evaluation_steps = 300
output_path = 'models/end2end_bert'  # where the BERT model is saved

# STEP 1: Split data in training and evaluation set
training_sentences_1, training_sentences_2, training_scores, \
validation_sentences_1, validation_sentences_2, validation_scores, \
test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw \
    = load_data()

print(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences_1)}\n"
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
# test_dataset = SentenceDataset(test_sentences_1, test_sentences_2, test_scores_normalized)
# test_dataloader = DataLoader(test_dataset)

# bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("Starting BERT training...")
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=5e-2)


def train(model, loss_fn, optimizer, device, result_path):
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
            print(f"loss: {loss_sum / i:>7f}  [{batch_index + 1:>5d}/{num_batches:>5d}]")
            loss_sum = 0
            i = 0


def validate(model, device, result_path):
    print("validate")
    model.eval()
    size = len(validation_dataloader_1.dataset)
    num_batches = math.ceil(size / validation_dataloader_1.batch_size)
    model.train()
    sentence_dl_iter_1 = iter(validation_dataloader_1)
    sentence_dl_iter_2 = iter(validation_dataloader_2)
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
    print(metrics.classification_report(torch.argmax(l, dim=1).detach(), torch.argmax(p, dim=1).detach()))
    # print(f"accuracy: {accuracy}")
    print(f"\npredicted labels {torch.unique(torch.argmax(p, dim=1), return_counts=True)}")
    print(f"actual labels {torch.unique(torch.argmax(l, dim=1), return_counts=True)}")


def train_one_epoch(model, optimizer, loss_fn, epoch_index, result_path,
                    device):
    start = time.time()
    print(f"Epoch {epoch_index + 1}\n-------------------------------")
    train(model, loss_fn, optimizer, device, result_path=f"{result_path}/train/{epoch_index}")
    print(f"Validation {epoch_index + 1}\n-------------------------------")
    validate(model, device, result_path=f"{result_path}/test/{epoch_index}")
    end = time.time()
    print(f"Epoch{epoch_index + 1} duration was {end - start}\n\n")


for t in range(epochs):
    train_one_epoch(model=network, optimizer=optimizer, loss_fn=loss, epoch_index=t,
                    result_path="", device='cpu')
print("Done!")
