import os
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

import time

from sentence_transformers import SentenceTransformer
from data_set import SentenceDataset, my_collate
from preprocess import preprocess_data
from models.lstm import lstm_model
from models.sim_cnn import  SimCnn
from train import train, validate

device = "cuda" if torch.cuda.is_available() else "cpu"
device_pl = "gpu" if device == "cuda" else device
print("Using {} device".format(device))

"""
---------------------------------------------------------------------
|                                                                   |
|                           Parameters                              |
|                                                                   |
---------------------------------------------------------------------
"""
log_path = os.path.join("..", "..", "logs", "recursive")
log_path_tb = os.path.join(log_path, "tb_logs")
log_name = "recursive"
base_path = os.path.join("..", "..", "data")
data_path = os.path.join(base_path, "processed", "training_data")
CSV_PATH = os.path.join(base_path, "semeval-2022_task8_train-data_batch.csv")

evaluation_ratio = 0.3  # ~20% of pairs for evaluation
create_test_set = False
test_ratio = 0.01  # ~20% of pairs for testing if desired

preprocess = True

# training parameters
batch_size = 16
epochs = 200
lr = 0.001

es_epochs = 20
"""
---------------------------------------------------------------------
|                                                                   |
|                            Code                                   |
|                                                                   |
---------------------------------------------------------------------
"""

bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')

training_sentences_1, training_sentences_2, training_scores, training_ids, \
evaluation_sentences_1, evaluation_sentences_2, evaluation_scores, evaluation_ids, \
test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids \
    = preprocess_data(data_path, CSV_PATH, base_path, create_test_set=True, validation_ratio=evaluation_ratio,
                      test_ratio=test_ratio)

print(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences_1)}\n"
      f"# evaluation sentence pairs: {len(evaluation_sentences_1)}\n"
      f"# test sentence pairs: {len(test_sentences_1)}")
train_ds = SentenceDataset(training_sentences_1, training_sentences_2, training_scores, bert)
val_ds = SentenceDataset(evaluation_sentences_1, evaluation_sentences_2, evaluation_scores, bert)
test_ds = SentenceDataset(test_sentences_1, test_sentences_2, test_scores_normalized, bert)

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=my_collate)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)

loss_fn = nn.MSELoss().to(device)
network = SimCnn(loss_fn, device=device).to(device)
summary(network, input_size=(batch_size, 1, 20, 100))
optimizer = SGD(network.parameters(), lr=lr)

print("Start training model!")

writer = SummaryWriter(os.path.join(log_path, "tb_logs"))

best_metric = 0
best_index = 0
epochs_not_improved = 0
for t in range(epochs):
    start = time.time()
    train(network, loss_fn, optimizer, device, train_dl, writer, epoch=t)
    metric = validate(network, device, val_dl, save_predictions=True,
                      result_path=os.path.join(log_path, f"predictions_epoch_{t}.csv"),
                      pbar_description=f"Validate epoch {t}")
    if metric <= best_metric:
        epochs_not_improved += 1
        if epochs_not_improved >= es_epochs:
            break
    else:
        best_index = t
        best_metric = metric
        epochs_not_improved = 0
    end = time.time()

writer.flush()
writer.close()

print(f"Finished training model! Best loss was {best_metric} at epoch index {best_index}")
print("Start testing...")
validate(network, device, train_dl, save_predictions=True, ids=training_ids,
         result_path=os.path.join(log_path, "predictions_train.csv"),
         pbar_description="Test network with train data set")
validate(network, device, val_dl, save_predictions=True, ids=evaluation_ids,
         result_path=os.path.join(log_path, "predictions_test.csv"),
         pbar_description="Test network with validation data set")
# trainer.test(model, module)
print("Done!")
