import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary

import time

from sentence_transformers import SentenceTransformer
from data_set import SentenceDataset
from preprocess import preprocess_data
from text_cnn import TextCnn
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
log_path = os.path.join("..", "..", "logs")
log_path_tb = os.path.join(log_path, "tb_logs")
log_name = "text_cnn"
base_path = os.path.join("..", "..", "data")
data_path = os.path.join(base_path, "processed", "training_data")
CSV_PATH = os.path.join(base_path, "semeval-2022_task8_train-data_batch.csv")

evaluation_ratio = 0.2  # ~20% of pairs for evaluation
create_test_set = True
test_ratio = 0.2  # ~20% of pairs for testing if desired

preprocess = True

# training parameters
batch_size = 2
epochs = 20

"""
---------------------------------------------------------------------
|                                                                   |
|                            Code                                   |
|                                                                   |
---------------------------------------------------------------------
"""
if not os.path.exists(log_path):
    os.makedirs(log_path)

# get latest checkpoint
checkpoint = None
checkpoint_dir = os.path.join(log_path, log_name)
if os.path.exists(checkpoint_dir):
    versions = [os.path.join(checkpoint_dir, dir, "checkpoints") for dir in os.listdir(checkpoint_dir) if
                os.path.isdir(os.path.join(checkpoint_dir, dir, "checkpoints"))]
    versions.sort(reverse=True)
    versions = list(
        filter(lambda cpdir: len([os.path.join(cpdir, _) for _ in os.listdir(cpdir) if _.endswith(".ckpt")]) > 0,
               versions))
    if len(versions) > 0:
        checkpoint_dir = versions[0]
        checkpoints = [os.path.join(checkpoint_dir, _) for _ in os.listdir(checkpoint_dir) if _.endswith(".ckpt")]
        if len(checkpoints) > 0:
            checkpoints.sort(reverse=True)
            checkpoint = checkpoints[0]
print(checkpoint)

"""
---------------------------------------------------------------------
|                                                                   |
|                     data loader                                   |
|                                                                   |
---------------------------------------------------------------------
"""

bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
bert.max_seq_length = 512

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

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

loss_fn = nn.MSELoss().to(device)
network = TextCnn(loss_fn, device).to(device)
summary(network,(2,40 ,512))
optimizer = Adam(network.parameters(), 1e-3)

print("Start training model!")

for t in range(epochs):
    start = time.time()
    train(network, loss_fn, optimizer, device, train_dl, epoch=t)
    validate(network, device, val_dl, save_predictions=False, pbar_description=f"Validate epoch {t}")
    end = time.time()

print("Finished training model!")
print("Start testing...")
validate(network, device, test_dl, save_predictions=True, ids=test_ids,
         result_path=os.path.join(log_path, "predictions.csv"), pbar_description="Test network")
# trainer.test(model, module)
print("Done!")
