import os
import torch
from torch import nn
from pytorch_lightning import Trainer

from pl_network import PytorchLightningModule
from data_module import DataModule
from preprocess import preprocess_data

from pytorch_lightning.loggers import TensorBoardLogger
from sentence_transformers import SentenceTransformer

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
batch_size = 8
epochs = 1

loss_fn = nn.BCELoss(reduction='mean').to(device)

"""
---------------------------------------------------------------------
|                                                                   |
|                            Code                                   |
|                                                                   |
---------------------------------------------------------------------
"""
if not os.path.exists(log_path):
    os.makedirs(log_path)

if (preprocess):
    bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    train_s1, train_s2, train_scores, train_ids, val_s1, val_s2, val_scores, val_ids, test_s1, test_s2, test_scores_normalized, test_scores_raw, test_ids \
        = preprocess_data(data_path, CSV_PATH,base_path, bert, create_test_set=create_test_set, validation_ratio=evaluation_ratio,
                          test_ratio=test_ratio)

    print()
    #save_data(base_path, train_s1, train_s2, train_scores, train_ids, val_s1, val_s2, val_scores, val_ids, test_s1,
     #         test_s2,
      #        test_scores_normalized, test_scores_raw, test_ids)

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

logger = TensorBoardLogger(log_path, name=log_name)
if checkpoint is not None:
    model = PytorchLightningModule.load_from_checkpoint(checkpoint, loss_fn=loss_fn, device=device)
    trainer = Trainer(max_epochs=epochs, logger=logger, resume_from_checkpoint=checkpoint, gpus=1)
else:
    model = PytorchLightningModule(loss_fn=loss_fn, device=device)
    trainer = Trainer(max_epochs=epochs, logger=logger, gpus=1)
module = DataModule(embeddings_path=os.path.join(base_path), batch_size=batch_size)
print("Start training model!")
trainer.fit(model, module)
print("Finished training model!")

trainer.test(model, module)
print("Done!")
