import os
import torch
from torch import nn
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from pl_network import PytorchLightningModule
from data_module import DataModule

from pytorch_lightning.loggers import TensorBoardLogger

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
batch_size = 4
epochs = 3

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
    trainer = Trainer(logger=logger, resume_from_checkpoint=checkpoint)
else:
    model = PytorchLightningModule(loss_fn=loss_fn, device=device)
    trainer = Trainer(logger=logger)

module = DataModule(data_path, CSV_PATH, base_path, evaluation_ratio, test_ratio, batch_size)
print("Start training model!")
model.train()
trainer.fit(model, module)
print("Finished training model!")

# trainer.test(model, module)
print("Done!")
