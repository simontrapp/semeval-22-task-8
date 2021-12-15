import os.path

import pandas
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, IoU
import numpy as np
from torchmetrics.functional import accuracy, iou
from sentence_transformers import SentenceTransformer
from util import MaxOverTimePooling


class PytorchLightningModule(LightningModule):
    def __init__(self, loss_fn, device="cuda"):
        super().__init__()

        self.loss_fn = loss_fn
        self.embdg_1 = bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.embdg_2 = bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.network_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3, 3)),
            nn.Dropout(0.2),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3, 3)),
            nn.Dropout(0.2),
            nn.ReLU6(),
            MaxOverTimePooling()
        )

        self.final_network = nn.Sequential(
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU6(),
            nn.Linear(in_features=300, out_features=200),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU6(),
            nn.Linear(in_features=100, out_features=4),
            nn.Dropout(0.2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_1 = x[:, 0, :, :]
        x_1 = self.network_1(x_1.resize(x_1.size()[0],1, *x_1.size()[-2:]))
        x_2 = x[:, 1, :, :]
        x_2 = self.network_2(x_2.resize(x_2.size()[0],1, *x_2.size()[-2:]))
        x = torch.cat((x_1, x_2), dim=1).resize(x_1.size()[0], 400)
        return self.final_network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        logs = {"train_loss": loss}
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
        }
        return batch_dictionary

    def configure_optimizers(self):
        return Adam(self.parameters(), 1e-3)

    def validation_step(self, batch, batch_idx):
        pred, loss, acc, iou = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_iou", iou)
        logs = {"train_loss": loss, "accuracy": acc, "iou": iou}
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            "pred": pred,
        }
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        pred, loss, acc, iou = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_iou", iou)
        logs = {"train_loss": loss, "accuracy": acc, "iou": iou}
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            "pred": pred,
        }
        return batch_dictionary

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y = y.int()
        acc = accuracy(y_hat, y)
        _iou = iou(y_hat, y, num_classes=2)

        return y_hat, loss, acc, _iou

    def validation_epoch_end(self, validation_step_outputs):
        dict = {}
        lenght = len(validation_step_outputs)
        for key in validation_step_outputs[0]["log"]:
            dict[key] = np.sum([x["log"][key].cpu().numpy() for x in validation_step_outputs]) / lenght

        return dict

    def test_epoch_end(self, validation_step_outputs):
        dict = {}
        lenght = len(validation_step_outputs)
        pred = [torch.argmax(step['pred'], dim=1) for step in validation_step_outputs]
        pred = torch.concat(pred, dim=0) if len(pred) > 1 else pred[0]
        csv = pandas.read_csv(os.path.join("..", "..", "data", "embeddings", "test_ids.csv"), index_col=False)
        csv['predictions'] = pred
        csv.to_csv(os.path.join("..", "..", "logs", "predictions.csv"), index=False)

        for key in validation_step_outputs[0]["log"]:
            dict[key] = np.sum([x["log"][key].cpu().numpy() for x in validation_step_outputs]) / lenght

        return dict
