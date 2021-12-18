import math
import torch
import pandas
from tqdm import tqdm
from torchmetrics.functional import accuracy, iou
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def _shared_eval_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    y = y.int()
    acc = accuracy(y_hat, y)
    _iou = iou(y_hat, y, num_classes=2)

    return y_hat, loss, acc, _iou


def train(model, loss_fn, optimizer, device, train_dataloader, epoch=0, result_path=None):
    writer = SummaryWriter("../../logs/tb_logs/text-cnn")
    size = len(train_dataloader.dataset)
    num_batches = math.ceil(size / train_dataloader.batch_size)
    model.train()
    pbar = tqdm(train_dataloader, file=sys.stdout)
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(f"Train epoch {epoch}")
        sys.stdout.flush()
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        writer.add_scalar("Loss/train", loss, batch_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 20 == 0:
            print("")
    writer.flush()
    writer.close()


def validate(model, device, dataloader, result_path=None, save_predictions=False, ids=None, pbar_description=""):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = math.ceil(size / dataloader.batch_size)
    p = None
    l = None
    acc = np.zeros(num_batches)
    pbar = tqdm(dataloader, file=sys.stdout)
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(pbar_description)
        sys.stdout.flush()
        X, y = X.to(device), y.to(device)
        pred = model(X).detach()
        acc[batch_index] = accuracy(pred, y.int()).detach().cpu().numpy()
        if p is None:
            p = pred
            l = y
        else:
            p = torch.cat((p, pred), dim=0)
            l = torch.cat((l, y), dim=0)
        if batch_index % 50 == 0:
            print("")

    p = torch.argmax(p, dim=1).detach()
    l = torch.argmax(l, dim=1).detach()
    acc = np.average(acc)
    print(f"accuracy: {acc}")

    if save_predictions:
        pandas.DataFrame({'pair_id': ids, 'predictions': p.numpy().tolist()}).to_csv(result_path, index=False)
