import math
import torch
import pandas
from tqdm import tqdm
from torchmetrics.functional import accuracy, iou
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from util import unnormalize_scores


def _shared_eval_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    y = y.int()
    acc = accuracy(y_hat, y)
    _iou = iou(y_hat, y, num_classes=2)

    return y_hat, loss, acc, _iou


def train(model, loss_fn, optimizer, device, train_dataloader, writer, epoch=0, result_path=None):
    size = len(train_dataloader.dataset)
    num_batches = math.ceil(size / train_dataloader.batch_size)
    model.train()
    pbar = tqdm(train_dataloader, file=sys.stdout)
    la = np.zeros(num_batches)
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(f"Train epoch {epoch}")
        sys.stdout.flush()
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        la[batch_index] = loss.detach().cpu()
        writer.add_scalar("Loss/train", loss, batch_index + epoch * num_batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 20 == 0:
            print("")
    print(f"Epoch loss is {np.average(la)}")


def validate(model, device, dataloader, result_path=None, save_predictions=False, ids=None, pbar_description=""):
    model.eval()
    p = []
    l = []
    pbar = tqdm(dataloader, file=sys.stdout)
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(pbar_description)
        sys.stdout.flush()
        X, y = X.to(device), y.to(device)
        pred = model(X).detach()
        p.extend(pred.detach())
        l.extend(y.detach())
        if batch_index % 20 == 0:
            print("")

    p = unnormalize_scores([i.item() for i in p])
    l = unnormalize_scores([i.item() for i in l])

    mse = mean_squared_error(l, p)
    pears = pearsonr(p, l)
    mae = mean_absolute_error(l, p)
    print(f"mse: {mse}")
    print(f"pcc: {pears}")
    print(f"mae: {mae}")

    if save_predictions:
        pandas.DataFrame({'pair_id': ids, 'predictions': p}).to_csv(result_path, index=False)

    return pears[0]
