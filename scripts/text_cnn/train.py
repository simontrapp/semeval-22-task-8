import math
import torch
from sklearn import metrics
import pandas
import time
from tqdm import tqdm
from torchmetrics.functional import accuracy, iou


def _shared_eval_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    y = y.int()
    acc = accuracy(y_hat, y)
    _iou = iou(y_hat, y, num_classes=2)

    return y_hat, loss, acc, _iou

def train(model, loss_fn, optimizer, device, train_dataloader, epoch=0, result_path=None):
    size = len(train_dataloader.dataset)
    num_batches = math.ceil(size / train_dataloader.batch_size)
    model.train()
    loss_sum = 0
    pbar = tqdm(train_dataloader)
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(f"Train epoch {epoch}")
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        #if batch_index % 1 == 0 or batch_index + 1 == num_batches:
            #print(f"loss: {loss_sum / (batch_index + 1):>7f}  [{batch_index + 1:>5d}/{num_batches:>5d}]")


def validate(model, device, dataloader, result_path=None, save_predictions=False, ids=None, pbar_description=""):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = math.ceil(size / dataloader.batch_size)
    p = None
    l = None
    pbar = tqdm(dataloader)
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(pbar_description)
        X, y = X.to(device), y.to(device)
        pred = model(X)
        if p is None:
            p = pred
            l = y
        else:
            p = torch.cat((p, pred), dim=0)
            l = torch.cat((l, y), dim=0)

    p = torch.argmax(p, dim=1).detach()
    l = torch.argmax(l, dim=1).detach()
    #print(metrics.classification_report(l, p))
    acc = accuracy(p, l)
    print(f"accuracy: {accuracy}")

    if save_predictions:
        pandas.DataFrame({'pair_id': ids, 'predictions': p.numpy().tolist()}).to_csv(result_path, index=False)
