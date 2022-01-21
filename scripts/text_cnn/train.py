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


def train(model, loss_fn, optimizer, device, train_dataloader, writer, epoch=0, result_path=None):
    size = len(train_dataloader.dataset)
    num_batches = math.ceil(size / train_dataloader.batch_size)
    model.train()
    pbar = tqdm(train_dataloader, file=sys.stdout)
    total_loss = 0
    for batch_index, (X, y) in enumerate(pbar):
        pbar.set_description(f"Train epoch {epoch}")
        sys.stdout.flush()
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        writer.add_scalar("Loss/train", float(loss), batch_index + epoch * num_batches)
    print(f"Epoch loss is {total_loss / num_batches}")


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
