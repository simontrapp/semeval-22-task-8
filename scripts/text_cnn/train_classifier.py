from bert_sdr.util import load_data, DATA_PAIR_ID_1, DATA_PAIR_ID_2
from .data_set import SentenceDataset, my_collate, pad_data
from .models.sim_cnn import SimCnn
from .train import train, validate

# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import torch
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from text_cnn.util import unnormalize_scores
# from tqdm import tqdm
# import sys
# import pandas
# import numpy as np

# network_name = "sim_cnn_big"
# batch_size = 8
epochs = 1000
# lr = 0.05
es_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(training_data_path: str, sim_matrix_folder_train: str, validation_data_path:str, sim_matrix_folder_validation:str, name: str, lr: float, batch_size: int, dropout: float):
    log_path = os.path.join("models", name)
    log_path_tb = os.path.join(log_path, "tb_logs")

    x_train, y_train, pairs_train = load_data(training_data_path, True)
    pairs_train = list(zip(map(int, pairs_train[DATA_PAIR_ID_1]), map(int, pairs_train[DATA_PAIR_ID_2])))
    y_train = list((y_train - 1) / 3)

    x_validation, y_validation, pairs_validation = load_data(validation_data_path, True)
    pairs_validation = list(zip(map(int, pairs_validation[DATA_PAIR_ID_1]), map(int, pairs_validation[DATA_PAIR_ID_2])))
    y_validation = list((y_validation - 1) / 3)

    # x_train, x_test, y_train, y_test = train_test_split(pairs, y, test_size=0.2)
    # y_test = (y_test - 1) / 3
    # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)
    # y_train = (y_train - 1) / 3
    # y_validation = (y_validation - 1) / 3

    train_ds = SentenceDataset(pairs_train, y_train, sim_matrix_folder_train)
    val_ds = SentenceDataset(pairs_validation, y_validation, sim_matrix_folder_validation)
    # test_ds = SentenceDataset(x_test, y_test, sim_matrix_folder)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
    # test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)

    print("Start training model!")

    loss_fn = nn.MSELoss().to(device)
    network = SimCnn(dropout=dropout).to(device)
    summary(network, input_size=(batch_size, 2, 100, 100))
    optimizer = SGD(network.parameters(), lr=lr)

    writer = SummaryWriter(os.path.join(log_path, "tb_logs"))

    best_metric = 0
    best_index = 0
    epochs_not_improved = 0

    for t in range(epochs):
        train(network, loss_fn, optimizer, device, train_dl, writer, epoch=t)
        metric = validate(network, device, val_dl, save_predictions=False,
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
            torch.save(network.state_dict(), os.path.join(log_path, f"epoch_{t}"))

    writer.flush()
    writer.close()

    print(f"Finished training model! Best loss was {best_metric} at epoch index {best_index}")
    print("Start testing...")
    network = load_model(os.path.join(log_path, f"epoch_{best_index}"), dropout=dropout)

    validate(network, device, train_dl, save_predictions=False, ids=None,
             result_path=os.path.join(log_path, "predictions_train.csv"),
             pbar_description="Test network with train data set")
    validate(network, device, val_dl, save_predictions=False, ids=None,
             result_path=os.path.join(log_path, "predictions_validation.csv"),
             pbar_description="Test network with validation data set")
    # validate(network, device, test_dl, save_predictions=False, ids=None,
    #          result_path=os.path.join(log_path, "predictions_test.csv"),
    #          pbar_description="Test network with test data set")

    # trainer.test(model, module)
    print("Done!")


def predict_score(network: torch.nn.Module, x: torch.Tensor):
    with torch.no_grad():
        x = pad_data(torch.unsqueeze(x, 0)).to(device)
        network.eval()
        network.to(device)
        y = network(x).detach().cpu()
        return [score * 3 + 1 for score in y][0]


def load_model(model_path: str, dropout: float):
    network = SimCnn(dropout)
    network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return network.to(device)
