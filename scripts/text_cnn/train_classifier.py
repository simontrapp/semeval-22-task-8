from scripts.bert_sdr.util import load_data, DATA_PAIR_ID_1, DATA_PAIR_ID_2
from .data_set import SentenceDataset, my_collate
from .models.sim_cnn import SimCnn
from .train import train, validate

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import torch
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from tqdm import tqdm
import sys
import pandas

network_name = "sim_cnn_big"
batch_size = 8
epochs = 1000
lr = 0.05
es_epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

log_path = os.path.join("..", "..", "logs", network_name)
log_path_tb = os.path.join(log_path, "tb_logs")
base_path = os.path.join("..", "..", "data")
data_path = os.path.join(base_path, "processed", "training_data")
CSV_PATH = os.path.join(base_path, "semeval-2022_task8_train-data_batch.csv")


def train_model(training_data_path: str):
    x, y, _ = load_data(training_data_path, True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

    train_ds = SentenceDataset(x_train, y_train)
    val_ds = SentenceDataset(x_validation, y_validation)
    test_ds = SentenceDataset(x_test, y_test)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)

    print("Start training model!")

    loss_fn = nn.MSELoss().to(device)
    network = SimCnn().to(device)
    summary(network, input_size=(batch_size, 1, 100, 100))
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
    validate(network, device, train_dl, save_predictions=True, ids=None,
             result_path=os.path.join(log_path, "predictions_train.csv"),
             pbar_description="Test network with train data set")
    validate(network, device, val_dl, save_predictions=False, ids=None,
             result_path=os.path.join(log_path, "predictions_validation.csv"),
             pbar_description="Test network with validation data set")

    network = SimCnn(loss_fn, device=device)
    network.load_state_dict(torch.load(os.path.join(log_path, f"epoch_{best_index}")))
    network.to(device)
    validate(network, device, test_dl, save_predictions=False, ids=None,
             result_path=os.path.join(log_path, "predictions_test.csv"),
             pbar_description="Test network with test data set")

    # trainer.test(model, module)
    print("Done!")


def predict_scores(model_path: str, test_data_path: str, output_path: str):
    network = load_model(model_path)
    x, y, pairs = load_data(test_data_path)

    data_set = SentenceDataset(x, y)
    data_loader = DataLoader(data_set, shuffle=True, batch_size=batch_size, collate_fn=my_collate)

    pred = []
    for batch_index, (X, y) in enumerate(pbar:=tqdm(data_loader, file=sys.stdout)):
        pbar.set_description("Predict scores")
        X = X.to(device)
        pred.extend(network(X).detach())
    predictions = network.predict(x)
    out_data = pandas.DataFrame(
        pairs[DATA_PAIR_ID_1].combine(pairs[DATA_PAIR_ID_2], lambda p1, p2: f"{int(p1)}_{int(p2)}"))
    out_data['prediction'] = predictions
    # noinspection PyTypeChecker
    out_data.to_csv(output_path, header=['pair_id', 'Overall'], index=False)
    # write_metrics_to_file(output_path, y, predictions)


def load_model(model_path: str):
    network = SimCnn()
    network.load_state_dict(torch.load(model_path))
    network.to(device)
