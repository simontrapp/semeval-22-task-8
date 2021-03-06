import os
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import time

from sentence_transformers import SentenceTransformer
from .data_set import SentenceDataset, my_collate
from .preprocess import preprocess_data
from .models.lstm import lstm_model
from .models.sim_cnn import SimCnn
from .train import train, validate
import torch

if __name__ == "__main__":
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
    log_path = os.path.join("..", "..", "logs", "sim-cnn-big")
    log_path_tb = os.path.join(log_path, "tb_logs")
    log_name = "sim_cnn_big"
    base_path = os.path.join("..", "..", "data")
    data_path = os.path.join(base_path, "processed", "training_data")
    CSV_PATH = os.path.join(base_path, "semeval-2022_task8_train-data_batch.csv")

    evaluation_ratio = 0.2  # ~20% of pairs for evaluation
    create_test_set = True
    test_ratio = 0.2  # ~20% of pairs for testing if desired

    preprocess = True

    # training parameters
    batch_size = 8
    epochs = 1000
    lr = 0.05

    es_epochs = 100
    """
    ---------------------------------------------------------------------
    |                                                                   |
    |                            Code                                   |
    |                                                                   |
    ---------------------------------------------------------------------
    """

    # bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    # use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')

    train_ds, training_scores, training_ids, \
    eval_ds, evaluation_scores, evaluation_ids, \
    test_ds, test_scores_normalized, test_scores_raw, test_ids \
        = preprocess_data(data_path, CSV_PATH, base_path, create_test_set=True, validation_ratio=evaluation_ratio,
                          test_ratio=test_ratio)

    print(f"Finished reading the data!\n# training sentence pairs: {len(train_ds)}\n"
          f"# evaluation sentence pairs: {len(eval_ds)}\n"
          f"# test sentence pairs: {len(test_ds)}")
    train_ds = SentenceDataset(train_ds, training_scores)
    val_ds = SentenceDataset(eval_ds, evaluation_scores)
    test_ds = SentenceDataset(test_ds, test_scores_normalized)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=my_collate)

    loss_fn = nn.MSELoss().to(device)
    network = SimCnn(loss_fn, device=device).to(device)
    summary(network, input_size=(batch_size, 1, 100, 100))
    optimizer = SGD(network.parameters(), lr=lr)

    print("Start training model!")

    writer = SummaryWriter(os.path.join(log_path, "tb_logs"))

    best_metric = 0
    best_index = 0
    epochs_not_improved = 0

    for t in range(epochs):
        train(network, loss_fn, optimizer, device, train_dl, writer, epoch=t)
        metric = validate(network, device, val_dl, save_predictions=True,
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
    validate(network, device, train_dl, save_predictions=True, ids=training_ids,
             result_path=os.path.join(log_path, "predictions_train.csv"),
             pbar_description="Test network with train data set")
    validate(network, device, val_dl, save_predictions=True, ids=evaluation_ids,
             result_path=os.path.join(log_path, "predictions_validation.csv"),
             pbar_description="Test network with validation data set")

    network = SimCnn(loss_fn, device=device)
    network.load_state_dict(torch.load(os.path.join(log_path, f"epoch_{best_index}")))
    network.to(device)
    validate(network, device, test_dl, save_predictions=True, ids=test_ids,
             result_path=os.path.join(log_path, "predictions_test.csv"),
             pbar_description="Test network with test data set")

    # trainer.test(model, module)
    print("Done!")
