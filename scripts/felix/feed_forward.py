import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from util import PREPROCESSING_RESULT_PATH, PreprocessingModel, PREPROCESSING_RESULT_CSV_PATH
from torchvision import transforms

# embeddings = np.load(BERT_EMBEDDINGS_PATH)
# sorted_ids = pd.load(BERT_SORTED_PAIR_IDS_PATH)
# training_pairs = pd.load(PREPROCESSING_RESULT_CSV_PATH)
input_size = 1024


def balance_dataset(pairs_path):
    data = pd.read_csv(pairs_path)
    one = data[data['score_overall'] == 1]
    two = data[data['score_overall'] == 2]
    three = data[data['score_overall'] == 3]
    four = data[data['score_overall'] == 4]
    if len(one) < len(two):
        smallest = len(one)
    else:
        smallest = len(two)
    if len(three) < smallest:
        smallest = len(three)
    if len(four) < smallest:
        smallest = len(four)
    smallest = smallest - 1
    one = one.iloc[:smallest]
    two = two.iloc[:smallest]
    three = three.iloc[:smallest]
    four = four.iloc[:smallest]
    all = pd.concat([one, two, three, four], axis=0)
    all = all.sample(frac=1).reset_index(drop=True)
    return all


PREPROCESS_MODEL = PreprocessingModel.BERT_D2
result_path = PREPROCESSING_RESULT_PATH.format(PREPROCESS_MODEL.value)
preprocess_result_path = PREPROCESSING_RESULT_CSV_PATH.format(result_path)
balance_dataset(preprocess_result_path)


class TrainDataset(Dataset):
    def __init__(self, embeddings_path, sorted_ids_path, pairs_data, test_size, is_test, transform=None,
                 target_transform=None):
        self.training_pairs = pairs_data
        self.len = len(self.training_pairs);
        self.isTest = is_test
        self.test_size = test_size

        ts = self.len * self.test_size
        if not self.isTest:
            self.len = int(self.len - ts)
            self.embeddings = np.load(embeddings_path)
            self.sorted_ids = pd.read_csv(sorted_ids_path)
            self.training_pairs = self.training_pairs.iloc[:self.len]
        else:
            self.embeddings = np.load(embeddings_path)
            self.sorted_ids = pd.read_csv(sorted_ids_path)
            self.training_pairs = self.training_pairs.iloc[int(self.len - ts + 1):]
            self.len = int(ts - 1)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            pair_id, pair_id_1, pair_id_2, language_1, language_2, score_overall = self.training_pairs.iloc[idx]

            c = self.sorted_ids["pair_id"] == pair_id_1
            embdg1 = self.embeddings[self.sorted_ids.index[c][0]]
            c = self.sorted_ids["pair_id"] == pair_id_2
            embdg2 = self.embeddings[self.sorted_ids.index[c][0]]
            embdg = np.concatenate((embdg1, embdg2)).reshape((input_size, 1))
        except IndexError:
            print(idx)
        score_overall = round(score_overall)
        ohe = np.zeros((4))
        ohe[score_overall - 1] = 1

        if self.transform:
            embdg = self.transform(embdg)
        if self.target_transform:
            ohe = self.target_transform(ohe)
        return embdg, ohe


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = []
        in_size = input_size
        out_size = round(input_size / 2)
        # for i in range(50):
        #     if i % 10 == 0:
        #         out_size = round(out_size / 2)
        self.layers.append(
            nn.Linear(in_size, out_size)
        )
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Sigmoid())
        self.layers.append(
            nn.Linear(out_size, 100)
        )
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(100, 4))
        self.l = nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.flatten(x)
        return self.l(x)


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # print(pred.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_with_feed_forward(epochs, preprocessing_result_csv_path, embeddings_path, sorted_pair_ids_path, batch_size=64,
                            learning_rate=3e-2, test_set_size=0.2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = balance_dataset(preprocessing_result_csv_path)

    dataset_train = TrainDataset(embeddings_path, sorted_pair_ids_path, data, test_set_size,
                                 False)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
    dataset_test = TrainDataset(embeddings_path, sorted_pair_ids_path, data, test_set_size,
                                True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size)

    model = NeuralNetwork().to(device)
    print(model)

    counts = [0, 0, 0, 0, 0]
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        unique, counts1 = np.unique(y.argmax(1), return_counts=True)
        for i in range(len(unique)):
            counts[unique[i]] += counts1[i]
    print(unique, counts)

    loss_fn = nn.CrossEntropyLoss()
    print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        # print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")
