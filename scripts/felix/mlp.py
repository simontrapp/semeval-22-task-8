from bert import BERT_EMBEDDINGS_PATH, BERT_SORTED_PAIR_IDS_PATH
from util import PREPROCESSING_RESULT_CSV_PATH
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn.neural_network import MLPClassifier


# embeddings = np.load(BERT_EMBEDDINGS_PATH)
# sorted_ids = pd.load(BERT_SORTED_PAIR_IDS_PATH)
# training_pairs = pd.load(PREPROCESSING_RESULT_CSV_PATH)

class TrainDataset(Dataset):
    def __init__(self, embeddings_path, sorted_ids_path, pairs_path, test_size, is_test, transform=None,
                 target_transform=None):
        self.training_pairs = pd.read_csv(pairs_path).sample(frac=1).reset_index(drop=True)
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
        pair_id, pair_id_1, pair_id_2, language_1, language_2, score_overall = self.training_pairs.iloc[idx]

        c = self.sorted_ids["pair_id"] == pair_id_1
        embdg1 = self.embeddings[self.sorted_ids.index[c][0]]
        c = self.sorted_ids["pair_id"] == pair_id_2
        embdg2 = self.embeddings[self.sorted_ids.index[c][0]]
        embdg = np.concatenate((embdg1, embdg2))

        score_overall = round(score_overall)
        ohe = np.zeros((4))
        ohe[score_overall - 1] = 1

        if self.transform:
            embdg = self.transform(embdg)
        if self.target_transform:
            ohe = self.target_transform(ohe)
        return embdg, score_overall


def train_with_mlp(preprocessing_results_path, embeddings_path, sorted_pair_ids_path, hidden_layers=5, batch_size=64,
                   test_set_size=0.2):
    dataset_train = TrainDataset(embeddings_path, sorted_pair_ids_path, preprocessing_results_path, test_set_size,
                                 False)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
    dataset_test = TrainDataset(embeddings_path, sorted_pair_ids_path, preprocessing_results_path, test_set_size, True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size)

    classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=200, activation='relu', solver='sgd',
                               verbose=True,
                               random_state=762, learning_rate='invscaling', n_iter_no_change=50)
    train_x = [x for index, (x, y) in enumerate(dataset_train)]
    train_y = [y for index, (x, y) in enumerate(dataset_train)]
    classifier.fit(train_x, train_y)

    test_x = [x for index, (x, y) in enumerate(dataset_test)]
    test_y = [y for index, (x, y) in enumerate(dataset_test)]
    predictions = classifier.predict(test_x)
    score = metrics.accuracy_score(test_y, predictions)
    return score
