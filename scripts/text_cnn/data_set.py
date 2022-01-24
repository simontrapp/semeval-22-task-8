from torch.utils.data import Dataset
import torch
from .util import pad_input
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer, SimpleImputer


class SentenceDataset(Dataset):

    def __init__(self, dataset, label, sim_matrix_path: str):
        super(SentenceDataset, self).__init__()
        self.dataset = dataset
        self.labels = label
        self.len = len(dataset)
        self.sim_matrix_path = sim_matrix_path
        self.imputer = SimpleImputer(strategy='constant', fill_value=0.0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        e1 = self.dataset[idx]  # create_universal_sentence_encoder_embeddings(self.encoder, self.sentences_1[idx])

        # ms_0 = np.max(matrix, axis=0)
        # ms_1 = np.max(matrix, axis=1)
        # x =np.concat([ms_0, ms_1])
        label = self.labels[idx]
        sim = np.load(f"{self.sim_matrix_path}/{e1[0]}_{e1[1]}.npy")
        sim = [self.imputer.fit_transform(x) for x in sim]
        sim = torch.Tensor(sim)
        return torch.Tensor(sim), torch.Tensor([label]).float()


def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    data = pad_data(data)
    target = np.array([item[1].numpy() for item in batch])
    return [data, torch.Tensor(target)]


def pad_data(data):
    max_w = 100
    max_h = max(np.max([x.shape[2] for x in data]), 20)
    data = [np.pad(x, ((0, 0), (0, max_w - x.shape[1]), (0, max_h - x.shape[2]))) for x in data]
    return torch.Tensor(data)
