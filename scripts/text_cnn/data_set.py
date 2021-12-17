from torch.utils.data import Dataset
import torch
from util import pad_input
import numpy as np


class SentenceDataset(Dataset):

    def __init__(self, sentences_1, sentences_2, label, encoder):
        super(SentenceDataset, self).__init__()
        max_1, max_2 = np.max([len(s) for s in sentences_1]), np.max([len(s) for s in sentences_2])
        self.max = np.max([max_1,max_2])
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.labels = label
        self.encoder = encoder
        self.len = len(sentences_1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        e1 = self.encoder.encode(self.sentences_1[idx])
        e1 = np.concatenate((e1, np.zeros((self.max - e1.shape[0], e1.shape[1]))))
        e2 = self.encoder.encode(self.sentences_2[idx])
        e2 = np.concatenate((e2, np.zeros((self.max - e2.shape[0], e2.shape[1]))))
        embedding = np.array([e1,e2])
        label = self.labels[idx]
        return torch.Tensor(embedding), torch.Tensor(label).float()
