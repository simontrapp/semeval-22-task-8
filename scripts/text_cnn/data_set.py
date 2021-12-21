from torch.utils.data import Dataset
import torch
from util import pad_input
import numpy as np


class SentenceDataset(Dataset):

    def __init__(self, sentences_1, sentences_2, label, encoder):
        super(SentenceDataset, self).__init__()
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.labels = label
        self.encoder = encoder
        self.len = len(sentences_1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        e1 = self.encoder.encode(self.sentences_1[idx])
        e2 = self.encoder.encode(self.sentences_2[idx])
        m = max(e1.shape[0], e2.shape[0])
        if e1.shape[0] < m:
            e1 = np.concatenate((e1, np.zeros((m - e1.shape[0], e1.shape[1]))))
        if e2.shape[0] < m:
            e2 = np.concatenate((e2, np.zeros((m - e2.shape[0], e2.shape[1]))))

        embedding = np.array([e1, e2])
        label = self.labels[idx]
        return torch.Tensor(embedding), torch.Tensor([label]).float()


def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    m = max(np.max([x.shape[1] for x in data]), 5)
    # input can't be smaller than biggest kernel size of conv of text cnn
    padded = [np.concatenate((x, np.zeros((x.shape[0], m - x.shape[1], x.shape[2]))), axis=1) for x in data]
    padded = torch.Tensor(np.array(padded))
    target = np.array([item[1].numpy() for item in batch])
    target = torch.Tensor(target)
    return [padded, target]
