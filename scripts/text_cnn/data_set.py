from torch.utils.data import Dataset
import torch


class SentenceDataset(Dataset):

    def __init__(self, sentences_1, sentences_2, label):
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.labels = label
        self.len = len(sentences_1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        s1 = self.sentences_1[idx]
        s2 = self.sentences_2[idx]
        label = self.labels[idx]
        return torch.tensor([s1, s2]), torch.Tensor(label).float()