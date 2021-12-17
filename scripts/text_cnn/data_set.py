from torch.utils.data import Dataset
import torch
from util import pad_input


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
        pad = pad_input([torch.Tensor([s1]),torch.tensor([s2])])
        pad = pad.resize(2, pad.size()[2], pad.size()[3])
        return pad, torch.Tensor(label).float()