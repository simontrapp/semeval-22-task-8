from torch.utils.data import Dataset
import torch
from util import pad_input


class SentenceDataset(Dataset):

    def __init__(self, sentences, label):
        super(SentenceDataset, self).__init__()
        self.sentences = sentences
        self.labels = label
        self.len = len(sentences)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        s = self.sentences[idx]
        label = self.labels[idx]
        return torch.Tensor(s), torch.Tensor(label).float()
