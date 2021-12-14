from torch.utils.data import Dataset
import torch

class SentenceDataset(Dataset):

    def __init__(self, sentences, label, model):
        self.sentences = sentences
        self.labels = label
        self.len = len(sentences)
        self.model = model

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.Tensor(self.sentences[idx])
        y = torch.Tensor(self.labels[idx])
        # try:
        #   if x.size()[1]<3:
        #     print(x.size)
        #     print(self.sentences[idx])
        # except:
        #   print(self.sentences[idx])
        # print(x.size())
        return x.resize(1,*x.size()),y