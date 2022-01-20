from torch.utils.data import Dataset
import torch
from util import pad_input
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SentenceDataset(Dataset):

    def __init__(self, dataset, label):
        super(SentenceDataset, self).__init__()
        self.dataset = dataset
        self.labels = label
        self.len = len(dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        e1 = self.dataset[idx] #create_universal_sentence_encoder_embeddings(self.encoder, self.sentences_1[idx])

        # ms_0 = np.max(matrix, axis=0)
        # ms_1 = np.max(matrix, axis=1)
        # x =np.concat([ms_0, ms_1])
        label = self.labels[idx]
        return torch.Tensor(e1), torch.Tensor([label]).float()



def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    max_w = 100# max(np.max([x.shape[0] for x in data]),20)
    max_h = 100#max(np.max([x.shape[1] for x in data]),20)
    data = [np.pad(x, ((0, max_w - x.shape[0]), (0, max_h - x.shape[1]))) for x in data]
    [t.resize((1,max_w, max_h)) for t in data]
    target = np.array([item[1].numpy() for item in batch])
    return [torch.Tensor(data), torch.Tensor(target)]
