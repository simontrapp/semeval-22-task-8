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


def create_universal_sentence_encoder_embeddings(model, input_sentences: list, batch_size: int = 50):
    if len(input_sentences) > batch_size:  # prevent memory error by limiting number of sentences
        res = []
        for i in range(0, len(input_sentences), batch_size):
            res.extend(model(input_sentences[i:min(i + batch_size, len(input_sentences))]))
        return res
    else:
        return model(input_sentences)


def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    max_w = max(np.max([x.shape[1] for x in data]),20)
    max_h = max(np.max([x.shape[2] for x in data]),20)
    data = [np.pad(x, ((0,0), (0, max_w - x.shape[1]), (0, max_h - x.shape[2]))) for x in data]
    target = np.array([item[1].numpy() for item in batch])
    return [torch.Tensor(data), torch.Tensor(target)]
