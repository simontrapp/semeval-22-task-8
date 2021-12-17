import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocess import load_data, preprocess_data
from data_set import SentenceDataset
import numpy as np
import torch
from util import pad_input


class DataModule(pl.LightningDataModule):
    def __init__(self, embeddings_path: str, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.embeddings_path = embeddings_path

    def setup(self, stage):
        training_sentences_1, training_sentences_2, training_scores, training_ids, \
        validation_sentences_1, validation_sentences_2, validation_scores, validation_ids, \
        test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids = load_data(
            self.embeddings_path)
        print(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences_1)}\n"
              f"# evaluation sentence pairs: {len(validation_sentences_1)}\n"
              f"# test sentence pairs: {len(test_sentences_1)}")
        self.train_dl = DataLoader(SentenceDataset(training_sentences_1, training_sentences_2, training_scores),
                                   shuffle=True, batch_size=self.batch_size, collate_fn=my_collate)
        self.val_dl = DataLoader(SentenceDataset(validation_sentences_1, validation_sentences_2, validation_scores),
                                 shuffle=False, batch_size=self.batch_size, collate_fn=my_collate)
        self.test_dl = DataLoader(SentenceDataset(test_sentences_1, test_sentences_2, test_scores_normalized),
                                  shuffle=False, batch_size=self.batch_size, collate_fn=my_collate)
        self.pred_dl = self.test_dl

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def predict_dataloader(self):
        return self.pred_dl


def my_collate(batch):

    data = [item[0].numpy() for item in batch]
    # print(data.shape)
    padded = pad_input(data)
    target = np.array([item[1].numpy() for item in batch])
    target = torch.Tensor(target)
    return [padded, target]

