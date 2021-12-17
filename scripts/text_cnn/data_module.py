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
        self.train_ds = SentenceDataset(training_sentences_1, training_sentences_2, training_scores)
        self.val_ds = SentenceDataset(validation_sentences_1, validation_sentences_2, validation_scores)
        self.test_ds = SentenceDataset(test_sentences_1, test_sentences_2, test_scores_normalized)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, collate_fn=my_collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size, collate_fn=my_collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size, collate_fn=my_collate)


def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    # print(data.shape)
    padded = pad_input(data)
    target = np.array([item[1].numpy() for item in batch])
    target = torch.Tensor(target)
    return [padded, target]

