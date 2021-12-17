import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocess import load_data, preprocess_data
from data_set import SentenceDataset


class DataModule(pl.LightningDataModule):
    train_ds = None
    val_ds = None
    test_ds = None

    def __init__(self, embeddings_path: str, batch_size: int = 32):
        super(DataModule, self).__init__()
        self.batch_size = batch_size
        self.embeddings_path = embeddings_path

    def setup(self, stage):
        training_sentences, training_scores, training_ids, \
        validation_sentences, validation_scores, validation_ids, \
        test_sentences, test_scores_normalized, test_scores_raw, test_ids = load_data(
            self.embeddings_path)

        print(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences)}\n"
              f"# evaluation sentence pairs: {len(validation_sentences)}\n"
              f"# test sentence pairs: {len(test_sentences)}")
        self.train_ds = SentenceDataset(training_sentences, training_scores)
        self.val_ds = SentenceDataset(validation_sentences, validation_scores)
        self.test_ds = SentenceDataset(test_sentences, test_scores_normalized)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)
