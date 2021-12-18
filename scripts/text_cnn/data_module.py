import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocess import  preprocess_data
from data_set import SentenceDataset
from sentence_transformers import SentenceTransformer


class DataModule(pl.LightningDataModule):
    train_ds = None
    val_ds = None
    test_ds = None

    def __init__(self, data_path, CSV_PATH, base_path, evaluation_ratio=0.2, test_ratio=0.2, batch_size: int = 32):
        super(DataModule, self).__init__()
        self.batch_size = batch_size

        self.CSV_PATH = CSV_PATH
        self.data_path = data_path
        self.base_path = base_path
        self.evaluation_ratio = evaluation_ratio
        self.test_ratio = test_ratio

    def setup(self, stage):
        bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        bert.max_seq_length = 512

        training_sentences_1, training_sentences_2, training_scores, training_ids, \
        evaluation_sentences_1, evaluation_sentences_2, evaluation_scores, evaluation_ids, \
        test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids \
            = preprocess_data(self.data_path, self.CSV_PATH, self.base_path, bert, create_test_set=True,
                              validation_ratio=self.evaluation_ratio, test_ratio=self.test_ratio)

        print(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences_1)}\n"
              f"# evaluation sentence pairs: {len(evaluation_sentences_1)}\n"
              f"# test sentence pairs: {len(test_sentences_1)}")
        self.train_ds = SentenceDataset(training_sentences_1, training_sentences_2, training_scores, bert)
        self.val_ds = SentenceDataset(evaluation_sentences_1, evaluation_sentences_2, evaluation_scores, bert)
        self.test_ds = SentenceDataset(test_sentences_1, test_sentences_2, test_scores_normalized, bert)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)
