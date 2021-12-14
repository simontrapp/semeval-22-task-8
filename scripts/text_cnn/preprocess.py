import numpy as np
import os
from util import preprocess_data
from sentence_transformers import SentenceTransformer


bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
base_path = os.path.join("..","..","data")
# folder where the web articles were downloaded to
DATA_DIR = os.path.join(base_path, "processed", "training_data")
# the file containing the links for the download script
CSV_PATH = os.path.join(base_path, "semeval-2022_task8_train-data_batch.csv")

evaluation_ratio = 0.2  # ~20% of pairs for evaluation
create_test_set = False
test_ratio = 0.2  # ~20% of pairs for testing if desired

training_sentences_1, training_sentences_2, training_scores, \
validation_sentences_1, validation_sentences_2, validation_scores, \
test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw \
    = preprocess_data(DATA_DIR, CSV_PATH, bert, create_test_set=create_test_set, validation_ratio=evaluation_ratio,
                      test_ratio=test_ratio)

training_sentences_1_out = os.path.join(base_path, "embeddings", "train_sentence_1.npy")
np.save(training_sentences_1_out, training_sentences_1)

training_sentences_2_out = os.path.join(base_path, "embeddings", "train_sentence_2.npy")
np.save(training_sentences_2_out, training_sentences_2)

training_scores_out = os.path.join(base_path, "embeddings", "train_scores.npy")
np.save(training_scores_out, training_scores)

validation_sentences_1_out = os.path.join(base_path, "embeddings", "validation_sentence_1.npy")
np.save(validation_sentences_1_out, validation_sentences_1)

validation_sentences_2_out = os.path.join(base_path, "embeddings", "validation_sentence_2.npy")
np.save(validation_sentences_2_out, validation_sentences_2)

validation_scores_out = os.path.join(base_path, "embeddings", "validation_scores.npy")
np.save(validation_scores_out, validation_scores)

test_sentences_1_out = os.path.join(base_path, "embeddings", "test_sentence_1.npy")
np.save(test_sentences_1_out, test_sentences_1)

test_sentences_2_out = os.path.join(base_path, "embeddings", "test_sentence_2.npy")
np.save(test_sentences_2_out, test_sentences_2)

test_scores_normalized_out = os.path.join(base_path, "embeddings", "test_scores_normalized.npy")
np.save(test_scores_normalized_out, test_scores_normalized)

test_scores_raw_out = os.path.join(base_path, "embeddings", "test_scores_raw.npy")
np.save(test_scores_raw_out, test_scores_raw)
