import numpy as np
import os
import pandas
import random
from util import lable2ohe, process_json_to_sentences, process_article_to_encoding
from sentence_transformers import SentenceTransformer
import logging as log

base_path = os.path.join("..", "..", "data")
# folder where the web articles were downloaded to
DATA_DIR = os.path.join(base_path, "processed", "training_data")
# the file containing the links for the download script
CSV_PATH = os.path.join(base_path, "semeval-2022_task8_train-data_batch.csv")

evaluation_ratio = 0.2  # ~20% of pairs for evaluation
create_test_set = True
test_ratio = 0.2  # ~20% of pairs for testing if desired


def preprocess_data(DATA_DIR, CSV_PATH, model, create_test_set=True, validation_ratio=0.2, test_ratio=0.2):
    training_ids = []
    training_sentences_1 = []
    training_sentences_2 = []
    training_scores = []

    evaluation_ids = []
    evaluation_sentences_1 = []
    evaluation_sentences_2 = []
    evaluation_scores = []

    test_ids = []
    test_sentences_1 = []
    test_sentences_2 = []
    test_scores_normalized = []
    test_scores_raw = []

    log.info("Starting reading the data")
    sentence_pairs = pandas.read_csv(CSV_PATH).iloc[:10]
    for index, row in sentence_pairs.iterrows():
        # if(index%100 == 0):
        log.info(f"[{index:>5d}/{len(sentence_pairs):>5d}]")
        pair_id = row['pair_id']
        overall_score = row['Overall']
        pair_ids = pair_id.split('_')
        if len(pair_ids) != 2:
            raise ValueError('ID Pair doesnt contain 2 IDs!')
        # read the data and create the models
        first_json_path = f"{DATA_DIR}/{pair_ids[0]}.json"
        second_json_path = f"{DATA_DIR}/{pair_ids[1]}.json"
        if os.path.exists(first_json_path) and os.path.exists(
                second_json_path):  # only add pair to data if pair was actually downloaded
            sentence_1 = process_article_to_encoding(first_json_path, model)
            sentence_2 = process_article_to_encoding(second_json_path, model)
            if len(sentence_1) == 0 or len(sentence_2) == 0:
                continue
            score = lable2ohe(overall_score)
            r = random.random()
            if r < validation_ratio:
                evaluation_ids.append(pair_id)
                evaluation_sentences_1.append(sentence_1)
                evaluation_sentences_2.append(sentence_2)
                evaluation_scores.append(score)
            elif create_test_set and r < validation_ratio + test_ratio:
                test_ids.append(pair_id)
                test_sentences_1.append(sentence_1)
                test_sentences_2.append(sentence_2)
                test_scores_normalized.append(score)
                test_scores_raw.append(overall_score)
            else:
                training_ids.append(pair_id)
                training_sentences_1.append(sentence_1)
                training_sentences_2.append(sentence_2)
                training_scores.append(score)

    return training_sentences_1, training_sentences_2, training_scores, training_ids, \
           evaluation_sentences_1, evaluation_sentences_2, evaluation_scores, evaluation_ids, \
           test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids


def save_data(training_sentences_1, training_sentences_2, training_scores, training_ids, validation_sentences_1,
              validation_sentences_2, validation_scores, validation_ids, test_sentences_1, test_sentences_2,
              test_scores_normalized,
              test_scores_raw, test_ids):
    training_ids_out = os.path.join(base_path, "embeddings", "train_ids.csv")
    pandas.DataFrame({"pair_id": training_ids}).to_csv(training_ids_out, index=False)

    training_sentences_1_out = os.path.join(base_path, "embeddings", "train_sentence_1.npy")
    np.save(training_sentences_1_out, training_sentences_1)

    training_sentences_2_out = os.path.join(base_path, "embeddings", "train_sentence_2.npy")
    np.save(training_sentences_2_out, training_sentences_2)

    training_scores_out = os.path.join(base_path, "embeddings", "train_scores.npy")
    np.save(training_scores_out, training_scores)

    validation_ids_out = os.path.join(base_path, "embeddings", "validation_ids.csv")
    pandas.DataFrame({"pair_id": validation_ids}).to_csv(validation_ids_out, index=False)

    validation_sentences_1_out = os.path.join(base_path, "embeddings", "validation_sentence_1.npy")
    np.save(validation_sentences_1_out, validation_sentences_1)

    validation_sentences_2_out = os.path.join(base_path, "embeddings", "validation_sentence_2.npy")
    np.save(validation_sentences_2_out, validation_sentences_2)

    validation_scores_out = os.path.join(base_path, "embeddings", "validation_scores.npy")
    np.save(validation_scores_out, validation_scores)

    test_ids_out = os.path.join(base_path, "embeddings", "test_ids.csv")
    pandas.DataFrame({"pair_id": test_ids}).to_csv(test_ids_out, index=False)

    test_sentences_1_out = os.path.join(base_path, "embeddings", "test_sentence_1.npy")
    np.save(test_sentences_1_out, test_sentences_1)

    test_sentences_2_out = os.path.join(base_path, "embeddings", "test_sentence_2.npy")
    np.save(test_sentences_2_out, test_sentences_2)

    test_scores_normalized_out = os.path.join(base_path, "embeddings", "test_scores_normalized.npy")
    np.save(test_scores_normalized_out, test_scores_normalized)

    test_scores_raw_out = os.path.join(base_path, "embeddings", "test_scores_raw.npy")
    np.save(test_scores_raw_out, test_scores_raw)


def load_data():
    base_path = os.path.join("..", "..", "data")

    training_ids_out = os.path.join(base_path, "embeddings", "train_ids.csv")
    training_ids = pandas.read_csv(training_ids_out)

    training_sentences_1_out = os.path.join(base_path, "embeddings", "train_sentence_1.npy")
    training_sentences_1 = np.load(training_sentences_1_out, allow_pickle=True)

    training_sentences_2_out = os.path.join(base_path, "embeddings", "train_sentence_2.npy")
    training_sentences_2 = np.load(training_sentences_2_out, allow_pickle=True)

    training_scores_out = os.path.join(base_path, "embeddings", "train_scores.npy")
    training_scores = np.load(training_scores_out, allow_pickle=True)

    validation_ids_out = os.path.join(base_path, "embeddings", "validation_ids.csv")
    validation_ids = pandas.read_csv(validation_ids_out)

    validation_sentences_1_out = os.path.join(base_path, "embeddings", "validation_sentence_1.npy")
    validation_sentences_1 = np.load(validation_sentences_1_out, allow_pickle=True)

    validation_sentences_2_out = os.path.join(base_path, "embeddings", "validation_sentence_2.npy")
    validation_sentences_2 = np.load(validation_sentences_2_out, allow_pickle=True)

    validation_scores_out = os.path.join(base_path, "embeddings", "validation_scores.npy")
    validation_scores = np.load(validation_scores_out, allow_pickle=True)

    test_ids_out = os.path.join(base_path, "embeddings", "test_ids.csv")
    test_ids = pandas.read_csv(test_ids_out)

    test_sentences_1_out = os.path.join(base_path, "embeddings", "test_sentence_1.npy")
    test_sentences_1 = np.load(test_sentences_1_out, allow_pickle=True)

    test_sentences_2_out = os.path.join(base_path, "embeddings", "test_sentence_2.npy")
    test_sentences_2 = np.load(test_sentences_2_out, allow_pickle=True)

    test_scores_normalized_out = os.path.join(base_path, "embeddings", "test_scores_normalized.npy")
    test_scores_normalized = np.load(test_scores_normalized_out, allow_pickle=True)

    test_scores_raw_out = os.path.join(base_path, "embeddings", "test_scores_raw.npy")
    test_scores_raw = np.load(test_scores_raw_out, allow_pickle=True)

    return training_sentences_1, training_sentences_2, training_scores, training_ids, \
           validation_sentences_1, validation_sentences_2, validation_scores, validation_ids, \
           test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids


if __name__ == "__main__":
    log_base_path = os.path.join("..", "..", "logs")
    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path)
    log.basicConfig(filename=os.path.join(log_base_path, "preprocess.py.log"), encoding='utf-8', level=log.DEBUG)

    bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    train_s1, train_s2, train_scores, train_ids, val_s1, val_s2, val_scores, val_ids, test_s1, test_s2, test_scores_normalized, test_scores_raw, test_ids \
        = preprocess_data(DATA_DIR, CSV_PATH, bert, create_test_set=create_test_set, validation_ratio=evaluation_ratio,
                          test_ratio=test_ratio)

    save_data(train_s1, train_s2, train_scores, train_ids, val_s1, val_s2, val_scores, val_ids, test_s1, test_s2,
              test_scores_normalized, test_scores_raw, test_ids)
