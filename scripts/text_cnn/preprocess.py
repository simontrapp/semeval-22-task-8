import numpy as np
import os
import pandas
import random
from util import lable2ohe, process_json_to_sentences, process_article_to_encoding
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt')


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

    print("Starting reading the data")
    sentence_pairs = pandas.read_csv(CSV_PATH)
    for index, row in sentence_pairs.iterrows():
        if index % 50 == 0:
            print(f"[{index:>5d}/{len(sentence_pairs):>5d}]")
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
            sentence_1 = process_json_to_sentences(
                first_json_path)  # process_article_to_encoding(first_json_path, model)
            sentence_2 = process_json_to_sentences(
                first_json_path)  # process_article_to_encoding(second_json_path, model)
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

    training_sentences_1, training_sentences_2 = pad_len(training_sentences_1, training_sentences_2)
    evaluation_sentences_1, evaluation_sentences_2 = pad_len(evaluation_sentences_1, evaluation_sentences_2)
    test_sentences_1, test_sentences_2 = pad_len(test_sentences_1, test_sentences_2)

    print("start calculating embeddings")
    training_sentences_1 = encode_sentences(training_sentences_1, model)
    training_sentences_2 = encode_sentences(training_sentences_2, model)
    print("calculated train embeddings")

    test_sentences_1 = encode_sentences(test_sentences_1, model)
    test_sentences_2 = encode_sentences(test_sentences_2, model)
    print("calculated validation embeddings")

    evaluation_sentences_1 = encode_sentences(evaluation_sentences_1, model)
    evaluation_sentences_2 = encode_sentences(evaluation_sentences_2, model)
    print("calculated test embeddings")

    return training_sentences_1, training_sentences_2, training_scores, training_ids, \
           evaluation_sentences_1, evaluation_sentences_2, evaluation_scores, evaluation_ids, \
           test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw, test_ids


def pad_len(sentences_1, sentences_2):
    m_1 = max([len(s) for s in sentences_1])
    m_2 = max([len(s) for s in sentences_2])
    m = max([m_1, m_2])
    sentences_1 = [s + ([""] * (m - len(s))) for s in sentences_1]
    sentences_2 = [s + ([""] * (m - len(s))) for s in sentences_2]
    return sentences_1, sentences_2


def encode_sentences(sentences, model):
    sentences = [model.encode(s, show_progress_bar=False) for s in sentences]
    return np.array(sentences)


def save_data(base_path, training_sentences_1, training_sentences_2, training_scores, training_ids,
              validation_sentences_1,
              validation_sentences_2, validation_scores, validation_ids, test_sentences_1, test_sentences_2,
              test_scores_normalized,
              test_scores_raw, test_ids):
    if not os.path.exists(os.path.join(base_path, "embeddings")):
        os.makedirs(os.path.join(base_path, "embeddings"))
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


def load_data(embeddings_path):
    base_path = embeddings_path

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
