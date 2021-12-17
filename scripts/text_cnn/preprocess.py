import math

import numpy as np
import os
import pandas
import random
from util import lable2ohe, process_json_to_sentences, process_article_to_encoding
from sentence_transformers import SentenceTransformer
import nltk
import shutil

nltk.download('punkt')


def preprocess_data(DATA_DIR, CSV_PATH, result_base_path, model, create_test_set=True, validation_ratio=0.2,
                    test_ratio=0.2):
    training_ids_out = os.path.join(result_base_path, "embeddings", "train_ids.csv")
    validation_ids_out = os.path.join(result_base_path, "embeddings", "validation_ids.csv")
    test_ids_out = os.path.join(result_base_path, "embeddings", "test_ids.csv")
    ids_exist = os.path.exists(training_ids_out) and os.path.exists(validation_ids_out) and os.path.exists(test_ids_out)

    training_sentences_1_out = os.path.join(result_base_path, "embeddings", "train_sentence_1.npy")
    training_sentences_2_out = os.path.join(result_base_path, "embeddings", "train_sentence_2.npy")
    training_scores_out = os.path.join(result_base_path, "embeddings", "train_scores.npy")
    validation_sentences_1_out = os.path.join(result_base_path, "embeddings", "validation_sentence_1.npy")
    validation_sentences_2_out = os.path.join(result_base_path, "embeddings", "validation_sentence_2.npy")
    validation_scores_out = os.path.join(result_base_path, "embeddings", "validation_scores.npy")
    test_sentences_1_out = os.path.join(result_base_path, "embeddings", "test_sentence_1.npy")
    test_sentences_2_out = os.path.join(result_base_path, "embeddings", "test_sentence_2.npy")
    test_scores_normalized_out = os.path.join(result_base_path, "embeddings", "test_scores_normalized.npy")
    test_scores_raw_out = os.path.join(result_base_path, "embeddings", "test_scores_raw.npy")
    all_files_exist = os.path.exists(training_sentences_1_out) and os.path.exists(training_sentences_2_out) \
                      and os.path.exists(training_scores_out) and os.path.exists(
        validation_sentences_1_out) and os.path.exists(validation_sentences_2_out) \
                      and os.path.exists(validation_scores_out) and os.path.exists(
        validation_scores_out and os.path.exists(test_sentences_1_out)) \
                      and os.path.exists(test_sentences_2_out) and os.path.exists(
        test_scores_normalized_out) and os.path.exists(test_scores_raw_out)

    #if ids_exist and all_files_exist:
    #    shutil.rmtree(os.path.join(result_base_path, "embeddings"))

    if not os.path.exists(os.path.join(result_base_path, "embeddings")):
        os.makedirs(os.path.join(result_base_path, "embeddings"))

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

    if not ids_exist:
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
                    evaluation_scores.append(score)
                elif create_test_set and r < validation_ratio + test_ratio:
                    test_ids.append(pair_id)
                    test_scores_normalized.append(score)
                    test_scores_raw.append(overall_score)
                else:
                    training_ids.append(pair_id)
                    training_scores.append(score)

        # save pair ids split
        pandas.DataFrame({"pair_id": training_ids}).to_csv(training_ids_out, index=False)
        pandas.DataFrame({"pair_id": evaluation_ids}).to_csv(validation_ids_out, index=False)
        pandas.DataFrame({"pair_id": test_ids}).to_csv(test_ids_out, index=False)

        # save scores
        np.save(training_scores_out, training_scores)
        np.save(validation_scores_out, evaluation_scores)
        np.save(test_scores_normalized_out, test_scores_normalized)
        np.save(test_scores_raw_out, test_scores_raw)

    if len(training_ids) == 0 or len(evaluation_ids) == 0 or len(test_ids) == 0:
        test_ids = pandas.read_csv(test_ids_out)["pair_id"]
        evaluation_ids = pandas.read_csv(validation_ids_out)["pair_id"]
        training_ids = pandas.read_csv(training_ids_out)["pair_id"]

        training_scores = np.load(training_scores_out, allow_pickle=True)
        evaluation_scores = np.load(validation_scores_out, allow_pickle=True)
        test_scores_normalized = np.load(test_scores_normalized_out, allow_pickle=True)
        test_scores_raw = np.load(test_scores_raw_out, allow_pickle=True)

    print("start calculating train embeddings")

    training_sentences_1, training_sentences_2 = load_sentences(training_ids, DATA_DIR)
    #training_sentences_1, training_sentences_2 = pad_len(training_sentences_1, training_sentences_2)
    training_sentences_1 = encode_sentences(training_sentences_1, model, training_sentences_1_out)
    training_sentences_2 = encode_sentences(training_sentences_2, model, training_sentences_2_out)
    print("calculated train embeddings")

    evaluation_sentences_1, evaluation_sentences_2 = load_sentences(evaluation_ids, DATA_DIR)
    #evaluation_sentences_1, evaluation_sentences_2 = pad_len(evaluation_sentences_1, evaluation_sentences_2)
    evaluation_sentences_1 = encode_sentences(evaluation_sentences_1, model, validation_sentences_1_out)
    evaluation_sentences_2 = encode_sentences(evaluation_sentences_2, model, validation_sentences_2_out)
    print("calculated validation embeddings")

    test_sentences_1, test_sentences_2 = load_sentences(test_ids, DATA_DIR)
    #test_sentences_1, test_sentences_2 = pad_len(test_sentences_1, test_sentences_2)
    test_sentences_1 = encode_sentences(test_sentences_1, model, test_sentences_1_out)
    test_sentences_2 = encode_sentences(test_sentences_2, model, test_sentences_2_out)
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


def load_sentences(pair_ids, data_path):
    s_1 = []
    s_2 = []
    for id in pair_ids:
        pair_ids = id.split('_')
        if len(pair_ids) != 2:
            raise ValueError('ID Pair doesnt contain 2 IDs!')
        # read the data and create the models
        first_json_path = f"{data_path}/{pair_ids[0]}.json"
        second_json_path = f"{data_path}/{pair_ids[1]}.json"
        if os.path.exists(first_json_path) and os.path.exists(
                second_json_path):  # only add pair to data if pair was actually downloaded
            sentence_1 = process_json_to_sentences(
                first_json_path)  # process_article_to_encoding(first_json_path, model)
            s_1.append(sentence_1)
            sentence_2 = process_json_to_sentences(
                second_json_path)  # process_article_to_encoding(second_json_path, model)
            s_2.append(sentence_2)
    return s_1, s_2


def encode_sentences(sentences, model, result_path):
    print(f"start calculation for {len(sentences)} sentences to {result_path}")
    batch_size = 100
    if os.path.exists(result_path):
        loaded = np.load(result_path, allow_pickle=True)
        print(f"loaded {loaded.shape[0]} from {result_path}")
    else:
        loaded = None
        print("Nothing loaded. start calculating")

    to_process = sentences[loaded.shape[0] if loaded is not None else 0:]
    calculated = []
    i = 0
    for index, row in enumerate(to_process):
        calculated.append(model.encode(row))
        i += 1
        if i == batch_size:
            if loaded is None:
                loaded = np.array(calculated)
            else:
                loaded = np.concatenate((loaded, np.array(calculated)))
            print(f"calculated {loaded.shape[0]}/{len(sentences)}")
            np.save(result_path, loaded)
            calculated = []
            i = 0
    if len(calculated) > 0:
        if loaded is None:
            loaded = np.array(calculated)
        else:
            loaded = np.concatenate((loaded, np.array(calculated)))
    np.save(result_path, loaded)

    assert len(sentences) == loaded.shape[0]
    return loaded


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
