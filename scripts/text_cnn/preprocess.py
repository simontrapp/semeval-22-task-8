import numpy as np
import os
import pandas
import random
from util import lable2ohe, process_json_to_sentences, normalize_score, process_json_to_keywords
import nltk
from tqdm import tqdm
import sys
from time import time
from sklearn.metrics.pairwise import cosine_similarity

# import tensorflow_hub as hub
# from tensorflow_text import SentencepieceTokenizer

from sentence_transformers import SentenceTransformer

nltk.download('punkt')
SIMILARITY_TYPE = 'cosine'


def preprocess_data(data_dir, csv_path, result_base_path, create_test_set=True, validation_ratio=0.2, test_ratio=0.2):
    training_ids_out = os.path.join(result_base_path, "embeddings", "train_ids.csv")
    validation_ids_out = os.path.join(result_base_path, "embeddings", "validation_ids.csv")
    test_ids_out = os.path.join(result_base_path, "embeddings", "test_ids.csv")
    ids_exist = os.path.exists(training_ids_out) and os.path.exists(validation_ids_out) and os.path.exists(test_ids_out)

    training_scores_out = os.path.join(result_base_path, "embeddings", "train_scores.npy")
    validation_scores_out = os.path.join(result_base_path, "embeddings", "validation_scores.npy")
    test_scores_normalized_out = os.path.join(result_base_path, "embeddings", "test_scores_normalized.npy")
    test_scores_raw_out = os.path.join(result_base_path, "embeddings", "test_scores_raw.npy")

    if not os.path.exists(os.path.join(result_base_path, "embeddings")):
        os.makedirs(os.path.join(result_base_path, "embeddings"))

    training_ids = []
    training_scores = []

    evaluation_ids = []
    evaluation_scores = []

    test_ids = []
    test_scores_normalized = []
    test_scores_raw = []

    if not ids_exist:

        print("Starting reading the data")
        sentence_pairs = pandas.read_csv(csv_path)
        pbar = tqdm(sentence_pairs.iterrows(), total=sentence_pairs.shape[0], file=sys.stdout)
        for index, row in pbar:
            pbar.set_description("Preprocessing Data")
            pair_id = row['pair_id']
            overall_score = row['Overall']
            pair_ids = pair_id.split('_')
            if len(pair_ids) != 2:
                raise ValueError('ID Pair doesnt contain 2 IDs!')
            # read the data and create the models
            first_json_path = f"{data_dir}/{pair_ids[0]}.json"
            second_json_path = f"{data_dir}/{pair_ids[1]}.json"
            if os.path.exists(first_json_path) and os.path.exists(second_json_path):
                sentence_1 = process_json_to_sentences(first_json_path)
                sentence_2 = process_json_to_sentences(second_json_path)

                if len(sentence_1) == 0 or len(sentence_2) == 0:
                    continue
                if (len(sentence_1) > 100) or (len(sentence_2) > 100):
                    continue

                score = normalize_score(overall_score)
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

    else:
        test_ids = pandas.read_csv(test_ids_out)["pair_id"]
        evaluation_ids = pandas.read_csv(validation_ids_out)["pair_id"]
        training_ids = pandas.read_csv(training_ids_out)["pair_id"]

        training_scores = np.load(training_scores_out, allow_pickle=True)
        evaluation_scores = np.load(validation_scores_out, allow_pickle=True)
        test_scores_normalized = np.load(test_scores_normalized_out, allow_pickle=True)
        test_scores_raw = np.load(test_scores_raw_out, allow_pickle=True)

    # use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')
    use_batch_size = 16
    bert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # train dataset
    training_sentences_1, train_keywords_1, training_sentences_2, train_keywords_2 = load_sentences(training_ids,
                                                                                                    data_dir,
                                                                                                    description="Load train sentences")
    train_s1 = sentences_2_embedding(training_sentences_1, use_batch_size, bert)
    train_s2 = sentences_2_embedding(training_sentences_2, use_batch_size, bert)
    # train_k1 = sentences_2_embedding(train_keywords_1, use_batch_size, use_model)
    # train_k2 = sentences_2_embedding(train_keywords_2, use_batch_size, use_model)
    train_ds = embeddings_2_similarity(train_s1, train_s2)
    # add_keywords(train_ds, train_k1, train_k2)

    # validation dataset
    evaluation_sentences_1, evaluation_keywords_1, evaluation_sentences_2, evaluation_keywords_2 = load_sentences(
        evaluation_ids, data_dir,
        description="Load validation sentences")
    eval_s1 = sentences_2_embedding(evaluation_sentences_1, use_batch_size, bert)
    eval_s2 = sentences_2_embedding(evaluation_sentences_2, use_batch_size, bert)
    # eval_k1 = sentences_2_embedding(evaluation_keywords_1, use_batch_size, use_model)
    # eval_k2 = sentences_2_embedding(evaluation_keywords_2, use_batch_size, use_model)
    eval_ds = embeddings_2_similarity(eval_s1, eval_s2)
    # add_keywords(eval_ds, eval_k1, eval_k2)

    # test dataset
    test_sentences_1, test_keywords_1, test_sentences_2, test_keywords_2 = load_sentences(test_ids, data_dir,
                                                                                          description="Load test sentences")
    test_s1 = sentences_2_embedding(test_sentences_1, use_batch_size, bert)
    test_s2 = sentences_2_embedding(test_sentences_2, use_batch_size, bert)
    # test_k1 = sentences_2_embedding(test_keywords_1, use_batch_size, use_model)
    # test_k2 = sentences_2_embedding(test_keywords_2, use_batch_size, use_model)
    test_ds = embeddings_2_similarity(test_s1, test_s2)
    # add_keywords(test_ds, test_k1, test_k2)

    del bert

    return train_ds, training_scores, training_ids, \
           eval_ds, evaluation_scores, evaluation_ids, \
           test_ds, test_scores_normalized, test_scores_raw, test_ids


def add_keywords(dataset, keywords_1, keywords_2):
    for index, (k1, k2) in enumerate(tqdm(zip(keywords_1, keywords_2))):
        if (len(k1) == 0) or (len(k2) == 0):
            dataset[index] = np.pad(np.array([dataset[index]]), ((0, 1), (0, 0), (0, 0)))
            continue
        sim = create_similarity_matrix(k1, k2, type=SIMILARITY_TYPE)
        np.fill_diagonal(sim, 0)
        max_w = max(np.max([sim.shape[0], dataset[index].shape[0]]), 20)
        max_h = max(np.max([sim.shape[1], dataset[index].shape[1]]), 20)
        sim = np.pad(sim, ((0, max_w - sim.shape[0]), (0, max_h - sim.shape[1])))
        sen = np.pad(dataset[index], ((0, max_w - dataset[index].shape[0]), (0, max_h - dataset[index].shape[1])))
        dataset[index] = np.array([sen, sim])
        del sim
        del sen
    del keywords_1
    del keywords_2


def embeddings_2_similarity(s1, s2):
    dataset = []
    for (s1, s2) in tqdm(zip(s1, s2)):
        e1 = create_similarity_matrix(s1, s2, type=SIMILARITY_TYPE)
        np.fill_diagonal(e1, 0)
        dataset.append(e1)
    del s1
    del s2
    return dataset


def create_similarity_matrix(embeddings_1, embeddings_2, type='cosine'):
    if 'cosine' == type:
        return create_cosine_similarity_matrix(embeddings_1, embeddings_2)
    if 'arccosine' == type:
        return create_arccosine_similarity_matrix(embeddings_1, embeddings_2)
    else:
        raise Exception('e')


# arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
def create_cosine_similarity_matrix(embeddings_1, embeddings_2):
    return cosine_similarity(X=embeddings_1, Y=embeddings_2)


# arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
def create_arccosine_similarity_matrix(embeddings_1, embeddings_2):
    return 1 - np.arccos(cosine_similarity(embeddings_1, embeddings_2)) / np.pi


def sentences_2_embedding(sentences, use_batch_size, use_model):
    train_s1 = [None] * len(sentences)
    for index, x in enumerate(tqdm(sentences)):
        train_s1[index] = use_model.encode(x)# create_universal_sentence_encoder_embeddings(use_model, x, batch_size=use_batch_size)
    del sentences
    return train_s1


def load_sentences(pair_ids, data_path, description=""):
    s_1 = []
    k_1 = []
    s_2 = []
    k_2 = []

    pbar = tqdm(pair_ids, file=sys.stdout)
    for id in pbar:
        pbar.set_description(description)
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
            k_1.append(process_json_to_keywords(first_json_path))
            sentence_2 = process_json_to_sentences(
                second_json_path)  # process_article_to_encoding(second_json_path, model)
            s_2.append(sentence_2)
            k_2.append(process_json_to_keywords(second_json_path))
    return s_1, k_1, s_2, k_2


def create_universal_sentence_encoder_embeddings(model, input_sentences: list, batch_size: int = 50):
    if len(input_sentences) > batch_size:  # prevent memory error by limiting number of sentences
        res = []
        for i in range(0, len(input_sentences), batch_size):
            res.extend(model(input_sentences[i:min(i + batch_size, len(input_sentences))]).numpy())
        return np.array(res)
    else:
        return model(input_sentences).numpy()
