import json
import nltk
import sentence_transformers
from torch import nn
import pandas
import os
import random
import torch
import numpy as np


def process_json_to_sentences(path: str):
    with open(path, 'r') as file:
        article_data = json.load(file)
        title = article_data['title']
        text_sentences = nltk.sent_tokenize(article_data['text'])
        res = []
        # if len(title) > 0:
        #     res.append(title)
        res.extend(text_sentences)
        # if len(res) == 0:
        #   print(article_data['text'], article_data)
        #   raise Exception()
        kt = article_data['keywords']
        kt.extend(article_data['tags'])
        return res


def process_article_to_sentence(article):
    return nltk.sent_tokenize(article)

def process_article_to_encoding(path: str, model):
    sentences = process_json_to_sentences(path)
    return model.encode(sentences)

def create_sbert_embeddings(model: sentence_transformers.SentenceTransformer, sentence):
    sentences = process_article_to_sentence(sentence)
    return model.encode(sentences)


def process_json_to_text(path: str):
    with open(path, 'r') as file:
        article_data = json.load(file)
        return article_data['text']  # TODO: maybe append title, creation date etc., check missing values...


class MaxOverTimePooling(nn.Module):

    def __init__(self):
        super(MaxOverTimePooling, self).__init__()

    def forward(self, x):
        pool = nn.MaxPool2d(kernel_size=(x.size()[2], x.size()[3]))
        return pool(x)


# convert the 1-4 score to 0-1 for BERT
def normalize_score(semeval_score: float):
    return (semeval_score - 1) / 3


# convert the 1-4 score to an one hot encoded vector
def lable2ohe(semeval_score: float):
    ohe = np.zeros((4))
    ohe[int(round(semeval_score)) - 1] = 1
    return ohe


# return one hot encoded vector to 1-4 scores
def ohe2lable(ohe):
    return torch.argmax(ohe, dim=1) + 1


# return 0-1 scores to 1-4 form
def unnormalize_scores(scores: list):
    return [s * 3 + 1 for s in scores]  # TODO: convert to integer scores ( round() )


def preprocess_data(DATA_DIR, CSV_PATH, model, create_test_set=True, validation_ratio=0.2, test_ratio=0.2):
    training_sentences_1 = []
    training_sentences_2 = []
    training_scores = []

    evaluation_sentences_1 = []
    evaluation_sentences_2 = []
    evaluation_scores = []

    test_sentences_1 = []
    test_sentences_2 = []
    test_scores_normalized = []
    test_scores_raw = []

    print("Starting reading the data")
    sentence_pairs = pandas.read_csv(CSV_PATH).iloc[:50]
    for index, row in sentence_pairs.iterrows():
        # if(index%100 == 0):
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
            sentence_1 = process_article_to_encoding(first_json_path,model)
            sentence_2 = process_article_to_encoding(second_json_path, model)
            if len(sentence_1) == 0 or len(sentence_2)==0:
              continue
            score = lable2ohe(overall_score)
            r = random.random()
            if r < validation_ratio:
                evaluation_sentences_1.append(sentence_1)
                evaluation_sentences_2.append(sentence_2)
                evaluation_scores.append(score)
            elif create_test_set and r < validation_ratio + test_ratio:
                test_sentences_1.append(sentence_1)
                test_sentences_2.append(sentence_2)
                test_scores_normalized.append(score)
                test_scores_raw.append(overall_score)
            else:
                training_sentences_1.append(sentence_1)
                training_sentences_2.append(sentence_2)
                training_scores.append(score)

    return training_sentences_1, training_sentences_2,training_scores,\
           evaluation_sentences_1, evaluation_sentences_2, evaluation_scores,\
           test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw


def load_data():
    base_path = os.path.join("..","..","data")
    training_sentences_1_out = os.path.join(base_path, "embeddings", "train_sentence_1.npy")
    training_sentences_1 = np.load(training_sentences_1_out, allow_pickle=True)

    training_sentences_2_out = os.path.join(base_path, "embeddings", "train_sentence_2.npy")
    training_sentences_2 = np.load(training_sentences_2_out, allow_pickle=True)

    training_scores_out = os.path.join(base_path, "embeddings", "train_scores.npy")
    training_scores = np.load(training_scores_out, allow_pickle=True)

    validation_sentences_1_out = os.path.join(base_path, "embeddings", "validation_sentence_1.npy")
    validation_sentences_1 = np.load(validation_sentences_1_out, allow_pickle=True)

    validation_sentences_2_out = os.path.join(base_path, "embeddings", "validation_sentence_2.npy")
    validation_sentences_2 = np.load(validation_sentences_2_out, allow_pickle=True)

    validation_scores_out = os.path.join(base_path, "embeddings", "validation_scores.npy")
    validation_scores = np.load(validation_scores_out, allow_pickle=True)

    test_sentences_1_out = os.path.join(base_path, "embeddings", "test_sentence_1.npy")
    test_sentences_1 = np.load(test_sentences_1_out, allow_pickle=True)

    test_sentences_2_out = os.path.join(base_path, "embeddings", "test_sentence_2.npy")
    test_sentences_2 = np.load(test_sentences_2_out, allow_pickle=True)

    test_scores_normalized_out = os.path.join(base_path, "embeddings", "test_scores_normalized.npy")
    test_scores_normalized = np.load(test_scores_normalized_out, allow_pickle=True)

    test_scores_raw_out = os.path.join(base_path, "embeddings", "test_scores_raw.npy")
    test_scores_raw = np.load(test_scores_raw_out, allow_pickle=True)

    return training_sentences_1, training_sentences_2, training_scores, \
           validation_sentences_1, validation_sentences_2, validation_scores, \
           test_sentences_1, test_sentences_2, test_scores_normalized, test_scores_raw
