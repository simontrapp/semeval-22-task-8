import json
import nltk
import sentence_transformers
from torch import nn
import torch
import numpy as np


def process_json_to_sentences(path: str):
    with open(path, 'r') as file:
        res = []
        article_data = json.load(file)
        title = article_data['title']
        if title is not None and title.strip() != "":
            res.append(title)
        text = article_data['text']

        if text is not None and text.strip() != "":
            text_sentences = nltk.sent_tokenize(article_data['text'])
            res.extend(text_sentences)

        kt = article_data['keywords']
        kt.extend(article_data['tags'])
        return res


def process_json_to_keywords(path: str):
    with open(path, 'r') as file:
        article_data = json.load(file)
        kt = article_data['keywords']
        kt.extend(article_data['tags'])
        return kt


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


def pad_input(data):
    max = np.max([x.shape[1] for x in data])
    padded = [np.concatenate((x, np.zeros((x.shape[0], max - x.shape[1], x.shape[2]))), axis=1) for x in data]
    return torch.Tensor(np.array(padded))
