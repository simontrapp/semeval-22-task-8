import numpy as np
import pandas
import os
import json
import random
import math

import torch
from torch import nn
from sentence_transformers import SentenceTransformer, SentencesDataset, \
    InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from NeuralNetworkLoss import NeuralNetworkLoss
from NeuralNetworkEvaluator import NeuralNetworkEvaluator
import sys

# folder where the web articles were downloaded to
DATA_DIR = 'data/processed/training_data'
# the file containing the links for the download script
CSV_PATH = 'data/semeval-2022_task8_train-data_batch.csv'

evaluation_ratio = 0.2  # ~20% of pairs for evaluation
create_test_set = True
test_ratio = 0.2  # ~20% of pairs for testing if desired

train_batch_size = 16  # after how many samples to update weights (default = 1)
epochs = 5
warmup_steps = 50
evaluation_steps = 300
output_path = 'models/end2end_bert'  # where the BERT model is saved


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


def process_json_to_text(path: str):
    with open(path, 'r') as file:
        article_data = json.load(file)
        return article_data['text']  # TODO: maybe append title, creation date etc., check missing values...


# STEP 1: Split data in training and evaluation set

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
sentence_pairs = pandas.read_csv(CSV_PATH).iloc[0:5]
for index, row in sentence_pairs.iterrows():
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
        sentence_1 = process_json_to_text(first_json_path)
        sentence_2 = process_json_to_text(second_json_path)
        score = lable2ohe(overall_score)
        r = random.random()
        if r < evaluation_ratio:
            evaluation_sentences_1.append(sentence_1)
            evaluation_sentences_2.append(sentence_2)
            evaluation_scores.append(score)
        elif create_test_set and r < evaluation_ratio + test_ratio:
            test_sentences_1.append(sentence_1)
            test_sentences_2.append(sentence_2)
            test_scores_normalized.append(score)
            test_scores_raw.append(overall_score)
        else:
            training_sentences_1.append(sentence_1)
            training_sentences_2.append(sentence_2)
            training_scores.append(score)

print(f"Finished reading the data!\n# training sentence pairs: {len(training_sentences_1)}\n"
      f"# evaluation sentence pairs: {len(evaluation_sentences_1)}\n"
      f"# test sentence pairs: {len(test_sentences_1)}")

# STEP 2: Train BERT

print("Starting BERT training...")
# https://www.sbert.net/examples/training/sts/README.html
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
train_examples = [InputExample(
    texts=[training_sentences_1[i], training_sentences_2[i]],
    label=training_scores[i]) for i in range(len(training_sentences_1))
]
train_dataset = SentencesDataset(train_examples, model)
# https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss
train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

# network for classifying (maps to 4 classes)
network = nn.Sequential(
    nn.Linear(in_features=1536, out_features=768),
    nn.ReLU6(),
    nn.Linear(in_features=768, out_features=384),
    nn.Dropout(0.2),
    nn.ReLU6(),
    nn.Linear(in_features=384, out_features=100),
    nn.ReLU6(),
    nn.Linear(in_features=100, out_features=4),
    nn.Dropout(0.2),
    nn.Softmax(dim=1)
)
# loss function:
# - to instantiate it expects an sbert model, a network for classification and an loss function
# - expects 2 sentences and a loss as ohe vector
# - calculates embeddings -> concatenates at dim=1 -> pushes to network -> calculates loss
train_loss = NeuralNetworkLoss(network=network, model=model, loss_fct=nn.CrossEntropyLoss())

# https://www.sbert.net/docs/training/overview.html
evaluator = NeuralNetworkEvaluator(evaluation_sentences_1, evaluation_sentences_2, evaluation_scores,
                                   network)  # evaluation.EmbeddingSimilarityEvaluator(evaluation_sentences_1, evaluation_sentences_2, evaluation_scores)
model.fit(
    train_objectives=[(train_data_loader, train_loss)],
    epochs=epochs, warmup_steps=warmup_steps, evaluator=evaluator,
    evaluation_steps=evaluation_steps, output_path=output_path)
print(f"Finished BERT training. Model saved to \"{output_path}\"!")

# STEP 3: Evaluate model on test set (optional)

if create_test_set:
    train_loss.eval()
    print("Evaluate test set...")
    embeddings_1 = model.encode(test_sentences_1, convert_to_tensor=True)
    embeddings_2 = model.encode(test_sentences_1, convert_to_tensor=True)
    c = network(torch.cat((embeddings_1, embeddings_2), dim=1))
    # model = SentenceTransformer(output_path)
    # embeddings_1 = model.encode(test_sentences_1, convert_to_tensor=True)
    # embeddings_2 = model.encode(test_sentences_2, convert_to_tensor=True)
    # cosine_scores = util.pytorch_cos_sim(embeddings_1, embeddings_2).cpu().tolist()
    unnormalized_scores = ohe2lable(c)
    print(torch.unique(unnormalized_scores, return_counts=True))
    accuracy = torch.sum(unnormalized_scores == torch.Tensor(test_scores_raw)) / len(test_scores_raw)
    # plot
    sns.set_style("whitegrid")
    plt.title('BERT end-to-end model performance')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    sns.distplot(test_scores_raw, color='red', label='Labels')
    sns.distplot(unnormalized_scores, color='blue', label='Predictions')
    errors = abs(torch.Tensor(
        test_scores_raw) - unnormalized_scores)  # [abs(test_score - predicted_score).numpy().tolist() for (test_score, predicted_score) in zip(torch.Tensor(test_scores_raw), unnormalized_scores)]
    sns.distplot(errors, color='green', label='Deviation')
    plt.legend()
    plt.savefig('./plot.pdf')
    plt.savefig('./plot.png')
    print(f"Max. Error: {torch.max(errors)}")
    print(f"Mean Error: {torch.mean(errors)}")

