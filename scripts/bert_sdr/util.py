from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
import pandas
from sklearn import metrics
from scipy.stats import pearsonr

# data names
DATA_PAIR_ID_1 = 'pair_id_1'
DATA_PAIR_ID_2 = 'pair_id_2'
DATA_OVERALL_SCORE = 'overall_score'
DATA_BERT_SIM_21 = 'bert_sentence_similarity_2_to_1'
DATA_BERT_SIM_12 = 'bert_sentence_similarity_1_to_2'
DATA_USE_SIM_21 = 'universal_sentence_encoder_similarity_2_to_1'
DATA_USE_SIM_12 = 'universal_sentence_encoder_similarity_1_to_2'
DATA_TEXT_CNN_SCORE = 'text_cnn_score'


# if knn_imputer = True, compute missing keyword similarities from k nearest neighbors, else fill with zero
def load_data(data_path: str, knn_imputer: bool = False):
    preprocessed_data = pandas.read_csv(data_path, na_values=['NULL'])
    if knn_imputer:
        imputer = KNNImputer(n_neighbors=3)
    else:
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    preprocessed_data = pandas.DataFrame(imputer.fit_transform(preprocessed_data), columns=preprocessed_data.columns)
    x = preprocessed_data[[DATA_BERT_SIM_21, DATA_BERT_SIM_12, DATA_USE_SIM_21, DATA_USE_SIM_12, DATA_TEXT_CNN_SCORE]]
    y = preprocessed_data[DATA_OVERALL_SCORE]
    pairs = preprocessed_data[[DATA_PAIR_ID_1, DATA_PAIR_ID_2]]
    return x, y, pairs


def write_metrics_to_file(path: str, y_test, y_predictions):
    with open(f"{path}.txt", 'w') as file:
        file.write(f"Mean squared error: {metrics.mean_squared_error(y_test, y_predictions)}\n"
                   f"Mean absolute error: {metrics.mean_absolute_error(y_test, y_predictions)}\n"
                   f"Pearson correlation coefficient (r, p-value): {pearsonr(y_test, y_predictions)}")


# cosine scores for all embedding pairs
def create_cosine_similarity_matrix(embeddings_1, embeddings_2):
    return cosine_similarity(X=embeddings_1, Y=embeddings_2)


# arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
def create_arccosine_similarity_matrix(embeddings_1, embeddings_2):
    return 1 - np.arccos(cosine_similarity(embeddings_1, embeddings_2)) / np.pi


# takes a similarity matrix and computes the mean of every row/column's max value
# noinspection PyArgumentList
def score_similarity_matrix(matrix):
    matrix = np.asarray(matrix)
    avg_max_2_to_1 = np.average(matrix.max(axis=1))
    avg_max_1_to_2 = np.average(matrix.max(axis=0))
    return avg_max_2_to_1, avg_max_1_to_2


# convenience function to compute cosine similarity matrices of 2 sets of embeddings and calculate the respective scores
def embeddings_to_scores(embeddings_1, embeddings_2, similarity_type: str = 'cosine'):
    if similarity_type == 'cosine':
        sim_matrix = create_cosine_similarity_matrix(embeddings_1, embeddings_2)
    elif similarity_type == 'arccosine':
        sim_matrix = create_arccosine_similarity_matrix(embeddings_1, embeddings_2)
    else:
        raise NotImplementedError('similarity_type has to be cosine or arccosine!')
    similarity_2_to_1, similarity_1_to_2 = score_similarity_matrix(sim_matrix)
    return similarity_2_to_1, similarity_1_to_2, sim_matrix
