import os

import numpy
import pandas
from bert import compare_bert_cosine, compare_bert_euclidean, get_cosine_similarity_matrix, get_euclidean_distance_matrix, get_pair_id_list
from doc2vec import compare_d2v
from util import COMPARISON_MODEL, PREPROCESSING_RESULT_CSV_PATH, ComparisonModel, COMPARISON_RESULT_CSV_PATH, COMPARISON_RESULT_PATH, PREPROCESSING_MODEL, PreprocessingModel


# take a score from a model in the specified range and convert it to integers in [1, 2, 3, 4]
def score_to_int(min_val: float, max_val: float, val: float) -> float:
    return (val - min_val) / (max_val - min_val) * 3 + 1


os.makedirs(COMPARISON_RESULT_PATH, exist_ok=True)
PREP_CSV = pandas.read_csv(PREPROCESSING_RESULT_CSV_PATH)
SUCCESSFUL_DATA = pandas.DataFrame(columns=['pair_id', 'pair_id_1', 'pair_id_2', 'language_1', 'language_2', 'score_overall', 'score_model'])


IS_BERT = (PREPROCESSING_MODEL == PreprocessingModel.BERT_D1 or PREPROCESSING_MODEL == PreprocessingModel.BERT_D2 or PREPROCESSING_MODEL == PreprocessingModel.BERT_MLM or PREPROCESSING_MODEL == PreprocessingModel.BERT_MPNET)
if IS_BERT:
    pair_id_list = get_pair_id_list()
    if COMPARISON_MODEL == ComparisonModel.COSINE_SIM:
        cs_matrix = get_cosine_similarity_matrix()
    elif COMPARISON_MODEL == ComparisonModel.EUCLIDEAN_DIST:
        ed_matrix = get_euclidean_distance_matrix()
        ed_matrix_max = numpy.amax(ed_matrix)

# write results to file
for index, row in PREP_CSV.iterrows():
    pair_id = row['pair_id']
    pair_id_1 = row['pair_id_1']
    pair_id_2 = row['pair_id_2']
    language_1 = row['language_1']
    language_2 = row['language_2']
    score_overall = row['score_overall']
    similarity_score = None
    if COMPARISON_MODEL == ComparisonModel.DOC2VEC:
        similarity_score = score_to_int(0, 1, compare_d2v(pair_id))
    elif COMPARISON_MODEL == ComparisonModel.COSINE_SIM:
        if IS_BERT:
            similarity_score = score_to_int(0, 1, compare_bert_cosine(pair_id_1, pair_id_2, cs_matrix, pair_id_list))
    elif COMPARISON_MODEL == ComparisonModel.EUCLIDEAN_DIST:
        if IS_BERT:
            similarity_score = 5 - score_to_int(0, ed_matrix_max, compare_bert_euclidean(pair_id_1, pair_id_2, ed_matrix, pair_id_list))
    else:
        raise Exception('Not implemented!')

    # write successful pairs to file
    SUCCESSFUL_DATA = SUCCESSFUL_DATA.append(pandas.DataFrame({
        'pair_id': [pair_id],
        'pair_id_1': [pair_id_1],
        'pair_id_2': [pair_id_2],
        'language_1': [language_1],
        'language_2': [language_2],
        'score_overall': [score_overall],
        'score_model': [similarity_score]
    }), ignore_index=True)

SUCCESSFUL_DATA.to_csv(COMPARISON_RESULT_CSV_PATH, index=False)
