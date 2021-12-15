from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# data names
DATA_PAIR_ID_1 = 'pair_id_1'
DATA_PAIR_ID_2 = 'pair_id_2'
DATA_OVERALL_SCORE = 'overall_score'
DATA_BERT_SIM_21 = 'bert_sentence_similarity_2_to_1'
DATA_BERT_SIM_12 = 'bert_sentence_similarity_1_to_2'
DATA_USE_SIM_21 = 'universal_sentence_encoder_similarity_2_to_1'
DATA_USE_SIM_12 = 'universal_sentence_encoder_similarity_1_to_2'


# cosine scores for all embedding pairs
def create_cosine_similarity_matrix(embeddings_1, embeddings_2):
    return cosine_similarity(X=embeddings_1, Y=embeddings_2)


# arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
def create_arccosine_similarity_matrix(embeddings_1, embeddings_2):
    return 1 - np.arccos(cosine_similarity(embeddings_1, embeddings_2)) / np.pi


# takes a similarity matrix and computes the mean of every row's max value (by column if by_row = False)
def score_similarity_matrix(matrix, by_row: bool = True):
    n = len(matrix)
    if n == 0:
        return None
    if by_row:
        s = sum([max(row) for row in matrix])
        return s / n
    else:
        m = len(matrix[0])
        if m == 0:
            return None
        s = sum([max([matrix[row][column] for row in range(n)]) for column in range(m)])
        return s / m


# convenience function to compute cosine similarity matrices of 2 sets of embeddings and calculate the respective scores
def embeddings_to_scores(embeddings_1, embeddings_2, similarity_type: str = 'cosine'):
    if similarity_type == 'cosine':
        sim_matrix = create_cosine_similarity_matrix(embeddings_1, embeddings_2)
    elif similarity_type == 'arccosine':
        sim_matrix = create_arccosine_similarity_matrix(embeddings_1, embeddings_2)
    else:
        raise NotImplementedError('similarity_type has to be cosine or arccosine!')
    similarity_2_to_1 = score_similarity_matrix(sim_matrix, by_row=True)
    similarity_1_to_2 = score_similarity_matrix(sim_matrix, by_row=False)
    return similarity_2_to_1, similarity_1_to_2
