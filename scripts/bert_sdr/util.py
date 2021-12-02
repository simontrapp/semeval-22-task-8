from sklearn.metrics.pairwise import cosine_similarity


# cosine scores for all embedding pairs
def create_similarity_matrix(embeddings_1, embeddings_2):
    return cosine_similarity(X=embeddings_1, Y=embeddings_2)


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
def embeddings_to_scores(embeddings_1, embeddings_2):
    sim_matrix = create_similarity_matrix(embeddings_1, embeddings_2)
    similarity_2_to_1 = score_similarity_matrix(sim_matrix, by_row=True)
    similarity_1_to_2 = score_similarity_matrix(sim_matrix, by_row=False)
    return similarity_2_to_1, similarity_1_to_2
