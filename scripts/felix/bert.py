import re
from util import acronym_to_language, PREPROCESSING_RESULT_PATH
import nltk
import pandas
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy


BERT_EMBEDDINGS_PATH = "{}/bert_embeddings.npy" # PREPROCESSING_RESULT_PATH.format(model.value)
BERT_SORTED_PAIR_IDS_PATH = "{}/sorted_pair_ids.csv" #PREPROCESSING_RESULT_PATH.format(model.value)


def preprocess_bert_text(text: str, lang: str, model) -> str:
    BERT_EMBEDDINGS_PATH = f"{PREPROCESSING_RESULT_PATH.format(model.value)}/bert_embeddings.npy"
    BERT_SORTED_PAIR_IDS_PATH = f"{PREPROCESSING_RESULT_PATH.format(model.value)}/sorted_pair_ids.csv"

    tmp = re.sub(r'\w*\.\w*', '.', text)  # remove spaces around points
    language = acronym_to_language(lang)
    stop_words = []
    if language is not None:
        stop_words = stopwords.words(language)
    tmp = " ".join(re.sub(r'[^a-zA-Z]', ' ', w).lower() for w in tmp.split() if re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words)
    return tmp


def preprocess_bert(pretrained_model: str, documents_dict, model):
    embeddings_path = BERT_EMBEDDINGS_PATH.format(PREPROCESSING_RESULT_PATH.format(model.value))
    sorted_pair_ids_path = BERT_SORTED_PAIR_IDS_PATH.format(PREPROCESSING_RESULT_PATH.format(model.value))

    # nltk.download('stopwords')
    documents_df = pandas.DataFrame([(pid, preprocess_bert_text(doc['text'], doc['lang'], model)) for pid, doc in documents_dict.items()], columns=['pair_id', 'documents'])
    sbert_model = SentenceTransformer(pretrained_model)
    document_embeddings = sbert_model.encode(documents_df['documents'])
    documents_df[['pair_id']].to_csv(sorted_pair_ids_path, index=False)
    numpy.save(embeddings_path, document_embeddings)


def get_cosine_similarity_matrix(embeddings_path):
    document_embeddings = numpy.load(embeddings_path)
    return cosine_similarity(document_embeddings)


def get_euclidean_distance_matrix(embeddings_path):
    document_embeddings = numpy.load(embeddings_path)
    return euclidean_distances(document_embeddings)


def get_pair_id_list(embeddings_path):
    return pandas.read_csv(embeddings_path)


def get_index(id_list, pair_id):
    return id_list.loc[id_list['pair_id'] == pair_id].index[0]


def compare_bert_cosine(pair_id_1, pair_id_2, matrix, pair_ids):
    return matrix[get_index(pair_ids, pair_id_1)][get_index(pair_ids, pair_id_2)]


def compare_bert_euclidean(pair_id_1, pair_id_2, matrix, pair_ids):
    return matrix[get_index(pair_ids, pair_id_1)][get_index(pair_ids, pair_id_2)]
