import pandas
import os
import json
import nltk
import sentence_transformers
from util import embeddings_to_scores
import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors

# folder where the web articles were downloaded to
DATA_DIR = '../../data/processed/train'
# the file containing the links for the download script
CSV_PATH = '../../data/semeval-2022_task8_train-data_batch.csv'
# Output file for the similarity scores
OUTPUT_CSV_PATH = '../../models/sdr_sbert_document_similarities.csv'


# process article title (first) + text to a list of sentences.
def process_json_to_sentences(path: str):
    with open(path, 'r') as file:
        article_data = json.load(file)
        title = article_data['title']
        text_sentences = nltk.sent_tokenize(article_data['text'])
        res = []
        if len(title) > 0:
            res.append(title)
        res.extend(text_sentences)
        kt = article_data['keywords']
        kt.extend(article_data['tags'])
        return res, list(set(kt))


def create_sbert_embeddings(model: sentence_transformers.SentenceTransformer, sentences: list):
    return model.encode(sentences)


def create_word_embeddings(model: KeyedVectors, words: list):
    result = []
    for word in words:
        try:
            result.append(model.get_vector(word))
        except KeyError:    # if no vector for word exists
            continue
    return result


def append_output_sample(output_data: dict, pair_id_1: int, pair_id_2: int, ov_score: float, ss_2_1: float, ss_1_2: float, ks_2_1, ks_1_2):
    output_data['pair_id_1'].append(pair_id_1)
    output_data['pair_id_2'].append(pair_id_2)
    output_data['overall_score'].append(ov_score)
    output_data['sentence_similarity_2_to_1'].append(ss_2_1)
    output_data['sentence_similarity_1_to_2'].append(ss_1_2)
    output_data['keyword_similarity_2_to_1'].append(ks_2_1)
    output_data['keyword_similarity_1_to_2'].append(ks_1_2)


def compute_similarities(data_folder: str, data_csv: str, output_csv: str, sentence_embedding_model, word_embedding_model):
    output_data = {
        'pair_id_1': [],
        'pair_id_2': [],
        'overall_score': [],
        'sentence_similarity_2_to_1': [],
        'sentence_similarity_1_to_2': [],
        'keyword_similarity_2_to_1': [],
        'keyword_similarity_1_to_2': []
    }
    print("Start reading the data...")
    sentence_pairs = pandas.read_csv(data_csv)
    for index, row in sentence_pairs.iterrows():
        pair_id = row['pair_id']
        overall_score = row['Overall']
        pair_ids = pair_id.split('_')
        if len(pair_ids) != 2:
            raise ValueError('ID Pair doesnt contain 2 IDs!')
        # read the data and create the models
        first_json_path = f"{data_folder}/{pair_ids[0]}.json"
        second_json_path = f"{data_folder}/{pair_ids[1]}.json"
        # only add pair to data if pair was actually downloaded
        if os.path.exists(first_json_path) and os.path.exists(second_json_path):
            # read data
            sentences_1, keywords_1 = process_json_to_sentences(first_json_path)
            sentences_2, keywords_2 = process_json_to_sentences(second_json_path)
            # create embeddings
            sentence_embeddings_1 = create_sbert_embeddings(sentence_embedding_model, sentences_1)
            sentence_embeddings_2 = create_sbert_embeddings(sentence_embedding_model, sentences_2)
            keyword_embeddings_1 = create_word_embeddings(word_embedding_model, keywords_1)
            keyword_embeddings_2 = create_word_embeddings(word_embedding_model, keywords_2)
            # score similarities
            if len(sentence_embeddings_1) > 0 and len(sentence_embeddings_2) > 0:
                sentence_sim_2_to_1, sentence_sim_1_to_2 = embeddings_to_scores(sentence_embeddings_1, sentence_embeddings_2)
            else:
                print(f"No sentences in sample {index}!")
                continue        # abort if no sentences in one of the articles (data is messed up?)
            if len(keyword_embeddings_1) > 0 and len(keyword_embeddings_2) > 0:
                keyword_sim_2_to_1, keyword_sim_1_to_2 = embeddings_to_scores(keyword_embeddings_1, keyword_embeddings_2)
            else:
                keyword_sim_2_to_1 = None
                keyword_sim_1_to_2 = None
            # append result to output file
            append_output_sample(output_data, int(pair_ids[0]), int(pair_ids[1]), overall_score, sentence_sim_2_to_1, sentence_sim_1_to_2, keyword_sim_2_to_1, keyword_sim_1_to_2)
            print(f"Processed sample {index}")
    # save results as csv
    result_df = pandas.DataFrame(output_data)
    # noinspection PyTypeChecker
    result_df.to_csv(output_csv, index=False, na_rep='NULL')


if __name__ == "__main__":
    nltk.download('punkt')
    sbert_model = sentence_transformers.SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    sbert_model.max_seq_length = 512
    word2vec_model = api.load('conceptnet-numberbatch-17-06-300')
    compute_similarities(DATA_DIR, CSV_PATH, OUTPUT_CSV_PATH, sbert_model, word2vec_model)
