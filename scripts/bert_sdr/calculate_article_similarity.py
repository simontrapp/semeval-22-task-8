import os
import json
import nltk
from .util import DATA_PAIR_ID_1, DATA_PAIR_ID_2, DATA_OVERALL_SCORE, DATA_BERT_SIM_21, DATA_BERT_SIM_12, \
    DATA_USE_SIM_21, DATA_USE_SIM_12, DATA_TEXT_CNN_SCORE, embeddings_to_scores
import pandas as pd
import torch
import sentence_transformers
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
from tensorflow_text import SentencepieceTokenizer
from tqdm import tqdm
import numpy as np

# folder where the web articles were downloaded to
DATA_DIR = 'data/processed/train'
# the file containing the links for the download script
CSV_PATH = 'data/semeval-2022_task8_train-data_batch.csv'
# Output file for the similarity scores
OUTPUT_CSV_PATH = 'models/sdr_sbert_document_similarities.csv'


# process article title (first) + text to a list of sentences.
def process_json_to_sentences(path: str, filter_sentence_length: bool = False, min_sentence_len: int = 15,
                              max_sentence_len: int = 1500):
    with open(path, 'r') as file:
        article_data = json.load(file)
        title = article_data['title']
        text_sentences = nltk.sent_tokenize(article_data['text'])
        res = []
        if len(title) > 0:
            res.append(title)
        res.extend(text_sentences)
        del article_data, title
        if filter_sentence_length:
            tmp = []
            for sentence in res:
                if len(sentence) > min_sentence_len:
                    if len(sentence) < max_sentence_len:
                        tmp.append(sentence)
                    else:
                        tmp.extend([sentence[i:min(i + max_sentence_len, len(sentence))] for i in
                                    range(0, len(sentence), max_sentence_len)])
            return tmp
        return res


def create_sbert_embeddings(sbert_models: dict, sentences: list, language_1: str, language_2: str):
    with torch.no_grad():  # avoid changes to the model
        if language_1 == language_2 and language_1 in sbert_models:
            return sbert_models[language_1].encode(sentences, batch_size=4)
        else:
            return sbert_models['default'].encode(sentences, batch_size=4)


def create_universal_sentence_encoder_embeddings(use_model, input_sentences: list, batch_size: int = 50):
    if len(input_sentences) > batch_size:  # prevent memory error by limiting number of sentences
        res = []
        for i in range(0, len(input_sentences), batch_size):
            res.extend(use_model(input_sentences[i:min(i + batch_size, len(input_sentences))]))
        return res
    else:
        return use_model(input_sentences)


def append_output_sample(output_data: dict, pair_id_1: int, pair_id_2: int, ov_score: float, ss_2_1: float,
                         ss_1_2: float, us_2_1: float, us_1_2: float, tcs: float = 0.0):
    output_data[DATA_PAIR_ID_1].append(pair_id_1)
    output_data[DATA_PAIR_ID_2].append(pair_id_2)
    output_data[DATA_OVERALL_SCORE].append(ov_score)
    output_data[DATA_BERT_SIM_21].append(ss_2_1)
    output_data[DATA_BERT_SIM_12].append(ss_1_2)
    output_data[DATA_USE_SIM_21].append(us_2_1)
    output_data[DATA_USE_SIM_12].append(us_1_2)
    output_data[DATA_TEXT_CNN_SCORE].append(tcs)


def save_sim_matrix(use_sim_matrix, sbert_sim_matrix, path: str):
    arr = np.array([sbert_sim_matrix, use_sim_matrix], dtype='float32')
    np.save(path, arr)


def compute_similarities(data_folder: str, data_csv: str, output_csv: str, sbert_embedding_model: dict,
                         use_embedding_model, similarity_matrix_path: str, is_eval: bool = False):
    output_data = {
        DATA_PAIR_ID_1: [],
        DATA_PAIR_ID_2: [],
        DATA_OVERALL_SCORE: [],
        DATA_BERT_SIM_21: [],
        DATA_BERT_SIM_12: [],
        DATA_USE_SIM_21: [],
        DATA_USE_SIM_12: [],
        DATA_TEXT_CNN_SCORE: []
    }
    print("Start reading the data...")
    sentence_pairs = pd.read_csv(data_csv)
    # noinspection PyBroadException
    try:
        for index, row in tqdm(sentence_pairs.iterrows()):
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
                sentences_1 = process_json_to_sentences(first_json_path, True)
                sentences_2 = process_json_to_sentences(second_json_path, True)
                # score similarities
                if len(sentences_1) > 0 and len(sentences_2) > 0:
                    if len(sentences_1) > 100 or len(sentences_2) > 100:
                        continue
                    # create embeddings
                    sbert_embeddings_1 = create_sbert_embeddings(sbert_embedding_model, sentences_1, row['url1_lang'],
                                                                 row['url2_lang'])
                    sbert_embeddings_2 = create_sbert_embeddings(sbert_embedding_model, sentences_2, row['url1_lang'],
                                                                 row['url2_lang'])
                    use_embeddings_1 = create_universal_sentence_encoder_embeddings(use_embedding_model, sentences_1)
                    use_embeddings_2 = create_universal_sentence_encoder_embeddings(use_embedding_model, sentences_2)
                    assert len(sentences_1) == len(sbert_embeddings_1)
                    assert len(sentences_1) == len(use_embeddings_1)
                    assert len(sentences_2) == len(sbert_embeddings_2)
                    assert len(sentences_2) == len(use_embeddings_2)
                    # create scores
                    sbert_sim_2_to_1, sbert_sim_1_to_2, sbert_sim_matrix = embeddings_to_scores(sbert_embeddings_1,
                                                                                                sbert_embeddings_2,
                                                                                                similarity_type='cosine')
                    use_sim_2_to_1, use_sim_1_to_2, use_sim_matrix = embeddings_to_scores(use_embeddings_1,
                                                                                          use_embeddings_2,
                                                                                          similarity_type='arccosine')
                    # text_cnn_score =    # TODO: implement felix's model
                    # append result to output file
                    pair_id_1 = int(pair_ids[0])
                    pair_id_2 = int(pair_ids[1])
                    append_output_sample(output_data, pair_id_1, pair_id_2, overall_score, sbert_sim_2_to_1,
                                         sbert_sim_1_to_2, use_sim_2_to_1, use_sim_1_to_2)  # , text_cnn_score)
                    save_sim_matrix(use_sim_matrix, sbert_sim_matrix, f"{similarity_matrix_path}/{pair_id}")
                    # print(f"Processed {index}: #sentences_1: {len(sentences_1)}, #sentences_2: {len(sentences_2)}")
                    del sentences_1, sentences_2, sbert_embeddings_1, sbert_embeddings_2, use_embeddings_1, use_embeddings_2
                elif is_eval:
                    print("EVAL ERROR: sentence amount of one article is zero!")
                    raise ValueError('eval error: sentence amount of one article is zero!')
            elif is_eval:
                print("EVAL ERROR: article file doesn't exist!")
                raise ValueError('eval error: file doesnt exist!')
    except AssertionError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error occurred! Saving results...\n Exception: {e}")
    finally:
        # save results as csv
        result_df = pd.DataFrame(output_data)
        # noinspection PyTypeChecker
        result_df.to_csv(output_csv, index=False, na_rep='NULL')


if __name__ == "__main__":
    nltk.download('punkt')
    sbert_models = {
        'default': sentence_transformers.SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu'),
        'en': sentence_transformers.SentenceTransformer('all-mpnet-base-v2', device='cpu'),
        'es': sentence_transformers.SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cpu'),
        'fr': sentence_transformers.SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
    }
    for model in sbert_models.values():
        model.max_seq_length = 512
    universal_sentence_encoder_model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')
    compute_similarities(DATA_DIR, CSV_PATH, OUTPUT_CSV_PATH, sbert_models, universal_sentence_encoder_model)
