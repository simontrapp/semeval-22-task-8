from bert_sdr.calculate_article_similarity import compute_similarities
# from bert_sdr.train_classifier import train_random_forest, predict_scores
from text_cnn.train_classifier import train_model, load_model

import nltk
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import tensorflow as tf
import sys

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# STEP 0: initialize environment
TRAINING_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_training.csv'
VALIDATION_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_validation.csv'
RANDOM_FOREST_FILE = 'models/random_forest_evaluation.joblib'
EVAL_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_eval.csv'
EVAL_DATA_CSV_PATH_CNN = 'models/cnn_document_similarities_eval.csv'
SIM_MATRIX_OUTPUT_FOLDER_TRAIN = 'models/sim_matrix_train'
SIM_MATRIX_OUTPUT_FOLDER_VALIDATION = 'models/sim_matrix_validation'
# CNN_MODEL_PATH = 'models/text_cnn_final'

# nltk.download('punkt')
# sbert_models = {
#     'default': SentenceTransformer('models/sbert/paramulti160-overall'),
#     # SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'),
#     'en': SentenceTransformer('models/sbert/allmpnet160-overall'),  # SentenceTransformer('all-mpnet-base-v2'),
#     'es': SentenceTransformer('models/sbert/distilmulti160-overall'),
#     # SentenceTransformer('distiluse-base-multilingual-cased-v1'),
#     'fr': SentenceTransformer('sentence-transformers/LaBSE')
# }
# for model in sbert_models.values():
#     model.max_seq_length = 512
# universal_sentence_encoder_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')

# text_cnn = load_model(CNN_MODEL_PATH, 0.0)

# # STEP 1: create training data for random forest regressor
# compute_similarities('./data/processed/train', './data/split/train.csv', TRAINING_DATA_CSV_PATH,
#                      sbert_models, universal_sentence_encoder_model, None, SIM_MATRIX_OUTPUT_FOLDER_TRAIN) #text_cnn)


# create validation data
# compute_similarities('./data/processed/train', './data/split/test.csv', VALIDATION_DATA_CSV_PATH,
#                      sbert_models, universal_sentence_encoder_model, None, SIM_MATRIX_OUTPUT_FOLDER_VALIDATION) #text_cnn)

# STEP 2: train random forest regressor
# print("start training random forest ...")
# train_random_forest(TRAINING_DATA_CSV_PATH, 'models/random_forest_test.joblib',
#                     True)  # train and evaluate on test set (create visualization/data for paper)
# train_random_forest(TRAINING_DATA_CSV_PATH, RANDOM_FOREST_FILE,
#                     False)  # use the whole data for training the random forest
args = sys.argv[1:]
lr = float(args[0])
name = f"nlpprak_lr_{lr}"
train_model(TRAINING_DATA_CSV_PATH, SIM_MATRIX_OUTPUT_FOLDER_TRAIN, VALIDATION_DATA_CSV_PATH, SIM_MATRIX_OUTPUT_FOLDER_VALIDATION, 'nlpprak-final', lr, 8, 0.5)

# STEP 3: process evaluation data
# compute_similarities('data/processed/eval', 'data/semeval-2022_task8_eval_data_202201.csv', EVAL_DATA_CSV_PATH,
#                      sbert_models, universal_sentence_encoder_model, text_cnn, True)

# STEP 4: predict similarity scores of evaluation data
# predict_scores(RANDOM_FOREST_FILE, EVAL_DATA_CSV_PATH, 'models/predictions.csv')
