from bert_sdr.calculate_article_similarity import compute_similarities, add_cnn_score
from bert_sdr.train_classifier import train_random_forest, predict_scores
from text_cnn.train_classifier import train_model, load_model
from bert_sdr.util import write_metrics_to_file
import nltk
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import tensorflow as tf
# import sys
import os
import pandas as pd

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

folder = 'final'
if not os.path.exists(f'models/{folder}'):
    os.makedirs(f'models/{folder}')
# STEP 0: initialize environment
TRAINING_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_training.csv'
TRAINING_DATA_CSV_PATH_WITH_CNN = 'models/sdr_sbert_document_similarities_training_cnn.csv'
VALIDATION_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_validation.csv'
VALIDATION_DATA_CSV_PATH_WITH_CNN = 'models/sdr_sbert_document_similarities_validation_cnn.csv'
RANDOM_FOREST_FILE = f'models/{folder}/random_forest_evaluation.joblib'
EVAL_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_eval.csv'
EVAL_DATA_CSV_PATH_CNN = 'models/cnn_document_similarities_eval.csv'
SIM_MATRIX_OUTPUT_FOLDER_TRAIN = 'models/sim_matrix_train'
SIM_MATRIX_OUTPUT_FOLDER_VALIDATION = 'models/sim_matrix_validation'
SIM_MATRIX_OUTPUT_FOLDER_EVAL = 'models/sim_matrix_eval'
CNN_MODEL_PATH = 'models/cnn-ft'

nltk.download('punkt')
sbert_models = {
    'default': SentenceTransformer('models/sbert/paramulti160-overall'),
    # SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'),
    'en': SentenceTransformer('models/sbert/allmpnet160-overall'),  # SentenceTransformer('all-mpnet-base-v2'),
    'es': SentenceTransformer('models/sbert/distilmulti160-overall'),
    # SentenceTransformer('distiluse-base-multilingual-cased-v1'),
    'fr': SentenceTransformer('sentence-transformers/LaBSE')
}
# for model in sbert_models.values():
#     model.max_seq_length = 512
universal_sentence_encoder_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')
#
text_cnn = load_model(CNN_MODEL_PATH, 0.0)
# #
# sentence_pairs = pd.read_csv(VALIDATION_DATA_CSV_PATH)
# predictions = pd.read_csv('models/cnn_pred/predictions_cnn_validation.csv')
# write_metrics_to_file('models/cnn_pred/predictions.csv',sentence_pairs['overall_score'], predictions['Overall'])
# sentence_pairs['text_cnn_score'] = predictions['Overall']
# sentence_pairs.to_csv(VALIDATION_DATA_CSV_PATH_WITH_CNN, index=False, na_rep='NULL')

# add_cnn_score(TRAINING_DATA_CSV_PATH, TRAINING_DATA_CSV_PATH_WITH_CNN, 'models/cnn_pred/predictions_cnn_train.csv', text_cnn,
#               SIM_MATRIX_OUTPUT_FOLDER_TRAIN)
# add_cnn_score(VALIDATION_DATA_CSV_PATH, VALIDATION_DATA_CSV_PATH_WITH_CNN, 'models/cnn_pred/predictions_cnn_validation.csv',
#               text_cnn, SIM_MATRIX_OUTPUT_FOLDER_VALIDATION)

# # STEP 1: create training data for random forest regressor
# compute_similarities('./data/processed/train', './data/split/train.csv', TRAINING_DATA_CSV_PATH,
#                      sbert_models, universal_sentence_encoder_model, None, SIM_MATRIX_OUTPUT_FOLDER_TRAIN) #text_cnn)


# create validation data
# compute_similarities('./data/processed/train', './data/split/test.csv', VALIDATION_DATA_CSV_PATH,
#                      sbert_models, universal_sentence_encoder_model, None, SIM_MATRIX_OUTPUT_FOLDER_VALIDATION) #text_cnn)

# STEP 2: train random forest regressor
# print("start training random forest ...")
train_random_forest(TRAINING_DATA_CSV_PATH_WITH_CNN, f'models/{folder}/random_forest_test.joblib',
                    False)  # train and evaluate on test set (create visualization/data for paper)
# train_random_forest(TRAINING_DATA_CSV_PATH, RANDOM_FOREST_FILE,
#                     False)  # use the whole data for training the random forest
predict_scores(f'models/{folder}/random_forest_test.joblib', VALIDATION_DATA_CSV_PATH_WITH_CNN, f'models/{folder}/predictions-validation.csv')

# args = sys.argv[1:]
# lr = float(args[0])
# name = f"nlpprak-final-{lr}"
# train_model(TRAINING_DATA_CSV_PATH, SIM_MATRIX_OUTPUT_FOLDER_TRAIN, VALIDATION_DATA_CSV_PATH, SIM_MATRIX_OUTPUT_FOLDER_VALIDATION, name, lr, 8, 0.5)

# STEP 3: process evaluation data
compute_similarities('data/processed/eval', 'data/semeval-2022_task8_eval_data_202201.csv', EVAL_DATA_CSV_PATH,
                     sbert_models, universal_sentence_encoder_model, text_cnn, SIM_MATRIX_OUTPUT_FOLDER_EVAL, True)

# STEP 4: predict similarity scores of evaluation data
predict_scores(f'models/{folder}/random_forest_test.joblib', EVAL_DATA_CSV_PATH, f'models/{folder}/predictions-eval.csv')

# sentence_pairs = pd.read_csv(EVAL_DATA_CSV_PATH)
# # predictions = pd.read_csv('models/cnn_pred/predictions_cnn_validation.csv')
# out_data = pd.DataFrame(
#         sentence_pairs['pair_id_1'].combine(sentence_pairs['pair_id_2'], lambda p1, p2: f"{int(p1)}_{int(p2)}"))
# out_data['Overall'] = sentence_pairs['text_cnn_score']
# out_data.to_csv(f'models/{folder}/predictions-eval-cnn.csv', header=['pair_id', 'Overall'], index=False, na_rep='NULL')