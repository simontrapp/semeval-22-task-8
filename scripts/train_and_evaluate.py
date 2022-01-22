import nltk
import sentence_transformers
import tensorflow_hub as hub

from bert_sdr.calculate_article_similarity import compute_similarities
from bert_sdr.train_classifier import train_random_forest, predict_scores
from text_cnn.train_classifier import train_model, predict_scores as predict_scores_cnn

# STEP 0: initialize environment
TRAINING_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities.csv'
RANDOM_FOREST_FILE = 'models/random_forest_evaluation.joblib'
EVAL_DATA_CSV_PATH = 'models/sdr_sbert_document_similarities_eval.csv'
EVAL_DATA_CSV_PATH_CNN = 'models/cnn_document_similarities_eval.csv'

nltk.download('punkt')
sbert_models = {  # TODO: implement Dirk's fine-tuned models
    'default': sentence_transformers.SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu'),
    'en': sentence_transformers.SentenceTransformer('all-mpnet-base-v2', device='cpu'),
    'es': sentence_transformers.SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cpu'),
    'fr': sentence_transformers.SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
}
for model in sbert_models.values():
    model.max_seq_length = 512
universal_sentence_encoder_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')
# TODO: implement felix's textCNN model (also in calculate_article_similarity.py)

# STEP 1: create training data for random forest regressor
compute_similarities('../data/processed/train', '../data/semeval-2022_task8_train-data_batch.csv', TRAINING_DATA_CSV_PATH,
                     sbert_models, universal_sentence_encoder_model)

# STEP 2: train random forest regressor
train_random_forest(TRAINING_DATA_CSV_PATH, '../models/random_forest_test.joblib',
                    True)  # train and evaluate on test set (create visualization/data for paper)
train_random_forest(TRAINING_DATA_CSV_PATH, RANDOM_FOREST_FILE,
                    False)  # use the whole data for training the random forest

train_model(TRAINING_DATA_CSV_PATH)

# STEP 3: process evaluation data
compute_similarities('../data/processed/eval', '../data/semeval-2022_task8_eval_data_202201.csv', EVAL_DATA_CSV_PATH,
                     sbert_models, universal_sentence_encoder_model, True)

# STEP 4: predict similarity scores of evaluation data
predict_scores(RANDOM_FOREST_FILE, EVAL_DATA_CSV_PATH, '../models/predictions.csv')
predict_scores(RANDOM_FOREST_FILE, EVAL_DATA_CSV_PATH_CNN, '../models/predictions_cnn.csv')
