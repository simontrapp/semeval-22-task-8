from util import PreprocessingModel, PREPROCESSING_RESULT_CSV_PATH, PREPROCESSING_RESULT_PATH
from preprocess import preprocess_data
from feed_forward import train_with_feed_forward
from conv import train_with_feed_conv
from mlp import train_with_mlp
from bert import BERT_EMBEDDINGS_PATH, BERT_SORTED_PAIR_IDS_PATH

preprocess = False
PREPROCESS_MODEL = PreprocessingModel.BERT_D2
DATA_DIR = '../data/downloader_output_subsets'
CSV_PATH = '../data/semeval-2022_task8_train-data_batch.csv'

if preprocess:
    preprocess_data(PREPROCESS_MODEL, CSV_PATH, DATA_DIR)

embeddings_path = BERT_EMBEDDINGS_PATH.format(PREPROCESSING_RESULT_PATH.format(PREPROCESS_MODEL.value))
sorted_pair_ids_path = BERT_SORTED_PAIR_IDS_PATH.format(PREPROCESSING_RESULT_PATH.format(PREPROCESS_MODEL.value))
result_path = PREPROCESSING_RESULT_PATH.format(PREPROCESS_MODEL.value)
preprocess_result_path = PREPROCESSING_RESULT_CSV_PATH.format(result_path)

# train_with_feed_forward(1000, preprocess_result_path, embeddings_path, sorted_pair_ids_path, batch_size=64,
#                         learning_rate=0.0005, test_set_size=0.2)

train_with_feed_conv(100, preprocess_result_path, embeddings_path, sorted_pair_ids_path, batch_size=64,
                        learning_rate=0.0005, test_set_size=0.2)

# best = 0
# for i in [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
#     score = train_with_mlp(preprocess_result_path, embeddings_path, sorted_pair_ids_path, hidden_layers=i)
#     if score > best:
#         best = score
#     print(score)
# print(best)
