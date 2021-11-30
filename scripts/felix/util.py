from enum import Enum


LANGUAGE_DICT = {
    'ar': 'arabic',
    'nl': 'dutch',
    'en': 'english',
    'fr': 'french',
    'de': 'german',
    'hu': 'hungarian',
    'pt': 'portuguese',
    'ru': 'russian',
    'es': 'spanish',
    'tr': 'turkish'
}


# only for nltk stopwords compatible languages!
def acronym_to_language(ac: str):
    if ac in LANGUAGE_DICT:
        return LANGUAGE_DICT[ac]
    else:
        return None


class PreprocessingModel(Enum):
    DOC2VEC = 'doc2vec'
    BERT_D1 = 'BERT-D1'             # uses 'distiluse-base-multilingual-cased-v1'
    BERT_D2 = 'BERT-D2'             # uses 'distiluse-base-multilingual-cased-v2'
    BERT_MLM = 'BERT-MiniLM'        # uses 'paraphrase-multilingual-MiniLM-L12-v2'
    BERT_MPNET = 'BERT-mpnet'       # uses 'paraphrase-multilingual-mpnet-base-v2'


class ComparisonModel(Enum):
    DOC2VEC = 'doc2vec'             # only works with doc2vec preprocessing model!
    COSINE_SIM = 'cosine-similarity'
    EUCLIDEAN_DIST = 'euclidean-distance'


# Configuration for all other steps (configure here)
PREPROCESSING_MODEL = PreprocessingModel.BERT_MPNET
COMPARISON_MODEL = ComparisonModel.COSINE_SIM

# Fixed values! Do not change!
DATA_DIR = '../data/downloader_output_subsets'
CSV_PATH = '../data/semeval-2022_task8_train-data_batch.csv'
PREPROCESSING_RESULT_PATH = "../results/{}/preprocessing" #.format(PREPROCESSING_MODEL.value)
PREPROCESSING_RESULT_CSV_PATH = "{}/successfully_preprocessed_pairs.csv" #.format(PREPROCESSING_RESULT_PATH)
COMPARISON_RESULT_PATH = "../results/{}/{}" #.format(PREPROCESSING_MODEL.value,COMPARISON_MODEL.value)
COMPARISON_RESULT_CSV_PATH = "{}/successfully_compared_pairs.csv" # .format(COMPARISON_RESULT_PATH)
