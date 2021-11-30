import pandas
import json
import os
import os.path
import sys
from util import PreprocessingModel, CSV_PATH, PREPROCESSING_MODEL, DATA_DIR, PREPROCESSING_RESULT_PATH, \
    PREPROCESSING_RESULT_CSV_PATH
from doc2vec import preprocess_d2v
from bert import preprocess_bert


# read scraped article data
def get_data(directory: str, identifier: str):
    file_path = f"{directory}{identifier}"
    if os.path.exists(f"{file_path}.json"):
        with open(f"{file_path}.json", 'r') as file:
            article_data = json.load(file)
            title = article_data['title']
            text = article_data['text']
            language = article_data['meta_lang']
            if len(text) == 0:
                raise ValueError(f"{file_path}.json has empty text!")
            return title, text, language
    else:
        raise ValueError(f"{file_path}.json doesnt exist!")


def create_document_dict(title, text, lang):
    return {
        'title': title,
        'text': text,
        'lang': lang
    }


# read provided training data csv and extract valid pairs
def filter_data(data, model, data_dir):
    result_path = PREPROCESSING_RESULT_PATH.format(model.value)
    os.makedirs(result_path, exist_ok=True)
    successful_data = pandas.DataFrame(
        columns=['pair_id', 'pair_id_1', 'pair_id_2', 'language_1', 'language_2', 'score_overall'])
    documents = {}
    for index, row in data.iterrows():
        pair_id = row['pair_id']
        overall_score = row['Overall']
        pair_ids = pair_id.split('_')
        if len(pair_ids) != 2:
            raise ValueError('ID Pair doesnt contain 2 IDs!')
        # read the data and create the models
        try:
            title_1, text_1, language_1 = get_data(f"{data_dir}/{pair_ids[0]}/", pair_ids[0])
            title_2, text_2, language_2 = get_data(f"{data_dir}/{pair_ids[1]}/", pair_ids[1])
            documents[pair_ids[0]] = create_document_dict(title_1, text_1, language_1)
            documents[pair_ids[1]] = create_document_dict(title_2, text_2, language_2)
        except ValueError as ex:
            print(f"Pair {pair_id} not processable! {ex}", file=sys.stderr)
        else:
            # write successful pairs to file
            successful_data = successful_data.append(pandas.DataFrame({
                'pair_id': [pair_id],
                'pair_id_1': [pair_ids[0]],
                'pair_id_2': [pair_ids[1]],
                'language_1': [language_1],
                'language_2': [language_2],
                'score_overall': [overall_score]
            }), ignore_index=True)

    successful_data.to_csv(PREPROCESSING_RESULT_CSV_PATH.format(result_path), index=False)
    return successful_data, documents


def preprocess_data(preprocessing_model, csv_path, data_dir):
    DATA = pandas.read_csv(csv_path)
    successful_data, documents = filter_data(DATA, preprocessing_model,data_dir)

    # create model of documents
    if preprocessing_model == PreprocessingModel.DOC2VEC:
        preprocess_d2v(successful_data, documents)
    elif preprocessing_model == PreprocessingModel.BERT_D1:
        preprocess_bert('distiluse-base-multilingual-cased-v1', documents, preprocessing_model)
    elif preprocessing_model == PreprocessingModel.BERT_D2:
        preprocess_bert('distiluse-base-multilingual-cased-v2', documents, preprocessing_model)
    elif preprocessing_model == PreprocessingModel.BERT_MLM:
        preprocess_bert('paraphrase-multilingual-MiniLM-L12-v2', documents, preprocessing_model)
    elif preprocessing_model == PreprocessingModel.BERT_MPNET:
        preprocess_bert('paraphrase-multilingual-mpnet-base-v2', documents, preprocessing_model)
    else:
        raise Exception('Not implemented yet!')


if __name__ == "__main__":
    for model in [PreprocessingModel.BERT_D1, PreprocessingModel.BERT_D2,
                  PreprocessingModel.BERT_MLM, PreprocessingModel.BERT_MPNET]:  # PreprocessingModel.DOC2VEC
        preprocess_data(model, CSV_PATH, DATA_DIR)
