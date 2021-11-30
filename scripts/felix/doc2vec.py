from util import PREPROCESSING_RESULT_PATH, PreprocessingModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
import gensim


def preprocess_d2v(successful_data, documents_dict):
    nltk.download('punkt')
    for index, row in successful_data.iterrows():
        pair_id = row['pair_id']
        pair_id_1 = row['pair_id_1']
        pair_id_2 = row['pair_id_2']
        model_path = f"{PREPROCESSING_RESULT_PATH.format(PreprocessingModel.DOC2VEC.value)}/{pair_id}.model"
        data = [documents_dict[pair_id_1]['text'], documents_dict[pair_id_2]['text']]
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        model = gensim.models.doc2vec.Doc2Vec(vector_size=30, min_count=2, epochs=80)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=80)
        model.save(model_path)


def compare_d2v(pair_id):
    model = Doc2Vec.load(f"{PREPROCESSING_RESULT_PATH.format(PreprocessingModel.DOC2VEC)}/{pair_id}.model")
    similar_doc = model.dv.most_similar('0')
    return similar_doc[0][1]
