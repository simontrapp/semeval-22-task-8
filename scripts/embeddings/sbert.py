from sentence_transformers import SentenceTransformer

PRETRAINED_MODEL_NAMES = ['distiluse-base-multilingual-cased-v1', 'distiluse-base-multilingual-cased-v2',
                          'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-multilingual-mpnet-base-v2']
DEFAULT_MODEL = SentenceTransformer(PRETRAINED_MODEL_NAMES[3])


def encode_sbert(text: str, model=DEFAULT_MODEL):
    return model.encode(text)
