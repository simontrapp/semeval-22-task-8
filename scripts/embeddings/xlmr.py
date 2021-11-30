import torch
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()     # turn off training mode


# Vector size: 1024
def encode_xlmr(text: str, model=xlmr):
    # create sentence-piece model --> each token gets an int value
    tokens = xlmr.encode(text)
    # convert sentence-piece model to matrix with torch.Size([1, len(tokens), 1024])
    last_layer_features = model.extract_features(tokens)[0]
    # calculate mean over all token embeddings as described in https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/xlm-r-encoder
    embedding = torch.mean(last_layer_features, 0)
    return embedding.tolist()
