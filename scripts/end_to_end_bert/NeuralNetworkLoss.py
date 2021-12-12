from torch import nn, Tensor
from typing import Iterable, Dict
import torch
from sentence_transformers import SentenceTransformer


class NeuralNetworkLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, network:nn.Module, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(NeuralNetworkLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.network = network


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        x = torch.concat((embeddings[0],embeddings[1]), dim=1)
        x = self.network(x)
        return self.loss_fct(x, labels)

