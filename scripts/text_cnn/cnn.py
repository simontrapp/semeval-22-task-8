from torch import nn
import torch
from sentence_transformers import SentenceTransformer
from util import MaxOverTimePooling


class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN,self).__init__()

        # self.bert_1 = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # self.bert_2 = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        self.network_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3,3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3,3)),
            nn.Dropout(0.2),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3,3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3,3)),
            nn.Dropout(0.2),
            nn.ReLU6(),
            MaxOverTimePooling()
        )

        self.final_network = nn.Sequential(
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU6(),
            nn.Linear(in_features=300, out_features=200),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU6(),
            nn.Linear(in_features=100, out_features=4),
            nn.Dropout(0.2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x_1 = x[0]#.resize(x[0].size()[0],1,*x[0].size()[-2:])
        x_2 = x[1]#.resize(x[1].size()[0], 1, *x[1].size()[-2:])
        c_1 = self.network_1(x_1)
        c_2 = self.network_2(x_2)
        c_out = torch.cat((c_1, c_2), dim=1).resize(c_1.size()[0],400)
        return self.final_network(c_out)
