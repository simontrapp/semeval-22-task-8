import torch
from torch import nn
from util import MaxOverTimePooling


class lstm_model(nn.Module):
    def __init__(self, loss_fn, batch_size, embedding_length=768, device="cuda"):
        super(lstm_model, self).__init__()
        self.embedding_length = embedding_length
        self.loss_fn = loss_fn
        self.network_1 = InputNetwork(200, 2, embedding_length=self.embedding_length)
        self.network_2 = InputNetwork(200, 2, embedding_length=self.embedding_length)

        self.final_network = nn.Sequential(
            #nn.Linear(in_features=800, out_features=200),
            #nn.Dropout(0.2),
            #nn.ReLU6(),
            #nn.BatchNorm1d(200),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU6(),
            nn.BatchNorm1d(100),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x_1 = x[:, 0, :, :]
        x_1 = self.network_1(x_1)
        x_2 = x[:, 1, :, :]
        x_2 = self.network_2(x_2)
        x = self.flatten(torch.cat((x_1, x_2), dim=1))
        return self.final_network(x)


class InputNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_length=768):
        super(InputNetwork, self).__init__()

        self.lstm = nn.LSTM(input_size=embedding_length, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True)
        self.network = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 10), stride=(2, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 100, kernel_size=(5, 10), stride=(2, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(100),
            MaxOverTimePooling()
        )

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)

        return self.network(output.resize(output.size()[0], 1, *output.size()[-2:]))
