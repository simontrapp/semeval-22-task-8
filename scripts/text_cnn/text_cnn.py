import torch
from torch import nn
from util import MaxOverTimePooling


class TextCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(TextCnn, self).__init__()

        self.loss_fn = loss_fn
        self.network_1 = InputNetwork()
        self.network_2 = InputNetwork()

        self.final_network = nn.Sequential(
            nn.Linear(in_features=800, out_features=400),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.Linear(in_features=400, out_features=200),
            nn.ReLU6(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU6(),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_1 = x[:, 0, :, :]
        x_1 = self.network_1(x_1.reshape(x_1.size()[0], 1, *x_1.size()[-2:]))
        x_2 = x[:, 1, :, :]
        x_2 = self.network_2(x_2.reshape(x_2.size()[0], 1, *x_2.size()[-2:]))
        x = torch.cat((x_1, x_2), dim=1).reshape(x_1.size()[0], 800)
        return self.final_network(x)


class InputNetwork(nn.Module):
    def __init__(self):
        super(InputNetwork, self).__init__()

        self.network_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 5), stride=(1, 3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=(1, 5), stride=(1, 5)),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(2, 5), stride=(1, 3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=(2, 5), stride=(1, 5)),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 5), stride=(1, 3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=(3, 5), stride=(1, 5)),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 5), stride=(1, 3)),
            nn.ReLU6(),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=(4, 5), stride=(1, 5)),
            nn.ReLU6(),
            MaxOverTimePooling()
        )

    def forward(self, x):
        x_1 = self.network_1(x)
        x_2 = self.network_1(x)
        x_3 = self.network_1(x)
        x_4 = self.network_1(x)
        return torch.cat((x_1, x_2, x_3, x_4), dim=1).reshape(x_1.size()[0], 400)
