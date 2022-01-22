import torch
from torch import nn
from ..util import MaxOverTimePooling


class TextCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(TextCnn, self).__init__()

        self.loss_fn = loss_fn
        self.network_1 = InputNetwork()
        self.network_2 = InputNetwork()

        self.final_network = nn.Sequential(
            nn.Linear(in_features=400, out_features=200),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.BatchNorm1d(200),
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
        x_1 = self.network_1(x_1.reshape(x_1.size()[0], 1, *x_1.size()[-2:]))
        x_2 = x[:, 1, :, :]
        x_2 = self.network_2(x_2.reshape(x_2.size()[0], 1, *x_2.size()[-2:]))
        x = self.flatten(torch.cat((x_1, x_2), dim=1))
        return self.final_network(x)


class InputNetwork(nn.Module):
    def __init__(self):
        super(InputNetwork, self).__init__()

        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(2, 8), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(10, eps=1e-3, momentum=0.999),
            nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(2, 191), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(50, eps=1e-3, momentum=0.999),
            MaxOverTimePooling()
        )
        self.network_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 8), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(10, eps=1e-3, momentum=0.999),
            nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(3, 191), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(50, eps=1e-3, momentum=0.999),
            MaxOverTimePooling()
        )
        self.network_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4, 8), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(10, eps=1e-3, momentum=0.999),
            nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(4, 191), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(50, eps=1e-3, momentum=0.999),
            MaxOverTimePooling()
        )
        self.network_5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 8), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(10, eps=1e-3, momentum=0.999),
            nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(5, 191), stride=(1, 4)),
            nn.ReLU6(),
            nn.BatchNorm2d(50, eps=1e-3, momentum=0.999),
            MaxOverTimePooling()
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x_2 = self.network_2(x)
        x_3 = self.network_3(x)
        x_4 = self.network_4(x)
        x_5 = self.network_5(x)
        return self.flatten(torch.cat((x_2, x_3, x_4, x_5), dim=1))
