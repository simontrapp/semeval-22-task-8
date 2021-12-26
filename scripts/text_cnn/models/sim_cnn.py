import torch
from torch import nn
from util import MaxOverTimePooling


class SimCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(SimCnn, self).__init__()

        self.final_network = nn.Sequential(
            CnnBlock(in_channels=2, out_channels=16, expand=32),
            CnnBlock(in_channels=16, out_channels=32, expand=64),
            nn.MaxPool2d(kernel_size=(3,3)),
            CnnBlock(in_channels=32, out_channels=64, expand=128),
            CnnBlock(in_channels=64, out_channels=128, expand=256),
            nn.MaxPool2d(kernel_size=(3,3)),
            CnnBlock(in_channels=128, out_channels=256, expand=512),
            CnnBlock(in_channels=256, out_channels=512, expand=1024),
            MaxOverTimePooling(),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(0.5),
            nn.ReLU6(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU6(),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(0.5),
            nn.ReLU6(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU6(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )
        self.loss_fn = loss_fn

        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.final_network(x)


class CnnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand: int):
        super(CnnBlock, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=expand, kernel_size=3, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(expand),
            nn.Conv2d(in_channels=expand, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.c1(x)
        exp = self.c2(x)
        x = torch.add(x, exp)
        del exp
        return self.c3(x)
