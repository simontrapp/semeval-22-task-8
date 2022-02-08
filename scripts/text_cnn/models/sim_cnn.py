import torch
from torch import nn
from ..util import MaxOverTimePooling


class SimCnn(nn.Module):
    def __init__(self, dropout):
        super(SimCnn, self).__init__()

        self.in_2 = SimCnnSubPart(kernel_size=2, dropout=dropout)
        self.in_3 = SimCnnSubPart(kernel_size=3, dropout=dropout)
        self.in_4 = SimCnnSubPart(kernel_size=4, dropout=dropout)
        self.in_5 = SimCnnSubPart(kernel_size=5, dropout=dropout)
        self.in_6 = SimCnnSubPart(kernel_size=6, dropout=dropout)
        self.in_7 = SimCnnSubPart(kernel_size=7, dropout=dropout)
        self.in_8 = SimCnnSubPart(kernel_size=8, dropout=dropout)
        self.in_9 = SimCnnSubPart(kernel_size=9, dropout=dropout)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU6(),
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(dropout),
            nn.ReLU6(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU6(),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(dropout),
            nn.ReLU6(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU6(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_2 = self.in_2(x)
        x_3 = self.in_3(x)
        x_4 = self.in_4(x)
        x_5 = self.in_5(x)
        x_6 = self.in_6(x)
        x_8 = self.in_8(x)
        x_7 = self.in_7(x)
        x_9 = self.in_9(x)
        x = torch.cat([x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], dim=1)
        return self.out(x)


class SimCnnSubPart(nn.Module):
    def __init__(self, kernel_size: int, dropout: float):
        super(SimCnnSubPart, self).__init__()
        self.network_1 = nn.Sequential(
            CnnBlock(in_channels=2, out_channels=32, kernel_size=kernel_size, dropout=(dropout / 2.0)),
            nn.Dropout(dropout / 2),
            CnnBlock(in_channels=32, out_channels=64, kernel_size=kernel_size, dropout=(dropout / 2.0)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(100, kernel_size)),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            MaxOverTimePooling(),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(dropout),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.network_1(x)
        # x = self.n_2(x)
        # return x


class CnnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, dropout: int):
        super(CnnBlock, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.c1(x)
