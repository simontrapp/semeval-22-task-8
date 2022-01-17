import torch
from torch import nn
from util import MaxOverTimePooling

class SimCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(SimCnn, self).__init__()

        self.in_3 = SimCnnPart(kernel_size=3, device=device)
        self.in_6 = SimCnnPart(kernel_size=6, device=device)
        self.in_9 = SimCnnPart(kernel_size=9, device=device)

        self.in_2 = SimCnnPart(kernel_size=2, device=device)
        self.in_4 = SimCnnPart(kernel_size=4, device=device)
        self.in_8 = SimCnnPart(kernel_size=8, device=device)


        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1536, out_features=1024),
            nn.Dropout(0.5),
            nn.ReLU6(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU6(),
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

    def forward(self, x):
        x_3 = self.in_3(x)
        x_6 = self.in_6(x)
        x_9 = self.in_9(x)
        x_2 = self.in_2(x)
        x_4 = self.in_4(x)
        x_8 = self.in_8(x)
        x = torch.cat([x_3, x_6, x_9, x_2, x_4, x_8], dim=1)
        return self.out(x)

class SimCnnPart(nn.Module):
    def __init__(self, kernel_size,  device="cuda"):
        super(SimCnnPart, self).__init__()

        self.final_network = nn.Sequential(
            CnnBlock(in_channels=1, out_channels=32, expand=128, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1),
            CnnBlock(in_channels=32, out_channels=64, expand=128, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            CnnBlock(in_channels=64, out_channels=128, expand=256, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            CnnBlock(in_channels=128, out_channels=256, expand=512, kernel_size=kernel_size),
            nn.Dropout(0.3),
            MaxOverTimePooling(),
        )

    def forward(self, x):
        return self.final_network(x)


class CnnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size,  expand: int):
        super(CnnBlock, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=expand, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(expand),
            nn.Conv2d(in_channels=expand, out_channels=expand, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(expand),
            nn.Conv2d(in_channels=expand, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU6(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.c1(x)
        exp = self.c2(x)
        x = torch.add(x, exp)
        del exp
        return self.c3(x)
