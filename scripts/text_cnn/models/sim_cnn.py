import torch
from torch import nn
from util import MaxOverTimePooling


class SimCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(SimCnn, self).__init__()

        self.in_2 = SimCnnSubPart(kernel_size=2)
        self.in_3 = SimCnnSubPart(kernel_size=3)
        self.in_4 = SimCnnSubPart(kernel_size=4)
        self.in_5 = SimCnnSubPart(kernel_size=5)
        self.in_6 = SimCnnSubPart(kernel_size=6)
        self.in_7 = SimCnnSubPart(kernel_size=7)
        self.in_8 = SimCnnSubPart(kernel_size=8)
        self.in_9 = SimCnnSubPart(kernel_size=9)


        self.out = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features=2560, out_features=2048),
            # nn.ReLU6(),
            # nn.Linear(in_features=2048, out_features=1024),
            # nn.ReLU6(),
            # nn.Dropout(0.5),
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


# class SimCnnPart(nn.Module):
#     def __init__(self, kernel_size, amount, device='cuda'):
#         super(SimCnnPart, self).__init__()
#
#         self.n1 = SimCnnSubPart(kernel_size=kernel_size)
#         self.n2 = SimCnnSubPart(kernel_size=kernel_size)
#         self.n3 = SimCnnSubPart(kernel_size=kernel_size)
#         self.n4 = SimCnnSubPart(kernel_size=kernel_size)
#         self.n5 = SimCnnSubPart(kernel_size=kernel_size)
#
#     def forward(self, x):
#         x_1 = [self.n1(x), self.n2(x), self.n3(x), self.n4(x), self.n5(x)]
#         return torch.cat(x_1, dim=1)


class SimCnnSubPart(nn.Module):
    def __init__(self, kernel_size, device='cuda'):
        super(SimCnnSubPart, self).__init__()
        self.network_1 = nn.Sequential(
            # CnnBlock(in_channels=1, out_channels=128, kernel_size=kernel_size, expand=64),
            # nn.Dropout(0.2),
            # CnnBlock(in_channels=64, out_channels=128, kernel_size=kernel_size, expand=128),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(100, kernel_size)),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            MaxOverTimePooling(),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(0.5),
            nn.ReLU6(),
            # nn.Linear(in_features=128, out_features=128),
            # nn.ReLU6(),
        )

    def forward(self, x):
        return self.network_1(x)
        # x = self.n_2(x)
        # return x


class CnnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, expand: int):
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
        # x = torch.cat([x, exp], dim=1)
        del exp
        return self.c3(x)
