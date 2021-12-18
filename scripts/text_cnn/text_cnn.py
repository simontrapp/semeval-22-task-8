import torch
from torch import nn
from util import MaxOverTimePooling


class TextCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(TextCnn, self).__init__()

        self.loss_fn = loss_fn
        self.network_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(in_channels=5, out_channels=20, kernel_size=3),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=3),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(in_channels=5, out_channels=20, kernel_size=3),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=3),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU6(),
            MaxOverTimePooling()
        )

        self.final_network = nn.Sequential(
            nn.Linear(in_features=400, out_features=100, bias=False),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.Linear(in_features=100, out_features=1, bias=False),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_1 = x[:, 0, :, :]
        x_1 = self.network_1(x_1.reshape(x_1.size()[0], 1, *x_1.size()[-2:]))
        x_2 = x[:, 1, :, :]
        x_2 = self.network_2(x_2.reshape(x_2.size()[0], 1, *x_2.size()[-2:]))
        x = torch.cat((x_1, x_2), dim=1).reshape(x_1.size()[0], 400)
        return self.final_network(x)
