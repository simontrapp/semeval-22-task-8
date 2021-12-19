import torch
from torch import nn
from util import MaxOverTimePooling


class TextCnn(nn.Module):
    def __init__(self, loss_fn, device="cuda"):
        super(TextCnn, self).__init__()

        self.loss_fn = loss_fn
        self.network_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 5), stride=(2, 3)),
            nn.BatchNorm2d(20, eps=1e-3, momentum=0.999),
            nn.ReLU6(),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=(3, 4), stride=(1, 2)),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3, 3)),
            nn.BatchNorm2d(200, eps=1e-3, momentum=0.999),
            nn.ReLU6(),
            MaxOverTimePooling()
        )
        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 5), stride=(2, 3)),
            nn.BatchNorm2d(20, eps=1e-3, momentum=0.999),
            nn.ReLU6(),
            nn.Conv2d(in_channels=20, out_channels=100, kernel_size=(3, 4), stride=(1, 2)),
            nn.Dropout(0.2),
            nn.ReLU6(),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3, 3)),
            nn.BatchNorm2d(200, eps=1e-3, momentum=0.999),
            nn.ReLU6(),
            MaxOverTimePooling()
        )

        self.final_network = nn.Sequential(
            nn.Linear(in_features=400, out_features=100),
            nn.Dropout(0.2),
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
        x = torch.cat((x_1, x_2), dim=1).reshape(x_1.size()[0], 400)
        return self.final_network(x)


class DeepWiseSeperabel(nn.Module):
    # def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DeepWiseSeperabel, self).__init__()
        layers = []

        # Depthwise
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,
                      bias=False, padding='same'))
        layers.append(nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.999))
        layers.append(nn.ReLU6())

        # Pointwise
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      bias=False, padding='same'))
        layers.append(nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out
