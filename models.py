from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class BaseNet(nn.Module):
    def normalize(self, n: int, norm: str = "bn", num_groups: int = 2):
        if norm == "bn":
            layer = nn.BatchNorm2d(n)
        elif norm == "ln":
            layer = nn.GroupNorm(1, n)
        elif norm == "gn":
            layer = nn.GroupNorm(num_groups, n)

        return layer

    def summarize(self, device: torch.device, input_size: tuple = (1, 3, 32, 32)):
        print(summary(self.to(device), input_size=input_size))


class Net(BaseNet):
    def __init__(self, drop: float = 0, norm: str = "bn", num_groups: int = 2):
        super().__init__()

        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(8, norm, num_groups),
            nn.Dropout2d(drop),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(16, norm, num_groups),
            nn.Dropout2d(drop),
        )

        # Transition Block 1x1
        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 8, 1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.MaxPool2d(2, 2),
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(16, norm, num_groups),
            nn.Dropout2d(drop),
            nn.Conv2d(16, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(24, norm, num_groups),
            nn.Dropout2d(drop),
            nn.Conv2d(24, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(32, norm, num_groups),
            nn.Dropout2d(drop),
        )

        # Transition Block (1x1)
        self.trans2 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.MaxPool2d(2, 2),
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(16, norm, num_groups),
            nn.Dropout2d(drop),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            self.normalize(32, norm, num_groups),
            nn.Dropout2d(drop),
            nn.Conv2d(32, 48, 3, bias=False),
            nn.ReLU(),
            self.normalize(48, norm, num_groups),
            nn.Dropout2d(drop),
        )

        # Fully connected layer
        self.out = nn.Sequential(
            nn.AvgPool2d(6),
            nn.Conv2d(
                in_channels=48, out_channels=10, kernel_size=(1, 1), bias=False
            ),  # output  RF: 28
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)

        x = self.out(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)


