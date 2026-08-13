import torch
from torch import nn
import torch.nn.functional as F


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze, expand_1x1, expand_3x3):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.expand_1x1 = nn.Conv2d(squeeze, expand_1x1, kernel_size=1)
        self.expand_3x3 = nn.Conv2d(squeeze, expand_3x3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x), inplace=True)
        return torch.cat(
            [F.relu(self.expand_1x1(x), inplace=True), F.relu(self.expand_3x3(x), inplace=True)],
            dim=1,
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
