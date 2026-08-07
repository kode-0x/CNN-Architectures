import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels),
            nn.SiLU(),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio
        reduced_channels = max(1, int(in_channels * se_ratio))

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(),
            ]
        layers += [
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
        ]
        self.conv = nn.Sequential(*layers)
        self.se = SqueezeExcitation(mid_channels, reduced_channels)
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.se(self.conv(x))
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetB0(nn.Module):
    _STAGE_CFG = [
        (1,  16, 1, 3, 1),
        (6,  24, 2, 3, 2),
        (6,  40, 2, 5, 2),
        (6,  80, 3, 3, 2),
        (6, 112, 3, 5, 1),
        (6, 192, 4, 5, 2),
        (6, 320, 1, 3, 1),
    ]

    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        stages = []
        in_channels = 32
        for expand_ratio, out_channels, num_layers, kernel_size, stride in self._STAGE_CFG:
            for i in range(num_layers):
                stages.append(MBConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                ))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
