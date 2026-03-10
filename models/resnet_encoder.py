from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class ResNetShallowEncoder(nn.Module):
    """Keep only shallow stages from ResNet34."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x0 = self.relu(self.bn1(self.conv1(x)))     # 1/2, 64
        x1 = self.layer1(self.maxpool(x0))          # 1/4, 64
        x2 = self.layer2(x1)                        # 1/8, 128
        return x0, x1, x2
