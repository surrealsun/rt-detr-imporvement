from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models


RESNET_FACTORY: dict[str, tuple[Any, Any, dict[str, int]]] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights, {"c2": 64, "c3": 128, "c4": 256, "c5": 512}),
    "resnet34": (models.resnet34, models.ResNet34_Weights, {"c2": 64, "c3": 128, "c4": 256, "c5": 512}),
    "resnet50": (models.resnet50, models.ResNet50_Weights, {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}),
}


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, freeze_at: int = 0) -> None:
        super().__init__()
        if name not in RESNET_FACTORY:
            raise ValueError(f"Unsupported backbone '{name}'.")

        builder, weights_enum, channels = RESNET_FACTORY[name]
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = builder(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = channels

        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages[:freeze_at]:
            for parameter in stage.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
