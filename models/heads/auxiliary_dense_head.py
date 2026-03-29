from __future__ import annotations

from torch import nn


class AuxiliaryDenseHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.box = nn.Conv2d(hidden_dim, 4, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        return {
            "heatmap_logits": self.heatmap(x),
            "box_map": self.box(x).sigmoid(),
        }
