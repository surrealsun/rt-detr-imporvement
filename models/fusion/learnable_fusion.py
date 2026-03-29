from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LearnableScaleFusion(nn.Module):
    def __init__(self, channels: int, num_scales: int = 3, dynamic_gate: bool = True) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.dynamic_gate = dynamic_gate
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))
        self.post_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        if dynamic_gate:
            self.gate = nn.Sequential(
                nn.Linear(channels * num_scales, channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, num_scales),
            )
        else:
            self.gate = None

        self._last_weights: torch.Tensor | None = None

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {len(features)}.")

        batch_size = features[0].shape[0]
        logits = self.scale_logits.unsqueeze(0).expand(batch_size, -1)
        if self.gate is not None:
            pooled = [feature.mean(dim=(2, 3)) for feature in features]
            gate_input = torch.cat(pooled, dim=-1)
            logits = logits + self.gate(gate_input)

        weights = logits.softmax(dim=-1)
        fused = sum(
            feature * weights[:, index].view(batch_size, 1, 1, 1)
            for index, feature in enumerate(features)
        )
        fused = self.post_fusion(fused)
        self._last_weights = weights.detach().mean(dim=0).cpu()
        return fused, weights

    def get_last_weights(self) -> torch.Tensor | None:
        return self._last_weights
