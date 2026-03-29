from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = x.relu()
        return x


class DetectionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, decoder_outputs):
        return {
            "pred_logits_all": self.class_embed(decoder_outputs),
            "pred_boxes_all": self.bbox_embed(decoder_outputs).sigmoid(),
        }
