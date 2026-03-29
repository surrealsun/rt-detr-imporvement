from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from torch import nn

from .heads.rtdetr_head import MLP


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(num_layers)])


def _batch_index_select(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    return tensor[batch_indices, indices]


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, target: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor | None = None):
        output = target
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            intermediate.append(self.norm(output) if self.norm is not None else output)
        return torch.stack(intermediate)


class RTDETRTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        nheads: int,
        encoder_layers: int,
        decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_feature_levels: int,
        num_queries: int,
        query_select_topk: int | None = None,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(hidden_dim))
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, hidden_dim))
        self.query_pos_mlp = MLP(4, hidden_dim, hidden_dim, 2)
        self.enc_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.enc_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_queries = num_queries
        self.query_select_topk = query_select_topk or num_queries

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        positions: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        src_flatten = []
        mask_flatten = []
        pos_flatten = []

        for level, (src, mask, pos) in enumerate(zip(sources, masks, positions)):
            batch_size, channels, height, width = src.shape
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos = pos.flatten(2).transpose(1, 2) + self.level_embed[level].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            pos_flatten.append(pos)

        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        pos_flatten = torch.cat(pos_flatten, dim=1)

        memory = self.encoder(src_flatten + pos_flatten, src_key_padding_mask=mask_flatten)
        encoder_logits = self.enc_class_embed(memory)
        encoder_boxes = self.enc_box_embed(memory).sigmoid()

        class_prob = encoder_logits.softmax(dim=-1)[..., :-1]
        topk_scores = class_prob.max(dim=-1).values
        topk = min(self.num_queries, self.query_select_topk, topk_scores.shape[1])
        topk_indices = topk_scores.topk(topk, dim=1).indices
        query_content = _batch_index_select(memory, topk_indices)
        query_boxes = _batch_index_select(encoder_boxes, topk_indices)
        query_pos = self.query_pos_mlp(query_boxes)

        decoder_outputs = self.decoder(
            query_content + query_pos,
            memory + pos_flatten,
            memory_key_padding_mask=mask_flatten,
        )
        return {
            "decoder_outputs": decoder_outputs,
            "encoder_logits": encoder_logits,
            "encoder_boxes": encoder_boxes,
            "topk_indices": topk_indices,
        }
