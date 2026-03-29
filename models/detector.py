from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.backbone import ResNetBackbone
from models.fusion import LearnableScaleFusion
from models.heads import AuxiliaryDenseHead, DetailEnhancementBranch, DetectionHead
from models.position_encoding import PositionEmbeddingSine
from models.transformer import RTDETRTransformer
from utils.misc import NestedTensor, nested_tensor_from_tensor_list


class TinyRTDETRDetector(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        backbone_cfg = model_cfg["backbone"]
        hidden_dim = int(model_cfg["hidden_dim"])

        self.backbone = ResNetBackbone(
            name=backbone_cfg["name"],
            pretrained=bool(backbone_cfg.get("pretrained", True)),
            freeze_at=int(backbone_cfg.get("freeze_at", 0)),
        )
        channels = self.backbone.out_channels

        self.input_proj = nn.ModuleDict(
            {
                "c3": nn.Conv2d(channels["c3"], hidden_dim, kernel_size=1),
                "c4": nn.Conv2d(channels["c4"], hidden_dim, kernel_size=1),
                "c5": nn.Conv2d(channels["c5"], hidden_dim, kernel_size=1),
            }
        )
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)
        self.use_detail_branch = bool(model_cfg.get("use_detail_branch", True))
        self.use_auxiliary_dense = bool(model_cfg.get("use_auxiliary_dense", True))
        self.fusion_mode = str(model_cfg.get("fusion_mode", "none")).lower()

        self.detail_branch = (
            DetailEnhancementBranch(channels["c2"], hidden_dim)
            if self.use_detail_branch
            else None
        )
        self.fusion = (
            LearnableScaleFusion(
                channels=hidden_dim,
                num_scales=3,
                dynamic_gate=bool(model_cfg.get("fusion", {}).get("dynamic_gate", True)),
            )
            if self.fusion_mode == "learnable"
            else None
        )
        self.auxiliary_dense_head = (
            AuxiliaryDenseHead(hidden_dim, num_classes)
            if self.use_auxiliary_dense
            else None
        )

        self.transformer = RTDETRTransformer(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            nheads=int(model_cfg["nheads"]),
            encoder_layers=int(model_cfg["encoder_layers"]),
            decoder_layers=int(model_cfg["decoder_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            num_feature_levels=3,
            num_queries=int(model_cfg["num_queries"]),
            query_select_topk=int(model_cfg.get("query_select_topk", model_cfg["num_queries"])),
        )
        self.detection_head = DetectionHead(hidden_dim=hidden_dim, num_classes=num_classes)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].to(torch.bool)

    def get_fusion_weights(self) -> torch.Tensor | None:
        if self.fusion is None:
            return None
        return self.fusion.get_last_weights()

    def forward(self, samples: list[torch.Tensor] | NestedTensor) -> dict[str, torch.Tensor]:
        if isinstance(samples, (list, tuple)):
            samples = nested_tensor_from_tensor_list(samples)

        features = self.backbone(samples.tensors)
        p3 = self.input_proj["c3"](features["c3"])
        p4 = self.input_proj["c4"](features["c4"])
        p5 = self.input_proj["c5"](features["c5"])

        mid_size = p4.shape[-2:]
        aligned_scales = [
            F.interpolate(p3, size=mid_size, mode="bilinear", align_corners=False),
            p4,
            F.interpolate(p5, size=mid_size, mode="bilinear", align_corners=False),
        ]
        if self.fusion_mode == "learnable" and self.fusion is not None:
            fused_mid, fusion_weights = self.fusion(aligned_scales)
        elif self.fusion_mode == "fixed":
            fused_mid = sum(aligned_scales) / len(aligned_scales)
            fusion_weights = torch.full(
                (p4.shape[0], len(aligned_scales)),
                1.0 / len(aligned_scales),
                device=p4.device,
            )
        else:
            fused_mid = p4
            fusion_weights = None

        detail_feature = None
        if self.detail_branch is not None:
            detail_feature = self.detail_branch(features["c2"], target_size=mid_size)
            fused_mid = fused_mid + detail_feature

        source_features = [p3, fused_mid, p5]
        masks = [self._resize_mask(samples.mask, feature.shape[-2:]) for feature in source_features]
        positions = [self.position_embedding(mask) for mask in masks]

        transformer_outputs = self.transformer(source_features, masks, positions)
        head_outputs = self.detection_head(transformer_outputs["decoder_outputs"])
        pred_logits_all = head_outputs["pred_logits_all"]
        pred_boxes_all = head_outputs["pred_boxes_all"]

        outputs: dict[str, Any] = {
            "pred_logits": pred_logits_all[-1],
            "pred_boxes": pred_boxes_all[-1],
            "pred_logits_all": pred_logits_all,
            "pred_boxes_all": pred_boxes_all,
            "encoder_logits": transformer_outputs["encoder_logits"],
            "encoder_boxes": transformer_outputs["encoder_boxes"],
        }

        if fusion_weights is not None:
            outputs["fusion_weights"] = fusion_weights
        if detail_feature is not None:
            outputs["detail_feature"] = detail_feature
        if self.training and self.auxiliary_dense_head is not None:
            outputs["auxiliary"] = self.auxiliary_dense_head(fused_mid)
        else:
            outputs["auxiliary"] = None
        return outputs
