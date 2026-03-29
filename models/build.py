from __future__ import annotations

from typing import Any

from models.detector import TinyRTDETRDetector
from models.losses import DetectionCriterion, HungarianMatcher


def build_model(cfg: dict[str, Any], num_classes: int):
    model = TinyRTDETRDetector(cfg, num_classes=num_classes)
    training_cfg = cfg["training"]
    matcher = HungarianMatcher(
        cost_class=float(training_cfg["cls_loss_coef"]),
        cost_bbox=float(training_cfg["bbox_loss_coef"]),
        cost_giou=float(training_cfg["giou_loss_coef"]),
    )
    criterion = DetectionCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": float(training_cfg["cls_loss_coef"]),
            "loss_bbox": float(training_cfg["bbox_loss_coef"]),
            "loss_giou": float(training_cfg["giou_loss_coef"]),
        },
        eos_coef=float(training_cfg["eos_coef"]),
        aux_loss_weight=float(training_cfg["aux_loss_weight"]),
        aux_heatmap_weight=float(training_cfg["aux_heatmap_weight"]),
        aux_box_weight=float(training_cfg["aux_box_weight"]),
        focal_alpha=float(training_cfg["focal_alpha"]),
        focal_gamma=float(training_cfg["focal_gamma"]),
    )
    return model, criterion
