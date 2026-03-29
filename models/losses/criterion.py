from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou, normalize_boxes_xyxy


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_factor * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class DetectionCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict[str, float],
        eos_coef: float,
        aux_loss_weight: float,
        aux_heatmap_weight: float,
        aux_box_weight: float,
        focal_alpha: float,
        focal_gamma: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.aux_loss_weight = aux_loss_weight
        self.aux_heatmap_weight = aux_heatmap_weight
        self.aux_box_weight = aux_box_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, index) for index, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[batch_index, src_idx] = targets[batch_index]["labels"][tgt_idx]
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

    def loss_boxes(self, outputs, targets, indices, num_boxes: float):
        src_idx = self._get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return zero, zero

        src_boxes = outputs["pred_boxes"][src_idx]
        target_boxes_list = []
        for target, (_, tgt_idx) in zip(targets, indices):
            if tgt_idx.numel() == 0:
                continue
            size = target["size"].to(src_boxes.device).float()
            normalized_xyxy = normalize_boxes_xyxy(target["boxes"][tgt_idx].to(src_boxes.device), size)
            target_boxes_list.append(box_xyxy_to_cxcywh(normalized_xyxy))
        target_boxes = torch.cat(target_boxes_list, dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou.sum() / num_boxes
        return loss_bbox, loss_giou

    def loss_auxiliary_dense(self, auxiliary_outputs: dict[str, Tensor], targets: list[dict[str, Any]]) -> Tensor:
        heatmap_logits = auxiliary_outputs["heatmap_logits"]
        box_map = auxiliary_outputs["box_map"]
        batch_size, num_classes, height, width = heatmap_logits.shape
        device = heatmap_logits.device

        heatmap_target = torch.zeros_like(heatmap_logits)
        box_target = torch.zeros_like(box_map)
        positive_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=device)

        for batch_index, target in enumerate(targets):
            if target["boxes"].numel() == 0:
                continue
            image_size = target["size"].to(device).float()
            normalized_xyxy = normalize_boxes_xyxy(target["boxes"].to(device), image_size)
            normalized_boxes = box_xyxy_to_cxcywh(normalized_xyxy)
            for box, label in zip(normalized_boxes, target["labels"].to(device)):
                grid_x = min(width - 1, max(0, int(box[0].item() * width)))
                grid_y = min(height - 1, max(0, int(box[1].item() * height)))
                heatmap_target[batch_index, label, grid_y, grid_x] = 1.0
                box_target[batch_index, :, grid_y, grid_x] = box
                positive_mask[batch_index, 0, grid_y, grid_x] = True

        num_positive = positive_mask.sum().clamp(min=1).float()
        heatmap_loss = sigmoid_focal_loss(
            heatmap_logits,
            heatmap_target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / num_positive

        if positive_mask.any():
            box_mask = positive_mask.expand_as(box_map)
            box_loss = F.l1_loss(box_map[box_mask], box_target[box_mask], reduction="sum") / num_positive
        else:
            box_loss = box_map.sum() * 0.0
        return self.aux_heatmap_weight * heatmap_loss + self.aux_box_weight * box_loss

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = float(sum(target["labels"].numel() for target in targets))
        num_boxes = max(num_boxes, 1.0)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices, num_boxes)
        total = (
            self.weight_dict["loss_ce"] * loss_ce
            + self.weight_dict["loss_bbox"] * loss_bbox
            + self.weight_dict["loss_giou"] * loss_giou
        )

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

        if outputs.get("auxiliary") is not None:
            loss_aux = self.loss_auxiliary_dense(outputs["auxiliary"], targets)
            losses["loss_aux"] = loss_aux
            total = total + self.aux_loss_weight * loss_aux
        else:
            losses["loss_aux"] = loss_ce.new_tensor(0.0)

        losses["loss_total"] = total
        return losses
