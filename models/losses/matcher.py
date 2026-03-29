from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou, normalize_boxes_xyxy


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        out_prob = pred_logits.softmax(-1)

        indices = []
        for batch_index in range(pred_logits.shape[0]):
            target_labels = targets[batch_index]["labels"].to(pred_logits.device)
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty))
                continue

            size = targets[batch_index]["size"].to(pred_boxes.device).float()
            target_boxes_xyxy = normalize_boxes_xyxy(targets[batch_index]["boxes"].to(pred_boxes.device), size)
            target_boxes = box_xyxy_to_cxcywh(target_boxes_xyxy)

            cost_class = -out_prob[batch_index][:, target_labels]
            cost_bbox = torch.cdist(pred_boxes[batch_index], target_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_index]),
                box_cxcywh_to_xyxy(target_boxes),
            )
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            src_indices, tgt_indices = linear_sum_assignment(cost.cpu())
            indices.append(
                (
                    torch.as_tensor(src_indices, dtype=torch.int64, device=pred_boxes.device),
                    torch.as_tensor(tgt_indices, dtype=torch.int64, device=pred_boxes.device),
                )
            )
        return indices
