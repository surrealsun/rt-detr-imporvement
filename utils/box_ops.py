from __future__ import annotations

import torch
from torch import Tensor


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0), dim=-1)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def clip_boxes_to_image(boxes: Tensor, height: int, width: int) -> Tensor:
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, width)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, height)
    return boxes


def normalize_boxes_xyxy(boxes: Tensor, size: Tensor) -> Tensor:
    height, width = size.unbind(-1)
    scale = torch.stack((width, height, width, height), dim=-1)
    return boxes / scale


def denormalize_boxes_xyxy(boxes: Tensor, size: Tensor) -> Tensor:
    height, width = size.unbind(-1)
    scale = torch.stack((width, height, width, height), dim=-1)
    return boxes * scale
