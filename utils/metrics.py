from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pycocotools.cocoeval import COCOeval

from utils.box_ops import box_cxcywh_to_xyxy, clip_boxes_to_image


def decode_predictions(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    conf_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    probabilities = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    scores, labels = probabilities.max(dim=-1)
    pred_boxes = outputs["pred_boxes"]

    decoded = []
    for batch_index, target in enumerate(targets):
        keep = scores[batch_index] >= conf_threshold
        if keep.sum() == 0:
            decoded.append(
                {
                    "image_id": int(target["image_id"]),
                    "boxes": torch.empty((0, 4)),
                    "labels": torch.empty((0,), dtype=torch.int64),
                    "scores": torch.empty((0,)),
                }
            )
            continue

        boxes = box_cxcywh_to_xyxy(pred_boxes[batch_index][keep])
        scores_kept = scores[batch_index][keep]
        labels_kept = labels[batch_index][keep]

        resized_size = target["size"].to(pred_boxes.device).float()
        original_size = target["orig_size"].to(pred_boxes.device).float()
        scale_resized = pred_boxes.new_tensor([resized_size[1], resized_size[0], resized_size[1], resized_size[0]])
        scale_back = pred_boxes.new_tensor(
            [
                original_size[1] / resized_size[1],
                original_size[0] / resized_size[0],
                original_size[1] / resized_size[1],
                original_size[0] / resized_size[0],
            ]
        )
        boxes = boxes * scale_resized
        boxes = boxes * scale_back
        boxes = clip_boxes_to_image(boxes, int(original_size[0].item()), int(original_size[1].item()))

        order = scores_kept.argsort(descending=True)[:max_detections]
        decoded.append(
            {
                "image_id": int(target["image_id"]),
                "boxes": boxes[order].detach().cpu(),
                "labels": labels_kept[order].detach().cpu(),
                "scores": scores_kept[order].detach().cpu(),
            }
        )
    return decoded


def decoded_to_coco_results(
    decoded_predictions: list[dict[str, Any]],
    contiguous_to_json_id: dict[int, int],
) -> list[dict[str, Any]]:
    coco_results: list[dict[str, Any]] = []
    for prediction in decoded_predictions:
        image_id = int(prediction["image_id"])
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            x0, y0, x1, y1 = box.tolist()
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": contiguous_to_json_id[int(label)],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                }
            )
    return coco_results


def evaluate_coco_predictions(coco_gt, coco_results: list[dict[str, Any]], output_path: str | Path) -> dict[str, float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(coco_results, handle)

    if len(coco_results) == 0:
        return {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APsmall": 0.0,
            "APmedium": 0.0,
            "APlarge": 0.0,
        }

    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    return {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APsmall": float(stats[3]),
        "APmedium": float(stats[4]),
        "APlarge": float(stats[5]),
    }
