from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from utils.logger import AverageMeter
from utils.metrics import decode_predictions, decoded_to_coco_results, evaluate_coco_predictions
from utils.misc import move_targets_to_device, nested_tensor_from_tensor_list
from utils.profiling import compute_flops, compute_latency, count_parameters
from utils.visualization import save_detection_visualization


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module | None,
    data_loader,
    device: torch.device,
    cfg: dict[str, Any],
    logger,
    output_dir: str | Path,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / cfg["logging"]["predictions_dir"]
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    if criterion is not None:
        criterion.eval()

    loss_meter = AverageMeter()
    coco_results: list[dict[str, Any]] = []
    decoded_predictions: list[dict[str, Any]] = []

    for images, targets in tqdm(data_loader, desc="Eval", leave=False):
        samples = nested_tensor_from_tensor_list(images).to(device)
        device_targets = move_targets_to_device(targets, device)
        outputs = model(samples)

        if criterion is not None:
            losses = criterion(outputs, device_targets)
            loss_meter.update(float(losses["loss_total"].item()), n=len(images))

        decoded = decode_predictions(
            outputs,
            device_targets,
            conf_threshold=float(cfg["evaluation"]["conf_threshold"]),
            max_detections=int(cfg["evaluation"]["max_detections"]),
        )
        decoded_predictions.extend(decoded)
        coco_results.extend(decoded_to_coco_results(decoded, data_loader.dataset.contiguous_to_json_id))

    metrics = evaluate_coco_predictions(
        data_loader.dataset.coco,
        coco_results,
        output_path=output_dir / "coco_predictions.json",
    )
    metrics["val_loss"] = loss_meter.avg
    metrics["params"] = float(count_parameters(model))

    input_size = (3, int(cfg["dataset"]["eval_scale"]), int(cfg["dataset"]["eval_scale"]))
    try:
        latency_ms, fps = compute_latency(
            model,
            device=device,
            input_size=input_size,
            warmup=int(cfg["evaluation"]["latency_warmup"]),
            iterations=int(cfg["evaluation"]["latency_iters"]),
        )
        metrics["latency_ms"] = latency_ms
        metrics["fps"] = fps
    except Exception as error:  # pragma: no cover
        logger.warning("Latency profiling skipped: %s", error)

    try:
        flops = compute_flops(model, device=device, input_size=input_size)
        if flops is not None:
            metrics["flops"] = flops
    except Exception as error:  # pragma: no cover
        logger.warning("FLOPs profiling skipped: %s", error)

    for prediction in decoded_predictions[: min(10, len(decoded_predictions))]:
        image_info = data_loader.dataset.coco.loadImgs([prediction["image_id"]])[0]
        image_path = data_loader.dataset.root / image_info["file_name"]
        ann_ids = data_loader.dataset.coco.getAnnIds(imgIds=[prediction["image_id"]])
        annotations = data_loader.dataset.coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(data_loader.dataset.json_id_to_contiguous[annotation["category_id"]])
        ground_truth = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_labels, dtype=torch.int64),
        }
        save_detection_visualization(
            image_path=image_path,
            output_path=prediction_dir / f"{prediction['image_id']}.jpg",
            predictions=prediction,
            class_names=data_loader.dataset.class_names,
            ground_truth=ground_truth,
        )

    with (output_dir / cfg["logging"]["metrics_file"]).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
