from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from utils.metrics import decode_predictions
from utils.visualization import save_detection_visualization


def _resize_for_eval(image: Image.Image, eval_scale: int, max_size: int) -> tuple[Image.Image, dict[str, torch.Tensor]]:
    width, height = image.size
    min_side = min(height, width)
    max_side = max(height, width)
    scale = eval_scale / float(min_side)
    if max_side * scale > max_size:
        scale = max_size / float(max_side)

    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    resized = TF.resize(image, [new_height, new_width])
    target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
        "area": torch.empty((0,), dtype=torch.float32),
        "iscrowd": torch.empty((0,), dtype=torch.int64),
        "image_id": 0,
        "orig_size": torch.tensor([height, width], dtype=torch.int64),
        "size": torch.tensor([new_height, new_width], dtype=torch.int64),
    }
    return resized, target


def _prepare_tensor(image: Image.Image, cfg: dict[str, Any]) -> torch.Tensor:
    tensor = TF.pil_to_tensor(image).float() / 255.0
    return TF.normalize(tensor, cfg["dataset"]["image_mean"], cfg["dataset"]["image_std"])


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    source: str | Path,
    output_dir: str | Path,
    device: torch.device,
    class_names: list[str],
) -> None:
    source_path = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        image_paths = sorted(path for path in source_path.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    else:
        image_paths = [source_path]

    model.eval()
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        resized_image, target = _resize_for_eval(
            image,
            eval_scale=int(cfg["dataset"]["eval_scale"]),
            max_size=int(cfg["dataset"]["max_size"]),
        )
        image_tensor = _prepare_tensor(resized_image, cfg).to(device)
        device_target = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}

        outputs = model([image_tensor])
        prediction = decode_predictions(
            outputs,
            [device_target],
            conf_threshold=float(cfg["evaluation"]["conf_threshold"]),
            max_detections=int(cfg["evaluation"]["max_detections"]),
        )[0]
        save_detection_visualization(
            image_path=image_path,
            output_path=output_dir / image_path.name,
            predictions=prediction,
            class_names=class_names,
        )
