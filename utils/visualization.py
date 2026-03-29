from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


COLORS = [
    (220, 20, 60),
    (0, 128, 255),
    (34, 139, 34),
    (255, 140, 0),
    (148, 0, 211),
    (0, 191, 165),
]


def _load_history(path: str | Path) -> list[dict[str, Any]]:
    history = []
    input_path = Path(path)
    if not input_path.exists():
        return history
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def save_detection_visualization(
    image_path: str | Path,
    output_path: str | Path,
    predictions: dict[str, Any],
    class_names: Sequence[str],
    ground_truth: dict[str, Any] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    canvas = ImageDraw.Draw(image)

    if ground_truth is not None:
        for box, label in zip(ground_truth["boxes"], ground_truth["labels"]):
            canvas.rectangle(box.tolist(), outline=(255, 255, 255), width=1)
            canvas.text((box[0], max(0, box[1] - 10)), f"GT:{class_names[int(label)]}", fill=(255, 255, 255))

    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        color = COLORS[int(label) % len(COLORS)]
        canvas.rectangle(box.tolist(), outline=color, width=2)
        canvas.text(
            (box[0], max(0, box[1] - 12)),
            f"{class_names[int(label)]}:{float(score):.2f}",
            fill=color,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def plot_training_curves(history_path: str | Path, output_path: str | Path) -> None:
    history = _load_history(history_path)
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry.get("train_loss_total", 0.0) for entry in history]
    map_values = [entry.get("val_mAP", 0.0) for entry in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, map_values)
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.grid(True, alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_fusion_weights(history_path: str | Path, output_path: str | Path) -> None:
    history = _load_history(history_path)
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    weights = [entry["weights"] for entry in history if "weights" in entry]
    if not weights:
        return

    plt.figure(figsize=(6, 4))
    for scale_index in range(len(weights[0])):
        plt.plot(epochs[: len(weights)], [entry[scale_index] for entry in weights], label=f"scale_{scale_index}")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Weight")
    plt.title("Fusion Weights")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
