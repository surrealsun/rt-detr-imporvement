from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .aitod import AITODDetection
from .collate import detection_collate_fn
from .transforms import build_transforms
from .visdrone import VisDroneDetection


DATASET_REGISTRY = {
    "aitod": AITODDetection,
    "visdrone": VisDroneDetection,
}


def build_dataset(cfg: dict[str, Any], split: str):
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"].lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    dataset_cls = DATASET_REGISTRY[dataset_name]
    image_dir_key = "train_images" if split == "train" else "val_images"
    ann_key = "train_annotations" if split == "train" else "val_annotations"
    transform = build_transforms(cfg, is_train=split == "train")

    return dataset_cls(
        root=Path(dataset_cfg["root"]) / dataset_cfg[image_dir_key],
        annotation_file=Path(dataset_cfg["root"]) / dataset_cfg[ann_key],
        transform=transform,
        min_box_size=float(dataset_cfg.get("min_box_size", 2.0)),
        use_crowd=bool(dataset_cfg.get("use_crowd", False)),
    )


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["runtime"]["workers"]),
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["workers"]),
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
