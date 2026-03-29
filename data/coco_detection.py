from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoStyleDetection(Dataset):
    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        transform: Callable[[Image.Image, dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]] | None = None,
        min_box_size: float = 2.0,
        use_crowd: bool = False,
    ) -> None:
        self.root = Path(root)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.min_box_size = float(min_box_size)
        self.use_crowd = use_crowd

        self.coco = COCO(str(self.annotation_file))
        self.ids = sorted(self.coco.imgs.keys())
        self.json_category_ids = sorted(self.coco.getCatIds())
        self.json_id_to_contiguous = {category_id: index for index, category_id in enumerate(self.json_category_ids)}
        self.contiguous_to_json_id = {index: category_id for category_id, index in self.json_id_to_contiguous.items()}
        self.class_names = [self.coco.cats[category_id]["name"] for category_id in self.json_category_ids]
        self.num_classes = len(self.class_names)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_annotations(self, image_id: int) -> list[dict[str, Any]]:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)
        if not self.use_crowd:
            annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        return annotations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_id = self.ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        annotations = self._load_annotations(image_id)

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            if w < self.min_box_size or h < self.min_box_size:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.json_id_to_contiguous[annotation["category_id"]])
            areas.append(annotation.get("area", w * h))
            iscrowd.append(annotation.get("iscrowd", 0))

        target: dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "image_id": image_id,
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
            "size": torch.tensor([image.height, image.width], dtype=torch.int64),
        }

        if self.transform is not None:
            image_tensor, target = self.transform(image, target)
        else:
            image_tensor = torch.as_tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3)
            image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        return image_tensor, target
