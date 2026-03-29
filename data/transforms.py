from __future__ import annotations

import random
from typing import Any, Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image


def _filter_small_boxes(target: dict[str, Any], min_size: float) -> dict[str, Any]:
    if target["boxes"].numel() == 0:
        return target

    boxes = target["boxes"]
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    for key in ("boxes", "labels", "area", "iscrowd"):
        target[key] = target[key][keep]
    return target


class Compose:
    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        if random.random() >= self.probability:
            return image, target

        image = TF.hflip(image)
        width, _ = image.size
        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            x0 = boxes[:, 0].clone()
            x1 = boxes[:, 2].clone()
            boxes[:, 0] = width - x1
            boxes[:, 2] = width - x0
            target["boxes"] = boxes
        return image, target


class RandomColorJitter:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.05) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
        image = TF.adjust_saturation(image, saturation_factor)
        return image, target


class RandomResize:
    def __init__(self, sizes: Sequence[int], max_size: int) -> None:
        self.sizes = list(sizes)
        self.max_size = int(max_size)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        size = random.choice(self.sizes)
        return resize(image, target, size=size, max_size=self.max_size)


class Resize:
    def __init__(self, size: int, max_size: int) -> None:
        self.size = int(size)
        self.max_size = int(max_size)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        return resize(image, target, size=self.size, max_size=self.max_size)


class ToTensor:
    def __call__(self, image: Image.Image, target: dict[str, Any]):
        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        return image_tensor, target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: torch.Tensor, target: dict[str, Any]):
        return TF.normalize(image, self.mean, self.std), target


class SanitizeBoxes:
    def __init__(self, min_size: float) -> None:
        self.min_size = float(min_size)

    def __call__(self, image, target: dict[str, Any]):
        target = _filter_small_boxes(target, self.min_size)
        return image, target


def resize(image: Image.Image, target: dict[str, Any], size: int, max_size: int):
    width, height = image.size
    min_side = min(height, width)
    max_side = max(height, width)
    scale = size / float(min_side)
    if max_side * scale > max_size:
        scale = max_size / float(max_side)

    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    image = TF.resize(image, [new_height, new_width], interpolation=Image.BILINEAR)

    if target["boxes"].numel() > 0:
        scale_tensor = torch.tensor([new_width / width, new_height / height, new_width / width, new_height / height])
        target["boxes"] = target["boxes"] * scale_tensor
        target["area"] = target["area"] * scale_tensor[0] * scale_tensor[1]
    target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)
    return image, target


def build_transforms(cfg: dict[str, Any], is_train: bool):
    dataset_cfg = cfg["dataset"]
    common = [
        SanitizeBoxes(min_size=float(dataset_cfg.get("min_box_size", 2.0))),
        ToTensor(),
        Normalize(mean=dataset_cfg["image_mean"], std=dataset_cfg["image_std"]),
    ]

    if is_train:
        return Compose(
            [
                RandomHorizontalFlip(probability=0.5),
                RandomColorJitter(),
                RandomResize(
                    sizes=dataset_cfg["train_scales"],
                    max_size=int(dataset_cfg["max_size"]),
                ),
                *common,
            ]
        )
    return Compose(
        [
            Resize(size=int(dataset_cfg["eval_scale"]), max_size=int(dataset_cfg["max_size"])),
            *common,
        ]
    )
