from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class NestedTensor:
    tensors: Tensor
    mask: Tensor

    def to(self, device: torch.device | str) -> "NestedTensor":
        return NestedTensor(self.tensors.to(device), self.mask.to(device))


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def nested_tensor_from_tensor_list(tensor_list: Sequence[Tensor], size_divisible: int = 32) -> NestedTensor:
    max_size = [max(s) for s in zip(*[list(img.shape) for img in tensor_list])]
    batch_shape = [len(tensor_list)] + max_size

    if size_divisible > 1:
        batch_shape[-2] = int((batch_shape[-2] + size_divisible - 1) // size_divisible * size_divisible)
        batch_shape[-1] = int((batch_shape[-1] + size_divisible - 1) // size_divisible * size_divisible)

    batch = tensor_list[0].new_zeros(batch_shape)
    mask = torch.ones((len(tensor_list), batch_shape[-2], batch_shape[-1]), dtype=torch.bool)

    for image, padded, padded_mask in zip(tensor_list, batch, mask):
        padded[: image.shape[0], : image.shape[1], : image.shape[2]].copy_(image)
        padded_mask[: image.shape[1], : image.shape[2]] = False
    return NestedTensor(batch, mask)


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device | str) -> list[dict[str, Any]]:
    moved: list[dict[str, Any]] = []
    for target in targets:
        moved_target: dict[str, Any] = {}
        for key, value in target.items():
            if isinstance(value, Tensor):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value
        moved.append(moved_target)
    return moved


def tensor_to_list(value: Tensor) -> list[float]:
    return value.detach().cpu().flatten().tolist()
