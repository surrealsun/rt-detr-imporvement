from __future__ import annotations

from typing import Any

import torch


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)
