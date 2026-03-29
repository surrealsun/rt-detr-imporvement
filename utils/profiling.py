from __future__ import annotations

import time

import torch
from torch import nn

try:
    from thop import profile
except ImportError:  # pragma: no cover
    profile = None


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class _ProfileWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model([x[0]])["pred_logits"]


@torch.no_grad()
def compute_latency(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, int, int],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, *input_size, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(warmup):
        _ = model([dummy[0]])
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = model([dummy[0]])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - start_time

    if was_training:
        model.train()

    latency_ms = total_time / iterations * 1000.0
    fps = iterations / total_time
    return latency_ms, fps


def compute_flops(model: nn.Module, device: torch.device, input_size: tuple[int, int, int]) -> float | None:
    if profile is None:
        return None

    was_training = model.training
    model.eval()
    wrapper = _ProfileWrapper(model).to(device)
    dummy = torch.randn(1, *input_size, device=device)
    flops, _ = profile(wrapper, inputs=(dummy,), verbose=False)
    if was_training:
        model.train()
    return float(flops)
