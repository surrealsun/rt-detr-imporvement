from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_dataloaders
from engine import train
from models import build_model
from utils import apply_overrides, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tiny RT-DETR for aerial tiny-object detection.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    parser.add_argument("--set", nargs="*", default=None, help="Config overrides in key=value form.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.resume:
        cfg["training"]["resume"] = args.resume
    set_seed(int(cfg["seed"]))

    train_loader, val_loader = build_dataloaders(cfg)
    num_classes = train_loader.dataset.num_classes
    cfg["dataset"]["num_classes"] = num_classes

    model, criterion = build_model(cfg, num_classes=num_classes)
    device_name = cfg["runtime"]["device"]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["project"]["output_dir"])

    train(
        cfg=cfg,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
