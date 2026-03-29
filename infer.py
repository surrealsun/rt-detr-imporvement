from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_dataset
from engine import run_inference
from models import build_model
from utils import apply_overrides, load_config, set_seed
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Tiny RT-DETR.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--source", required=True, help="Image path or folder.")
    parser.add_argument("--output-dir", default="outputs/inference", help="Where to save visualizations.")
    parser.add_argument("--set", nargs="*", default=None, help="Config overrides in key=value form.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(int(cfg["seed"]))

    dataset = build_dataset(cfg, split="val")
    model, _ = build_model(cfg, num_classes=dataset.num_classes)
    device_name = cfg["runtime"]["device"]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")

    run_inference(
        model=model,
        cfg=cfg,
        source=args.source,
        output_dir=Path(args.output_dir),
        device=device,
        class_names=dataset.class_names,
    )


if __name__ == "__main__":
    main()
