from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_dataset
from data.collate import detection_collate_fn
from engine import evaluate
from models import build_model
from utils import apply_overrides, load_config, set_seed
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tiny RT-DETR checkpoints.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--split", default="val", choices=["val"], help="Dataset split to evaluate.")
    parser.add_argument("--set", nargs="*", default=None, help="Config overrides in key=value form.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation artifacts.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(int(cfg["seed"]))

    dataset = build_dataset(cfg, split=args.split)
    data_loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["workers"]),
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    model, criterion = build_model(cfg, num_classes=dataset.num_classes)
    device_name = cfg["runtime"]["device"]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")

    output_dir = Path(args.output_dir or (Path(args.checkpoint).parent / "standalone_eval"))
    logger = setup_logger(output_dir, name="tiny_rtdetr_eval")
    metrics = evaluate(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        device=device,
        cfg=cfg,
        logger=logger,
        output_dir=output_dir,
    )
    print(metrics)


if __name__ == "__main__":
    main()
