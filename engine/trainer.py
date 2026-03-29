from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from engine.evaluator import evaluate
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import save_config
from utils.logger import AverageMeter, JsonlWriter, setup_logger
from utils.misc import move_targets_to_device, nested_tensor_from_tensor_list
from utils.visualization import plot_fusion_weights, plot_training_curves


def build_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    backbone_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": float(cfg["training"]["lr"])},
            {"params": backbone_params, "lr": float(cfg["training"]["lr_backbone"])},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    total_epochs = int(cfg["training"]["epochs"])
    warmup_epochs = int(cfg["training"]["warmup_epochs"])
    base_lr = float(cfg["training"]["lr"])
    min_lr = float(cfg["training"]["lr_min"])
    min_ratio = min_lr / max(base_lr, 1e-8)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _summarize_losses(meters: dict[str, AverageMeter]) -> dict[str, float]:
    return {name: meter.avg for name, meter in meters.items()}


def train(
    cfg: dict[str, Any],
    model: nn.Module,
    criterion: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    save_config(cfg, output_dir / "resolved_config.yaml")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(enabled=bool(cfg["runtime"]["amp"]) and device.type == "cuda")

    model.to(device)
    criterion.to(device)

    history_writer = JsonlWriter(output_dir / cfg["logging"]["history_file"])
    fusion_writer = JsonlWriter(output_dir / "fusion_weights.jsonl")
    best_map = 0.0
    start_epoch = 0

    resume_path = cfg["training"].get("resume") or ""
    if resume_path:
        epoch, best_metric, _ = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        start_epoch = epoch + 1
        best_map = best_metric
        logger.info("Resumed training from %s at epoch %d", resume_path, start_epoch)

    total_epochs = int(cfg["training"]["epochs"])
    print_freq = int(cfg["runtime"]["print_freq"])

    for epoch in range(start_epoch, total_epochs):
        model.train()
        criterion.train()
        meters = {
            "train_loss_total": AverageMeter(),
            "train_loss_ce": AverageMeter(),
            "train_loss_bbox": AverageMeter(),
            "train_loss_giou": AverageMeter(),
            "train_loss_aux": AverageMeter(),
        }

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for step, (images, targets) in enumerate(progress):
            samples = nested_tensor_from_tensor_list(images).to(device)
            targets = move_targets_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg["runtime"]["amp"]) and device.type == "cuda"):
                outputs = model(samples)
                losses = criterion(outputs, targets)
                total_loss = losses["loss_total"]

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if float(cfg["training"]["clip_max_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["clip_max_norm"]))
            scaler.step(optimizer)
            scaler.update()

            batch_size = len(images)
            for key, meter in meters.items():
                loss_key = key.replace("train_", "")
                meter.update(float(losses[loss_key].item()), n=batch_size)

            if (step + 1) % print_freq == 0 or (step + 1) == len(train_loader):
                progress.set_postfix(loss=f"{meters['train_loss_total'].avg:.4f}")

        scheduler.step()
        epoch_record: dict[str, Any] = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
        epoch_record.update(_summarize_losses(meters))

        fusion_weights = model.get_fusion_weights() if hasattr(model, "get_fusion_weights") else None
        if fusion_weights is not None:
            fusion_writer.write({"epoch": epoch + 1, "weights": fusion_weights.tolist()})

        if (epoch + 1) % int(cfg["training"]["eval_interval"]) == 0 or (epoch + 1) == total_epochs:
            metrics = evaluate(
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                device=device,
                cfg=cfg,
                logger=logger,
                output_dir=output_dir / "eval",
            )
            epoch_record.update({f"val_{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
            logger.info(
                "Epoch %d | train loss %.4f | mAP %.4f | APsmall %.4f",
                epoch + 1,
                epoch_record["train_loss_total"],
                metrics.get("mAP", 0.0),
                metrics.get("APsmall", 0.0),
            )
            current_map = float(metrics.get("mAP", 0.0))
            if current_map >= best_map:
                best_map = current_map
                save_checkpoint(
                    output_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_map,
                    cfg=cfg,
                )
        else:
            logger.info("Epoch %d | train loss %.4f", epoch + 1, epoch_record["train_loss_total"])

        history_writer.write(epoch_record)
        save_checkpoint(
            output_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_map,
            cfg=cfg,
        )

    plot_training_curves(output_dir / cfg["logging"]["history_file"], output_dir / "training_curves.png")
    plot_fusion_weights(output_dir / "fusion_weights.jsonl", output_dir / "fusion_weights.png")

    final_summary = {"best_mAP": best_map, "output_dir": str(output_dir)}
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)
    return final_summary
