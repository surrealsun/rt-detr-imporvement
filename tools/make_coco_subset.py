from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small COCO-style subset for laptop-friendly experiments.")
    parser.add_argument("--source-root", required=True, help="Root directory of the source dataset.")
    parser.add_argument("--output-root", required=True, help="Root directory for the subset dataset.")
    parser.add_argument("--train-images", default="train/images", help="Relative path to training images.")
    parser.add_argument("--val-images", default="val/images", help="Relative path to validation images.")
    parser.add_argument("--train-annotations", default="annotations/train.json", help="Relative path to training annotations.")
    parser.add_argument("--val-annotations", default="annotations/val.json", help="Relative path to validation annotations.")
    parser.add_argument("--train-count", type=int, default=200, help="Number of train images to keep.")
    parser.add_argument("--val-count", type=int, default=50, help="Number of val images to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def subset_split(
    images_dir: Path,
    annotations_path: Path,
    output_images_dir: Path,
    output_annotations_path: Path,
    keep_count: int,
    rng: random.Random,
) -> None:
    coco = load_json(annotations_path)
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    categories = list(coco.get("categories", []))

    if keep_count >= len(images):
        selected_images = images
    else:
        selected_images = rng.sample(images, keep_count)

    selected_ids = {image["id"] for image in selected_images}
    selected_annotations = [ann for ann in annotations if ann["image_id"] in selected_ids]

    output_images_dir.mkdir(parents=True, exist_ok=True)
    for image in selected_images:
        source_path = images_dir / image["file_name"]
        destination_path = output_images_dir / image["file_name"]
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    save_json(
        {
            "images": selected_images,
            "annotations": selected_annotations,
            "categories": categories,
        },
        output_annotations_path,
    )


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    subset_split(
        images_dir=source_root / args.train_images,
        annotations_path=source_root / args.train_annotations,
        output_images_dir=output_root / "train" / "images",
        output_annotations_path=output_root / "annotations" / "train.json",
        keep_count=args.train_count,
        rng=rng,
    )
    subset_split(
        images_dir=source_root / args.val_images,
        annotations_path=source_root / args.val_annotations,
        output_images_dir=output_root / "val" / "images",
        output_annotations_path=output_root / "annotations" / "val.json",
        keep_count=args.val_count,
        rng=rng,
    )


if __name__ == "__main__":
    main()
