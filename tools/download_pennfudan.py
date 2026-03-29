from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


PENNFUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Download Penn-Fudan and convert it to COCO format.")
    parser.add_argument("--output-root", default="data/datasets/PennFudanCOCO", help="Where to store the converted dataset.")
    parser.add_argument("--train-count", type=int, default=140, help="Number of training images.")
    parser.add_argument("--val-count", type=int, default=30, help="Number of validation images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip file.")
    return parser.parse_args()


def download_zip(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def build_annotations(image_paths: list[Path], raw_root: Path) -> tuple[list[dict], list[dict]]:
    images = []
    annotations = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        mask_path = raw_root / "PedMasks" / image_path.name.replace(".png", "_mask.png")
        with Image.open(image_path) as image:
            width, height = image.size
        mask = Image.open(mask_path)
        mask_width, mask_height = mask.size
        mask_pixels = mask.load()
        object_ids = sorted(
            {
                int(mask_pixels[x, y])
                for y in range(mask_height)
                for x in range(mask_width)
                if int(mask_pixels[x, y]) != 0
            }
        )

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for object_id in object_ids:
            x_min = mask_width
            y_min = mask_height
            x_max = -1
            y_max = -1
            area = 0

            for y in range(mask_height):
                for x in range(mask_width):
                    if int(mask_pixels[x, y]) != object_id:
                        continue
                    area += 1
                    if x < x_min:
                        x_min = x
                    if y < y_min:
                        y_min = y
                    if x > x_max:
                        x_max = x
                    if y > y_max:
                        y_max = y

            if area == 0:
                continue

            x_max += 1
            y_max += 1
            width_box = x_max - x_min
            height_box = y_max - y_min

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, width_box, height_box],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return images, annotations


def write_coco_json(image_paths: list[Path], raw_root: Path, json_path: Path) -> None:
    images, annotations = build_annotations(image_paths, raw_root)
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def copy_images(image_paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        shutil.copy2(image_path, destination / image_path.name)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    zip_path = raw_dir / "PennFudanPed.zip"

    download_zip(PENNFUDAN_URL, zip_path)
    extract_zip(zip_path, raw_dir)

    raw_root = raw_dir / "PennFudanPed"
    image_paths = sorted((raw_root / "PNGImages").glob("*.png"))
    if not image_paths:
        raise RuntimeError("PennFudan download finished, but no images were found after extraction.")

    rng = random.Random(args.seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)
    train_count = min(args.train_count, len(shuffled))
    val_count = min(args.val_count, max(0, len(shuffled) - train_count))

    train_images = sorted(shuffled[:train_count], key=lambda path: path.name)
    val_images = sorted(shuffled[train_count : train_count + val_count], key=lambda path: path.name)

    copy_images(train_images, output_root / "train" / "images")
    copy_images(val_images, output_root / "val" / "images")
    write_coco_json(train_images, raw_root, output_root / "annotations" / "train.json")
    write_coco_json(val_images, raw_root, output_root / "annotations" / "val.json")

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()

    print(f"PennFudan stored at: {output_root}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")


if __name__ == "__main__":
    main()
