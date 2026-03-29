from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class SceneImage:
    image_id: int
    file_name: str
    width: int
    height: int
    boxes: list[list[float]]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a tiny-object-like synthetic dataset from PennFudanCOCO.")
    parser.add_argument("--source-root", default="data/datasets/PennFudanCOCO", help="Source PennFudanCOCO root.")
    parser.add_argument("--output-root", default="data/datasets/TinyPennFudanCOCO", help="Output dataset root.")
    parser.add_argument("--train-scenes", type=int, default=240, help="Number of synthetic training images.")
    parser.add_argument("--val-scenes", type=int, default=60, help="Number of synthetic validation images.")
    parser.add_argument("--canvas-size", type=int, default=640, help="Synthetic canvas size.")
    parser.add_argument("--min-scale", type=float, default=0.06, help="Minimum scale for pasted mini-scenes.")
    parser.add_argument("--max-scale", type=float, default=0.16, help="Maximum scale for pasted mini-scenes.")
    parser.add_argument("--min-tiles", type=int, default=2, help="Minimum mini-scenes per synthetic image.")
    parser.add_argument("--max-tiles", type=int, default=5, help="Maximum mini-scenes per synthetic image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split(source_root: Path, split: str) -> list[SceneImage]:
    annotation_path = source_root / "annotations" / f"{split}.json"
    payload = load_json(annotation_path)
    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload["annotations"]:
        ann_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    scene_images: list[SceneImage] = []
    for image_entry in payload["images"]:
        image_id = int(image_entry["id"])
        boxes = []
        for annotation in ann_by_image.get(image_id, []):
            x, y, w, h = annotation["bbox"]
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
        scene_images.append(
            SceneImage(
                image_id=image_id,
                file_name=image_entry["file_name"],
                width=int(image_entry["width"]),
                height=int(image_entry["height"]),
                boxes=boxes,
            )
        )
    return scene_images


def make_background(scene: SceneImage, source_image_dir: Path, canvas_size: int) -> Image.Image:
    image = Image.open(source_image_dir / scene.file_name).convert("RGB")
    image = image.resize((canvas_size, canvas_size), resample=Image.BILINEAR)
    image = image.filter(ImageFilter.GaussianBlur(radius=10))
    image = ImageEnhance.Color(image).enhance(0.65)
    image = ImageEnhance.Contrast(image).enhance(0.85)
    return image


def try_place(existing: list[tuple[int, int, int, int]], width: int, height: int, canvas_size: int, rng: random.Random) -> tuple[int, int]:
    for _ in range(50):
        x0 = rng.randint(0, max(0, canvas_size - width))
        y0 = rng.randint(0, max(0, canvas_size - height))
        candidate = (x0, y0, x0 + width, y0 + height)
        too_much_overlap = False
        for placed in existing:
            inter_x0 = max(candidate[0], placed[0])
            inter_y0 = max(candidate[1], placed[1])
            inter_x1 = min(candidate[2], placed[2])
            inter_y1 = min(candidate[3], placed[3])
            inter_w = max(0, inter_x1 - inter_x0)
            inter_h = max(0, inter_y1 - inter_y0)
            if inter_w * inter_h > 0.2 * width * height:
                too_much_overlap = True
                break
        if not too_much_overlap:
            return x0, y0
    return rng.randint(0, max(0, canvas_size - width)), rng.randint(0, max(0, canvas_size - height))


def build_scene(
    split_name: str,
    index: int,
    scenes: list[SceneImage],
    source_image_dir: Path,
    output_root: Path,
    canvas_size: int,
    min_scale: float,
    max_scale: float,
    min_tiles: int,
    max_tiles: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    background_scene = rng.choice(scenes)
    canvas = make_background(background_scene, source_image_dir, canvas_size)
    placed_regions: list[tuple[int, int, int, int]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id_offset = index * 1000

    num_tiles = rng.randint(min_tiles, max_tiles)
    chosen_tiles = [rng.choice(scenes) for _ in range(num_tiles)]

    for tile_scene in chosen_tiles:
        tile_image = Image.open(source_image_dir / tile_scene.file_name).convert("RGB")
        scale = rng.uniform(min_scale, max_scale)
        new_width = max(24, int(round(tile_scene.width * scale)))
        new_height = max(24, int(round(tile_scene.height * scale)))
        tile_image = tile_image.resize((new_width, new_height), resample=Image.BILINEAR)
        x0, y0 = try_place(placed_regions, new_width, new_height, canvas_size, rng)
        placed_regions.append((x0, y0, x0 + new_width, y0 + new_height))

        # Paste a downscaled natural mini-scene onto the larger canvas.
        canvas.paste(tile_image, (x0, y0))

        for box in tile_scene.boxes:
            bx0, by0, bx1, by1 = box
            scaled_box = [
                x0 + bx0 * scale,
                y0 + by0 * scale,
                x0 + bx1 * scale,
                y0 + by1 * scale,
            ]
            width_box = scaled_box[2] - scaled_box[0]
            height_box = scaled_box[3] - scaled_box[1]
            area = width_box * height_box
            annotations.append(
                {
                    "id": annotation_id_offset + len(annotations) + 1,
                    "image_id": index + 1,
                    "category_id": 1,
                    "bbox": [scaled_box[0], scaled_box[1], width_box, height_box],
                    "area": area,
                    "iscrowd": 0,
                }
            )

    image_dir = output_root / split_name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{split_name}_{index + 1:05d}.png"
    canvas.save(image_dir / file_name)

    image_entry = {
        "id": index + 1,
        "file_name": file_name,
        "width": canvas_size,
        "height": canvas_size,
    }
    return image_entry, annotations


def build_split(
    split_name: str,
    scenes: list[SceneImage],
    source_image_dir: Path,
    output_root: Path,
    scene_count: int,
    canvas_size: int,
    min_scale: float,
    max_scale: float,
    min_tiles: int,
    max_tiles: int,
    rng: random.Random,
) -> None:
    images = []
    annotations = []
    for index in range(scene_count):
        image_entry, image_annotations = build_scene(
            split_name=split_name,
            index=index,
            scenes=scenes,
            source_image_dir=source_image_dir,
            output_root=output_root,
            canvas_size=canvas_size,
            min_scale=min_scale,
            max_scale=max_scale,
            min_tiles=min_tiles,
            max_tiles=max_tiles,
            rng=rng,
        )
        images.append(image_entry)
        annotations.extend(image_annotations)

    annotation_path = output_root / "annotations" / f"{split_name}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    train_scenes = load_split(source_root, split="train")
    val_scenes = load_split(source_root, split="val")

    build_split(
        split_name="train",
        scenes=train_scenes,
        source_image_dir=source_root / "train" / "images",
        output_root=output_root,
        scene_count=args.train_scenes,
        canvas_size=args.canvas_size,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        min_tiles=args.min_tiles,
        max_tiles=args.max_tiles,
        rng=rng,
    )
    build_split(
        split_name="val",
        scenes=val_scenes,
        source_image_dir=source_root / "val" / "images",
        output_root=output_root,
        scene_count=args.val_scenes,
        canvas_size=args.canvas_size,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        min_tiles=args.min_tiles,
        max_tiles=args.max_tiles,
        rng=rng,
    )

    print(f"TinyPennFudan stored at: {output_root}")
    print(f"Train scenes: {args.train_scenes}")
    print(f"Val scenes: {args.val_scenes}")


if __name__ == "__main__":
    main()
