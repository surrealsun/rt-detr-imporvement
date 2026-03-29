from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations into COCO JSON.")
    parser.add_argument("--images", required=True, help="Path to VisDrone image folder.")
    parser.add_argument("--annotations", required=True, help="Path to VisDrone txt annotation folder.")
    parser.add_argument("--output", required=True, help="Output COCO json file.")
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images)
    annotations_dir = Path(args.annotations)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    annotation_id = 1

    for image_id, image_path in enumerate(sorted(images_dir.iterdir()), start=1):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            continue

        with annotation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                values = [int(float(item)) for item in line.split(",")[:8]]
                x, y, w, h, score, category_id, truncation, occlusion = values
                if category_id <= 0 or w <= 0 or h <= 0:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "ignore": 0,
                        "score": score,
                        "truncation": truncation,
                        "occlusion": occlusion,
                    }
                )
                annotation_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_CATEGORIES,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(coco_json, handle)


if __name__ == "__main__":
    main()
