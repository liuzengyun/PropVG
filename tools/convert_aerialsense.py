#!/usr/bin/env python3
"""Convert AerialSense visual-grounding annotations to AerialVG JSONL.

The source dataset stores one referring expression and one xyxy box in each
``question/visual_grounding_*.json`` file.  Output records follow AerialVG's
ODVG layout, but contain exactly one region whose phrase is the complete
referring expression.
"""

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image


SOURCE_NAME = re.compile(r"^visual_grounding_(\d+)(?:_.*)?\.json$")


def natural_key(path):
    """Sort numbered annotation filenames in human/numeric order."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/AerialSense"),
        help="AerialSense root containing images/ and question/",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Fraction of naturally ordered annotations assigned to train",
    )
    return parser.parse_args()


def load_record(annotation_path, image_dir, image_sizes):
    match = SOURCE_NAME.match(annotation_path.name)
    if match is None:
        raise ValueError(f"Unexpected annotation filename: {annotation_path.name}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    expression = source.get("Question")
    bbox = source.get("Answer")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"Invalid Question in {annotation_path}")
    expression = expression.strip()
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Answer must be one xyxy box in {annotation_path}: {bbox!r}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise ValueError(f"Non-numeric box in {annotation_path}: {bbox!r}")
    if any(not math.isfinite(float(value)) for value in bbox):
        raise ValueError(f"Non-finite box in {annotation_path}: {bbox!r}")

    image_name = f"visual_grounding_{match.group(1)}.jpg"
    image_path = image_dir / image_name
    if image_name not in image_sizes:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {annotation_path}: {image_path}")
        with Image.open(image_path) as image:
            image_sizes[image_name] = image.size
    width, height = image_sizes[image_name]

    raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in bbox)
    # A few source annotations store the two corners in reverse order.  Sort
    # each axis first, then clip boundary-crossing boxes to the actual image.
    x1, x2 = sorted((raw_x1, raw_x2))
    y1, y2 = sorted((raw_y1, raw_y2))
    reordered = (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2)
    unclipped_bbox = (x1, y1, x2, y2)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    clipped = (x1, y1, x2, y2) != unclipped_bbox
    if not (x1 < x2 and y1 < y2):
        raise ValueError(
            f"Degenerate box after clipping in {annotation_path}: {bbox!r}, "
            f"image size=({width}, {height})"
        )

    output_bbox = [x1, y1, x2, y2]
    record = {
        "filename": image_name,
        "height": height,
        "width": width,
        "grounding": {
            "caption": expression,
            "regions": [{"bbox": output_bbox, "phrase": expression}],
        },
    }
    return record, reordered, clipped


def write_jsonl(path, records):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be strictly between 0 and 1")

    question_dir = args.data_root / "question"
    image_dir = args.data_root / "images"
    output_dir = args.data_root / "annotation"
    annotation_paths = sorted(question_dir.glob("visual_grounding_*.json"), key=natural_key)
    if not annotation_paths:
        raise FileNotFoundError(f"No source annotations found in {question_dir}")

    image_sizes = {}
    converted = [load_record(path, image_dir, image_sizes) for path in annotation_paths]
    records = [item[0] for item in converted]
    reordered_count = sum(item[1] for item in converted)
    clipped_count = sum(item[2] for item in converted)
    requested_train_count = int(len(records) * args.train_ratio)
    # Keep all expressions belonging to one image in the same split.  Pick the
    # image-group boundary nearest to the requested annotation ratio.
    image_boundaries = [
        index
        for index in range(1, len(records))
        if records[index - 1]["filename"] != records[index]["filename"]
    ]
    if not image_boundaries:
        raise ValueError("At least two source images are required to create a split")
    train_count = min(
        image_boundaries,
        key=lambda index: (abs(index - requested_train_count), index > requested_train_count),
    )
    train_records = records[:train_count]
    held_out_records = records[train_count:]

    train_images = {record["filename"] for record in train_records}
    held_out_images = {record["filename"] for record in held_out_records}
    overlapping_images = sorted(train_images & held_out_images)
    if overlapping_images:
        preview = ", ".join(overlapping_images[:5])
        raise ValueError(
            "The requested annotation-order split leaks images across train and val: "
            f"{preview}. Move the split boundary before converting."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "vg_train_odvg.jsonl", train_records)
    write_jsonl(output_dir / "vg_val_odvg.jsonl", held_out_records)
    write_jsonl(output_dir / "vg_test_odvg.jsonl", held_out_records)

    print(f"source annotations: {len(records)}")
    print(f"boxes reordered: {reordered_count}")
    print(f"boxes clipped to image: {clipped_count}")
    if train_count != requested_train_count:
        print(
            "split adjusted to avoid image leakage: "
            f"{requested_train_count} -> {train_count} train annotations"
        )
    print(f"train: {len(train_records)} annotations, {len(train_images)} images")
    print(f"val:   {len(held_out_records)} annotations, {len(held_out_images)} images")
    print(f"test:  {len(held_out_records)} annotations, {len(held_out_images)} images (same as val)")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
