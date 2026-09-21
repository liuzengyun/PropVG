"""Re-render saved CRV score diagnostics without loading a GPU model."""

import argparse
import csv
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate

from propvg.datasets import build_dataset
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from visualize_crv_scores import add_panel_title, denormalize_image, draw_box, make_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--core-expansion", type=float, default=1.5)
    parser.add_argument("--feather-expansion", type=float, default=2.0)
    return parser.parse_args()


def parse_box(value):
    return np.asarray([float(item) for item in value.split()], dtype=np.float32)


def image_space_soft_mask(box, height, width, core_expansion, feather_expansion):
    """Continuous equivalent of CRV's padded-map mask, after padding is cropped."""
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    box_width, box_height = x2 - x1, y2 - y1
    inner_width, inner_height = box_width * core_expansion, box_height * core_expansion
    outer_width, outer_height = box_width * feather_expansion, box_height * feather_expansion
    inner_x1, inner_x2 = center_x - inner_width * 0.5, center_x + inner_width * 0.5
    inner_y1, inner_y2 = center_y - inner_height * 0.5, center_y + inner_height * 0.5
    outer_x1, outer_x2 = center_x - outer_width * 0.5, center_x + outer_width * 0.5
    outer_y1, outer_y2 = center_y - outer_height * 0.5, center_y + outer_height * 0.5
    xx = torch.arange(width, dtype=torch.float32) + 0.5
    yy = torch.arange(height, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    def smoothstep(value):
        value = value.clamp(0, 1)
        return value * value * (3 - 2 * value)

    left = smoothstep((xx - outer_x1) / max(inner_x1 - outer_x1, 1e-6))
    right = smoothstep((outer_x2 - xx) / max(outer_x2 - inner_x2, 1e-6))
    top = smoothstep((yy - outer_y1) / max(inner_y1 - outer_y1, 1e-6))
    bottom = smoothstep((outer_y2 - yy) / max(outer_y2 - inner_y2, 1e-6))
    return (torch.minimum(left, right) * torch.minimum(top, bottom)).clamp(0, 1).numpy()


def apply_footprint(image, mask):
    neutral = image.reshape(-1, 3).mean(0).reshape(1, 1, 3)
    mask = mask[..., None]
    result = image.astype(np.float32) * (1 - mask) + neutral * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(getattr(cfg.data, args.split))
    grouped = defaultdict(list)
    with open(args.scores_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["sample_index"])].append(row)
    os.makedirs(args.output_dir, exist_ok=True)

    for index, rows in sorted(grouped.items()):
        candidates = sorted(
            (row for row in rows if row["mask_type"] == "candidate"),
            key=lambda row: int(row["raw_rank"]),
        )
        gt_row = next(row for row in rows if row["mask_type"] == "refer_gt")
        batch = collate([dataset[index]], samples_per_gpu=1)
        meta = batch["img_metas"].data[0][0]
        image_tensor = batch["img"].data[0][0]
        image = denormalize_image(image_tensor, cfg, meta["img_shape"])
        height, width = image.shape[:2]
        gt_box = parse_box(gt_row["box_xyxy"])

        original = image.copy()
        draw_box(original, gt_box, (0, 220, 0), "GT", 3)
        for row in candidates:
            draw_box(original, parse_box(row["box_xyxy"]), (20, 80, 240), row["raw_rank"], 1)
        panels = [add_panel_title(original, [
            "ORIGINAL SCENE",
            "relative support = 0.0000 (reference)",
            "green=refer GT; red=raw CRS Top-K",
        ])]

        gt_drop = float(gt_row["support_drop_logit"])
        gt_mask = image_space_soft_mask(
            gt_box, height, width, args.core_expansion, args.feather_expansion
        )
        gt_panel = apply_footprint(image, gt_mask)
        draw_box(gt_panel, gt_box, (0, 220, 0), "GT", 3)
        panels.append(add_panel_title(gt_panel, [
            "MASK REFER GT (corrected position)",
            f"relative support(mask-original)={-gt_drop:+.4f}",
            f"drop={gt_drop:+.4f} importance={float(gt_row['importance']):.4f}",
        ]))

        for row in candidates:
            box = parse_box(row["box_xyxy"])
            mask = image_space_soft_mask(
                box, height, width, args.core_expansion, args.feather_expansion
            )
            panel = apply_footprint(image, mask)
            draw_box(panel, gt_box, (0, 220, 0), "GT", 2)
            draw_box(panel, box, (20, 80, 240), f"C{row['raw_rank']}", 3)
            drop = float(row["support_drop_logit"])
            panels.append(add_panel_title(panel, [
                f"MASK C{row['raw_rank']} raw-rank={row['raw_rank']} final-rank={row['final_rank']}",
                f"IoU={float(row['iou_with_gt']):.3f} CRS={float(row['raw_crs']):.4f} relative-support={-drop:+.4f}",
                f"drop={drop:+.4f} imp={float(row['importance']):.4f} final={float(row['weighted_score']):.4f}",
            ]))

        grid = make_grid(panels, meta["expression"], index, osp.basename(meta["filename"]))
        output = osp.join(args.output_dir, f"{index:05d}_crv_scores.jpg")
        if not cv2.imwrite(output, grid):
            raise RuntimeError(f"failed to write {output}")
        print(output)


if __name__ == "__main__":
    main()
