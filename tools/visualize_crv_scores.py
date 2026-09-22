"""Visualize paired CRV support scores on a small AerialVG split.

For every sample this script compares the untouched scene, the scene after
masking the annotated referent, and the scenes after masking each raw-CRS
Top-K proposal.  The model itself is not modified: the script observes the
normal inference call and then uses the trained CRV verifier for the extra GT
intervention.
"""

import argparse
import csv
import json
import os
import os.path as osp
import textwrap
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/propvg_cache")

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate

from propvg.datasets import build_dataset, extract_data
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.models import build_model
from propvg.models.heads.crv import gather_candidates
from propvg.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=110)
    parser.add_argument("--output-dir", default="visualization/crv_score_comparison")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def denormalize_image(tensor, cfg, img_shape):
    mean = np.asarray(cfg.img_norm_cfg.mean, dtype=np.float32)
    std = np.asarray(cfg.img_norm_cfg.std, dtype=np.float32)
    image = tensor.detach().cpu().permute(1, 2, 0).numpy() * std + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = img_shape[:2]
    return image[:height, :width]


def cxcywh_to_xyxy(boxes, img_shape):
    height, width = img_shape[:2]
    boxes = boxes.detach().float().cpu().clone()
    result = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                        boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)
    result *= result.new_tensor([width, height, width, height])
    return result.numpy()


def box_iou_xyxy(boxes, target):
    boxes = np.asarray(boxes, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    left_top = np.maximum(boxes[:, :2], target[None, :2])
    right_bottom = np.minimum(boxes[:, 2:], target[None, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(1)
    box_area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return intersection / np.maximum(box_area + target_area - intersection, 1e-6)


def intervention_mask(crv, normalized_box, visual, visual_mask):
    height, width = visual.shape[-2:]
    if visual_mask is None:
        valid_ratio = visual.new_ones((1, 2))
    else:
        valid_h = (~visual_mask[:, :, 0]).sum(1).to(visual.dtype) / visual_mask.shape[1]
        valid_w = (~visual_mask[:, 0, :]).sum(1).to(visual.dtype) / visual_mask.shape[2]
        valid_ratio = torch.stack((valid_w, valid_h), dim=-1)
    return crv._soft_box_masks(
        normalized_box[:, None], valid_ratio, height, width,
        visual.dtype, visual.device
    )[0, 0]


def apply_mask_footprint(image, feature_mask, visual_mask):
    """Display the feature intervention footprint without pretending it is a pixel mask."""
    # CRV masks live on the padded FPN canvas.  The displayed image is cropped
    # to img_shape, so remove the bottom/right feature padding before resizing.
    # Resizing the whole padded mask into the valid image would shift/scale the
    # intervention relative to the box even though the model-side coordinates
    # are correct.
    if visual_mask is not None:
        valid_height = int((~visual_mask[0, :, 0]).sum().item())
        valid_width = int((~visual_mask[0, 0, :]).sum().item())
        feature_mask = feature_mask[:valid_height, :valid_width]
    mask = cv2.resize(
        feature_mask.detach().float().cpu().numpy(),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )[..., None]
    neutral = image.reshape(-1, 3).mean(0, keepdims=True).reshape(1, 1, 3)
    filled = image.astype(np.float32) * (1.0 - mask) + neutral * mask
    return np.clip(filled, 0, 255).astype(np.uint8)


def draw_box(image, box, color, label=None, thickness=2):
    box = np.rint(box).astype(int)
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, thickness)
    if label:
        origin = (max(2, box[0]), max(18, box[1] - 5))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def add_panel_title(image, lines):
    title_height = 86
    canvas = np.full((image.shape[0] + title_height, image.shape[1], 3), 248, np.uint8)
    canvas[title_height:] = image
    for line_index, line in enumerate(lines[:3]):
        cv2.putText(
            canvas, line[:92], (9, 24 + line_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return canvas


def make_grid(panels, expression, index, filename, columns=3):
    panel_width = 560
    resized = []
    for panel in panels:
        body_height = panel.shape[0] - 86
        target_body_height = max(1, round(body_height * panel_width / panel.shape[1]))
        body = cv2.resize(panel[86:], (panel_width, target_body_height))
        title = cv2.resize(panel[:86], (panel_width, 86))
        resized.append(np.concatenate((title, body), axis=0))
    cell_height = max(panel.shape[0] for panel in resized)
    rows = (len(resized) + columns - 1) // columns
    header_height = 92
    grid = np.full((header_height + rows * cell_height, columns * panel_width, 3), 255, np.uint8)
    for position, panel in enumerate(resized):
        row, column = divmod(position, columns)
        y = header_height + row * cell_height
        x = column * panel_width
        grid[y:y + panel.shape[0], x:x + panel.shape[1]] = panel
    caption = f"val #{index}  {filename}  |  {expression}"
    wrapped = textwrap.wrap(caption, width=155)[:2]
    for line_index, line in enumerate(wrapped):
        cv2.putText(grid, line, (12, 33 + line_index * 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.67, (15, 15, 15), 2, cv2.LINE_AA)
    return grid


@torch.no_grad()
def main():
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(torch.device(args.device))

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(getattr(cfg.data, args.split))
    indices = list(range(args.start_index, args.start_index + args.num_images))
    if not indices or indices[-1] >= len(dataset):
        raise IndexError("requested sample range exceeds the dataset")

    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(args.device)
    load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    model.eval()
    if not getattr(model.head, "enable_crv", False):
        raise RuntimeError("the supplied config does not enable CRV")
    crv = model.head.crv

    captured = {}

    def capture_crv(_module, _args, kwargs, output):
        captured["kwargs"] = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in kwargs.items()
        }
        captured["output"] = output

    hook = crv.register_forward_hook(capture_crv, with_kwargs=True)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_rows = []
    summaries = []

    for ordinal, index in enumerate(indices, 1):
        sample = dataset[index]
        batch = collate([sample], samples_per_gpu=1)
        img_meta = batch["img_metas"].data[0][0]
        gt_boxes = batch["gt_bbox"].data[0][0]
        refer_index = int(img_meta["refer_target_index"][0])
        gt_xyxy = gt_boxes[refer_index].float()
        inputs = extract_data(batch)
        image = denormalize_image(inputs["img"][0], cfg, img_meta["img_shape"])

        captured.clear()
        model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=False)
        if "output" not in captured:
            raise RuntimeError("CRV forward hook did not run")
        call = captured["kwargs"]
        output = captured["output"]

        visual = call["visual"]
        visual_mask = call["visual_mask"]
        text = call["text"]
        text_mask = call["text_mask"]
        selected_indices = output["indices"]
        selected_boxes = gather_candidates(call["boxes"], selected_indices)[0]
        selected_raw = call["refer_scores"].squeeze(-1).gather(1, selected_indices)[0]
        selected_weighted = output["weighted_scores"][0, :, 0]
        candidate_original_logits = output["original_support_logits"][0]
        candidate_original_support = candidate_original_logits.sigmoid()
        candidate_masked_logits = output["masked_support_logits"][0]
        candidate_masked_support = candidate_masked_logits.sigmoid()
        candidate_drop = output["support_drop_logits"][0]
        candidate_importance = output["importance_scores"][0]

        img_h, img_w = img_meta["img_shape"][:2]
        scale = gt_xyxy.new_tensor([img_w, img_h, img_w, img_h])
        gt_normalized_xyxy = gt_xyxy / scale
        gt_normalized = torch.cat(
            ((gt_normalized_xyxy[:2] + gt_normalized_xyxy[2:]) / 2,
             gt_normalized_xyxy[2:] - gt_normalized_xyxy[:2])
        )[None].to(visual.device)
        gt_original, gt_masked, gt_valid = crv._scene_features(
            visual, visual_mask, gt_normalized[:, None]
        )
        gt_support_logits = crv._paired_support(
            gt_original, gt_masked, gt_valid, text, text_mask
        )[0]
        original_logit = gt_support_logits[0]
        gt_masked_logit = gt_support_logits[1]
        original_support = original_logit.sigmoid()
        gt_masked_support = gt_masked_logit.sigmoid()
        gt_drop = original_logit - gt_masked_logit
        gt_residual = torch.tanh(gt_drop)
        gt_importance = gt_residual.exp()
        # Only the paired difference is trained.  Absolute sigmoid support can
        # therefore saturate at 1 because a shared logit offset cancels from
        # every loss term.  Preserve logits for an honest visual comparison.
        # The directly returned logits should equal original_logit-drop.  Use
        # the returned value so diagnostics also catch any future mismatch.

        candidate_xyxy = cxcywh_to_xyxy(selected_boxes, img_meta["img_shape"])
        gt_xyxy_np = gt_xyxy.cpu().numpy()
        candidate_ious = box_iou_xyxy(candidate_xyxy, gt_xyxy_np)
        final_order = selected_weighted.argsort(descending=True)
        final_rank = torch.empty_like(final_order)
        final_rank[final_order] = torch.arange(len(final_order), device=final_order.device)

        panels = []
        original_panel = image.copy()
        draw_box(original_panel, gt_xyxy_np, (0, 220, 0), "GT", 3)
        for candidate_index, box in enumerate(candidate_xyxy):
            draw_box(original_panel, box, (20, 80, 240), str(candidate_index + 1), 1)
        panels.append(add_panel_title(original_panel, [
            "ORIGINAL SCENE",
            f"support logit={float(original_logit):+.4f}",
            f"sigmoid={float(original_support):.6g}; green=GT, red=raw CRS Top-K",
        ]))

        gt_feature_mask = intervention_mask(crv, gt_normalized, visual, visual_mask)
        gt_panel = apply_mask_footprint(image, gt_feature_mask, visual_mask)
        draw_box(gt_panel, gt_xyxy_np, (0, 220, 0), "GT", 3)
        panels.append(add_panel_title(gt_panel, [
            "MASK REFER GT (feature-level)",
            f"support logit={float(gt_masked_logit):+.4f}  drop={float(gt_drop):+.4f}",
            f"odds weight=exp(bounded residual)={float(gt_importance):.4f}",
        ]))

        for candidate_index, (box, iou) in enumerate(zip(candidate_xyxy, candidate_ious)):
            normalized_box = selected_boxes[candidate_index:candidate_index + 1]
            feature_mask = intervention_mask(crv, normalized_box, visual, visual_mask)
            panel = apply_mask_footprint(image, feature_mask, visual_mask)
            draw_box(panel, gt_xyxy_np, (0, 220, 0), "GT", 2)
            draw_box(panel, box, (20, 80, 240), f"C{candidate_index + 1}", 3)
            panels.append(add_panel_title(panel, [
                f"MASK C{candidate_index + 1}  raw-rank={candidate_index + 1} final-rank={int(final_rank[candidate_index]) + 1}",
                f"IoU={iou:.3f} CRS={float(selected_raw[candidate_index]):.4f} mask-logit={float(candidate_masked_logits[candidate_index]):+.4f}",
                f"drop={float(candidate_drop[candidate_index]):+.4f} imp={float(candidate_importance[candidate_index]):.4f} final={float(selected_weighted[candidate_index]):.4f}",
            ]))

            csv_rows.append({
                "sample_index": index,
                "filename": img_meta["filename"],
                "expression": img_meta["expression"],
                "mask_type": "candidate",
                "raw_rank": candidate_index + 1,
                "final_rank": int(final_rank[candidate_index]) + 1,
                "iou_with_gt": float(iou),
                "raw_crs": float(selected_raw[candidate_index]),
                "original_support": float(candidate_original_support[candidate_index]),
                "original_support_logit": float(candidate_original_logits[candidate_index]),
                "masked_support": float(candidate_masked_support[candidate_index]),
                "masked_support_logit": float(candidate_masked_logits[candidate_index]),
                "support_drop_logit": float(candidate_drop[candidate_index]),
                "importance": float(candidate_importance[candidate_index]),
                "weighted_score": float(selected_weighted[candidate_index]),
                "box_xyxy": " ".join(f"{coordinate:.3f}" for coordinate in box),
            })

        csv_rows.append({
            "sample_index": index,
            "filename": img_meta["filename"],
            "expression": img_meta["expression"],
            "mask_type": "refer_gt",
            "raw_rank": "",
            "final_rank": "",
            "iou_with_gt": 1.0,
            "raw_crs": "",
            "original_support": float(original_support),
            "original_support_logit": float(original_logit),
            "masked_support": float(gt_masked_support),
            "masked_support_logit": float(gt_masked_logit),
            "support_drop_logit": float(gt_drop),
            "importance": float(gt_importance),
            "weighted_score": "",
            "box_xyxy": " ".join(f"{coordinate:.3f}" for coordinate in gt_xyxy_np),
        })

        grid = make_grid(
            panels, img_meta["expression"], index, osp.basename(img_meta["filename"])
        )
        output_path = osp.join(args.output_dir, f"{index:05d}_crv_scores.jpg")
        if not cv2.imwrite(output_path, grid):
            raise RuntimeError(f"failed to write {output_path}")
        summaries.append({
            "sample_index": index,
            "filename": img_meta["filename"],
            "expression": img_meta["expression"],
            "original_support": float(original_support),
            "original_support_logit": float(original_logit),
            "gt_masked_support": float(gt_masked_support),
            "gt_masked_support_logit": float(gt_masked_logit),
            "gt_support_drop_logit": float(gt_drop),
            "raw_top1_iou": float(candidate_ious[0]),
            "final_top1_iou": float(candidate_ious[int(final_order[0])]),
        })
        print(
            f"[{ordinal:02d}/{args.num_images:02d}] {output_path} "
            f"original_logit={float(original_logit):+.4f} "
            f"gt_mask_logit={float(gt_masked_logit):+.4f} "
            f"gt_drop={float(gt_drop):+.4f}"
        )

    hook.remove()
    csv_path = osp.join(args.output_dir, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary_path = osp.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    print(f"scores: {osp.abspath(csv_path)}")
    print(f"summary: {osp.abspath(summary_path)}")


if __name__ == "__main__":
    main()
