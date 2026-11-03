"""Compare all original proposals with hypothetical strictly disjoint neighbors.

This is a visualization-only policy. The trained scorer, its prediction and its
actual neighbor selection are NOT changed. New neighbors are not rescored.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from mmcv import Config
from mmcv.parallel import collate
from tqdm import tqdm

from visualize_relation_scoring import (
    COLORS, GT_COLOR, TOP_COLOR, font, wrap_text, scene_panel,
    box_iou, pixel_boxes, make_record,
)
from propvg.datasets import build_dataset, extract_data
from propvg.models import build_model
from propvg.utils import load_checkpoint


def intersection_area(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    return np.maximum(np.minimum(first[..., 2:], second[..., 2:]) -
                      np.maximum(first[..., :2], second[..., :2]), 0).prod(-1)


def select_disjoint_neighbors(boxes, valid, target_index, count=6, metric_scale=None,
                              foreground_scores=None, foreground_threshold=None):
    """Distance-ordered greedy selection with EXACT zero intersection area.

    Touching edges/corners are permitted (IoU=0); even tiny positive overlap
    is rejected. Optional DETECTOR foreground confidence gates the neighbor
    pool only, never the original target hypothesis. No context NMS or GT.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or valid.shape != boxes.shape[:1]:
        raise ValueError("Expected aligned N,4 boxes and N validity mask")
    if not 0 <= target_index < len(boxes) or not valid[target_index]:
        raise ValueError("The original Top-1 must be a valid query")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Neighbor count must be a positive integer")
    valid = valid & np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
    if not valid[target_index]:
        raise ValueError("Top-1 coordinates must be finite and have positive area")
    scale = np.ones(4) if metric_scale is None else np.asarray(metric_scale, dtype=np.float64).reshape(-1)
    if len(scale) == 1:
        scale = np.repeat(scale, 4)
    if len(scale) != 4 or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Expected positive scalar/four-element coordinate scale")
    # Select by distances in the SAME resized-valid-image metric as the model.
    metric_boxes = boxes * scale
    centers = (metric_boxes[:, :2] + metric_boxes[:, 2:]) / 2
    distances = np.linalg.norm(centers - centers[target_index], axis=1)
    allowed = valid & (intersection_area(boxes, boxes[target_index]) == 0)
    allowed[target_index] = False
    if (foreground_scores is None) != (foreground_threshold is None):
        raise ValueError("Foreground scores and threshold must be supplied together")
    if foreground_scores is not None:
        scores = np.asarray(foreground_scores, dtype=np.float64)
        if scores.shape != valid.shape or not 0 <= foreground_threshold <= 1:
            raise ValueError("Expected N foreground scores and a threshold in [0,1]")
        allowed &= np.isfinite(scores) & (scores >= foreground_threshold)
    candidates = np.flatnonzero(allowed)
    candidates = candidates[np.lexsort((candidates, distances[candidates]))]
    selected = []
    for query in candidates:
        if selected and np.any(intersection_area(boxes[selected], boxes[query]) > 0):
            continue
        selected.append(int(query))
        if len(selected) == count:
            break
    return selected, distances, int(allowed.sum())


def all_candidate_scene(image, boxes, valid, top_index, gt_box, size, foreground_scores=None):
    # Reuse exactly the same full-image fitting as the right panel.
    panel = scene_panel(image, [], size)
    scale = min(size[0] / image.width, size[1] / image.height)
    nw, nh = round(image.width * scale), round(image.height * scale)
    ox, oy = (size[0] - nw) // 2, (size[1] - nh) // 2
    sx, sy = nw / image.width, nh / image.height
    draw = ImageDraw.Draw(panel)
    small = font(11)
    for query, (box, keep) in enumerate(zip(boxes, valid)):
        if not keep:
            continue
        x1, y1, x2, y2 = box
        transformed = (x1 * sx + ox, y1 * sy + oy, x2 * sx + ox, y2 * sy + oy)
        draw.rectangle(transformed, outline="#3d9cff", width=1)
        label = str(query) if foreground_scores is None else "q{} fg{:.2f}".format(query, foreground_scores[query])
        draw.text((max(0, min(size[0] - 95, transformed[0])), max(0, min(size[1] - 14, transformed[1]))),
                  label, font=small, fill="#a6d4ff", stroke_width=1, stroke_fill="#0c2240")
    # Highlight GT and original Top-1 on the same canvas, without erasing any
    # proposal rectangle or changing the coordinate fitting.
    from visualize_relation_scoring import dashed_rectangle
    gt = (gt_box[0] * sx + ox, gt_box[1] * sy + oy, gt_box[2] * sx + ox, gt_box[3] * sy + oy)
    dashed_rectangle(draw, gt, GT_COLOR, width=5)
    top = boxes[top_index]
    top = (top[0] * sx + ox, top[1] * sy + oy, top[2] * sx + ox, top[3] * sy + oy)
    draw.rectangle(top, outline=TOP_COLOR, width=4)
    for label, box, color in [("GT", gt, GT_COLOR), ("Top1 q{}".format(top_index), top, TOP_COLOR)]:
        x = max(0, min(size[0] - 150, box[0]))
        y = box[3] + 2 if label.startswith("Top1") else box[1] - 24
        y = max(0, min(size[1] - 25, y))
        bbox = draw.textbbox((x + 2, y), label, font=font(18, True))
        draw.rectangle((x, y, bbox[2] + 3, y + 25), fill=color)
        draw.text((x + 2, y), label, font=font(18, True), fill="white")
    return panel


def render_comparison(record, output_dir, stem):
    image = Image.open(record["filename"]).convert("RGB")
    width = 1500
    height = min(1100, max(420, round(width * image.height / image.width)))
    lines = wrap_text("Description: " + record["expression"], width - 40, font(22))
    header = 110 + 29 * len(lines)
    footer = 280
    size = (width, header + height + footer)
    left, right = Image.new("RGB", size, "white"), Image.new("RGB", size, "white")
    threshold = record.get("foreground_threshold")
    fg_mode = threshold is not None
    title_left = "FOREGROUND POOL: p_fg >= {:.2f}".format(threshold) if fg_mode else "ALL 256 ORIGINAL CANDIDATES"
    for canvas, title in [(left, title_left), (right, "NEW NEIGHBORS: STRICT IoU = 0")]:
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), title, font=font(28, True), fill="#202020")
        status = "Test {} | original Top1 q{} | {} | IoU(GT)={:.4f}".format(
            record["dataset_index"], record["top1_query_slot"],
            "CORRECT" if record["top1_correct"] else "WRONG", record["top1_iou"])
        if fg_mode:
            status += " | Top1 p_fg={:.4f}".format(record["top1_foreground_score"])
        draw.text((20, 52), status,
            font=font(21), fill=GT_COLOR if record["top1_correct"] else TOP_COLOR)
        for line, text in enumerate(lines):
            draw.text((20, 84 + 29 * line), text, font=font(22), fill="#202020")
    boxes = np.asarray(record["all_boxes_xyxy"])
    valid = np.asarray(record["candidate_valid"])
    scores = np.asarray(record["all_foreground_scores"]) if fg_mode else None
    visible_pool = valid & np.isfinite(scores) & (scores >= threshold) if fg_mode else valid
    left.paste(all_candidate_scene(image, boxes, visible_pool, record["top1_query_slot"],
                                   record["gt_box_xyxy"], (width, height), scores), (0, header))
    overlays = []
    for i, n in enumerate(record["new_neighbors"]):
        label = "N{} q{}".format(n["slot"], n["query_slot"])
        if fg_mode:
            label += " fg{:.2f}".format(n["foreground_score"])
        overlays.append((label, n["box_xyxy"], COLORS[i % 6], False))
    overlays += [("GT", record["gt_box_xyxy"], GT_COLOR, True),
                 ("Top1 q{}".format(record["top1_query_slot"]), record["top1_box_xyxy"], TOP_COLOR, False)]
    right.paste(scene_panel(image, overlays, (width, height)), (0, header))
    dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
    y = header + height + 12
    legend = "Blue: detector p_fg >= {:.2f}; labels show query + p_fg. No extra NMS.".format(threshold) if fg_mode else "Blue thin boxes + numbers: ALL raw query slots, no score filtering or NMS."
    dl.text((20, y), legend, font=font(21), fill="#225d99")
    counts = "Total={} | valid={} | foreground survivors={}".format(len(boxes), int(valid.sum()), int(visible_pool.sum())) if fg_mode else "Total={} | valid={} | degenerate (not drawable)={}".format(len(boxes), int(valid.sum()), int((~valid).sum()))
    dl.text((20, y + 36), counts, font=font(21), fill="#202020")
    dl.text((20, y + 72), "Green dashed: GT | Red: ORIGINAL model Top1 (unchanged on both sides).", font=font(21), fill="#202020")
    dl.text((20, y + 108), "Image: " + Path(record["filename"]).name, font=font(19), fill="#555555")
    dl.text((20, y + 146), "All original-pixel boxes and query scores are preserved in predictions.jsonl.", font=font(20), fill="#555555")
    if fg_mode:
        dl.text((20, y + 182), "p_fg = softmax(DETR class logits)[foreground], NOT the referring score.", font=font(20), fill="#555555")
        dl.text((20, y + 218), "Threshold is diagnostic only; low-confidence original Top1 is still shown in red.", font=font(19), fill="#a13b00")
    dr.text((20, y), "Greedy nearest first; no overlap with Top1 OR any already chosen neighbor.", font=font(20, True), fill="#202020")
    dr.text((20, y + 30), "Selected {}/6 | Top1-disjoint pool={} | edge touching allowed (zero area).".format(
        len(record["new_neighbors"]), record["top1_disjoint_pool"]), font=font(20), fill="#202020")
    if fg_mode:
        for x, label in [(20, "Neighbor / query"), (285, "p_fg"), (480, "distance (resized px)"),
                         (830, "IoU(Top1)"), (1070, "max IoU(other neighbors)")]:
            dr.text((x, y + 60), label, font=font(18, True), fill="#303030")
    else:
        dr.text((20, y + 60), "Neighbor / query       center distance (resized px)       IoU(Top1)      max IoU(other selected)", font=font(18, True), fill="#303030")
    for i, n in enumerate(record["new_neighbors"]):
        if fg_mode:
            values = [(20, "N{} / q{}".format(n["slot"], n["query_slot"])), (285, "{:.4f}".format(n["foreground_score"])),
                      (480, "{:.3f}".format(n["distance_resized_px"])), (830, "{:.6f}".format(n["iou_with_top1"])),
                      (1070, "{:.6f}".format(n["max_iou_other_neighbors"]))]
            for x, text in values:
                dr.text((x, y + 88 + 24 * i), text, font=font(18), fill=COLORS[i % 6])
        else:
            text = "N{} / q{:<4}                 {:>9.3f}                              {:.6f}                    {:.6f}".format(
                n["slot"], n["query_slot"], n["distance_resized_px"], n["iou_with_top1"], n["max_iou_other_neighbors"])
            dr.text((20, y + 88 + 24 * i), text, font=font(18), fill=COLORS[i % 6])
    dr.text((20, y + 246), "Selection-only diagnostic: model/weights/Top1 unchanged; NEW pairs were NOT rescored.", font=font(18), fill="#a13b00")
    pool_folder = "foreground_candidates" if fg_mode else "all_candidates"
    paths = {"all_candidates": "{}/{}.jpg".format(pool_folder, stem),
             "strict_neighbors": "strict_neighbors/{}.jpg".format(stem),
             "combined": "combined/{}.jpg".format(stem)}
    left.save(output_dir / paths["all_candidates"], quality=94, subsampling=0)
    right.save(output_dir / paths["strict_neighbors"], quality=94, subsampling=0)
    combined = Image.new("RGB", (width * 2 + 12, size[1]), "#c5cbd3")
    combined.paste(left, (0, 0)); combined.paste(right, (width + 12, 0))
    combined.save(output_dir / paths["combined"], quality=94, subsampling=0)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-analysis", required=True, help="previous 100-sample analysis directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--foreground-threshold", type=float, default=None,
                        help="Optional DETR foreground probability threshold; None preserves the old diagnostic")
    args = parser.parse_args()
    if args.foreground_threshold is not None and not 0 <= args.foreground_threshold <= 1:
        raise ValueError("foreground-threshold must be in [0,1]")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite {}".format(output))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with access to the NVIDIA device")
    torch.cuda.set_device(torch.device(args.device)); torch.set_num_threads(4)
    torch.manual_seed(args.seed); torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.config)
    if cfg.model.head.get("enable_grab") or not cfg.model.head.get("enable_relation_scoring"):
        raise ValueError("This diagnostic requires all 256 original queries, Grab off, relation scorer on")
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    cfg.model.vis_enc.pretrain = None
    dataset = build_dataset(cfg.data.test)
    previous = [json.loads(line) for line in (Path(args.reference_analysis) / "predictions.jsonl").read_text().splitlines()]
    if not 0 < args.num_samples <= len(previous):
        raise ValueError("Invalid number of comparison samples")
    chosen = np.random.RandomState(args.seed).choice(len(previous), args.num_samples, replace=False)
    references = {previous[int(i)]["dataset_index"]: previous[int(i)] for i in chosen}
    indices = sorted(references)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(args.device)
    _, _, _, strict = load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    if not strict:
        raise RuntimeError("Full checkpoint must load strictly")
    model.eval()
    captured = {}
    def capture(_, inputs, outputs):
        probability, logits, _, details = outputs
        captured.clear()
        captured.update(boxes=inputs[1].detach().cpu().clone(), probability=probability.detach().cpu(),
                        logits=logits.detach().cpu(), details={k: v.detach().cpu() for k, v in details.items()},
                        foreground_scores=inputs[4].detach().float().softmax(-1)[..., 0].cpu())
    hook = model.head.relation_scorer.register_forward_hook(capture)
    output.mkdir(parents=True, exist_ok=True)
    pool_folder = "foreground_candidates" if args.foreground_threshold is not None else "all_candidates"
    for folder in (pool_folder, "strict_neighbors", "combined"):
        (output / folder).mkdir()
    records = []
    try:
        with torch.no_grad(), (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index in tqdm(indices, desc="256 proposals vs strict-disjoint neighbors", dynamic_ncols=True):
                batch = collate([dataset[index]], samples_per_gpu=1)
                meta = batch["img_metas"].data[0][0]
                for key in ("gt_bbox", "gt_mask_rle", "is_crowd"):
                    batch.pop(key, None)
                captured.clear()
                prediction = model(**extract_data(batch), return_loss=False, with_bbox=True, with_mask=False, rescale=True)
                row = make_record(index, dataset.samples[index], meta, prediction["pred_bboxes"][0], captured)
                if row["top1_query_slot"] != references[index]["top1_query_slot"]:
                    raise RuntimeError("Top1 changed from previous analysis; comparison must preserve it")
                all_boxes, valid = pixel_boxes(captured["boxes"][0], meta)
                if len(all_boxes) != 256:
                    raise RuntimeError("Expected exactly 256 original candidates")
                original_boxes, validity = all_boxes.numpy(), valid.numpy()
                fg_scores = captured["foreground_scores"][0].numpy()
                selected, distances, allowed_count = select_disjoint_neighbors(
                    original_boxes, validity, row["top1_query_slot"], count=6, metric_scale=meta["scale_factor"],
                    foreground_scores=fg_scores if args.foreground_threshold is not None else None,
                    foreground_threshold=args.foreground_threshold)
                new_neighbors = []
                for slot, query in enumerate(selected):
                    others = [box_iou(original_boxes[query], original_boxes[other]) for other in selected if other != query]
                    new_neighbors.append(dict(slot=slot + 1, query_slot=query, box_xyxy=original_boxes[query].tolist(),
                                              foreground_score=float(fg_scores[query]),
                                              distance_resized_px=float(distances[query]),
                                              iou_with_top1=box_iou(original_boxes[query], original_boxes[row["top1_query_slot"]]),
                                              max_iou_other_neighbors=max(others, default=0.0)))
                if any(n["iou_with_top1"] != 0 or n["max_iou_other_neighbors"] != 0 for n in new_neighbors):
                    raise RuntimeError("Strict zero-overlap verification failed")
                if args.foreground_threshold is not None and any(n["foreground_score"] < args.foreground_threshold for n in new_neighbors):
                    raise RuntimeError("Selected neighbor failed foreground confidence threshold")
                row["original_model_neighbors"] = row.pop("neighbors")
                row.update(all_boxes_xyxy=original_boxes.tolist(), candidate_valid=validity.tolist(),
                           all_query_scores=captured["probability"][0, :, 0].tolist(),
                           all_query_logits=captured["logits"][0, :, 0].tolist(),
                           all_foreground_scores=fg_scores.tolist(), foreground_threshold=args.foreground_threshold,
                           top1_foreground_score=float(fg_scores[row["top1_query_slot"]]),
                           foreground_survivors=int((validity & (fg_scores >= args.foreground_threshold)).sum()) if args.foreground_threshold is not None else int(validity.sum()),
                           new_neighbors=new_neighbors, top1_disjoint_pool=allowed_count)
                stem = "{:03d}_test{:05d}_{}".format(len(records), index, "correct" if row["top1_correct"] else "wrong")
                row.update(render_comparison(row, output, stem))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
                records.append(row)
    finally:
        hook.remove()
    summary = dict(samples=len(records), seed=args.seed, dataset_indices=indices,
                   source_analysis=str(Path(args.reference_analysis).resolve()),
                   checkpoint=str(Path(args.checkpoint).resolve()),
                   selected_neighbor_counts=[len(row["new_neighbors"]) for row in records],
                   foreground_threshold=args.foreground_threshold,
                   foreground_survivor_counts=[row["foreground_survivors"] for row in records],
                   samples_with_fewer_than_six=sum(len(row["new_neighbors"]) < 6 for row in records),
                   all_samples_have_256_candidates=all(len(row["all_boxes_xyxy"]) == 256 for row in records),
                   strict_zero_overlap_verified=True,
                   top1_preserved_from_previous_analysis=True,
                   rule="Optional DETR foreground confidence filter, then nearest-first greedy; IoU(target,neighbor)=0; pairwise neighbor IoU=0; never pad with overlapping/low-confidence boxes.",
                   note="Visualization only. No production model, training policy, weights or Top1 was changed; new pairs were not rescored.")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for row in records:
        cards.append('<article><h2>Test {} | {} | {} neighbors</h2><p>{}</p><p>'
                     '<a href="{}">所有256候选单图</a> | <a href="{}">严格无重叠邻居单图</a></p>'
                     '<a href="{}"><img loading="lazy" src="{}"></a></article>'.format(
                         row["dataset_index"], "正确" if row["top1_correct"] else "错误", len(row["new_neighbors"]),
                         html.escape(row["expression"]), row["all_candidates"], row["strict_neighbors"], row["combined"], row["combined"]))
    page = ('<!doctype html><html><meta charset="utf-8"><title>256候选与严格无重叠选邻</title>'
            '<style>body{font-family:sans-serif;background:#eef1f5;margin:24px}article{background:white;padding:15px;margin:20px 0}'
            'img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>全部256候选 vs 严格无重叠邻居</h1>'
            '<p>左：全部原始候选；右：原Top1与新规则选出的邻居。GT绿色虚线，Top1红色，邻居彩色。</p>'
            '<p>新邻居与Top1、邻居两两IoU均严格为0；按距离贪心选择，不足6个不强行补齐。'
            '只改变诊断选邻，未改变模型或Top1，也未用旧配对分数冒充新邻居分数。</p><pre>__SUMMARY__</pre>__CARDS__</html>')
    if args.foreground_threshold is not None:
        page = page.replace("全部256候选", "检测前景置信度过滤后候选")
        page = page.replace("左：全部原始候选", "左：p_fg≥{}的候选（蓝框标签标注query及前景置信度）".format(args.foreground_threshold))
        page = page.replace("新邻居与Top1", "置信度过滤只作用于邻居池，原Top1不变；不足6个不补低置信度框。新邻居与Top1")
        cards = [card.replace("所有256候选单图", "置信度筛选后前景候选单图") for card in cards]
    (output / "index.html").write_text(page.replace("__SUMMARY__", html.escape(json.dumps(summary, ensure_ascii=False, indent=2)))
                                     .replace("__CARDS__", "".join(cards)), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Gallery:", output / "index.html")


if __name__ == "__main__":
    main()
