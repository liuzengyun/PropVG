"""Inspect actual Top-1 relation pairs without changing the grounding model.

GT is used only after inference for drawing/metrics. A read-only scorer hook
captures the exact neighbors, pair logits and adoption weights used by the model.
"""
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/propvg_cache")
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from mmcv import Config
from mmcv.parallel import collate
from tqdm import tqdm

from propvg.datasets import build_dataset, extract_data
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.models import build_model
from propvg.utils import load_checkpoint


COLORS = ["#1677ff", "#9254de", "#fa8c16", "#08979c", "#eb2f96", "#b79400"]
GT_COLOR, TOP_COLOR = "#159447", "#e33036"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size=20, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def wrap_text(text, width, selected_font):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines, line = [], ""
    for word in str(text).split():
        candidate = (line + " " + word).strip()
        if line and draw.textbbox((0, 0), candidate, font=selected_font)[2] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def box_iou(box, target):
    box, target = np.asarray(box), np.asarray(target)
    intersection = np.maximum(np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2]), 0).prod()
    area = np.maximum(box[2:] - box[:2], 0).prod()
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return float(intersection / max(area + target_area - intersection, 1e-6))


def pixel_boxes(normalized_cxcywh, meta):
    """Use the evaluator's valid-image clipping BEFORE original-image rescale."""
    boxes = normalized_cxcywh.detach().float().cpu()
    xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)
    height, width = meta["img_shape"][:2]
    xyxy *= xyxy.new_tensor([width, height, width, height])
    xyxy[:, 0::2].clamp_(min=0, max=width)
    xyxy[:, 1::2].clamp_(min=0, max=height)
    valid = (xyxy[:, 2:] > xyxy[:, :2]).all(-1)
    scale = torch.as_tensor(meta["scale_factor"], dtype=xyxy.dtype).reshape(-1)
    if scale.numel() == 1:
        scale = scale.repeat(4)
    if scale.numel() != 4 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Expected positive scalar/four-element scale_factor")
    return xyxy / scale, valid


def make_record(index, sample, meta, prediction, captured):
    """Map post-clipping Top-1 back to its ORIGINAL bound-query slot exactly."""
    boxes, valid = pixel_boxes(captured["boxes"][0], meta)
    kept = valid.nonzero(as_tuple=True)[0]
    predicted_boxes = prediction["boxes"].detach().float().cpu()
    torch.testing.assert_close(boxes[kept], predicted_boxes, rtol=1e-5, atol=2e-3)
    logits = captured["logits"][0, :, 0]
    torch.testing.assert_close(logits[kept], prediction["ranking_logits"].detach().float().cpu(),
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(captured["probability"][0, kept, 0],
                               prediction["scores"].detach().float().cpu(), rtol=1e-5, atol=1e-6)
    if not len(kept):
        raise RuntimeError("No valid predicted box for sample {}".format(index))
    order = logits[kept].argsort(descending=True, stable=True)
    top_query = int(kept[order[0]])
    details = captured["details"]
    pair_logits = details["relation_pair_logits"][0, top_query]
    weights = details["relation_weights"][0, top_query]
    final_logit = float(logits[top_query])
    if not math.isclose(float((pair_logits * weights).sum()), final_logit, abs_tol=1e-5, rel_tol=1e-5):
        raise RuntimeError("Pair aggregation does not reproduce actual Top-1 score")
    gt_index = int(np.asarray(sample["refer_target_index"]).reshape(-1)[0])
    gt = sample["boxes"][gt_index]
    top_box = boxes[top_query].tolist()
    top_iou = box_iou(top_box, gt)
    neighbors = []
    for slot, (value, weight) in enumerate(zip(pair_logits, weights)):
        query = int(details["relation_neighbor_indices"][0, top_query, slot])
        real = bool(details["relation_neighbor_valid"][0, top_query, slot])
        effective = bool(details["relation_effective_mask"][0, top_query, slot])
        neighbor_box = boxes[query].tolist() if real else None
        neighbors.append(dict(
            slot=slot + 1, query_slot=query if real else None,
            valid=real, used=effective, box_xyxy=neighbor_box,
            pair_logit=float(value), pair_sigmoid=float(value.sigmoid()),
            weight=float(weight), weighted_logit=float(weight * value),
            iou_with_refer_gt=box_iou(neighbor_box, gt) if real else None,
            geometry=details["relation_geometry"][0, top_query, slot].tolist(),
        ))
    ranked_ious = [box_iou(boxes[q].tolist(), gt) for q in kept[order[:5]]]
    return dict(
        dataset_index=index, annotation_line=sample.get("line_number"),
        filename=str(Path(meta["filename"]).resolve()), expression=meta["expression"],
        gt_box_xyxy=list(map(float, gt)), top1_query_slot=top_query,
        top1_box_xyxy=top_box, top1_logit=final_logit,
        top1_score=float(captured["probability"][0, top_query, 0]),
        top1_iou=top_iou, top1_correct=top_iou >= .5,
        top5_correct=max(ranked_ious) >= .5,
        fallback=bool(details["relation_fallback"][0, top_query]),
        img_shape=list(meta["img_shape"]), ori_shape=list(meta["ori_shape"]),
        scale_factor=np.asarray(meta["scale_factor"]).reshape(-1).tolist(),
        neighbors=neighbors,
    )


def dashed_rectangle(draw, box, color, width=5, dash=12):
    x1, y1, x2, y2 = box
    for x in range(round(x1), round(x2) + 1, dash * 2):
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
    for y in range(round(y1), round(y2) + 1, dash * 2):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)


def scene_panel(image, boxes, size, crop=None):
    """Draw all boxes with one explicit crop/scale/letterbox transformation."""
    if crop is None:
        crop = (0, 0, image.width, image.height)
    crop = tuple(map(int, crop))
    cropped = image.crop(crop)
    scale = min(size[0] / cropped.width, size[1] / cropped.height)
    new_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    panel = Image.new("RGB", size, "#eef1f5")
    ox, oy = (size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2
    panel.paste(resized, (ox, oy))
    sx, sy = new_size[0] / cropped.width, new_size[1] / cropped.height
    draw, selected_font = ImageDraw.Draw(panel), font(17, True)
    for label, box, color, dashed in boxes:
        x1, y1 = max(box[0], crop[0]), max(box[1], crop[1])
        x2, y2 = min(box[2], crop[2]), min(box[3], crop[3])
        if x2 <= x1 or y2 <= y1:
            continue
        xyxy = ((x1 - crop[0]) * sx + ox, (y1 - crop[1]) * sy + oy,
                (x2 - crop[0]) * sx + ox, (y2 - crop[1]) * sy + oy)
        if dashed:
            dashed_rectangle(draw, xyxy, color)
        else:
            draw.rectangle(xyxy, outline=color, width=3)
        tx = min(max(0, xyxy[0]), max(0, size[0] - 175))
        # GT above and Top-1 below to keep both readable when boxes overlap.
        ty = xyxy[3] + 2 if label.startswith("Top1") else xyxy[1] - 23
        ty = min(max(0, ty), size[1] - 24)
        bbox = draw.textbbox((tx + 3, ty), label, font=selected_font)
        draw.rectangle((tx, ty, min(size[0] - 1, bbox[2] + 4), ty + 23), fill=color)
        draw.text((tx + 3, ty), label, font=selected_font, fill="white")
    return panel


def pair_crop(image, target, neighbor):
    boxes = np.asarray([target, neighbor if neighbor is not None else target])
    lo, hi = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    center, extent = (lo + hi) / 2, np.maximum((hi - lo) * 1.5, [150, 120])
    lo = np.maximum(np.floor(center - extent / 2), [0, 0])
    hi = np.minimum(np.ceil(center + extent / 2), [image.width, image.height])
    return tuple(np.concatenate((lo, hi)).astype(int))


def render_record(record, output):
    image = Image.open(record["filename"]).convert("RGB")
    width = 1500
    overview_height = min(1000, max(420, round(width * image.height / image.width)))
    description = wrap_text("Description: " + record["expression"], width - 40, font(22))
    header_height = 115 + 30 * len(description)
    table_height, pair_height = 275, 390
    rows = math.ceil(len(record["neighbors"]) / 3)
    canvas = Image.new("RGB", (width, header_height + overview_height + table_height + rows * pair_height + 45), "white")
    draw = ImageDraw.Draw(canvas)
    correct = record["top1_correct"]
    draw.text((20, 12), "Test index {} | Top1 {} | IoU={:.4f} | p={:.4f} | logit={:.4f}".format(
        record["dataset_index"], "CORRECT" if correct else "WRONG", record["top1_iou"],
        record["top1_score"], record["top1_logit"]), font=font(25, True), fill=GT_COLOR if correct else TOP_COLOR)
    draw.text((20, 48), "Image: " + Path(record["filename"]).name, font=font(18), fill="#444444")
    for index, line in enumerate(description):
        draw.text((20, 77 + 30 * index), line, font=font(22), fill="#202020")
    neighbors = record["neighbors"]
    overlays = [("N{} q{}".format(n["slot"], n["query_slot"]), n["box_xyxy"], COLORS[i % 6], False)
                for i, n in enumerate(neighbors) if n["valid"]]
    overlays += [("GT", record["gt_box_xyxy"], GT_COLOR, True),
                 ("Top1 q{}".format(record["top1_query_slot"]), record["top1_box_xyxy"], TOP_COLOR, False)]
    canvas.paste(scene_panel(image, overlays, (width, overview_height)), (0, header_height))
    y = header_height + overview_height + 10
    draw.text((20, y), "Green dashed = GT | Red = Top1 | N1-N6 = ACTUAL selected neighbors (nearest first)",
              font=font(20, True), fill="#202020")
    y += 32
    draw.text((20, y), "Neighbor / query       Pair logit      Pair sigmoid       Weight alpha       alpha * logit      Neighbor IoU(GT)",
              font=font(19, True), fill="#303030")
    y += 30
    columns = [20, 300, 500, 735, 975, 1245]
    for i, n in enumerate(neighbors):
        status = ("q{}".format(n["query_slot"]) if n["valid"] else "fallback" if n["used"] else "PADDING")
        values = ["N{} / {}".format(n["slot"], status), "{:+.5f}".format(n["pair_logit"]),
                  "{:.5f}".format(n["pair_sigmoid"]), "{:.5f}".format(n["weight"]),
                  "{:+.5f}".format(n["weighted_logit"]),
                  "{:.4f}".format(n["iou_with_refer_gt"]) if n["valid"] else "--"]
        for x, value in zip(columns, values):
            draw.text((x, y), value, font=font(20), fill=COLORS[i % 6] if n["used"] else "#999999")
        y += 29
    pair_start = header_height + overview_height + table_height
    for i, n in enumerate(neighbors):
        x, y = (i % 3) * 500, pair_start + (i // 3) * pair_height
        title = "N{} | {} | alpha={:.4f}".format(n["slot"], "q{}".format(n["query_slot"]) if n["valid"] else "NO NEIGHBOR", n["weight"])
        draw.text((x + 10, y + 5), title, font=font(19, True), fill=COLORS[i % 6])
        draw.text((x + 10, y + 32), "pair p={:.4f}; logit={:+.4f}".format(n["pair_sigmoid"], n["pair_logit"]),
                  font=font(18), fill="#333333")
        pair_boxes = [("GT", record["gt_box_xyxy"], GT_COLOR, True),
                      ("Top1", record["top1_box_xyxy"], TOP_COLOR, False)]
        if n["valid"]:
            pair_boxes.insert(0, ("N{}".format(n["slot"]), n["box_xyxy"], COLORS[i % 6], False))
        crop = pair_crop(image, record["top1_box_xyxy"], n["box_xyxy"])
        canvas.paste(scene_panel(image, pair_boxes, (490, 320), crop), (x + 5, y + 60))
    draw.text((20, canvas.height - 34), "Final logit = SUM(alpha * pair logit); final p = sigmoid(final logit). Pair sigmoid is NOT separately supervised.",
              font=font(18), fill="#444444")
    canvas.save(output, quality=92, subsampling=0)


def write_gallery(output_dir, records, summary):
    cards = []
    for row in records:
        outcome = "correct" if row["top1_correct"] else "wrong"
        cards.append('<article class="{}"><h3>Test {} / {} / IoU {:.4f}</h3><p>{}</p>'
                     '<a href="{}"><img loading="lazy" src="{}"></a></article>'.format(
                         outcome, row["dataset_index"], outcome, row["top1_iou"], html.escape(row["expression"]),
                         row["visualization"], row["visualization"]))
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>Relation scorer inspection</title>'
            '<style>body{font-family:sans-serif;background:#eef1f5;margin:24px}article{background:white;padding:15px;'
            'margin:18px 0;border-left:6px solid #159447}article.wrong{border-color:#e33036}img{max-width:100%;height:auto}'
            'button{padding:10px;margin:6px}pre{white-space:pre-wrap}</style></head><body>'
            '<h1>关系评分头：__COUNT__个样本诊断</h1><p>固定随机抽样，不按GT或预测对错筛选。绿色虚线GT；红色Top1；彩色N1-N6为模型实际邻居。'
            '下方六个面板为Top1与各邻居的局部放大。</p>'
            '<p>pair sigmoid只是配对logit的sigmoid，不是独立监督的关系概率；最终分数是加权logit再sigmoid。'
            '局部放大框之外的GT不会完整显示，请结合全图。</p>'
            '<button onclick="show(\'all\')">全部</button><button onclick="show(\'wrong\')">仅错误</button>'
            '<button onclick="show(\'correct\')">仅正确</button><pre>__SUMMARY__</pre>__CARDS__'
            '<script>function show(k){document.querySelectorAll("article").forEach(e=>'
            'e.style.display=(k==="all"||e.classList.contains(k))?"block":"none")}</script></body></html>')
    page = page.replace("__SUMMARY__", html.escape(json.dumps(summary, ensure_ascii=False, indent=2)))
    page = page.replace("__CARDS__", "".join(cards))
    page = page.replace("__COUNT__", str(summary["samples"]))
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing analysis: {}".format(output_dir))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run outside a sandbox that hides the NVIDIA device")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.config)
    if not cfg.model.head.get("enable_relation_scoring", False):
        raise ValueError("This visualization requires the trained relation scoring head")
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    # Full checkpoint supplies ALL weights; do not reload unrelated pretraining.
    cfg.model.vis_enc.pretrain = None
    dataset = build_dataset(getattr(cfg.data, args.split))
    if not 0 < args.num_samples <= len(dataset):
        raise ValueError("num-samples must be between 1 and dataset size")
    indices = sorted(np.random.RandomState(args.seed).choice(len(dataset), args.num_samples, replace=False).tolist())
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(args.device)
    _, _, _, strict_loaded = load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    if not strict_loaded:
        raise RuntimeError("Checkpoint must load strictly; refusing partially initialized visualizations")
    model.eval()
    captured = {}
    def capture(_, inputs, outputs):
        probability, logits, _, details = outputs
        captured.clear()
        captured.update(boxes=inputs[1].detach().cpu().clone(), probability=probability.detach().cpu(),
                        logits=logits.detach().cpu(), details={k: v.detach().cpu() for k, v in details.items()})
    hook = model.head.relation_scorer.register_forward_hook(capture)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir()
    (output_dir / "selection.json").write_text(json.dumps(dict(split=args.split, seed=args.seed, dataset_indices=indices), indent=2), encoding="utf-8")
    records = []
    try:
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index in tqdm(indices, desc="Relation scoring visualization", dynamic_ncols=True):
                batch = collate([dataset[index]], samples_per_gpu=1)
                meta = batch["img_metas"].data[0][0]
                # Refer GT is read only from dataset.samples AFTER inference.
                batch.pop("gt_bbox", None)
                batch.pop("gt_mask_rle", None)
                batch.pop("is_crowd", None)
                captured.clear()
                outputs = model(**extract_data(batch), return_loss=False, with_bbox=True, with_mask=False, rescale=True)
                if not captured:
                    raise RuntimeError("Relation scorer hook did not run")
                record = make_record(index, dataset.samples[index], meta, outputs["pred_bboxes"][0], captured)
                record["visualization"] = "images/{:04d}_test{:05d}_{}.jpg".format(
                    len(records), index, "correct" if record["top1_correct"] else "wrong")
                render_record(record, output_dir / record["visualization"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
                records.append(record)
    finally:
        hook.remove()
    summary = dict(
        split=args.split, samples=len(records), seed=args.seed,
        top1_correct=sum(r["top1_correct"] for r in records), top1_wrong=sum(not r["top1_correct"] for r in records),
        top1_acc_at_0_5=100 * np.mean([r["top1_correct"] for r in records]),
        top5_acc_at_0_5=100 * np.mean([r["top5_correct"] for r in records]),
        box_miou=100 * np.mean([r["top1_iou"] for r in records]),
        checkpoint=str(Path(args.checkpoint).resolve()), config=str(Path(args.config).resolve()),
        note="Metrics describe only this random subset, not the full test set. All boxes are original-image xyxy pixels.",
    )
    with (output_dir / "pairs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["dataset_index", "top1_correct", "top1_iou", "top1_query_slot", "top1_score", "slot", "query_slot",
                  "valid", "used", "pair_logit", "pair_sigmoid", "weight", "weighted_logit", "iou_with_refer_gt", "expression"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in records:
            for neighbor in record["neighbors"]:
                row = {k: record[k] for k in fields if k in record}
                row.update({k: neighbor[k] for k in fields if k in neighbor})
                writer.writerow(row)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery(output_dir, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Gallery:", output_dir / "index.html")


if __name__ == "__main__":
    main()
