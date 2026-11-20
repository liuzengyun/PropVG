"""Visualize CRS Top-1 misses whose second-ranked box matches the refer GT.

Read-only diagnostic: use the experiment's archived source/configuration and
strictly load its checkpoint. GT is removed before inference, then used only
for offline selection and drawing. No training/model/config code is changed.
"""
import argparse
import csv
from functools import partial
import gc
import html
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

COLORS = {"GT": "#00c46b", "Top-1": "#f04444", "Top-2": "#159cff"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def overlaps(boxes, target):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    target = np.asarray(target, dtype=np.float64)
    intersection = np.maximum(
        np.minimum(boxes[:, 2:], target[2:]) - np.maximum(boxes[:, :2], target[:2]), 0
    ).prod(-1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return intersection / np.maximum(area + target_area - intersection, 1e-12)


def top2_rescues(ious, threshold=0.5):
    return len(ious) >= 2 and float(ious[0]) < threshold <= float(ious[1])


def font(size):
    return ImageFont.truetype(FONT, size)


def wrapped_lines(draw, text, text_font, width):
    lines, line = [], ""
    for word in str(text).split():
        proposal = (line + " " + word).strip()
        if line and draw.textlength(proposal, font=text_font) > width:
            lines.append(line)
            line = word
        else:
            line = proposal
    return lines + [line]


def dashed_box(draw, box, color, width=3, dash=10):
    x1, y1, x2, y2 = box
    for start, stop, fixed, horizontal in (
        (x1, x2, y1, True), (x1, x2, y2, True),
        (y1, y2, x1, False), (y1, y2, x2, False),
    ):
        for value in range(round(start), round(stop) + 1, dash * 2):
            end = min(value + dash, stop)
            points = (value, fixed, end, fixed) if horizontal else (fixed, value, fixed, end)
            draw.line(points, fill=color, width=width)


def draw_box(draw, box, name, width=3, label=True, dashed=False):
    color = COLORS[name]
    if dashed:
        dashed_box(draw, box, color, width=width)
    else:
        draw.rectangle(tuple(box), outline=color, width=width)
    if label:
        text_font = font(20)
        # Different sides keep GT and Top-2 labels readable when boxes overlap.
        x = max(0, box[0])
        y = max(0, box[1] - 26) if name != "Top-2" else box[3] + 3
        y = min(max(0, draw._image.height - 27), y)
        tw = draw.textlength(name, font=text_font)
        x = min(x, max(0, draw._image.width - tw - 8))
        draw.rectangle((x, y, x + tw + 8, y + 26), fill="#101820")
        draw.text((x + 4, y + 1), name, font=text_font, fill=color)


def crop_window(boxes, image_size):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    center = (low + high) / 2
    side = max(float((high - low).max()) * 2.0, 100.0)
    w, h = image_size
    side = min(side, max(w, h))
    x1 = min(max(0.0, center[0] - side / 2), max(0.0, w - side))
    y1 = min(max(0.0, center[1] - side / 2), max(0.0, h - side))
    return (math.floor(x1), math.floor(y1), min(w, math.ceil(x1 + side)), min(h, math.ceil(y1 + side)))


def render(record, destination):
    with Image.open(record["filename"]) as handle:
        image = handle.convert("RGB")
    boxes = {
        "GT": np.asarray(record["gt_box"]),
        "Top-1": np.asarray(record["top_boxes"][0]),
        "Top-2": np.asarray(record["top_boxes"][1]),
    }
    page_width = 1440
    probe = ImageDraw.Draw(Image.new("RGB", (page_width, 1)))
    lines = wrapped_lines(probe, record["expression"], font(26), page_width - 48)
    header_height = 114 + 34 * len(lines)
    overview_height = round(image.height * page_width / image.width)
    zoom_height = 424
    page = Image.new("RGB", (page_width, header_height + overview_height + zoom_height), "#f6f8fb")
    draw = ImageDraw.Draw(page)
    title = "CRS Top-1 wrong / Top-2 correct  |  sample #{:04d}".format(record["index"])
    draw.text((24, 12), title, font=font(27), fill="#182534")
    for k, line in enumerate(lines):
        draw.text((24, 52 + 34 * k), line, font=font(26), fill="#17212e")
    y = header_height - 43
    labels = (
        ("GT: green solid", "GT"),
        ("Top-1: red | score={:.6f}  IoU={:.3f}".format(record["top_scores"][0], record["top_ious"][0]), "Top-1"),
        ("Top-2: blue dashed | score={:.6f}  IoU={:.3f}".format(record["top_scores"][1], record["top_ious"][1]), "Top-2"),
    )
    x = 24
    for label, name in labels:
        draw.text((x, y), label, font=font(20), fill=COLORS[name])
        x += draw.textlength(label, font=font(20)) + 30

    overview = image.resize((page_width, overview_height), Image.Resampling.LANCZOS)
    pen = ImageDraw.Draw(overview)
    scale = np.asarray([page_width / image.width, overview_height / image.height] * 2)
    for name in ("GT", "Top-1", "Top-2"):
        draw_box(pen, boxes[name] * scale, name, width=4, dashed=name == "Top-2")
    page.paste(overview, (0, header_height))

    # Separate GT and Top-2 panels share exactly the same crop. This reveals
    # overlapping predictions without shifting any box coordinates for display.
    target_crop = crop_window([boxes["GT"], boxes["Top-2"]], image.size)
    wrong_crop = crop_window([boxes["Top-1"]], image.size)
    panel_width, canvas_size = 480, 360
    for column, (name, crop) in enumerate((("GT", target_crop), ("Top-1", wrong_crop), ("Top-2", target_crop))):
        left, top = column * panel_width, header_height + overview_height
        draw.text((left + 18, top + 10), name + " - local detail", font=font(23), fill=COLORS[name])
        view = image.crop(crop)
        scale = min(canvas_size / view.width, canvas_size / view.height)
        size = (max(1, round(view.width * scale)), max(1, round(view.height * scale)))
        view = view.resize(size, Image.Resampling.LANCZOS)
        pen = ImageDraw.Draw(view)
        box = (boxes[name] - np.asarray([crop[0], crop[1], crop[0], crop[1]]))
        box *= np.asarray([size[0] / (crop[2] - crop[0]), size[1] / (crop[3] - crop[1])] * 2)
        draw_box(pen, box, name, width=3, label=False, dashed=name == "Top-2")
        page.paste(view, (left + (panel_width - size[0]) // 2, top + 49))
    page.save(destination, quality=95, subsampling=0)


def write_gallery(output, selected, summary):
    cards = []
    for record in selected:
        filename = html.escape(record["visualization"], quote=True)
        cards.append('<article><h2>Sample #{}</h2><p>{}</p><a href="{}"><img loading="lazy" src="{}"></a></article>'.format(
            record["index"], html.escape(record["expression"]), filename, filename))
    document = ('<!doctype html><meta charset="utf-8"><title>CRS Top-2 correct</title>'
                '<style>body{{font:18px sans-serif;background:#eef1f5;max-width:1440px;margin:24px auto}}'
                'article{{background:white;padding:16px;margin:24px 0}}img{{width:100%;height:auto}}'
                'p{{line-height:1.5}}h2{{font-size:22px}}</style>'
                '<h1>CRS: Top-1 incorrect, Top-2 correct</h1><p>{} matched / {} validation samples; {} figures.</p>'
                '<p>Green = GT; Red = Top-1; Blue dashed = Top-2. Scores are CRS scores, not calibrated confidence.</p>{}').format(
                    summary["matched_samples"], summary["samples"], len(selected), "".join(cards))
    (output / "index.html").write_text(document, encoding="utf-8")
    if selected:
        thumb_width, thumb_height = 360, 350
        sheet = Image.new("RGB", (4 * thumb_width, math.ceil(len(selected) / 4) * thumb_height), "white")
        for k, record in enumerate(selected):
            with Image.open(output / record["visualization"]) as handle:
                thumb = handle.copy()
            thumb.thumbnail((thumb_width - 8, thumb_height - 8), Image.Resampling.LANCZOS)
            sheet.paste(thumb, ((k % 4) * thumb_width, (k // 4) * thumb_height))
        sheet.save(output / "contact_sheet.jpg", quality=90)


def render_cached(args, experiment, output):
    """Render an already sufficient prediction prefix, with no model/GPU load."""
    source = Path(args.cached_predictions).resolve()
    source.relative_to(experiment)  # Refuse caches outside the selected experiment.
    selected, inspected = [], 0
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            inspected += 1
            if top2_rescues(record["top_ious"]):
                np.testing.assert_allclose(
                    overlaps(record["top_boxes"], record["gt_box"]),
                    record["top_ious"], atol=2e-5, rtol=1e-5,
                )
                selected.append(record)
                if args.max_images and len(selected) >= args.max_images:
                    break
    if not selected:
        raise ValueError("No Top-1-wrong / Top-2-correct samples in cache")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    from tqdm import tqdm
    for record in tqdm(selected, desc="Rendering cached Top-2 matches"):
        record["visualization"] = "images/{:04d}_{}.jpg".format(record["index"], Path(record["filename"]).stem)
        render(record, output / record["visualization"])
    with (output / "matched_samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "matched_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["index", "filename", "expression", "top1_score", "top2_score", "top1_iou", "top2_iou", "visualization"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            writer.writerow(dict(index=record["index"], filename=record["filename"], expression=record["expression"],
                                 top1_score=record["top_scores"][0], top2_score=record["top_scores"][1],
                                 top1_iou=record["top_ious"][0], top2_iou=record["top_ious"][1],
                                 visualization=record["visualization"]))
    summary = dict(experiment=str(experiment), split=args.split, checkpoint=str(experiment / args.checkpoint),
                   source_predictions=str(source), samples=inspected, complete_split=False,
                   matched_samples=len(selected), visualized_samples=len(selected),
                   selection="First requested number with Top-1 IoU < 0.5 AND Top-2 IoU >= 0.5",
                   additional_model_inference=False, coordinate_check="passed for every figure")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery(output, selected, summary)
    report = ("# CRS Top-2 正确的 100 个样本\n\n"
              "实验：`{experiment}`，使用此前以备份源码严格加载的 `top1_best.pth` 预测。\n\n"
              "已停止继续推理。本次没有加载模型或使用 GPU，直接复用已有预测，按验证集顺序取前 {visualized_samples} 个符合条件的样本；"
              "对应缓存前 {samples} 条记录，并非完整验证集统计。\n\n"
              "条件：Top-1 IoU < 0.5，Top-2 IoU >= 0.5。GT 仅用于离线筛选与绘图，不参与模型推理。\n\n"
              "GT 绿色实线；Top-1 红色实线；Top-2 蓝色虚线。每图包含完整文本、CRS 分数、IoU 和局部放大。"
              "GT 与 Top-2 放大图裁剪范围相同，没有移动框的位置。\n\n"
              "打开 `index.html` 浏览；图片在 `images/`；总览为 `contact_sheet.jpg`；"
              "坐标、文本、分数见 `matched_samples.jsonl`，简明清单见 `matched_samples.csv`。\n").format(**summary)
    (output / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="top1_best.pth")
    parser.add_argument("--max-images", type=int, default=50, help="0 renders every matching sample")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the entire split")
    parser.add_argument("--stop-after-matches", type=int, default=0, help="Stop inference after this many matches")
    parser.add_argument("--cached-predictions", help="Render existing predictions only; never load a model")
    args = parser.parse_args()
    if args.max_images < 0 or args.limit < 0:
        parser.error("max-images and limit must be >= 0")
    experiment, output = Path(args.experiment).resolve(), Path(args.output).resolve()
    if not (experiment / "propvg").is_dir():
        raise FileNotFoundError("An archived propvg package is required")
    if args.cached_predictions:
        render_cached(args, experiment, output)
        return
    sys.path.insert(0, str(experiment))
    import torch
    from mmcv import Config
    from mmcv.parallel import collate
    from tqdm import tqdm
    import propvg
    from propvg.datasets import build_dataset, extract_data
    from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
    from propvg.models import build_model

    if Path(propvg.__file__).resolve().parent != experiment / "propvg":
        raise RuntimeError("The archived experiment source was not imported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback for the full model")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_paths = list(experiment.glob("*_PropVG-aerialvg.py"))
    if len(config_paths) != 1:
        raise RuntimeError("Expected exactly one archived AerialVG configuration")
    config_path = config_paths[0]
    cfg = Config.fromfile(str(config_path))
    for name in ("enable_grab", "enable_bcem", "enable_relation_scoring"):
        if cfg.model.head.get(name, False):
            raise ValueError("This CRS-only BASE diagnostic requires {}=False".format(name))
    cfg.model.vis_enc.pretrain = None  # The trained checkpoint already includes BEiT-3.
    cfg.model.process_visual = False
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    dataset = build_dataset(cfg.data[args.split])
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    checkpoint_path = experiment / args.checkpoint
    print("Loading archived BASE checkpoint strictly: " + str(checkpoint_path), flush=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    meta = dict(experiment=str(experiment), configuration=str(config_path), checkpoint=str(checkpoint_path),
                checkpoint_epoch=int(checkpoint.get("epoch", -1)) + 1, imported_source=propvg.__file__,
                physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), split=args.split, batch_size=1,
                selection="Top-1 IoU < 0.5 AND Top-2 IoU >= 0.5; original CRS slots, no IoU grouping/NMS",
                ranking="CPU torch.argsort descending on postprocessed CRS probabilities, as in official evaluation",
                coordinate_system="original image pixels (xyxy); selection uses the evaluator's resized-image coordinates",
                annotation_file=str(cfg.data[args.split].annsfile), seed=args.seed,
                archived_best_top1=checkpoint.get("top1_acc"), archived_best_top5=checkpoint.get("top5_acc"))
    del checkpoint, state
    gc.collect()
    model = model.cuda().eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                                         collate_fn=partial(collate, samples_per_gpu=1))
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    records, matched = [], []
    start = time.time()
    with torch.no_grad(), (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, batch in enumerate(tqdm(loader, total=min(args.limit or len(dataset), len(dataset)), desc="CRS Top-2 analysis")):
            metadata = batch["img_metas"].data[0][0]
            target_index = int(np.asarray(metadata["refer_target_index"]).reshape(-1)[0])
            gt = batch.pop("gt_bbox").data[0][0][target_index].cpu().numpy()
            batch.pop("gt_mask_rle", None)
            batch.pop("is_crowd", None)
            prediction = model(**extract_data(batch), return_loss=False, with_bbox=True, with_mask=False,
                               rescale=False)["pred_bboxes"][0]
            if "ranking_logits" in prediction:
                raise RuntimeError("BASE should use the original CRS probability ordering")
            scores = prediction["scores"].detach().float().reshape(-1).cpu()
            boxes = prediction["boxes"].detach().float().cpu().numpy()
            if len(boxes) != len(scores) or len(boxes) < 2:
                raise RuntimeError("Missing or misaligned postprocessed boxes/scores")
            if not np.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite predictions")
            order = scores.argsort(descending=True).numpy()[:5]
            ranked_boxes = boxes[order]
            ious = overlaps(ranked_boxes, gt)
            scale = np.asarray(metadata["scale_factor"], dtype=np.float64).reshape(-1)
            if scale.size == 1:
                scale = np.repeat(scale, 4)
            original_gt = np.asarray(dataset.samples[index]["boxes"][dataset.samples[index]["refer_target_index"]], dtype=float)
            if not np.allclose(gt / scale, original_gt, atol=0.003, rtol=1e-6):
                raise RuntimeError("GT original/resized coordinate conversion mismatch")
            original_boxes = ranked_boxes / scale
            if not np.allclose(overlaps(original_boxes, original_gt), ious, atol=2e-5, rtol=1e-5):
                raise RuntimeError("IoU changed unexpectedly during coordinate rescaling")
            record = dict(index=index, filename=str(Path(metadata["filename"]).resolve()),
                          expression=metadata["expression"], gt_box=original_gt.tolist(),
                          top_boxes=original_boxes.tolist(), top_scores=scores[order].tolist(),
                          top_ious=ious.tolist(), filtered_candidate_indices=order.tolist(),
                          top1_correct=bool(ious[0] >= 0.5), top5_correct=bool(ious.max() >= 0.5),
                          top2_rescues_top1=bool(top2_rescues(ious)), visualization=None)
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["top2_rescues_top1"]:
                matched.append(record)
            if args.stop_after_matches and len(matched) >= args.stop_after_matches:
                break
            if (index + 1) % 100 == 0:
                stream.flush()
                print("Processed {}; matching samples {}".format(index + 1, len(matched)), flush=True)
            if args.limit and index + 1 >= args.limit:
                break
    # Release the GPU before CPU-only image rendering.
    del model, prediction, batch
    gc.collect()
    torch.cuda.empty_cache()
    selected = matched
    if args.max_images and len(matched) > args.max_images:
        selected = sorted(random.Random(args.seed).sample(matched, args.max_images), key=lambda row: row["index"])
    for record in tqdm(selected, desc="Drawing GT / Top-1 / Top-2"):
        record["visualization"] = "images/{:04d}_{}.jpg".format(record["index"], Path(record["filename"]).stem)
        render(record, output / record["visualization"])
    with (output / "matched_samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in matched:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "matched_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["index", "filename", "expression", "top1_score", "top2_score", "top1_iou", "top2_iou", "visualization"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in matched:
            writer.writerow(dict(index=record["index"], filename=record["filename"], expression=record["expression"],
                                 top1_score=record["top_scores"][0], top2_score=record["top_scores"][1],
                                 top1_iou=record["top_ious"][0], top2_iou=record["top_ious"][1],
                                 visualization=record["visualization"]))
    summary = dict(meta, samples=len(records), complete_split=len(records) == len(dataset),
                   matched_samples=len(matched), visualized_samples=len(selected),
                   top1_acc=100 * float(np.mean([row["top1_correct"] for row in records])),
                   top5_acc=100 * float(np.mean([row["top5_correct"] for row in records])),
                   box_miou=100 * float(np.mean([row["top_ious"][0] for row in records])),
                   elapsed_seconds=time.time() - start,
                   peak_cuda_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery(output, selected, summary)
    report = """# CRS Top-2 正确样本可视化

- 实验：`{experiment}`
- 模型：`{checkpoint}`，第 {checkpoint_epoch} epoch 的最佳模型；严格加载当次备份代码。
- 数据：{split}，{samples} 个样本；无分布式补齐重复样本。
- 筛选：Top-1 IoU < 0.5，Top-2 IoU >= 0.5；共 **{matched_samples}** 个。
- 已绘制：{visualized_samples} 张，固定随机种子 {seed}，不是按错误程度挑选。
- 全集指标：Top-1 {top1_acc:.4f}%，Top-5 {top5_acc:.4f}%，box mIoU {box_miou:.4f}%。
- 颜色：GT 绿色实线；Top-1 红色实线；Top-2 蓝色虚线。
- 图上保留完整文本、CRS 原始分数和 IoU。CRS 分数不等于校准的正确概率。
- 下方 GT 与 Top-2 放大图使用完全相同的裁剪范围，所有框的位置均未移动。
- 排序遵循原评估：CPU 上按 CRS 概率降序，不改为 logit，不做去重或新 NMS。
- GT 只用于离线筛选、IoU 和画图，不送入模型。
- 全量清单：`matched_samples.csv` / `matched_samples.jsonl`；全验证集 Top-5 预测：`predictions.jsonl`。
- 浏览入口：`index.html`；图片目录：`images/`；总览：`contact_sheet.jpg`。
- 单卡去掉了多卡评估的尾部补齐样本，指标可能与日志有约 0.02 个百分点的差异。
""".format(**summary)
    (output / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
