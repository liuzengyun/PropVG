"""Visualize real AerialVG predictions, proposal coverage and BCEM diagnostics.

Use the experiment's archived config and top1_best.pth. GT annotations are
removed before inference; attention weights are observations, not causal tests.
The training model, losses and CRS mask behavior are not modified.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import html
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/propvg_cache")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate

from propvg.datasets import build_dataset
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.models import build_model
from visualize_aerialvg import box_iou


DIRECTIONS = ("up", "down", "left", "right", "upper_left", "upper_right",
              "lower_left", "lower_right")
LABELS = {"correct": "预测正确", "proposal_miss": "候选覆盖不足",
          "ranking_top5": "排序错误：正确候选在 Top5",
          "ranking_beyond5": "排序错误：正确候选在 Top5 之外"}


def analyze_candidates(boxes, scores, gt_box, ranking=None, stable=False):
    """Classify observable failures using ALL postprocessed candidates."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    gt_box = np.asarray(gt_box, dtype=np.float32)
    scores = np.asarray(scores).reshape(-1)
    ranking = scores if ranking is None else np.asarray(ranking).reshape(-1)
    if len(boxes) != len(scores) or len(scores) != len(ranking):
        raise ValueError("candidate boxes, scores and ranking must align")
    order = torch.as_tensor(ranking).argsort(descending=True, stable=stable).numpy()
    ious = np.asarray([box_iou(box, gt_box) for box in boxes], dtype=np.float32)
    top1_iou = float(ious[order[0]]) if len(order) else 0.0
    correct_ranks = np.flatnonzero(ious[order] >= 0.5) + 1
    first_rank = int(correct_ranks[0]) if len(correct_ranks) else None
    best = int(ious.argmax()) if len(ious) else None
    best_rank = int(np.flatnonzero(order == best)[0] + 1) if best is not None else None
    kind = ("correct" if top1_iou >= 0.5 else "proposal_miss" if first_rank is None
            else "ranking_top5" if first_rank <= 5 else "ranking_beyond5")
    return {
        "failure_type": kind, "top1_iou": top1_iou,
        "top1_correct_at_0.5": top1_iou >= 0.5,
        "top5_correct_at_0.5": first_rank is not None and first_rank <= 5,
        "candidate_count": len(boxes),
        "max_candidate_iou": float(ious[best]) if best is not None else 0.0,
        "first_correct_rank": first_rank, "best_iou_rank": best_rank,
    }, order, ious, best


def move_batch(batch, device):
    """Unwrap a single collate group without the training helper's CUDA call."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value
    return {key: value.data[0] if value.cpu_only else move(value.data[0])
            for key, value in batch.items()}


@contextmanager
def capture_head(model):
    """Capture existing diagnostics in the SAME forward, then restore method."""
    original = model.head.forward_test
    captured = {}
    def forward(*args, **kwargs):
        prediction, extra = original(*args, **kwargs)
        captured.clear()
        captured["boxes"] = prediction["pred_bbox_allobj"].detach().cpu()
        captured["extra"] = {key: value.detach().cpu() for key, value in extra.items()
                             if key.startswith("bcem_") and torch.is_tensor(value)}
        return prediction, extra
    model.head.forward_test = forward
    try:
        yield captured
    finally:
        model.head.forward_test = original


def kept_candidate_ids(raw_boxes, meta, post_boxes):
    """Reproduce clipping/nonempty filtering; assert alignment before plotting."""
    h, w = meta["img_shape"][:2]
    boxes = torch.cat((raw_boxes[:, :2] - raw_boxes[:, 2:] / 2,
                       raw_boxes[:, :2] + raw_boxes[:, 2:] / 2), -1).numpy()
    boxes *= np.asarray([w, h, w, h])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    kept = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    scale = np.asarray(meta["scale_factor"]).reshape(-1)
    original_boxes = boxes[kept] / (np.repeat(scale, 4) if scale.size == 1 else scale[:4])
    if original_boxes.shape != post_boxes.shape or not np.allclose(original_boxes, post_boxes, atol=0.01):
        raise RuntimeError("candidate/BCEM alignment failed; refusing misleading diagnostics")
    return kept


def background_info(extra, batch_index, candidate_index):
    fields = {"direction_weights": "bcem_direction_weights", "null_weight": "bcem_null_weight",
              "gate": "bcem_background_weight", "residual_ratio": "bcem_residual_ratio",
              "valid_regions": "bcem_valid_regions", "coverage": "bcem_region_coverage"}
    return {name: extra[key][batch_index, candidate_index].tolist()
            for name, key in fields.items() if key in extra}


def sampling_points(module, raw_box, meta):
    """Reuse the module's actual geometry, in original image coordinates."""
    device = module.sector_rays.device
    sizes = torch.tensor([meta["img_shape"][:2]], dtype=torch.float32, device=device)
    points = module._sampling_geometry(raw_box.to(device).reshape(1, 1, 4), sizes)[0][0, 0]
    scale = np.asarray(meta["scale_factor"]).reshape(-1)
    scale_xy = np.repeat(scale, 2) if scale.size == 1 else scale[:2]
    return points.cpu().numpy() / scale_xy


def draw_box(ax, box, color, label=None):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
    if label:
        ax.text(x1, y1, label, color="white", fontsize=8, va="bottom",
                bbox={"facecolor": color, "alpha": 0.8, "pad": 1})


def crop(ax, image, box, gt_box, pred_box, title):
    ax.imshow(image)
    draw_box(ax, gt_box, "lime")
    draw_box(ax, pred_box, "red")
    center = (np.asarray(box[:2]) + box[2:]) / 2
    half = max(float(np.max(np.asarray(box[2:]) - box[:2])) * 1.8, 45)
    ax.set_xlim(max(0, center[0] - half), min(image.shape[1], center[0] + half))
    ax.set_ylim(min(image.shape[0], center[1] + half), max(0, center[1] - half))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def sector_panel(ax, image, box, bg, points, title, direction_angles=None):
    ax.imshow(image)
    draw_box(ax, box, "red")
    if not bg:
        ax.set_title(title + " (BCEM disabled)")
        ax.axis("off")
        return
    weights = np.asarray(bg["direction_weights"])
    valid = np.asarray(bg["valid_regions"])
    # Shading shows selected sectors. The dots are ACTUAL sampling positions;
    # region boundaries are schematic ray-wise fractions, not semantic masks.
    center = (np.asarray(box[:2]) + box[2:]) / 2
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    angles = np.arctan2(yy - center[1], xx - center[0])
    if direction_angles is None:
        direction_angles = np.asarray([-np.pi / 2, np.pi / 2, np.pi, 0,
                                       -3 * np.pi / 4, -np.pi / 4, 3 * np.pi / 4, np.pi / 4])
    if len(direction_angles) != len(weights):
        raise ValueError("sector plot angles must match the model's sector count")
    distance = np.abs((angles[..., None] - direction_angles + np.pi) % (2 * np.pi) - np.pi)
    sectors = distance.argmin(-1)
    alpha = weights[sectors] * 0.6 / max(weights.max(), 1e-8)
    alpha[(xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])] = 0
    overlay = np.zeros((*alpha.shape, 4), dtype=np.float32)
    overlay[..., 1:3] = 1
    overlay[..., 3] = alpha
    ax.imshow(overlay)
    for direction in range(len(weights)):
        for region in range(points.shape[1]):
            p = points[direction, region]
            ax.scatter(p[:, 0], p[:, 1], s=5,
                       c="cyan" if valid[direction, region] else "gray", alpha=0.65)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title(title + "\nsector selection (schematic); gray dots = invalid region", fontsize=9)
    ax.axis("off")


def draw_case(record, image, ranked_boxes, ranked_scores, ious, best_box, backgrounds,
              points, output_dir, direction_names=DIRECTIONS, direction_angles=None):
    pred_box, gt_box = ranked_boxes[0], np.asarray(record["gt_box"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes[0, 0].imshow(image)
    for rank in range(min(5, len(ranked_boxes)) - 1, -1, -1):
        draw_box(axes[0, 0], ranked_boxes[rank], "red" if rank == 0 else "orange", str(rank + 1))
    draw_box(axes[0, 0], gt_box, "lime", "GT")
    axes[0, 0].set_title("Full image: green = GT, red = Top1, orange = Top2-5")
    axes[0, 0].axis("off")
    crop(axes[0, 1], image, gt_box, gt_box, pred_box, "GT target and surrounding context")
    crop(axes[0, 2], image, pred_box, gt_box, pred_box,
         "Top1 prediction: IoU={:.3f}".format(record["top1_iou"]))
    sector_panel(axes[1, 0], image, pred_box, backgrounds[0], points[0], "Top1 background", direction_angles)
    sector_panel(axes[1, 1], image, best_box, backgrounds[1], points[1],
                 "Best-IoU proposal background (CRS rank {})".format(record["best_iou_rank"]), direction_angles)
    ax = axes[1, 2]
    ticks = np.arange(len(direction_names) + 1)
    for k, (bg, label, color) in enumerate(zip(backgrounds, ("Top1", "Best IoU"), ("tomato", "steelblue"))):
        if bg:
            weights = np.r_[bg["direction_weights"], bg["null_weight"]]
            ax.barh(ticks + (k - .5) * .35, weights, height=.35, label=label, color=color)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tuple(direction_names) + ("null",))
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Sparse sector selection weight")
    if backgrounds[0]:
        ax.legend()
        desc = " | ".join("{}: gate={:.3f}, residual ratio={:.3f}".format(label, bg["gate"][0], bg["residual_ratio"][0])
                          for label, bg in zip(("Top1", "Best IoU"), backgrounds))
        ax.set_title(desc, fontsize=8, wrap=True)
    import textwrap
    title = "Sample {} | {} | first IoU>=0.5 rank={} | max candidate IoU={:.3f}\n{}".format(
        record["index"], record["failure_type"], record["first_correct_rank"],
        record["max_candidate_iou"], "\n".join(textwrap.wrap(record["expression"], 130)))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90))
    filename = "case_{:06d}.jpg".format(record["index"])
    fig.savefig(output_dir / filename, dpi=135)
    plt.close(fig)
    # Give each Top5 candidate a separate magnified view for tiny aerial targets.
    fig, axes = plt.subplots(1, min(5, len(ranked_boxes)), figsize=(15, 3), squeeze=False)
    for rank, ax in enumerate(axes[0]):
        crop(ax, image, ranked_boxes[rank], gt_box, ranked_boxes[rank],
             "Top{}: score={:.4f}, IoU={:.3f}".format(rank + 1, ranked_scores[rank], ious[rank]))
    fig.tight_layout()
    strip = "top5_{:06d}.jpg".format(record["index"])
    fig.savefig(output_dir / strip, dpi=130)
    plt.close(fig)
    record["visualization"] = filename
    record["top5_crops"] = strip


def write_gallery(records, summary, output_dir):
    cards = []
    for record in records:
        if not record.get("visualization"):
            continue
        category = record["failure_type"]
        table = "".join("<tr><td>{}</td><td>{:.6f}</td><td>{:.3f}</td></tr>".format(
            i + 1, score, iou) for i, (score, iou) in enumerate(zip(record["top_scores"], record["top_ious"])))
        cards.append('<article data-kind="{}"><h2>#{} · {}</h2><p>{}</p>'
                     '<p>Top1 IoU {:.3f}；候选最大 IoU {:.3f}；首个合格候选排名 {}；最佳 IoU 候选排名 {}</p>'
                     '<a href="{}" target="_blank"><img loading="lazy" src="{}"></a>'
                     '<img loading="lazy" src="{}"><details><summary>Top5 分数与 IoU</summary>'
                     '<table><tr><th>排名</th><th>CRS 分数</th><th>IoU</th></tr>{}</table></details></article>'.format(
                         category, record["index"], LABELS[category], html.escape(record["expression"]),
                         record["top1_iou"], record["max_candidate_iou"], record["first_correct_rank"],
                         record["best_iou_rank"], record["visualization"], record["visualization"],
                         record["top5_crops"], table))
    buttons = '<button onclick="filter(\'all\')">全部</button>' + "".join(
        '<button onclick="filter(\'{}\')">{} ({})</button>'.format(key, label, summary["failure_counts"].get(key, 0))
        for key, label in LABELS.items())
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>BCEM 错误案例</title>
<style>body{max-width:1450px;margin:28px auto;padding:0 18px;font:16px/1.7 sans-serif;background:#f4f6f9;color:#17212f}
article{background:white;margin:24px 0;padding:20px;border-radius:12px}img{width:100%%;height:auto}
button{padding:8px 16px;margin:5px;cursor:pointer}td,th{padding:4px 18px;text-align:left}.note{background:#e5edf8;padding:16px}
code{overflow-wrap:anywhere}</style><h1>BCEM 预测与错误案例</h1>
<p>样本数：%d / %d · Top1：%.2f%% · Top5：%.2f%% · mIoU：%.2f%%</p>
<p>范围：%s。<b>部分样本结果不能替代全量评估指标。</b></p>
<p>模型：<code>%s</code></p><div class="note">绿色是真实框，红色是预测框，橙色是其他 Top5 候选。
“候选覆盖不足”表示所有输出候选的 IoU 均小于 0.5；“排序错误”表示存在合格候选但 Top1 排错。
背景图比较预测候选与最佳 IoU 候选，显示真实采样位置、扇区选择权重和融合门控。
扇区着色是权重的几何示意，不是像素级注意力；边缘候选以实际采样点为准。
门控不是正确概率，注意力及残差比例也不能单独证明背景导致了错误。灰点表示所在区域无效。
单击大图可放大；下面的五格图用于检查小目标外观。</div>%s%s
<script>function filter(kind){document.querySelectorAll('article').forEach(el=>el.style.display=kind==='all'||el.dataset.kind===kind?'':'none')}</script></html>''' % (
        summary["samples"], summary["split_size"], summary["top1_acc_at_0.5"],
        summary["top5_acc_at_0.5"], summary["miou"], html.escape(summary["scope"]),
        html.escape(summary["checkpoint"]), buttons, "".join(cards))
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", default="visualization/bcem_errors")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8, help="CPU intra-op threads")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = remaining complete split")
    parser.add_argument("--max-images", type=int, default=60, help="cap saved cases; inference continues")
    parser.add_argument("--include-correct", action="store_true", help="also draw correct cases")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if min(args.batch_size, args.threads, args.max_images) < 1 or min(args.start_index, args.max_samples) < 0:
        raise ValueError("batch-size, threads and max-images must be positive; indices/limits nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; pass --device cpu for a small diagnostic run")
        torch.cuda.set_device(device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "predictions.jsonl").exists():
        raise FileExistsError("Use a new output-dir to preserve previous results")
    torch.set_num_threads(args.threads)
    cfg = Config.fromfile(args.config)
    if cfg.dataset != "AerialVG":
        raise ValueError("this script expects the AerialVG dataset")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset = build_dataset(getattr(cfg.data, args.split))
    if args.start_index >= len(dataset):
        raise ValueError("start-index outside dataset")
    end = min(len(dataset), args.start_index + args.max_samples) if args.max_samples else len(dataset)
    # The trained checkpoint includes the backbone; skip its initialization file.
    cfg.model.vis_enc.pretrain = None
    print("Building model and loading trained weights on CPU...", flush=True)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["state_dict"]
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    metadata = {key: ckpt[key] for key in ("epoch", "top1_acc", "top5_acc", "mIoU") if key in ckpt}
    del ckpt, state
    model.to(device).eval()
    module = getattr(model.head, "bcem", None)
    direction_names = module.direction_names if module is not None else DIRECTIONS
    direction_angles = module.sector_angles.detach().cpu().numpy() if module is not None else None
    if model.with_nms:
        raise ValueError("this analysis follows unfiltered baseline ranking; use an experiment with with_nms=False")
    print("Strict checkpoint load passed: {}. Running {} samples on {}.".format(metadata, end - args.start_index, device), flush=True)
    records, saved = [], 0
    with capture_head(model) as captured, (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for begin in range(args.start_index, end, args.batch_size):
            indices = list(range(begin, min(begin + args.batch_size, end)))
            batch = collate([dataset[index] for index in indices], samples_per_gpu=args.batch_size)
            for key in ("gt_bbox", "gt_mask_rle", "is_crowd"):
                batch.pop(key, None)
            inputs = move_batch(batch, device)
            outputs = model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=True)
            for bi, (index, prediction, meta) in enumerate(zip(indices, outputs["pred_bboxes"], inputs["img_metas"])):
                sample = dataset.samples[index]
                gt_box = np.asarray(sample["boxes"][sample["refer_target_index"]], dtype=np.float32)
                boxes = prediction["boxes"].detach().cpu().numpy()
                scores = prediction["scores"].detach().cpu().reshape(-1).numpy()
                ranking = prediction.get("ranking_logits", prediction["scores"]).detach().cpu().reshape(-1).numpy()
                record, order, ious, best = analyze_candidates(boxes, scores, gt_box, ranking, "ranking_logits" in prediction)
                raw_boxes = captured["boxes"][bi]
                kept = kept_candidate_ids(raw_boxes, meta, boxes)
                extra = captured["extra"]
                bg = [background_info(extra, bi, int(kept[ci])) for ci in (order[0], best)] if len(order) else [{}, {}]
                record.update(index=index, filename=meta["filename"], expression=meta["expression"], gt_box=gt_box.tolist(),
                              top_boxes=boxes[order[:5]].tolist(), top_scores=scores[order[:5]].tolist(),
                              top_ious=ious[order[:5]].tolist(), top_candidate_ids=kept[order[:5]].tolist(),
                              best_iou_box=boxes[best].tolist() if best is not None else None,
                              prediction_background=bg[0], best_iou_background=bg[1])
                if saved < args.max_images and len(order) and (args.include_correct or not record["top1_correct_at_0.5"]):
                    image = plt.imread(meta["filename"])
                    points = [sampling_points(module, raw_boxes[int(kept[ci])], meta) for ci in (order[0], best)] if bg[0] else [None, None]
                    draw_case(record, image, boxes[order], scores[order], ious[order], boxes[best], bg, points,
                              output_dir, direction_names, direction_angles)
                    saved += 1
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print("sample {}: {} | IoU {:.3f} | first correct rank {}".format(index, record["failure_type"], record["top1_iou"], record["first_correct_rank"]), flush=True)
    summary = {"samples": len(records), "split_size": len(dataset), "split": args.split,
               "scope": "indices [{}:{})".format(args.start_index, end),
               "full_split": args.start_index == 0 and end == len(dataset), "saved_cases": saved,
               "top1_acc_at_0.5": 100 * np.mean([r["top1_correct_at_0.5"] for r in records]),
               "top5_acc_at_0.5": 100 * np.mean([r["top5_correct_at_0.5"] for r in records]),
               "miou": 100 * np.mean([r["top1_iou"] for r in records]),
               "failure_counts": dict(Counter(r["failure_type"] for r in records)),
               "checkpoint": str(Path(args.checkpoint).resolve()), "config": str(Path(args.config).resolve()),
               "device": str(device), "checkpoint_metadata": metadata, "directions": direction_names,
               "num_sectors": len(direction_names) if module is not None else None,
               "sector_angles_radians": direction_angles.tolist() if direction_angles is not None else None,
               "notes": "Weights are observational diagnostics, not causal attribution. GT is never supplied to inference."}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery(records, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Open gallery: {}".format(output_dir / "index.html"))


if __name__ == "__main__":
    main()
