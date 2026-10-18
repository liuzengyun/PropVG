"""Read-only model diagnosis: archived CRS logits, entropy and grounding errors.

Uses the experiment's archived propvg package, not the modified training code.
GT is removed from inference inputs and is used only for offline statistics.
"""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time


def iou(boxes, target):
    import numpy as np
    lo = np.maximum(boxes[:, :2], target[:2])
    hi = np.minimum(boxes[:, 2:], target[2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    gt_area = np.maximum(target[2:] - target[:2], 0).prod()
    return inter / np.maximum(area + gt_area - inter, 1e-8)


def entropy(logits, temperature=1.):
    import numpy as np
    if len(logits) < 2:
        return 0.
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    q = np.exp(scaled - scaled.max())
    q /= q.sum()
    return float(-(q * np.log(np.maximum(q, 1e-30))).sum() / math.log(len(q)))


def representatives(boxes, threshold=.8):
    kept = []
    for index, box in enumerate(boxes):
        if not kept or iou(boxes[kept], box).max() < threshold:
            kept.append(index)
    return kept


def auc(labels, scores):
    import numpy as np
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    rank_sum, start = 0., 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        rank_sum += labels[order[start:end]].sum() * ((start + 1 + end) / 2)
        start = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def describe(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                quantiles={str(q): float(np.quantile(values, q)) for q in (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)})


def summarize(rows, meta, top_logits, top_boxes, output):
    import numpy as np
    correct = np.array([row["correct"] for row in rows], dtype=bool)
    reachable = np.array([row["top5_hit"] for row in rows], dtype=bool)
    result = dict(meta, samples=len(rows),
                  metrics=dict(top1=100 * float(correct.mean()), top5=100 * float(reachable.mean()),
                               miou=100 * float(np.mean([row["top1_iou"] for row in rows]))),
                  entropy={}, uncertainty_detection_auc={}, repairable_error_detection_auc={}, thresholds={}, temperatures={})
    groups = {"all": np.ones(len(rows), dtype=bool), "correct": correct, "wrong": ~correct,
              "wrong_but_top5_has_target": ~correct & reachable, "top5_has_no_target": ~reachable}
    for key in ("raw_entropy", "grouped_entropy", "probability_gap", "logit_gap"):
        values = np.array([row[key] for row in rows])
        result["entropy"][key] = {name: describe(values[mask]) for name, mask in groups.items()}
        result["uncertainty_detection_auc"][key] = auc(~correct, -values if "gap" in key else values)
        scores = -values if "gap" in key else values
        result["repairable_error_detection_auc"][key] = auc((~correct)[reachable], scores[reachable])
    for key in ("raw_entropy", "grouped_entropy"):
        values = np.array([row[key] for row in rows])
        entries = []
        for threshold in (.4, .8, .9, .95, .98, .99, .995, .999):
            selected = values > threshold
            entries.append(dict(threshold=threshold, selected_count=int(selected.sum()),
                                selected_pct=float(selected.mean() * 100),
                                selected_error_pct=float((~correct[selected]).mean() * 100) if selected.any() else None,
                                error_recall_pct=float((selected & ~correct).sum() / max((~correct).sum(), 1) * 100),
                                correct_exposure_pct=float((selected & correct).sum() / max(correct.sum(), 1) * 100)))
        result["thresholds"][key] = entries
    triggers = np.array([row["trigger"] for row in rows])
    active = triggers > 0
    result["current_gate"] = dict(active_count=int(active.sum()), active_pct=float(active.mean() * 100),
                                  mean_trigger=float(triggers.mean()),
                                  error_recall_pct=float((active & ~correct).sum() / max((~correct).sum(), 1) * 100),
                                  correct_exposure_pct=float((active & correct).sum() / max(correct.sum(), 1) * 100),
                                  selected_error_pct=float((~correct[active]).mean() * 100) if active.any() else None,
                                  single_group_count=sum(row["group_count"] < 2 for row in rows),
                                  confident_gap_protected_count=sum(row["group_probability_gap"] >= .35 for row in rows))
    result["group_count"] = {str(count): sum(row["group_count"] == count for row in rows) for count in range(1, 6)}
    result["cpu_gpu_top1_order_disagreements"] = sum(row["cpu_gpu_order_disagrees"] for row in rows)
    archived_world_size = int(meta.get("archived_world_size", 1))
    padding = (-len(rows)) % archived_world_size
    padded_rows = rows + rows[:padding]
    result["archived_sampler_reconstruction"] = dict(
        world_size=archived_world_size, padded_duplicates=padding,
        top1=float(np.mean([row["correct"] for row in padded_rows]) * 100),
        top5=float(np.mean([row["top5_hit"] for row in padded_rows]) * 100),
        miou=float(np.mean([row["top1_iou"] for row in padded_rows]) * 100))
    def selection(mask):
        count = int(mask.sum())
        repairable = mask & ~correct & reachable
        return dict(count=count, percent=float(mask.mean() * 100),
                    correct_count=int((mask & correct).sum()), wrong_count=int((mask & ~correct).sum()),
                    error_rate=float((~correct[mask]).mean() * 100) if count else None,
                    repairable_count=int(repairable.sum()),
                    error_recall=float((mask & ~correct).sum() / max((~correct).sum(), 1) * 100),
                    repairable_recall=float(repairable.sum() / max((~correct & reachable).sum(), 1) * 100))
    grouped_values = np.array([row["grouped_entropy"] for row in rows])
    result["routing_stages"] = {
        "entropy_gt_0.4_only": selection(grouped_values > .4),
        "current_entropy_and_gap": selection(active),
        "full_trigger": selection(triggers >= 1.),
        "partial_trigger": selection((triggers > 0) & (triggers < 1.)),
        "skipped": selection(~active),
    }
    required_alpha = np.full(len(rows), np.inf)
    for index, row in enumerate(rows):
        positive = [row["z%d" % rank] for rank in range(1, 6) if row["iou%d" % rank] >= .5]
        if positive and not correct[index]:
            required_alpha[index] = max(0., (row["z1"] - max(positive)) / 2)
    # Necessary, not sufficient: a bounded +/- alpha correction can change
    # a positive-vs-wrong logit difference by at most 2*alpha.
    result["bounded_fusion_feasibility"] = dict(
        note="Oracle upper bounds, not actual CRV gains; tanh drops need not attain +/-1.",
        repairable_total=int((~correct & reachable).sum()),
        selected_repairable=int((active & ~correct & reachable).sum()),
        required_alpha_for_repairable=describe(required_alpha[~correct & reachable]),
        upper_bounds={str(cap): dict(
            possible_count=int((active & (required_alpha < cap * triggers)).sum()),
            possible_top1_gain_pp=float((active & (required_alpha < cap * triggers)).sum() / len(rows) * 100))
            for cap in (.1, .25, .5, 1., 2.)})
    result["group_count_by_correctness"] = {
        str(count): selection(np.array([row["group_count"] == count for row in rows])) for count in range(1, 6)}
    for temperature in (.25, .5, 1., 2.):
        raw = np.array([entropy(z[:5], temperature) for z in top_logits])
        grouped = np.array([entropy(z[representatives(b[:5])], temperature) for z, b in zip(top_logits, top_boxes)])
        result["temperatures"][str(temperature)] = dict(raw_mean=float(raw.mean()), grouped_mean=float(grouped.mean()),
                                                       raw_error_auc=auc(~correct, raw), grouped_error_auc=auc(~correct, grouped))
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def plot(rows, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    correct = np.array([row["correct"] for row in rows], dtype=bool)
    raw = np.array([row["raw_entropy"] for row in rows])
    grouped = np.array([row["grouped_entropy"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, values, title in ((axes[0, 0], raw, "Top-5 entropy (no grouping)"),
                               (axes[0, 1], grouped, "Top-5 entropy (IoU grouping)")):
        for mask, label, color in ((correct, "CRS correct", "#2678b2"), (~correct, "CRS wrong", "#d55e00")):
            ax.hist(values[mask], bins=np.linspace(0, 1, 51), density=True, alpha=.5, label=label, color=color)
        for value in (.4, .8):
            ax.axvline(value, color="gray", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Normalized entropy", ylabel="Density", xlim=(0, 1))
        ax.legend()
    for mask, label, color in ((correct, "CRS correct", "#2678b2"), (~correct, "CRS wrong", "#d55e00")):
        values = np.sort(raw[mask])
        axes[1, 0].plot(values, np.arange(1, len(values) + 1) / max(len(values), 1) * 100,
                        color=color, label=label)
    for value in (.4, .8):
        axes[1, 0].axvline(value, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Cumulative distribution (no grouping)", xlabel="Normalized entropy", ylabel="Samples at or below threshold (%)")
    axes[1, 0].legend()
    for values, name in ((raw, "No grouping"), (grouped, "IoU grouping")):
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        chunks = [np.flatnonzero((values >= lo) & ((values <= hi) if i == len(edges) - 2 else (values < hi)))
                  for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
        chunks = [part for part in chunks if len(part)]
        x = [float(values[part].mean()) for part in chunks]
        y = [float((~correct[part]).mean() * 100) for part in chunks]
        axes[1, 1].plot(x, y, marker="o", label=name)
    axes[1, 1].axhline((~correct).mean() * 100, color="gray", linestyle="--", label="Overall error rate")
    axes[1, 1].set(title="Error rate by entropy decile", xlabel="Mean entropy in equal-count bin", ylabel="Top-1 error (%)")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output / "entropy_distribution.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not experiment.joinpath("propvg").is_dir():
        raise FileNotFoundError("Archived propvg source is required for faithful analysis")
    sys.path.insert(0, str(experiment))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
    import numpy as np
    import torch
    from mmcv import Config
    from propvg.datasets import build_dataset, build_dataloader, extract_data
    from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
    from propvg.models import build_model
    import propvg
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = next(experiment.glob("*_PropVG-aerialvg.py"))
    checkpoint_path = experiment / "top1_best.pth"
    cfg = Config.fromfile(str(config_path))
    archived_world_size = int(cfg.get("world_size", 1))
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    cfg.data.samples_per_gpu, cfg.data.workers_per_gpu = 1, 4
    cfg.model.vis_enc.pretrain = None  # Full checkpoint already contains BEiT-3.
    cfg.model.process_visual = False
    dataset = build_dataset(cfg.data[args.split])
    loader = build_dataloader(cfg, dataset)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    meta = dict(experiment=str(experiment), split=args.split, configuration=str(config_path),
                checkpoint=str(checkpoint_path), checkpoint_epoch=int(checkpoint.get("epoch", -1)) + 1,
                imported_source=propvg.__file__, batch_size=1,
                physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                entropy_topn=5, duplicate_iou=.8, temperature=1.,
                archived_world_size=archived_world_size,
                expected_val_top1=checkpoint.get("top1_acc"), expected_val_top5=checkpoint.get("top5_acc"))
    del state, checkpoint
    gc.collect()
    model = model.cuda().eval()
    captured = {}
    original = model.head.mrsm.forward
    def capture_logits(*positional, **kwargs):
        values = original(*positional, **dict(kwargs, return_logits=True))
        captured["logits"] = values[2].detach()
        captured["gpu_order"] = values[0].squeeze(-1).argsort(dim=1, descending=True).detach()
        return values[:2]
    model.head.mrsm.forward = capture_logits
    rows, top_logits, top_boxes = [], [], []
    start = time.time()
    writer = None
    with output.joinpath("samples.csv").open("w", newline="", encoding="utf-8") as stream, torch.no_grad():
        for index, inputs in enumerate(loader):
            metas = inputs["img_metas"].data[0]
            gt = inputs.pop("gt_bbox").data[0][0][metas[0]["refer_target_index"][0]].cpu().numpy()
            inputs.pop("gt_mask_rle", None)
            inputs.pop("is_crowd", None)
            inputs = extract_data(inputs)
            prediction = model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=False)["pred_bboxes"][0]
            boxes = prediction["boxes"].detach().cpu().numpy()
            probabilities = prediction["scores"].detach().cpu()
            logits = captured["logits"][0, :, 0].cpu().numpy()
            if len(boxes) != len(logits):
                raise RuntimeError("Degenerate box filtering needs explicit index alignment; no silent truncation allowed")
            order = probabilities.argsort(descending=True).numpy()
            gpu_order = captured["gpu_order"][0].cpu().numpy()
            overlaps = iou(boxes, gt)
            sorted_z, sorted_b = logits[order], boxes[order]
            p = probabilities.numpy()[order]
            kept = representatives(sorted_b[:5])
            h = entropy(sorted_z[:5])
            grouped_h = entropy(sorted_z[kept])
            group_gap = float(p[kept[0]] - p[kept[1]]) if len(kept) > 1 else 0.
            trigger = float(np.clip((grouped_h - .4) / .4, 0, 1) * (group_gap < .35)) if len(kept) > 1 else 0.
            row = dict(index=index, filename=metas[0]["filename"], expression=metas[0]["expression"],
                       correct=int(overlaps[order[0]] >= .5), top5_hit=int(overlaps[order[:5]].max() >= .5),
                       top8_hit=int(overlaps[order[:8]].max() >= .5), top1_iou=float(overlaps[order[0]]),
                       raw_entropy=h, grouped_entropy=grouped_h, group_count=len(kept),
                       probability_gap=float(p[0] - p[1]), group_probability_gap=group_gap,
                       logit_gap=float(sorted_z[0] - sorted_z[1]), trigger=trigger,
                       cpu_gpu_order_disagrees=int(order[0] != gpu_order[0]))
            for rank in range(5):
                row["p%d" % (rank + 1)] = float(p[rank])
                row["z%d" % (rank + 1)] = float(sorted_z[rank])
                row["iou%d" % (rank + 1)] = float(overlaps[order[rank]])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            rows.append(row)
            top_logits.append(sorted_z[:10])
            top_boxes.append(sorted_b[:10])
            if len(rows) % 100 == 0 or len(rows) == 1:
                stream.flush()
                progress = dict(processed=len(rows), total=min(len(dataset), args.limit or len(dataset)),
                                elapsed_seconds=round(time.time() - start, 1),
                                top1=round(np.mean([r["correct"] for r in rows]) * 100, 3),
                                mean_raw_entropy=round(np.mean([r["raw_entropy"] for r in rows]), 6),
                                mean_grouped_entropy=round(np.mean([r["grouped_entropy"] for r in rows]), 6),
                                active_pct=round(np.mean([r["trigger"] > 0 for r in rows]) * 100, 3))
                output.joinpath("progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                print(json.dumps(progress), flush=True)
            if args.limit and len(rows) >= args.limit:
                break
    meta["elapsed_seconds"] = time.time() - start
    meta["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
    meta["complete_split"] = len(rows) == len(dataset)
    np.savez_compressed(output / "top10_predictions.npz", logits=np.array(top_logits), boxes=np.array(top_boxes))
    result = summarize(rows, meta, top_logits, top_boxes, output)
    plot(rows, output)
    print(json.dumps(dict(output=str(output), samples=len(rows), metrics=result["metrics"],
                          gate=result["current_gate"], auc=result["uncertainty_detection_auc"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
