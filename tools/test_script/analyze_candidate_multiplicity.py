"""Frozen-model CRS context-multiplicity diagnosis; never edits model weights.

All 256 query/output slots, boxes, image and text are held fixed. Only CRS
self-attention key/value multiplicity is changed, via the exactly equivalent
additive log(multiplicity) key bias. GT is removed before model inference and
is used ONLY for offline evaluation. Run from the repository root.
"""
import argparse
from contextlib import contextmanager
import csv
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
warnings.filterwarnings("ignore")

import numpy as np
import torch


def overlaps(boxes, target):
    boxes, target = np.asarray(boxes, dtype=np.float64), np.asarray(target, dtype=np.float64)
    intersection = np.maximum(np.minimum(boxes[:, 2:], target[2:]) -
                              np.maximum(boxes[:, :2], target[:2]), 0).prod(-1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return intersection / np.maximum(area + target_area - intersection, 1e-12)


def group_candidates(boxes, foreground, valid, threshold):
    """Greedy representative grouping by detector confidence; no GT/text labels.

    Every member must overlap its representative >= threshold. This is NOT
    transitive connected-components grouping, and NOT proof of entity identity.
    Invalid slots remain singleton context entries (baseline keeps them too).
    """
    group_ids = np.full(len(boxes), -1, dtype=np.int64)
    representatives = []
    for candidate in np.argsort(-foreground, kind="stable"):
        if group_ids[candidate] >= 0:
            continue
        group_id = len(representatives)
        representatives.append(int(candidate))
        group_ids[candidate] = group_id
        if valid[candidate]:
            members = valid & (group_ids < 0) & (overlaps(boxes, boxes[candidate]) >= threshold)
            group_ids[members] = group_id
    counts = np.bincount(group_ids)[group_ids]
    return group_ids, counts, np.asarray(representatives, dtype=np.int64)


def score_order(scores, valid, stable=False):
    slots = np.flatnonzero(valid)
    order = torch.as_tensor(scores[slots]).argsort(descending=True, stable=stable).numpy()
    return slots[order]


def make_interventions(boxes, foreground, valid, native_scores):
    groups, counts, representatives = group_candidates(boxes, foreground, valid, .8)
    order = score_order(native_scores, valid)
    top1 = int(order[0])
    # Use a geometrically distinct, high-CRS competitor, not a GT-chosen wrong box.
    eligible = order[overlaps(boxes[order], boxes[top1]) < .3]
    rival = int(eligible[0]) if len(eligible) else int(order[-1])
    detector_top1 = int(np.flatnonzero(valid)[np.argmax(foreground[valid])])
    biases = {"sham": np.zeros(len(boxes), dtype=np.float32),
              "uniform_x8": np.full(len(boxes), math.log(8), dtype=np.float32)}
    selections = {"top1_slot": np.arange(len(boxes)) == top1,
                  "rival_slot": np.arange(len(boxes)) == rival}
    for policy, selection in selections.items():
        for multiplier in (2, 4, 8):
            biases[policy + "_x" + str(multiplier)] = selection.astype(np.float32) * math.log(multiplier)
    for name, selection in (("top1_group_x8", groups == groups[top1]),
                            ("rival_group_x8", groups == groups[rival]),
                            ("detector_top1_x8", np.arange(len(boxes)) == detector_top1)):
        biases[name] = selection.astype(np.float32) * math.log(8)
    for threshold in (.7, .8, .9):
        _, sizes, _ = group_candidates(boxes, foreground, valid, threshold)
        biases["balance_iou" + str(threshold)] = -np.log(sizes.astype(np.float32))
    dedup = np.full(len(boxes), -np.inf, dtype=np.float32)
    dedup[representatives] = 0
    biases["dedup_context_iou0.8"] = dedup
    return biases, groups, counts, top1, rival


class ContextIntervention:
    """Temporarily alter ONLY native CRS self-attention, restoring on exit."""
    def __init__(self, crs):
        self.modules = [layer.attentions[0].attn for layer in crs.word_attn.layers]
        self.bias = None
        self.attention = []

    @contextmanager
    def installed(self):
        originals = []
        for layer_index, module in enumerate(self.modules):
            original = module.forward
            originals.append(original)

            def forward(query, key, value, _original=original, _layer=layer_index,
                        _heads=module.num_heads, **kwargs):
                if self.bias is not None:
                    if kwargs.get("attn_mask") is not None:
                        raise RuntimeError("Unexpected original CRS self-attention mask")
                    batch, nq, _ = query.shape
                    if self.bias.shape != (batch, key.shape[1]):
                        raise RuntimeError("Intervention key bias is not batch/slot aligned")
                    kwargs["attn_mask"] = self.bias[:, None, None, :].expand(
                        batch, _heads, nq, -1).reshape(batch * _heads, nq, -1).contiguous()
                    # torch 2.0.0's native self-attention fast path erroneously
                    # casts floating masks to bool. An identity VIEW (same
                    # values/storage, different Python object) selects the
                    # correct functional path without changing any evidence.
                    key = key.view_as(key)
                output, weights = _original(query, key, value, **kwargs)
                if weights is None:
                    raise RuntimeError("Expected original CRS to return mean-head attention")
                if len(self.attention) != len(self.modules):
                    self.attention = [None] * len(self.modules)
                self.attention[_layer] = weights.detach().mean(dim=1)
                return output, weights

            module.forward = forward
        try:
            yield self
        finally:
            for module, original in zip(self.modules, originals):
                module.forward = original


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(rows):
    result = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        selected = [row for row in rows if row["mode"] == mode]
        count = len(selected)
        corrected = sum(row["corrected"] for row in selected)
        harmed = sum(row["harmed"] for row in selected)
        result[mode] = dict(
            samples=count, top1=100 * np.mean([r["correct"] for r in selected]),
            top5=100 * np.mean([r["top5_hit"] for r in selected]),
            miou=100 * np.mean([r["top1_iou"] for r in selected]),
            top1_logit_order=100 * np.mean([r["logit_correct"] for r in selected]),
            corrected=corrected, harmed=harmed, net=corrected - harmed,
            net_pp=100 * (corrected - harmed) / count,
            slot_flips=sum(r["slot_flip"] for r in selected),
            group_flips=sum(r["group_flip"] for r in selected),
            mean_abs_logit_change=float(np.mean([r["mean_logit_change"] for r in selected])),
            max_abs_logit_change=float(max(r["max_logit_change"] for r in selected)),
            mean_top1_group_size=float(np.mean([r["top1_group_size"] for r in selected])),
            mean_group_count=float(np.mean([r["group_count"] for r in selected])),
            attention_top1_group=[float(np.mean([r["attention_top1_l%d" % layer] for r in selected]))
                                  for layer in range(3)],
            attention_rival_group=[float(np.mean([r["attention_rival_l%d" % layer] for r in selected]))
                                   for layer in range(3)])
        # Approximate paired 95% CI; descriptive only, variants are not independent.
        differences = np.asarray([r["corrected"] - r["harmed"] for r in selected], dtype=float)
        radius = 1.96 * differences.std(ddof=1) / math.sqrt(count) * 100 if count > 1 else 0.
        result[mode]["paired_net_95ci_pp"] = [result[mode]["net_pp"] - radius, result[mode]["net_pp"] + radius]
    return result


def save_report(output, rows, meta):
    summary = dict(meta, modes=paired_summary(rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "metrics.csv", [dict(mode=mode, **{k: v for k, v in values.items()
                                                       if not isinstance(v, list)})
                                        for mode, values in summary["modes"].items()])
    write_csv(output / "changed_examples.csv", [row for row in rows if row["corrected"] or row["harmed"]])
    lines = ["# 候选重复数量敏感性诊断", "", "## 实验约束", "",
             "- 冻结归档 BASE 模型；不训练、不修改生产模型/配置。图像、文本、256 个 query/output 槽位和预测框保持固定。",
             "- 仅改变 CRS 三层候选自注意力的 K/V 表示次数；用 key 列的 log(m) 偏置等价实现复制，交叉注意力不变。",
             "- 所有干预选择均不使用 GT；GT 只离线统计 Top-1/Top-5@IoU≥0.5 和 Top-1 box mIoU。",
             "- 按检测置信度贪心分组，每个成员与代表框 IoU≥0.8；分组不是物体真值，也不做连通分量合并。",
             "- rival 为原始 CRS 最高分且与原 Top-1 的 IoU<0.3 的候选，不代表它一定是错误目标。",
             "- balance_iou0.8 是预先指定的主对照；0.7/0.9 仅稳健性检查，不按验证集挑参数。",
             "- dedup 只去掉 K/V 上下文中的重复，不减少任何目标 query/output 或修改框召回上限。",
             "- 未训练模型上的 balance/dedup 有分布偏移；此结果用于决定是否值得训练，不是最终模块消融。", "",
             "样本：{samples}/{dataset_size}，split={split}，随机种子={seed}。".format(**meta),
             "checkpoint：`{checkpoint}`。".format(**meta),
             "归档源码：`{imported_source}`。".format(**meta),
             "运行时长：{elapsed_seconds:.1f}s；CUDA 峰值显存：{peak_cuda_memory_mib:.1f} MiB。".format(**meta), "",
             "## 配对结果（%）", "", "|干预|Top-1|Top-5|mIoU|改对|改错|净变化（百分点）|槽位变化|跨组变化|平均logit漂移|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, value in summary["modes"].items():
        lines.append("|{}|{:.3f}|{:.3f}|{:.3f}|{}|{}|{:+.3f}|{}|{}|{:.6g}|".format(
            mode, value["top1"], value["top5"], value["miou"], value["corrected"], value["harmed"],
            value["net_pp"], value["slot_flips"], value["group_flips"], value["mean_abs_logit_change"]))
    lines += ["", "## 如何解读", "",
              "sham 与 uniform_x8 应近似 native；必须先检查数值控制，不能把数值误差当机制。",
              "复制单个 key 会改变它的相对注意力质量；slot flip 未必是语义错误，需看跨组变化与改对/改错。",
              "如果复制敏感但 balance/dedup 没有净改善，只能证明敏感性，不能证明它是主要性能瓶颈。",
              "若简单 dedup 与复杂设计同样有效，应优先简单方案，不能把去重或 log 次数校正本身宣称为原创。",
              "本诊断不验证联合训练后的收益或跨数据集泛化。", "", "## 数值与复现检查", "",
              "```json", json.dumps(meta.get("checks", {}), ensure_ascii=False, indent=2), "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=512, help="0 means full split; otherwise a GT-blind uniform subset")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant-batch", type=int, default=4)
    parser.add_argument("--modes", nargs="+", help="Optional predeclared subset of intervention names")
    args = parser.parse_args()
    if args.samples < 0 or args.variant_batch < 1:
        parser.error("samples >= 0 and variant-batch >= 1 required")
    experiment, output = Path(args.experiment).resolve(), Path(args.output).resolve()
    if not experiment.joinpath("propvg").is_dir():
        raise FileNotFoundError("An experiment with archived propvg source is required")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(experiment))
    from mmcv import Config
    from mmcv.parallel import collate
    from propvg.datasets import build_dataset, extract_data
    from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
    from propvg.models import build_model
    import propvg

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not run the full encoder on CPU")
    config = next(experiment.glob("*_PropVG-aerialvg.py"))
    checkpoint_path = experiment / "top1_best.pth"
    cfg = Config.fromfile(str(config))
    cfg.model.vis_enc.pretrain = None
    cfg.model.process_visual = False
    if cfg.model.head.get("enable_grab", False) or cfg.model.head.get("enable_bcem", False):
        raise ValueError("This diagnostic was designed for the unmodified CRS BASE control")
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    dataset = build_dataset(cfg.data[args.split])
    indices = np.arange(len(dataset))
    if args.samples and args.samples < len(indices):
        indices = np.sort(np.random.default_rng(args.seed).choice(indices, size=args.samples, replace=False))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1,
        num_workers=args.workers, collate_fn=partial(collate, samples_per_gpu=1), shuffle=False)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    print("Loading checkpoint on CPU...", flush=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    meta = dict(experiment=str(experiment), configuration=str(config), checkpoint=str(checkpoint_path),
                checkpoint_epoch=int(checkpoint.get("epoch", -1)) + 1,
                imported_source=propvg.__file__, split=args.split, dataset_size=len(dataset),
                samples=len(indices), seed=args.seed, indices=indices.tolist(),
                physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), batch_size=1,
                variant_batch=args.variant_batch, torch_version=torch.__version__,
                diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                checkpoint_size=checkpoint_path.stat().st_size)
    del state, checkpoint
    gc.collect()
    model = model.cuda().eval()
    crs = model.head.mrsm
    capture = {}
    original_crs, original_head = crs.forward, model.head.forward_test

    def capture_crs(query, text, mask, **kwargs):
        result = original_crs(query, text, mask, return_logits=True)
        capture.update(query=query.detach(), text=text.detach(), mask=mask.detach(),
                       native_scores=result[0][0, :, 0].detach().cpu().numpy(),
                       native_logits=result[2][0, :, 0].detach().cpu().numpy())
        return result[:2]

    def capture_head(*positional, **kwargs):
        result = original_head(*positional, **kwargs)
        capture["boxes"] = result[0]["pred_bbox_allobj"][0].detach().float().cpu()
        capture["fg"] = result[0]["pred_class_allobj"][0].softmax(-1)[:, :-1].max(-1).values.detach().cpu().numpy()
        return result

    crs.forward, model.head.forward_test = capture_crs, capture_head
    archived = {}
    archived_csv = experiment / ("entropy_analysis_" + args.split + "_20261016") / "samples.csv"
    if archived_csv.exists():
        with archived_csv.open(encoding="utf-8-sig") as stream:
            archived = {int(row["index"]): row for row in csv.DictReader(stream)}
    checks = dict(archive_samples_checked=0, archive_top1_correct_mismatches=0,
                  archive_top1_iou_max_error=0., native_box_alignment_max_error=0.,
                  native_score_alignment_max_error=0., max_sham_logit_error=0., max_uniform_logit_error=0.)
    rows, candidate_cache = [], []
    start = time.time()
    print("Strict checkpoint load passed; beginning frozen-model inference.", flush=True)
    with ContextIntervention(crs).installed() as intervention, torch.no_grad(), \
            (output / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = None
        for position, inputs in enumerate(loader):
            dataset_index = int(indices[position])
            metadata = inputs["img_metas"].data[0][0]
            gt = inputs.pop("gt_bbox").data[0][0][metadata["refer_target_index"][0]].cpu().numpy()
            inputs.pop("gt_mask_rle", None)
            inputs.pop("is_crowd", None)
            intervention.bias = None
            prediction = model(**extract_data(inputs), return_loss=False, with_bbox=True,
                               with_mask=False, rescale=False)["pred_bboxes"][0]
            native_attention = torch.stack(intervention.attention, dim=1)[0].cpu().numpy()
            raw = capture["boxes"]
            boxes = torch.cat((raw[:, :2] - raw[:, 2:] / 2, raw[:, :2] + raw[:, 2:] / 2), -1)
            h, w = metadata["img_shape"][:2]
            boxes *= torch.tensor([w, h, w, h])
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
            valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(-1).numpy()
            boxes = boxes.numpy()
            if not valid.any():
                raise RuntimeError("No valid predicted boxes")
            checks["native_box_alignment_max_error"] = max(checks["native_box_alignment_max_error"],
                float(np.max(np.abs(boxes[valid] - prediction["boxes"].cpu().numpy()))))
            checks["native_score_alignment_max_error"] = max(checks["native_score_alignment_max_error"],
                float(np.max(np.abs(capture["native_scores"][valid] - prediction["scores"].cpu().numpy()))))
            if checks["native_box_alignment_max_error"] > 1e-3 or checks["native_score_alignment_max_error"] > 1e-6:
                raise RuntimeError("Filtered native prediction is not aligned with original query slots")
            biases, group_ids, counts, native_top1, rival = make_interventions(
                boxes, capture["fg"], valid, capture["native_scores"])
            if args.modes:
                if set(args.modes) - set(biases):
                    raise ValueError("Unknown modes: " + str(set(args.modes) - set(biases)))
                biases = {name: biases[name] for name in args.modes}
            all_scores = [capture["native_scores"]]
            all_logits = [capture["native_logits"]]
            all_attention = [native_attention]
            names = ["native"] + list(biases)
            bias_array = np.stack(list(biases.values()))
            for offset in range(0, len(bias_array), args.variant_batch):
                bias_batch = bias_array[offset:offset + args.variant_batch]
                batch = len(bias_batch)
                intervention.bias = torch.as_tensor(bias_batch, device="cuda")
                scores, _, logits = original_crs(capture["query"].expand(batch, -1, -1).contiguous(),
                    capture["text"].expand(batch, -1, -1).contiguous(),
                    capture["mask"].expand(batch, -1).contiguous(), return_logits=True)
                all_scores.extend(scores[:, :, 0].cpu().numpy())
                all_logits.extend(logits[:, :, 0].cpu().numpy())
                all_attention.extend(torch.stack(intervention.attention, dim=1).cpu().numpy())
            if not np.isfinite(np.asarray(all_logits)).all() or not np.isfinite(np.asarray(all_attention)).all():
                raise RuntimeError("Non-finite replay result: check floating-mask compatibility")
            ious = overlaps(boxes, gt)
            candidate_cache.append(dict(boxes=boxes, foreground=capture["fg"], valid=valid,
                ious=ious, groups=group_ids, scores=capture["native_scores"], logits=np.asarray(all_logits)))
            native_correct = bool(ious[native_top1] >= .5)
            if dataset_index in archived:
                old = archived[dataset_index]
                if old["filename"] != metadata["filename"] or old["expression"] != metadata["expression"]:
                    raise RuntimeError("Archived annotation ordering/text differs")
                checks["archive_samples_checked"] += 1
                checks["archive_top1_correct_mismatches"] += int(native_correct != bool(int(old["correct"])))
                checks["archive_top1_iou_max_error"] = max(checks["archive_top1_iou_max_error"],
                    abs(float(ious[native_top1]) - float(old["top1_iou"])))
            for mode, scores, logits, attention in zip(names, all_scores, all_logits, all_attention):
                order = score_order(scores, valid)
                logit_order = score_order(logits, valid, stable=True)
                top1 = int(order[0])
                correct = bool(ious[top1] >= .5)
                changes = np.abs(logits - capture["native_logits"])
                row = dict(index=dataset_index, mode=mode, filename=metadata["filename"],
                    expression=metadata["expression"], native_top1=native_top1, top1=top1, rival=rival,
                    native_correct=int(native_correct), correct=int(correct),
                    top5_hit=int(ious[order[:5]].max() >= .5), top1_iou=float(ious[top1]),
                    logit_correct=int(ious[logit_order[0]] >= .5),
                    oracle_hit=int(ious[valid].max() >= .5),
                    corrected=int(correct and not native_correct), harmed=int(native_correct and not correct),
                    slot_flip=int(top1 != native_top1), group_flip=int(group_ids[top1] != group_ids[native_top1]),
                    max_logit_change=float(changes.max()), mean_logit_change=float(changes.mean()),
                    top1_group_size=int(counts[native_top1]), group_count=len(np.unique(group_ids)),
                    valid_count=int(valid.sum()), native_probability=float(capture["native_scores"][native_top1]),
                    new_probability=float(scores[top1]),
                    native_top1_logit_change=float(logits[native_top1] - capture["native_logits"][native_top1]),
                    rival_logit_change=float(logits[rival] - capture["native_logits"][rival]))
                for layer in range(3):
                    row["attention_top1_l%d" % layer] = float(attention[layer, group_ids == group_ids[native_top1]].sum())
                    row["attention_rival_l%d" % layer] = float(attention[layer, group_ids == group_ids[rival]].sum())
                if mode in ("sham", "uniform_x8"):
                    check_name = "max_sham_logit_error" if mode == "sham" else "max_uniform_logit_error"
                    checks[check_name] = max(checks[check_name], float(changes.max()))
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            processed = position + 1
            if processed % 25 == 0 or processed == 1 or processed == len(indices):
                stream.flush()
                native_rows = [r for r in rows if r["mode"] == "native"]
                progress = dict(processed=processed, total=len(indices), elapsed_seconds=round(time.time() - start, 1),
                    native_top1=100 * np.mean([r["correct"] for r in native_rows]), checks=checks)
                (output / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(progress, ensure_ascii=False), flush=True)
    crs.forward, model.head.forward_test = original_crs, original_head
    meta.update(elapsed_seconds=time.time() - start, peak_cuda_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                complete_split=len(indices) == len(dataset), checks=checks)
    np.savez_compressed(output / "candidate_statistics.npz", indices=indices, mode_names=np.asarray(names),
        **{key: np.asarray([sample[key] for sample in candidate_cache]) for key in candidate_cache[0]})
    summary = save_report(output, rows, meta)
    print(json.dumps(dict(output=str(output), modes=summary["modes"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
