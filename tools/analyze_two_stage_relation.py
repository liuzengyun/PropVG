"""Single forward pass: CRS/final ranking, reference evidence and repair/harm audit."""
import argparse
import csv
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from mmcv import Config, DictAction
from tqdm import tqdm

from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.datasets import build_dataset, build_dataloader, extract_data
from propvg.models import build_model
from propvg.utils import load_checkpoint
from propvg.evaluations.two_stage_relation import sample_record, summarize


def original_baseline_comparison(records, path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        baseline = {int(row["dataset_index"]): row for row in csv.DictReader(handle)}
    original = []
    for row in records:
        previous = baseline.get(row["dataset_index"])
        if (previous is None or Path(previous["filename"]).name != Path(row["filename"]).name
                or previous["expression"].strip() != row["expression"].strip()):
            raise ValueError("Original baseline sample/split/order does not match this analysis")
        original.append(float(previous["top1_iou"]) >= .5)
    result = dict(source=str(Path(path).resolve()), samples=len(records), original_top1=100 * np.mean(original))
    for mode in ("crs", "final"):
        correct = [row[mode + "_top1_correct"] for row in records]
        repaired = sum(not old and new for old, new in zip(original, correct))
        harmed = sum(old and not new for old, new in zip(original, correct))
        result[mode] = dict(top1=100 * np.mean(correct), corrected=repaired, harmed=harmed, net=repaired-harmed)
    return result


def write_reports(output, records, summary):
    with (output / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 两阶段关系评分诊断", "", "样本数：{}；split：{}。".format(len(records), summary["split"]), "",
             "| 排序模式 | Top-1 | Top-5 | box mIoU |", "|---|---:|---:|---:|"]
    for mode in ("crs", "crs_logit", "relation", "final"):
        m = summary[mode]
        lines.append("| {} | {:.4f}% | {:.4f}% | {:.4f}% |".format(mode, m["top1_acc"], m["top5_acc"], m["miou"]))
    lines += ["", "改对：{}；改错：{}；净改对：{}；净变化：{:.4f}个百分点。".format(
        summary["corrected"], summary["harmed"], summary["net"], summary["net_accuracy_points"]), "",
        "注意：这里CRS来自联合训练后的同一checkpoint。判断主干/CRS是否退化，还需要独立原基线对照。",
        "空参考权重不是‘无参考描述’的正确率。relation-only只是诊断，不能等同完整定位系统。", "",
        "逐样本CSV保存改对/改错、IoU、空参考权重、修正幅度和候选覆盖；details/*.npz保存框、原始索引、",
        "6维几何、有效配对、关系logit、支持矩阵、竞争先验、参考权重、文本注意力及全部排序。",
        "开启诊断不会重新运行主干，不改正常模型打分。详细CPU导出仅发生在本脚本，不在训练期间发生。"]
    if "original_baseline" in summary:
        lines += ["", "## 与独立原基线对照", "", "```json", json.dumps(summary["original_baseline"], indent=2), "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--baseline-report", help="Original baseline per_sample.csv from foreground-recall analysis")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty output: {}".format(output))
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if not cfg.model.head.get("enable_relation_refinement", False):
        raise ValueError("Use the saved config with enable_relation_refinement=True")
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    cfg.data.samples_per_gpu, cfg.data.workers_per_gpu = args.batch_size, args.workers
    cfg.model.vis_enc.pretrain = None
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-checkpoint analysis currently requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dataset = build_dataset(getattr(cfg.data, args.split))
    loader = build_dataloader(cfg, dataset)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(device)
    _, _, _, strict = load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    if not strict:
        raise ValueError("Analysis requires a strictly loaded checkpoint and its saved config")
    stage = model.head.relation_refiner
    if int(stage.trained_steps) <= 0:
        raise ValueError("This checkpoint has no supervised relation optimizer steps; normal inference falls back to CRS")
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    (output / "details").mkdir()
    captured = []
    original_forward = model.head.forward_test
    def capture(*inputs, **kwargs):
        result = original_forward(*inputs, **kwargs)
        captured[:] = result[1].get("relation_refinement_samples", [])
        return result
    records = []
    model.head.forward_test = capture
    try:
        with stage.diagnostics():
            for inputs in tqdm(loader, desc="Two-stage relation audit", dynamic_ncols=True):
                metas = inputs["img_metas"].data[0]
                gt_boxes = inputs["gt_bbox"].data[0]
                targets = []
                for boxes, meta in zip(gt_boxes, metas):
                    if len(meta["refer_target_index"]) != 1:
                        raise ValueError("Offline audit expects exactly one refer GT per expression")
                    h, w = meta["img_shape"][:2]
                    targets.append(boxes.reshape(-1, 4)[meta["refer_target_index"][0]].float().cpu() / torch.tensor([w,h,w,h]))
                for name in ("gt_bbox", "gt_mask_rle", "is_crowd"):
                    inputs.pop(name, None)
                captured.clear()
                predictions = model(**extract_data(inputs), return_loss=False, with_bbox=True, with_mask=False, rescale=False)
                if len(captured) != len(metas):
                    raise RuntimeError("Head diagnostics batch mismatch")
                for detail, target, meta, prediction in zip(captured, targets, metas, predictions["pred_bboxes"]):
                    index = len(records)
                    sample = dataset.samples[index]
                    if Path(sample["filename"]).name != Path(meta["filename"]).name:
                        raise RuntimeError("Test dataset/loader must preserve sequential sample order")
                    record = sample_record(detail, target, index, meta["filename"], sample["expression"])
                    actual = prediction.get("ranking_logits", prediction["scores"]).detach().cpu().argsort(descending=True, stable=True)
                    if len(actual):
                        h, w = meta["img_shape"][:2]
                        expected = detail["boxes"][record["final_top1_query"]] * torch.tensor([w,h,w,h])
                        torch.testing.assert_close(prediction["boxes"][actual[0]].float().cpu(), expected, rtol=1e-5, atol=2e-3)
                    arrays = {key: value.numpy() for key, value in detail.items() if torch.is_tensor(value)}
                    arrays["refer_gt_xyxy"] = target.numpy()
                    np.savez_compressed(output / "details" / "{:06d}.npz".format(index), **arrays)
                    records.append(record)
                    if args.max_samples is not None and len(records) >= args.max_samples:
                        break
                if args.max_samples is not None and len(records) >= args.max_samples:
                    break
    finally:
        model.head.forward_test = original_forward
        captured.clear()
    summary = summarize(records)
    summary.update(split=args.split, checkpoint=str(Path(args.checkpoint).resolve()), config=str(Path(args.config).resolve()))
    if args.baseline_report:
        summary["original_baseline"] = original_baseline_comparison(records, args.baseline_report)
    write_reports(output, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Report:", output / "report.md")


if __name__ == "__main__":
    main()
