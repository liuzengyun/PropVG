"""Evaluate a trained AerialVG CRV model using CRV drop alone for ranking.

The model and checkpoint are not modified. A forward hook reads CRV outputs
from the same inference pass, so fused and CRV-only metrics use identical
candidate boxes. CRS is still used inside the model to select its raw Top-K.
"""

import argparse
import csv
import os

import torch
from mmcv import Config
from tqdm import tqdm

from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.datasets import build_dataloader, build_dataset, extract_data
from propvg.models import build_model
from propvg.utils import load_checkpoint


def box_iou(boxes, target):
    left_top = torch.maximum(boxes[:, :2], target[:2])
    right_bottom = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (target[2:] - target[:2]).clamp(min=0).prod()
    return intersection / (areas + target_area - intersection).clamp(min=1e-6)


def summarize(records):
    if not records[0]:
        raise ValueError("Cannot evaluate an empty dataset")
    return tuple(100.0 * torch.stack(values).float().mean().item() for values in records)


@torch.no_grad()
def evaluate(model, loader, crv):
    captured = {}

    def capture_crv(_module, _args, output):
        captured["drop"] = output["support_drop_logits"].detach()
        raw_logits = output["raw_crs_logits"]
        captured["crs"] = raw_logits.sigmoid().detach()
        captured["crs_logits"] = raw_logits.detach()
        captured["alpha"] = output.get("fusion_alpha")

    hook = crv.register_forward_hook(capture_crv)
    fused_records = ([], [], [])
    crv_records = ([], [], [])
    crs_records = ([], [], [])
    sample_rows = []
    fused_correct_crv_wrong = 0
    fused_wrong_crv_correct = 0
    crs_wrong_fused_correct = 0
    crs_correct_fused_wrong = 0
    different_top1 = 0
    sample_count = 0
    model.eval()
    try:
        for inputs in tqdm(loader, total=len(loader), desc="CRS / fused / independent CRV", dynamic_ncols=True):
            img_metas = inputs["img_metas"].data[0]
            boxes_per_image = inputs["gt_bbox"].data[0]
            targets = [
                boxes[meta["refer_target_index"][0]]
                for boxes, meta in zip(boxes_per_image, img_metas)
            ]
            inputs.pop("gt_bbox", None)
            inputs.pop("gt_mask_rle", None)
            inputs.pop("is_crowd", None)
            inputs = extract_data(inputs)
            captured.clear()
            outputs = model(
                **inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=False
            )
            if "drop" not in captured:
                raise RuntimeError("CRV hook did not receive support_drop_logits")
            drops = captured["drop"]
            crs_scores = captured["crs"]
            for image_index, (prediction, target) in enumerate(zip(outputs["pred_bboxes"], targets)):
                boxes = prediction["boxes"]
                target = target.to(boxes.device)
                ious = box_iou(boxes, target)
                fused_order = prediction.get("ranking_logits", prediction["scores"]).reshape(-1).argsort(descending=True, stable=True)
                crv_order = drops[image_index].reshape(-1).argsort(descending=True)
                crs_order = captured["crs_logits"][image_index].reshape(-1).argsort(descending=True, stable=True)
                fused_ious = ious[fused_order]
                crv_ious = ious[crv_order]
                crs_ious = ious[crs_order]
                crs_ok = crs_ious[0] >= 0.5
                fused_ok = fused_ious[0] >= 0.5
                crv_ok = crv_ious[0] >= 0.5
                fused_records[0].append(fused_ok.float().cpu())
                fused_records[1].append((fused_ious[:5].max() >= 0.5).float().cpu())
                fused_records[2].append(fused_ious[0].cpu())
                crv_records[0].append(crv_ok.float().cpu())
                crv_records[1].append((crv_ious[:5].max() >= 0.5).float().cpu())
                crv_records[2].append(crv_ious[0].cpu())
                crs_records[0].append((crs_ious[0] >= 0.5).float().cpu())
                crs_records[1].append((crs_ious[:5].max() >= 0.5).float().cpu())
                crs_records[2].append(crs_ious[0].cpu())
                fused_wrong_crv_correct += int((~fused_ok & crv_ok).item())
                fused_correct_crv_wrong += int((fused_ok & ~crv_ok).item())
                crs_wrong_fused_correct += int((~crs_ok & fused_ok).item())
                crs_correct_fused_wrong += int((crs_ok & ~fused_ok).item())
                different_top1 += int((fused_order[0] != crv_order[0]).item())
                meta = img_metas[image_index]
                sample_rows.append({
                    "sample_index": sample_count,
                    "filename": meta.get("filename", ""),
                    "expression": meta.get("expression", ""),
                    "crs_top1_candidate": int(crs_order[0].item()),
                    "crs_top1_score": float(crs_scores[image_index, crs_order[0]].item()),
                    "crs_top1_logit": float(captured["crs_logits"][image_index, crs_order[0]].item()),
                    "crs_top1_iou": float(crs_ious[0].item()),
                    "crs_top1_correct": int((crs_ious[0] >= 0.5).item()),
                    "crs_top5_correct": int((crs_ious[:5].max() >= 0.5).item()),
                    "fused_top1_candidate": int(fused_order[0].item()),
                    "fused_top1_score": float(prediction["scores"].reshape(-1)[fused_order[0]].item()),
                    "fusion_alpha": float(captured["alpha"][image_index, 0]) if captured["alpha"] is not None else None,
                    "fused_top1_iou": float(fused_ious[0].item()),
                    "fused_top1_correct": int(fused_ok.item()),
                    "fused_top5_correct": int((fused_ious[:5].max() >= 0.5).item()),
                    "crv_top1_candidate": int(crv_order[0].item()),
                    "crv_top1_score": float(drops[image_index, crv_order[0]].item()),
                    "crv_top1_iou": float(crv_ious[0].item()),
                    "crv_top1_correct": int(crv_ok.item()),
                    "crv_top5_correct": int((crv_ious[:5].max() >= 0.5).item()),
                })
                sample_count += 1
    finally:
        hook.remove()

    return {
        "fused": summarize(fused_records),
        "crv_only": summarize(crv_records),
        "crs_only": summarize(crs_records),
        "sample_rows": sample_rows,
        "samples": sample_count,
        "different_top1": different_top1,
        "fused_wrong_crv_correct": fused_wrong_crv_correct,
        "fused_correct_crv_wrong": fused_correct_crv_wrong,
        "crs_wrong_fused_correct": crs_wrong_fused_correct,
        "crs_correct_fused_wrong": crs_correct_fused_wrong,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.distributed = False
    cfg.rank, cfg.world_size = 0, 1
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.workers
    if not cfg.model.head.get("enable_crv", False):
        raise ValueError("The supplied config does not enable CRV")
    dataset = build_dataset(getattr(cfg.data, args.split))
    loader = build_dataloader(cfg, dataset)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).cuda()
    load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset, expected_crv_version=2)
    results = evaluate(model, loader, model.head.crv)

    fused = results["fused"]
    crv_only = results["crv_only"]
    crs_only = results["crs_only"]
    lines = [
        f"split={args.split} samples={results['samples']}",
        f"crs_only: Top-1={crs_only[0]:.2f}% Top-5={crs_only[1]:.2f}% mIoU={crs_only[2]:.2f}%",
        f"fused: Top-1={fused[0]:.2f}% Top-5={fused[1]:.2f}% mIoU={fused[2]:.2f}%",
        f"crv_only: Top-1={crv_only[0]:.2f}% Top-5={crv_only[1]:.2f}% mIoU={crv_only[2]:.2f}%",
        f"different_top1={results['different_top1']}",
        f"fused_wrong_to_crv_correct={results['fused_wrong_crv_correct']}",
        f"fused_correct_to_crv_wrong={results['fused_correct_crv_wrong']}",
        f"crs_wrong_to_fused_correct={results['crs_wrong_fused_correct']}",
        f"crs_correct_to_fused_wrong={results['crs_correct_fused_wrong']}",
        f"fusion_net_corrections={results['crs_wrong_fused_correct'] - results['crs_correct_fused_wrong']}",
        "CRV-only score: support_drop_logits; CRS is retained only for raw Top-K selection.",
    ]
    report = "\n".join(lines)
    print(report)
    output = args.output or os.path.join(
        os.path.dirname(args.checkpoint), f"{args.split}_crs_fused_crv_eval.txt"
    )
    with open(output, "w", encoding="utf-8") as file:
        file.write(report + "\n")
    print(f"saved: {output}")

    output_stem = os.path.splitext(output)[0]
    summary_csv = output_stem + "_summary.csv"
    detail_csv = output_stem + "_samples.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=("score_mode", "top1_acc_at_0.5", "top5_acc_at_0.5", "miou")
        )
        writer.writeheader()
        for mode, values in (
            ("crs_only", crs_only), ("fused", fused), ("crv_only", crv_only)
        ):
            writer.writerow({
                "score_mode": mode,
                "top1_acc_at_0.5": f"{values[0]:.6f}",
                "top5_acc_at_0.5": f"{values[1]:.6f}",
                "miou": f"{values[2]:.6f}",
            })
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=results["sample_rows"][0].keys())
        writer.writeheader()
        writer.writerows(results["sample_rows"])
    print(f"saved: {summary_csv}")
    print(f"saved: {detail_csv}")


if __name__ == "__main__":
    main()
