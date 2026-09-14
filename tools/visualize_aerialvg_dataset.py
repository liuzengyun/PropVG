"""Run AerialVG inference over a complete split and save visualizations."""
import argparse
import json
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/propvg_cache")

import numpy as np
import torch
from mmcv import Config
from tqdm import tqdm

from propvg.datasets import build_dataset, build_dataloader, extract_data
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.models import build_model
from propvg.utils import load_checkpoint
from visualize_aerialvg import box_iou, draw_result


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a complete AerialVG split")
    parser.add_argument("config", help="AerialVG config file")
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", default="visualization/aerialvg_test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=5, help="number of candidate boxes to draw")
    parser.add_argument("--batch-size", type=int, help="override samples_per_gpu from the config")
    parser.add_argument("--workers", type=int, help="override workers_per_gpu from the config")
    parser.add_argument("--only-errors", action="store_true", help="save images only when Top-1 is incorrect")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing predictions.jsonl")
    return parser.parse_args()


def original_gt_box(boxes, meta):
    target_index = int(np.asarray(meta["refer_target_index"]).reshape(-1)[0])
    target = boxes[target_index].detach().cpu().numpy().astype(np.float32)
    scale = np.asarray(meta["scale_factor"], dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 4)
    return target / scale[:4]


@torch.no_grad()
def main():
    args = parse_args()
    if args.topk < 1:
        raise ValueError("--topk must be at least 1")

    cfg = Config.fromfile(args.config)
    if cfg.dataset != "AerialVG":
        raise ValueError(f"expected an AerialVG config, got dataset={cfg.dataset!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current PropVG inference path")
    torch.cuda.set_device(torch.device(args.device))
    cfg.distributed = False
    cfg.rank, cfg.world_size = 0, 1
    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers

    dataset = build_dataset(getattr(cfg.data, args.split))
    loader = build_dataloader(cfg, dataset)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(args.device)
    load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    model.eval()

    output_dir = osp.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_path = osp.join(output_dir, "predictions.jsonl")
    if osp.exists(results_path) and not args.overwrite:
        raise FileExistsError(f"{results_path} already exists; pass --overwrite to replace it")

    count = 0
    top1_hits = 0
    top5_hits = 0
    iou_sum = 0.0
    progress = tqdm(total=len(dataset), desc=f"Visualizing AerialVG-{args.split}", dynamic_ncols=True)
    with open(results_path, "w", encoding="utf-8") as result_file:
        for batch in loader:
            img_metas = batch["img_metas"].data[0]
            boxes_per_image = batch["gt_bbox"].data[0]
            gt_boxes = [original_gt_box(boxes, meta) for boxes, meta in zip(boxes_per_image, img_metas)]

            # Annotations are used above for reporting only and are never sent
            # into the visual-grounding model.
            batch.pop("gt_bbox", None)
            batch.pop("gt_mask_rle", None)
            batch.pop("is_crowd", None)
            inputs = extract_data(batch)
            outputs = model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=True)

            for meta, prediction, gt_box in zip(img_metas, outputs["pred_bboxes"], gt_boxes):
                scores = prediction["scores"].detach().reshape(-1).cpu()
                boxes = prediction["boxes"].detach().cpu()
                metric_k = min(5, len(scores))
                draw_k = min(args.topk, len(scores))
                order = scores.argsort(descending=True)[: max(metric_k, draw_k)]
                ranked_scores = scores[order].numpy()
                ranked_boxes = boxes[order].numpy()
                ious = np.asarray([box_iou(box, gt_box) for box in ranked_boxes], dtype=np.float32)
                top1_iou = float(ious[0])
                top1_correct = top1_iou >= 0.5
                top5_correct = bool(np.max(ious[:metric_k]) >= 0.5)

                stem = osp.splitext(osp.basename(meta["filename"]))[0]
                image_name = f"{count:06d}_{stem}.jpg"
                image_output = osp.join(output_dir, image_name)
                saved_image = None
                if not args.only_errors or not top1_correct:
                    saved_image = draw_result(
                        meta["filename"], meta["expression"], ranked_boxes[:draw_k],
                        ranked_scores[:draw_k], gt_box, image_output
                    )

                record = {
                    "index": count,
                    "filename": meta["filename"],
                    "expression": meta["expression"],
                    "gt_box": gt_box.tolist(),
                    "top_boxes": ranked_boxes[:draw_k].tolist(),
                    "top_scores": ranked_scores[:draw_k].tolist(),
                    "top1_iou": top1_iou,
                    "top1_correct_at_0.5": bool(top1_correct),
                    "top5_correct_at_0.5": top5_correct,
                    "visualization": saved_image,
                }
                result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                top1_hits += int(top1_correct)
                top5_hits += int(top5_correct)
                iou_sum += top1_iou
                progress.update(1)
    progress.close()

    top1 = 100.0 * top1_hits / count
    top5 = 100.0 * top5_hits / count
    miou = 100.0 * iou_sum / count
    summary = {
        "split": args.split,
        "samples": count,
        "top1_acc_at_0.5": top1,
        "top5_acc_at_0.5": top5,
        "miou": miou,
        "checkpoint": osp.abspath(args.checkpoint),
    }
    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    print(f"Top-1 Acc@0.5: {top1:.2f}%")
    print(f"Top-5 Acc@0.5: {top5:.2f}%")
    print(f"box mIoU: {miou:.2f}%")
    print(f"visualizations and predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
