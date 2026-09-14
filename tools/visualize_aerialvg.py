"""Single-image AerialVG inference and bounding-box visualization."""
import argparse
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/propvg_cache")

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate

from propvg.datasets import build_dataset, extract_data
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.datasets.pipelines.compose import Compose
from propvg.models import build_model
from propvg.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AerialVG grounding results")
    parser.add_argument("config", help="AerialVG config file")
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0, help="dataset sample index")
    parser.add_argument("--image", help="custom image; must be used together with --text")
    parser.add_argument("--text", help="custom referring expression")
    parser.add_argument("--output", default="visualization/aerialvg_result.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=5, help="draw the top-k candidates")
    return parser.parse_args()


def make_custom_sample(cfg, image_path, expression, split):
    image = mmcv.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    # The loader requires one box, but it is only a disposable placeholder:
    # gt_bbox is removed before the model is called.
    record = {
        "filename": osp.abspath(image_path),
        "width": w,
        "height": h,
        "boxes": [[0.0, 0.0, 1.0, 1.0]],
        "expression": expression,
        "target_phrase": expression,
        "refer_target_index": 0,
    }
    pipeline = Compose(getattr(cfg.data, split).pipeline)
    return pipeline({"ann": record, "imgsfile": ""}), osp.abspath(image_path), expression, None


def make_dataset_sample(cfg, split, index):
    dataset = build_dataset(getattr(cfg.data, split))
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} is outside [0, {len(dataset) - 1}]")
    record = dataset.samples[index]
    image_path = osp.join(dataset.imgsfile, record["filename"])
    expression = record["expression"].strip() or record.get("target_phrase", "")
    gt_box = np.asarray(record["boxes"][record["refer_target_index"]], dtype=np.float32)
    return dataset[index], image_path, expression, gt_box


def box_iou(box, target):
    lt = np.maximum(box[:2], target[:2])
    rb = np.minimum(box[2:], target[2:])
    intersection = np.maximum(rb - lt, 0).prod()
    box_area = np.maximum(box[2:] - box[:2], 0).prod()
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return float(intersection / max(box_area + target_area - intersection, 1e-6))


def draw_result(image_path, expression, boxes, scores, gt_box, output):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    if gt_box is not None:
        x1, y1, x2, y2 = np.rint(gt_box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(image, "GT", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    for rank in range(len(boxes) - 1, -1, -1):
        x1, y1, x2, y2 = np.rint(boxes[rank]).astype(int)
        is_top1 = rank == 0
        color = (0, 0, 255) if is_top1 else (0, 165, 255)
        thickness = 4 if is_top1 else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"Top-{rank + 1}: {scores[rank]:.3f}"
        cv2.putText(image, label, (x1, min(image.shape[0] - 8, max(20, y1 + 20))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

    cv2.putText(image, expression, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(image, expression, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1)
    output = osp.abspath(output)
    os.makedirs(osp.dirname(output), exist_ok=True)
    if not cv2.imwrite(output, image):
        raise RuntimeError(f"failed to save visualization to {output}")
    return output


@torch.no_grad()
def main():
    args = parse_args()
    if (args.image is None) != (args.text is None):
        raise ValueError("--image and --text must be provided together")
    if args.topk < 1:
        raise ValueError("--topk must be at least 1")

    cfg = Config.fromfile(args.config)
    if cfg.dataset != "AerialVG":
        raise ValueError(f"expected an AerialVG config, got dataset={cfg.dataset!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current PropVG inference path")
    torch.cuda.set_device(torch.device(args.device))

    if args.image is not None:
        sample, image_path, expression, gt_box = make_custom_sample(cfg, args.image, args.text, args.split)
    else:
        sample, image_path, expression, gt_box = make_dataset_sample(cfg, args.split, args.index)

    batch = collate([sample], samples_per_gpu=1)
    # Remove all annotations before inference. They are retained separately
    # only for drawing and IoU reporting.
    batch.pop("gt_bbox", None)
    batch.pop("gt_mask_rle", None)
    batch.pop("is_crowd", None)
    inputs = extract_data(batch)

    model = build_model(cfg.model, word_emb=None, num_token=64010).to(args.device)
    load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    model.eval()
    outputs = model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=True)
    prediction = outputs["pred_bboxes"][0]
    scores = prediction["scores"].detach().reshape(-1).cpu()
    boxes = prediction["boxes"].detach().cpu()
    order = scores.argsort(descending=True)[: min(args.topk, len(scores))]
    ranked_scores = scores[order].numpy()
    ranked_boxes = boxes[order].numpy()

    output = draw_result(image_path, expression, ranked_boxes, ranked_scores, gt_box, args.output)
    print(f"expression: {expression}")
    print(f"top-1 box: {ranked_boxes[0].tolist()}")
    print(f"top-1 score: {ranked_scores[0]:.6f}")
    if gt_box is not None:
        iou = box_iou(ranked_boxes[0], gt_box)
        print(f"GT box: {gt_box.tolist()}")
        print(f"top-1 IoU: {iou:.6f} ({'correct' if iou >= 0.5 else 'incorrect'} @ 0.5)")
    print(f"saved to: {output}")


if __name__ == "__main__":
    main()
