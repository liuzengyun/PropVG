"""Evaluate AerialVG without modifying the original PropVG evaluator."""
import argparse

import torch
from mmcv import Config
from tqdm import tqdm

# Register only the AerialVG additions.
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.datasets import build_dataset, build_dataloader, extract_data
from propvg.models import build_model
from propvg.utils import load_checkpoint


def box_iou(boxes, target):
    left_top = torch.maximum(boxes[:, :2], target[:2])
    right_bottom = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (target[2:] - target[:2]).clamp(min=0).prod()
    return intersection / (areas + target_area - intersection).clamp(min=1e-6)


@torch.no_grad()
def evaluate(model, loader):
    top1, top5, top1_iou = [], [], []
    model.eval()
    for inputs in tqdm(loader, total=len(loader), desc="Evaluating AerialVG", dynamic_ncols=True):
        img_metas = inputs["img_metas"].data[0]
        boxes_per_image = inputs["gt_bbox"].data[0]
        targets = [boxes[meta["refer_target_index"][0]] for boxes, meta in zip(boxes_per_image, img_metas)]
        # Ground truth is retained locally for metric computation only. Never
        # expose it to the model during visual-grounding inference.
        inputs.pop("gt_bbox", None)
        # These are needed only by the original segmentation evaluator, not
        # this box-grounding evaluation.
        inputs.pop("gt_mask_rle", None)
        inputs.pop("is_crowd", None)
        inputs = extract_data(inputs)
        outputs = model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=False)
        for prediction, target in zip(outputs["pred_bboxes"], targets):
            ranking = prediction["scores"].reshape(-1).argsort(descending=True)
            if ranking.numel() == 0:
                zero = prediction["boxes"].new_tensor(0.0)
                top1.append(zero)
                top5.append(zero)
                top1_iou.append(zero)
                continue
            ious = box_iou(prediction["boxes"][ranking], target.to(prediction["boxes"].device))
            top1.append((ious[0] >= 0.5).float())
            top5.append((ious[:5].max() >= 0.5).float())
            top1_iou.append(ious[0])
    return tuple(100.0 * torch.stack(values).mean().item() for values in (top1, top5, top1_iou))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.distributed = False
    cfg.rank, cfg.world_size = 0, 1
    dataset = build_dataset(getattr(cfg.data, args.split))
    loader = build_dataloader(cfg, dataset)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).cuda()
    load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    top1, top5, miou = evaluate(model, loader)
    print(f"AerialVG {args.split}: Top-1 Acc@0.5={top1:.2f}%, Top-5 Acc@0.5={top5:.2f}%, mIoU={miou:.2f}%")


if __name__ == "__main__":
    main()
