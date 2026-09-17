"""Visualize the largest PropVG FPN feature before/after masking refer GT."""

import argparse
import os
import os.path as osp
import textwrap
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/propvg_cache")

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate

from propvg.datasets import build_dataset, extract_data
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.models import build_model
from propvg.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=110)
    parser.add_argument("--output-dir", default="visualization/crv_gt_mask")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--core-expansion", type=float, default=1.5)
    parser.add_argument("--feather-expansion", type=float, default=2.0)
    return parser.parse_args()


def make_soft_mask(
    gt_xyxy, img_shape, pad_shape, feature_shape,
    core_expansion, feather_expansion, device, dtype
):
    """Match CRV's padding-aware, expanded sigmoid box mask."""
    img_h, img_w = img_shape[:2]
    pad_h, pad_w = pad_shape[:2]
    feat_h, feat_w = feature_shape
    x1, y1, x2, y2 = gt_xyxy.to(device=device, dtype=dtype)
    center_x = ((x1 + x2) * 0.5 / img_w) * (img_w / pad_w)
    center_y = ((y1 + y2) * 0.5 / img_h) * (img_h / pad_h)
    inner_width = ((x2 - x1) / img_w) * (img_w / pad_w)
    inner_height = ((y2 - y1) / img_h) * (img_h / pad_h)
    core_width = inner_width * core_expansion
    core_height = inner_height * core_expansion
    outer_width = inner_width * feather_expansion
    outer_height = inner_height * feather_expansion

    xx = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) / feat_w
    yy = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) / feat_h
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    inner_left, inner_right = center_x - core_width * 0.5, center_x + core_width * 0.5
    inner_top, inner_bottom = center_y - core_height * 0.5, center_y + core_height * 0.5
    outer_left, outer_right = center_x - outer_width * 0.5, center_x + outer_width * 0.5
    outer_top, outer_bottom = center_y - outer_height * 0.5, center_y + outer_height * 0.5

    def smoothstep(value):
        value = value.clamp(0, 1)
        return value * value * (3 - 2 * value)

    mask_x = torch.minimum(
        smoothstep((xx - outer_left) / max(float(inner_left - outer_left), 1e-6)),
        smoothstep((outer_right - xx) / max(float(outer_right - inner_right), 1e-6)),
    )
    mask_y = torch.minimum(
        smoothstep((yy - outer_top) / max(float(inner_top - outer_top), 1e-6)),
        smoothstep((outer_bottom - yy) / max(float(outer_bottom - inner_bottom), 1e-6)),
    )
    return (mask_x * mask_y).clamp(0, 1)


def valid_feature_mask(img_shape, pad_shape, feature_shape, device):
    img_h, img_w = img_shape[:2]
    pad_h, pad_w = pad_shape[:2]
    feat_h, feat_w = feature_shape
    valid_h = min(feat_h, int(np.ceil(img_h * feat_h / pad_h)))
    valid_w = min(feat_w, int(np.ceil(img_w * feat_w / pad_w)))
    valid = torch.zeros((feat_h, feat_w), dtype=torch.bool, device=device)
    valid[:valid_h, :valid_w] = True
    return valid


def feature_energy(feature):
    return feature.float().square().mean(0).sqrt()


def normalize_pair(original, masked, valid):
    """Normalize both RMS maps with the original valid map's color scale."""
    values = original[valid]
    low = torch.quantile(values, 0.01)
    high = torch.quantile(values, 0.99)
    scale = (high - low).clamp(min=1e-6)
    original = ((original - low) / scale).clamp(0, 1)
    masked = ((masked - low) / scale).clamp(0, 1)
    original[~valid] = 0
    masked[~valid] = 0
    return original.cpu().numpy(), masked.cpu().numpy()


def denormalize_image(tensor, cfg):
    mean = np.asarray(cfg.img_norm_cfg.mean, dtype=np.float32)
    std = np.asarray(cfg.img_norm_cfg.std, dtype=np.float32)
    image_rgb = tensor.detach().cpu().permute(1, 2, 0).numpy() * std + mean
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def heatmap_overlay(image, heatmap, alpha=0.55):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)


def colored_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_TURBO)


def add_title(image, title):
    canvas = np.full((image.shape[0] + 42, image.shape[1], 3), 245, dtype=np.uint8)
    canvas[42:] = image
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    return canvas


def draw_comparison(image, gt_box, original_map, masked_map, soft_mask, valid, expression, index):
    gt_box_int = np.rint(gt_box).astype(int)
    panels = []
    input_panel = image.copy()
    cv2.rectangle(input_panel, tuple(gt_box_int[:2]), tuple(gt_box_int[2:]), (0, 255, 0), 3)
    panels.append(add_title(input_panel, "Model input + refer GT"))

    original_panel = heatmap_overlay(image, original_map)
    cv2.rectangle(original_panel, tuple(gt_box_int[:2]), tuple(gt_box_int[2:]), (0, 255, 0), 3)
    panels.append(add_title(original_panel, "Original FPN feature RMS"))

    mask_full = cv2.resize(soft_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_panel = heatmap_overlay(image, mask_full, alpha=0.65)
    cv2.rectangle(mask_panel, tuple(gt_box_int[:2]), tuple(gt_box_int[2:]), (0, 255, 0), 3)
    panels.append(add_title(mask_panel, "Mean-fill core 1.5x; feather to 2.0x"))

    original_colored = colored_heatmap(original_map, image)
    masked_colored = colored_heatmap(masked_map, image)
    valid_full = cv2.resize(
        valid.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    # Derive the neutral display color from this sample's own original
    # heatmap. No fixed color is injected into the visualization.
    neutral_color = original_colored[valid_full].astype(np.float32).mean(0)
    display_mask = mask_full[..., None].astype(np.float32)
    masked_colored = (
        masked_colored.astype(np.float32) * (1.0 - display_mask)
        + neutral_color[None, None] * display_mask
    ).clip(0, 255).astype(np.uint8)
    masked_panel = cv2.addWeighted(image, 0.45, masked_colored, 0.55, 0)
    cv2.rectangle(masked_panel, tuple(gt_box_int[:2]), tuple(gt_box_int[2:]), (0, 255, 0), 3)
    panels.append(add_title(masked_panel, "Mean-filled feature RMS (sample heatmap mean)"))

    body = np.concatenate(panels, axis=1)
    caption_height = 72
    canvas = np.full((body.shape[0] + caption_height, body.shape[1], 3), 250, dtype=np.uint8)
    canvas[caption_height:] = body
    caption = f"#{index}  " + " | ".join(textwrap.wrap(expression, width=120))
    cv2.putText(canvas, caption[:220], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 15), 2)
    cv2.putText(canvas, "RMS maps share one scale; core color is computed from this sample's valid heatmap mean.", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    return canvas


@torch.no_grad()
def main():
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(torch.device(args.device))

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(getattr(cfg.data, args.split))
    end_index = min(args.start_index + args.num_images, len(dataset))
    indices = list(range(args.start_index, end_index))
    if len(indices) != args.num_images:
        raise IndexError("requested range exceeds the dataset")

    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(args.device)
    load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    model.eval()
    captured = {}

    def capture_neck(_module, _inputs, output):
        captured["largest"] = output[0][:, :256].detach()

    hook = model.head.neck.register_forward_hook(capture_neck)
    os.makedirs(args.output_dir, exist_ok=True)
    comparisons = []

    for index in indices:
        sample = dataset[index]
        batch = collate([sample], samples_per_gpu=1)
        img_meta = batch["img_metas"].data[0][0]
        gt_boxes = batch["gt_bbox"].data[0][0]
        refer_index = int(img_meta["refer_target_index"][0])
        gt_box = gt_boxes[refer_index].float()
        expression = img_meta["expression"]
        batch.pop("gt_bbox", None)
        batch.pop("gt_mask_rle", None)
        batch.pop("is_crowd", None)
        inputs = extract_data(batch)
        input_image = denormalize_image(inputs["img"][0], cfg)

        captured.clear()
        model(**inputs, return_loss=False, with_bbox=True, with_mask=False, rescale=False)
        feature = captured["largest"][0]
        feat_h, feat_w = feature.shape[-2:]
        valid = valid_feature_mask(img_meta["img_shape"], img_meta["pad_shape"],
                                   (feat_h, feat_w), feature.device)
        soft_mask = make_soft_mask(
            gt_box, img_meta["img_shape"], img_meta["pad_shape"], (feat_h, feat_w),
            args.core_expansion, args.feather_expansion, feature.device, feature.dtype
        )
        valid_float = valid[None].to(feature.dtype)
        neutral = (feature * valid_float).sum((1, 2), keepdim=True) / valid_float.sum().clamp(min=1)
        masked_feature = feature * (1.0 - soft_mask[None]) + neutral * soft_mask[None]

        original_energy, masked_energy = normalize_pair(
            feature_energy(feature), feature_energy(masked_feature), valid
        )
        comparison = draw_comparison(
            input_image, gt_box.cpu().numpy(), original_energy, masked_energy,
            soft_mask.float().cpu().numpy(), valid.cpu().numpy(), expression, index
        )
        output = osp.join(args.output_dir, f"{index:05d}_gt_mask_comparison.jpg")
        if not cv2.imwrite(output, comparison):
            raise RuntimeError(f"failed to write {output}")
        comparisons.append(comparison)
        print(f"[{len(comparisons):02d}/{args.num_images:02d}] {output}")

    hook.remove()
    thumb_width = 1000
    thumbnails = [cv2.resize(item, (thumb_width, max(1, round(item.shape[0] * thumb_width / item.shape[1]))))
                  for item in comparisons]
    cell_h = max(item.shape[0] for item in thumbnails)
    sheet = np.full((cell_h * 5, thumb_width * 2, 3), 255, dtype=np.uint8)
    for position, item in enumerate(thumbnails):
        row, col = divmod(position, 2)
        sheet[row * cell_h:row * cell_h + item.shape[0], col * thumb_width:(col + 1) * thumb_width] = item
    overview = osp.join(args.output_dir, "overview_10.jpg")
    if not cv2.imwrite(overview, sheet):
        raise RuntimeError(f"failed to write {overview}")
    print(f"overview: {osp.abspath(overview)}")


if __name__ == "__main__":
    main()
