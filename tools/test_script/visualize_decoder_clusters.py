"""Read-only BASE diagnostic: disjoint detector proposals and query-feature clusters.

Two stages avoid installing sklearn in the training environment:
  propvg Python: --stage extract (archived model, CUDA, only 30 sampled rows)
                 --stage add-scores (augment old caches without changing them)
  plotting Python with sklearn: --stage render (cached features, CPU only)

No production model/configuration is changed. Clustering is per image in the
original 256-D feature space, NOT on t-SNE or box coordinates. GT participates
only in offline IoU/annotation; never in selection, clustering, or inference.
"""
import argparse
import csv
from functools import partial
import gc
import hashlib
import html
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_decoder_cluster_matplotlib")
warnings.filterwarnings("ignore")

import numpy as np


COLORS = ["#00A6D6", "#F28E2B", "#B061D0", "#4CB140",
          "#E15759", "#4E79A7", "#C49A00", "#A26A42"]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def box_iou(boxes, target):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    target = np.asarray(target, dtype=np.float64)
    intersection = np.maximum(
        np.minimum(boxes[:, 2:], target[2:]) - np.maximum(boxes[:, :2], target[:2]), 0
    ).prod(-1)
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return intersection / np.maximum(areas + target_area - intersection, 1e-12)


def disjoint_by_confidence(boxes, scores):
    """Descending foreground confidence; reject ANY positive-area intersection.

    Original query index breaks score ties. Invalid/degenerate boxes are not
    selected. Boundary contact (zero intersection area) is allowed. No top-k,
    confidence cutoff, GT conditioning, transitive grouping or area heuristic.
    """
    boxes, scores = np.asarray(boxes), np.asarray(scores).reshape(-1)
    if boxes.shape != (len(scores), 4):
        raise ValueError("Boxes and detection scores must be query-aligned")
    valid = np.isfinite(boxes).all(1) & np.isfinite(scores)
    valid &= (boxes[:, 2:] > boxes[:, :2]).all(1)
    selected, suppressed_by = [], np.full(len(scores), -1, dtype=np.int64)
    for index in np.argsort(-scores, kind="stable"):
        if not valid[index]:
            continue
        if selected:
            extent = np.minimum(boxes[selected, 2:], boxes[index, 2:]) - np.maximum(
                boxes[selected, :2], boxes[index, :2])
            conflict = np.flatnonzero((extent > 0).all(1))
            if len(conflict):
                suppressed_by[index] = selected[int(conflict[0])]
                continue
        selected.append(int(index))
    return np.asarray(selected, dtype=np.int64), valid, suppressed_by


def max_pair_intersection(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) < 2:
        return 0.0
    intersection = np.maximum(
        np.minimum(boxes[:, None, 2:], boxes[None, :, 2:]) -
        np.maximum(boxes[:, None, :2], boxes[None, :, :2]), 0).prod(-1)
    np.fill_diagonal(intersection, 0)
    return float(intersection.max())


def l2_features(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected a finite query-by-channel feature matrix")
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def cluster_features(features, seed, clusters=4):
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    normalized = l2_features(features)
    count = len(normalized)
    if count < clusters:
        raise ValueError("Only {} disjoint boxes: cannot honestly form {} clusters".format(count, clusters))
    model = KMeans(n_clusters=clusters, init="k-means++", n_init=30, max_iter=500,
                   random_state=seed, algorithm="lloyd")
    labels = model.fit_predict(normalized)
    if len(np.unique(labels)) != clusters:
        raise ValueError("Fewer than {} distinct feature clusters".format(clusters))
    perplexity = min(15.0, max(2.0, (count - 1) / 3.0))
    options = dict(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                   random_state=seed, metric="euclidean", n_jobs=1)
    iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    options[iteration_key] = 1500
    projection = TSNE(**options).fit_transform(normalized)
    distances = ((normalized[:, None] - model.cluster_centers_[None]) ** 2).sum(-1)
    return labels, projection, dict(perplexity=perplexity, inertia=float(model.inertia_),
                                    centroid_squared_distances=distances)


def cluster_score_statistics(labels, scores, clusters=4):
    """Arithmetic mean of native CRS sigmoid scores in each retained cluster."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("CRS scores and cluster labels must be finite and query-aligned")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected original sigmoid probabilities in [0,1], not logits")
    result = []
    for cluster in range(clusters):
        values = scores[labels == cluster]
        if not len(values):
            raise ValueError("Cannot compute the mean of an empty feature cluster")
        result.append(dict(cluster=cluster + 1, count=int(len(values)), mean=float(values.mean()),
                           std=float(values.std()), min=float(values.min()), max=float(values.max())))
    return result


def crs_topk_annotations(scores, kept, topk=5):
    """Rank the ORIGINAL query set, then test visibility; never backfill."""
    scores = np.asarray(scores).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite native CRS scores")
    kept_set = set(np.asarray(kept, dtype=int).tolist())
    order = np.argsort(-scores, kind="stable")[:topk]
    return [dict(rank=rank + 1, query_id=int(query), score=float(scores[query]),
                 retained=int(query) in kept_set) for rank, query in enumerate(order)]


def extract(args):
    experiment, output = Path(args.experiment).resolve(), Path(args.output).resolve()
    adding_scores = args.stage == "add-scores"
    existing_metadata, existing_records = None, None
    if not (experiment / "propvg").is_dir():
        raise FileNotFoundError("An experiment with archived propvg source is required")
    if adding_scores:
        existing_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        existing_records = json.loads((output / "samples.json").read_text(encoding="utf-8"))
        if (experiment != Path(existing_metadata["experiment"]) or
                experiment / args.checkpoint != Path(existing_metadata["checkpoint"])):
            raise ValueError("Score augmentation must use the exact cached experiment/checkpoint")
        args.seed, args.split = existing_metadata["seed"], existing_metadata["split"]
    elif output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
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
        raise RuntimeError("Did not import the archived model source")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BASE inference")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = list(experiment.glob("*_PropVG-aerialvg.py"))
    if len(paths) != 1:
        raise RuntimeError("Expected exactly one archived configuration")
    config_path = paths[0]
    cfg = Config.fromfile(str(config_path))
    for flag in ("enable_grab", "enable_bcem", "enable_crv", "enable_relation_scoring",
                 "enable_relation_refinement", "enable_correctness"):
        if cfg.model.head.get(flag, False):
            raise ValueError("This BASE diagnostic requires {}=False".format(flag))
    cfg.model.vis_enc.pretrain = None  # Full trained weights include BEiT-3.
    cfg.model.process_visual = False
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    dataset = build_dataset(cfg.data[args.split])
    if args.samples > len(dataset):
        raise ValueError("Requested more samples than the dataset contains")
    indices = sorted(random.Random(args.seed).sample(range(len(dataset)), args.samples))
    if adding_scores:
        indices = [r["index"] for r in existing_records]
        if indices != existing_metadata["indices"]:
            raise RuntimeError("Existing sample identities are inconsistent")
        args.samples = len(indices)
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != existing_metadata["config_sha256"]:
            raise RuntimeError("Archived configuration changed since feature extraction")
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    checkpoint_path = experiment / args.checkpoint
    print("Strictly loading archived checkpoint: " + str(checkpoint_path), flush=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    metadata = dict(experiment=str(experiment), configuration=str(config_path),
                    checkpoint=str(checkpoint_path), checkpoint_epoch=int(checkpoint.get("epoch", -1)) + 1,
                    imported_source=propvg.__file__, split=args.split, dataset_size=len(dataset),
                    samples=args.samples, indices=indices, seed=args.seed, batch_size=1,
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    source_feature="decoder_transformer output inter_states[-1], BEFORE refer_adapter / CRS",
                    expected_feature_shape=[256, 256], cluster_count=4,
                    detection_score="softmax(last decoder class logits)[:, :-1].max(-1)",
                    selection="Greedy descending detection score; strictly zero positive-area pairwise intersection",
                    confidence_threshold=None, detector_topk=None,
                    gt_usage="Offline IoU and drawing only; removed from model inputs",
                    sampling="Uniform 30 validation rows without replacement; no GT or prediction filtering",
                    coordinate_system="Clipped xyxy in original image pixels; computed via resized-image scale factors",
                    annotation_file=str(cfg.data[args.split].annsfile),
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    del checkpoint, state
    gc.collect()
    model = model.cuda().eval()
    captured = {}

    def decoder_hook(module, inputs, outputs):
        queries = outputs[0][-1]
        if queries.ndim != 3 or tuple(queries.shape) != (1, 256, 256):
            raise RuntimeError("Unexpected decoder shape: " + str(tuple(queries.shape)))
        captured["features"] = queries[0].detach().float().cpu().numpy().copy()
        captured["decoder_layers"] = int(outputs[0].shape[0])
        captured["class_calls"] = 0

    def class_hook(module, inputs, outputs):
        # This archive shares the SAME classification module across layers.
        # Its forward hook consequently fires three times, not only at [-1].
        captured["class_calls"] += 1
        if captured["class_calls"] < captured["decoder_layers"]:
            return
        if captured["class_calls"] != captured["decoder_layers"]:
            raise RuntimeError("Unexpected extra classifier invocation")
        np.testing.assert_allclose(inputs[0][0].detach().float().cpu().numpy(),
                                   captured["features"], atol=0, rtol=0)
        captured["foreground"] = outputs[0].float().softmax(-1)[:, :-1].max(-1).values.detach().cpu().numpy()

    original_head = model.head.forward_test

    def head_capture(*positional, **kwargs):
        result = original_head(*positional, **kwargs)
        captured["normalized_boxes"] = result[0]["pred_bbox_allobj"][0].detach().float().cpu().numpy()
        captured["crs_scores"] = result[0]["refer_label"][0].detach().float().reshape(-1).cpu().numpy()
        if captured["crs_scores"].shape != (256,):
            raise RuntimeError("Final CRS scores are not aligned to 256 query slots")
        np.testing.assert_allclose(captured["foreground"], result[0]["pred_class_allobj"][0].float().softmax(-1)
                                   [:, :-1].max(-1).values.cpu().numpy(), atol=0, rtol=0)
        return result

    hooks = [model.head.decoder_transformer.register_forward_hook(decoder_hook),
             model.head.class_embed[-1].register_forward_hook(class_hook)]
    model.head.forward_test = head_capture
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1,
                                        shuffle=False, num_workers=args.workers,
                                        collate_fn=partial(collate, samples_per_gpu=1))
    if adding_scores:
        (output / "scores").mkdir(exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "cache").mkdir()
    records = []
    start = time.time()
    try:
        with torch.no_grad():
            for position, batch in enumerate(tqdm(loader, desc="Extracting decoder features", total=len(indices))):
                dataset_index = indices[position]
                meta = batch["img_metas"].data[0][0]
                target_index = int(np.asarray(meta["refer_target_index"]).reshape(-1)[0])
                gt = batch.pop("gt_bbox").data[0][0][target_index].cpu().numpy()
                batch.pop("gt_mask_rle", None)
                batch.pop("is_crowd", None)
                captured.clear()
                prediction = model(**extract_data(batch), return_loss=False, with_bbox=True,
                                   with_mask=False, rescale=False)["pred_bboxes"][0]
                raw = torch.from_numpy(captured["normalized_boxes"].copy())
                boxes = torch.cat([raw[:, :2] - raw[:, 2:] / 2, raw[:, :2] + raw[:, 2:] / 2], -1)
                height, width = meta["img_shape"][:2]
                boxes *= torch.tensor([width, height, width, height])
                boxes[:, 0::2].clamp_(0, width)
                boxes[:, 1::2].clamp_(0, height)
                boxes_resized = boxes.numpy().copy()
                valid = (boxes_resized[:, 2:] > boxes_resized[:, :2]).all(1)
                np.testing.assert_allclose(boxes_resized[valid], prediction["boxes"].float().cpu().numpy(),
                                           atol=0.0002, rtol=1e-6)
                np.testing.assert_allclose(captured["crs_scores"][valid], prediction["scores"].float().reshape(-1).cpu().numpy(),
                                           atol=1e-7, rtol=1e-6)
                scale = np.asarray(meta["scale_factor"], dtype=np.float64).reshape(-1)
                if scale.size == 1:
                    scale = np.repeat(scale, 4)
                if scale.size != 4 or (scale <= 0).any():
                    raise RuntimeError("Invalid scale factor")
                original_gt = np.asarray(dataset.samples[dataset_index]["boxes"]
                                         [dataset.samples[dataset_index]["refer_target_index"]], dtype=float)
                np.testing.assert_allclose(gt / scale, original_gt, atol=0.003, rtol=1e-6)
                boxes_original = boxes_resized.astype(np.float64) / scale
                ious = box_iou(boxes_original, original_gt)
                np.testing.assert_allclose(ious, box_iou(boxes_resized, gt), atol=2e-5, rtol=1e-5)
                kept, valid_selection, suppressed_by = disjoint_by_confidence(boxes_original, captured["foreground"])
                if not np.array_equal(valid, valid_selection):
                    raise RuntimeError("Inconsistent valid-box indexing")
                pair_area = max_pair_intersection(boxes_original[kept])
                if pair_area != 0:
                    raise RuntimeError("Nonzero overlap remained after strict filtering")
                if len(kept) < 4:
                    raise RuntimeError("Sample #{} has only {} retained boxes; cannot form 4 clusters".format(dataset_index, len(kept)))
                global_best = int(np.flatnonzero(valid)[np.argmax(ious[valid])])
                retained_best = int(kept[np.argmax(ious[kept])])
                filename = str(Path(meta["filename"]).resolve())
                record = dict(index=dataset_index, filename=filename, expression=meta["expression"],
                              gt_box=original_gt.tolist(), original_queries=256, valid_queries=int(valid.sum()),
                              retained_queries=int(len(kept)), retained_query_ids=kept.tolist(),
                              global_best_query=global_best, global_best_iou=float(ious[global_best]),
                              global_best_retained=bool(global_best in kept),
                              global_best_suppressed_by=int(suppressed_by[global_best]),
                              retained_best_query=retained_best, retained_best_iou=float(ious[retained_best]),
                              retained_best_detector_score=float(captured["foreground"][retained_best]),
                              max_retained_pair_intersection=pair_area,
                              cache="cache/{:04d}.npz".format(dataset_index))
                if adding_scores:
                    previous = existing_records[position]
                    if previous["filename"] != filename or previous["expression"] != meta["expression"]:
                        raise RuntimeError("Image/text identity changed")
                    with np.load(output / previous["cache"], allow_pickle=False) as old:
                        np.testing.assert_allclose(captured["features"], old["features"], atol=2e-5, rtol=1e-5)
                        np.testing.assert_allclose(boxes_original, old["boxes_xyxy"], atol=2e-4, rtol=1e-6)
                        np.testing.assert_allclose(captured["foreground"], old["foreground"], atol=1e-6, rtol=1e-5)
                        np.testing.assert_array_equal(kept, old["kept"])
                    np.savez_compressed(output / "scores" / "{:04d}.npz".format(dataset_index),
                                        crs_scores=captured["crs_scores"], query_ids=np.arange(256))
                else:
                    np.savez_compressed(str(output / record["cache"]), features=captured["features"],
                                        boxes_xyxy=boxes_original, foreground=captured["foreground"], ious=ious,
                                        kept=kept, valid=valid, suppressed_by=suppressed_by,
                                        crs_scores=captured["crs_scores"])
                records.append(record)
                print("sample #{:04d}: 256 -> {} disjoint; best IoU {:.3f} -> {:.3f}{}".format(
                    dataset_index, len(kept), ious[global_best], ious[retained_best],
                    " (global best suppressed)" if global_best not in kept else ""), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        model.head.forward_test = original_head
    metadata["extraction_seconds"] = time.time() - start
    metadata["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    metadata["checks"] = dict(strict_checkpoint_load=True, decoder_feature_class_head_alignment=True,
                               query_box_postprocessing_alignment=True, gt_coordinate_check=True,
                               iou_coordinate_invariance=True, all_retained_pairs_zero_intersection=True,
                               crs_score_postprocessing_alignment=True)
    if adding_scores:
        metadata["score_definition"] = "Original full-256-query forward final CRS sigmoid probability; no rescoring after dedup"
        metadata["existing_feature_box_and_kept_alignment"] = True
        (output / "scores_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (output / "samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(("Saved final CRS scores; original caches unchanged: " if adding_scores else
           "Saved decoder cache (no clustering yet): ") + str(output), flush=True)


def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_PATH, size)


def _wrap(draw, text, width, size):
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=_font(size)) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def _dashed_rectangle(draw, box, color, width=4):
    x1, y1, x2, y2 = box
    for start, end, fixed, horizontal in ((x1, x2, y1, True), (x1, x2, y2, True),
                                          (y1, y2, x1, False), (y1, y2, x2, False)):
        for position in np.arange(start, end, 16):
            finish = min(position + 9, end)
            coordinates = ((position, fixed, finish, fixed) if horizontal else
                           (fixed, position, fixed, finish))
            draw.line(coordinates, fill=color, width=width)


def render_one(record, cached, labels, projection, output):
    from io import BytesIO
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects

    kept, boxes = cached["kept"], cached["boxes_xyxy"]
    all_queries = record.get("all_queries", False)
    cluster_count = len(record["cluster_crs_statistics"])
    legend_rows = math.ceil(cluster_count / 4)
    if cluster_count > len(COLORS):
        raise ValueError("Insufficient distinct display colors")
    visible_top5 = {row["query_id"]: row for row in record["crs_top5"] if row["retained"]}
    best_local = int(np.flatnonzero(kept == record["retained_best_query"])[0])
    with Image.open(record["filename"]) as handle:
        source = handle.convert("RGB")
    panel_width, body_height = 1440, 1130
    page_width = 2 * panel_width
    probe = ImageDraw.Draw(Image.new("RGB", (page_width, 1)))
    caption_lines = _wrap(probe, record["expression"], page_width - 64, 29)
    header = 206 + 38 * len(caption_lines) + 40 * (legend_rows - 1)
    footer = 184
    page = Image.new("RGB", (page_width, header + body_height + footer), "#f4f6fa")
    draw = ImageDraw.Draw(page)
    title = ("BASE | All 256 decoder queries | {} feature clusters | val #{:04d}".format(
        cluster_count, record["index"]) if all_queries else
        "BASE | Decoder queries BEFORE CRS | val #{:04d} | {} / 256 retained".format(record["index"], len(kept)))
    draw.text((28, 16), title, font=_font(34), fill="#152638")
    for i, text in enumerate(caption_lines):
        draw.text((28, 66 + 38 * i), text, font=_font(29), fill="#23313f")
    legend_y = 74 + 38 * len(caption_lines)
    for cluster in range(cluster_count):
        x = 28 + (cluster % 4) * 710
        cy = legend_y + (cluster // 4) * 40
        draw.rectangle((x, cy + 4, x + 27, cy + 27), fill=COLORS[cluster])
        draw.text((x + 38, cy), "C{}: n={} | mean CRS={:.6f}".format(
            cluster + 1, int((labels == cluster).sum()), record["cluster_crs_statistics"][cluster]["mean"]),
                  font=_font(24), fill="#182534")
    draw.text((28, legend_y + 40 * legend_rows), "Dashed white = GT | star / thick edge = max GT-IoU query | CRS = native full-set referring probability (0-1)",
              font=_font(23), fill="#182534")
    draw.text((28, header - 38), ("IMAGE: all 256 boxes; emphasis on CRS Top-5 + GT" if all_queries else
              "IMAGE: strictly disjoint boxes, original query IDs"), font=_font(25), fill="#182534")
    draw.text((panel_width + 36, header - 38), "t-SNE: L2-normalized decoder features; K-Means in 256-D", font=_font(25), fill="#182534")

    usable_width, usable_height = panel_width - 40, body_height - 30
    scale = min(usable_width / source.width, usable_height / source.height)
    new_size = (round(source.width * scale), round(source.height * scale))
    view = source.resize(new_size, Image.Resampling.LANCZOS)
    pen = ImageDraw.Draw(view)
    box_scale = np.asarray([new_size[0] / source.width, new_size[1] / source.height] * 2)
    for local in np.argsort(cached["foreground"][kept], kind="stable"):
        query = int(kept[local])
        box = boxes[query] * box_scale
        color = COLORS[int(labels[local])]
        if local == best_local:
            pen.rectangle(tuple(box), outline="black", width=8)
            pen.rectangle(tuple(box), outline=color, width=4)
        elif all_queries:
            rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            ImageDraw.Draw(view, "RGBA").rectangle(tuple(box), outline=rgb + (100,), width=1)
        else:
            pen.rectangle(tuple(box), outline="black", width=4)
            pen.rectangle(tuple(box), outline=color, width=2)
        if all_queries and local != best_local:
            continue
        text = "q{}{}".format(query, " *" if local == best_local else "")
        text_width = pen.textlength(text, font=_font(19))
        tx = min(max(1, box[0]), max(1, view.width - text_width - 4))
        ty = max(1, min(box[1] - 23, view.height - 24))
        pen.text((tx, ty), text, font=_font(19), fill=color, stroke_width=2, stroke_fill="black")
    gt = np.asarray(record["gt_box"]) * box_scale
    _dashed_rectangle(pen, gt, "black", width=7)
    _dashed_rectangle(pen, gt, "white", width=3)
    pen.text((max(1, gt[0]), min(max(1, gt[3] + 3), view.height - 25)), "GT", font=_font(21),
             fill="white", stroke_width=2, stroke_fill="black")
    # Draw native Top-5 only AFTER greedy filtering; do not promote Top-6+.
    occupied = []
    for query, row in visible_top5.items():
        local = int(np.flatnonzero(kept == query)[0])
        box = boxes[query] * box_scale
        color = COLORS[int(labels[local])]
        pen.rectangle(tuple(box), outline="black", width=8)
        pen.rectangle(tuple(box), outline=color, width=5)
        text = "Top-{} q{} | {:.5f}".format(row["rank"], query, row["score"])
        tw = pen.textlength(text, font=_font(21)) + 12
        height = 30
        center = (box[:2] + box[2:]) / 2
        tx = min(max(1, box[0]), max(1, view.width - tw - 1))
        preferred = [box[1] - height - 3, box[3] + 3]
        preferred += [box[1] - height - 3 - height * shift for shift in range(1, 9)]
        preferred += [box[3] + 3 + height * shift for shift in range(1, 9)]
        label_box = None
        for candidate_y in preferred:
            ty = min(max(1, candidate_y), max(1, view.height - height - 1))
            trial = (tx, ty, tx + tw, ty + height)
            if not any(min(trial[2], old[2]) > max(trial[0], old[0]) and
                       min(trial[3], old[3]) > max(trial[1], old[1]) for old in occupied):
                label_box = trial
                break
        if label_box is None:
            label_box = (tx, 1, tx + tw, 1 + height)
        occupied.append(label_box)
        pen.line((center[0], center[1], (label_box[0] + label_box[2]) / 2,
                  (label_box[1] + label_box[3]) / 2), fill="black", width=4)
        pen.line((center[0], center[1], (label_box[0] + label_box[2]) / 2,
                  (label_box[1] + label_box[3]) / 2), fill=color, width=2)
        pen.rectangle(label_box, fill="#111b26", outline=color, width=2)
        pen.text((label_box[0] + 5, label_box[1] + 2), text, font=_font(21), fill="white")
    offset = ((panel_width - new_size[0]) // 2, header + (body_height - new_size[1]) // 2)
    page.paste(view, offset)

    fig, axis = plt.subplots(figsize=(panel_width / 140, body_height / 140), dpi=140)
    fig.patch.set_facecolor("#f4f6fa")
    axis.set_facecolor("white")
    for cluster in range(cluster_count):
        selected = labels == cluster
        axis.scatter(projection[selected, 0], projection[selected, 1], s=65 if all_queries else 115, c=COLORS[cluster],
                     edgecolors="white", linewidths=.8, label="C{}: n={}, mean CRS={:.6f}".format(
                         cluster + 1, int(selected.sum()), record["cluster_crs_statistics"][cluster]["mean"]), zorder=3)
    point = projection[best_local]
    axis.scatter([point[0]], [point[1]], s=620, marker="*", color=COLORS[int(labels[best_local])],
                 edgecolors="black", linewidths=1.8, zorder=6)
    for i, query in enumerate(kept):
        if int(query) in visible_top5:
            continue
        if all_queries and i != best_local:
            continue
        annotation = axis.annotate("q{}{}".format(query, " *" if i == best_local else ""),
                                   projection[i], xytext=(6, 6), textcoords="offset points",
                                   fontsize=10 if i == best_local else 8.5,
                                   fontweight="bold" if i == best_local else "normal", zorder=7)
        annotation.set_path_effects([effects.withStroke(linewidth=2.8, foreground="white")])
    offsets = [(24, 24), (28, -40), (-165, 44), (-165, -50), (28, 76)]
    for query, row in visible_top5.items():
        i = int(np.flatnonzero(kept == query)[0])
        color = COLORS[int(labels[i])]
        axis.scatter([projection[i, 0]], [projection[i, 1]], s=310, facecolors="none",
                     edgecolors="black", linewidths=1.5, zorder=8)
        position = (1.055, .94 - .18 * (row["rank"] - 1)) if all_queries else offsets[row["rank"] - 1]
        axis.annotate("Top-{} q{}{}\nCRS={:.5f}".format(
            row["rank"], query, " *" if i == best_local else "", row["score"]), projection[i],
            xytext=position, textcoords="axes fraction" if all_queries else "offset points", fontsize=10.5,
            annotation_clip=False,
            fontweight="bold", zorder=10, color="#162638",
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec=color, alpha=.96, lw=1.6),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.4))
    axis.set_xlabel("t-SNE 1 (not image x)", fontsize=12)
    axis.set_ylabel("t-SNE 2 (not image y)", fontsize=12)
    axis.grid(alpha=.13)
    axis.margins(.16)
    if not all_queries:
        axis.legend(loc="best", framealpha=.93, fontsize=11)
    axis.set_title("Star: q{} | GT IoU={:.3f} | CRS={:.6f}".format(
        record["retained_best_query"], record["retained_best_iou"], record["retained_best_crs_score"]), fontsize=14)
    if all_queries:
        # The eight cluster means already have a full-width legend above.
        # Reserve a separate right-hand column for five non-overlapping labels.
        fig.subplots_adjust(left=.10, right=.76, bottom=.10, top=.90)
        fig.text(.81, .94, "Native CRS Top-5", fontsize=12, fontweight="bold")
    else:
        fig.tight_layout(pad=2)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as plot:
        page.paste(plot.convert("RGB"), (panel_width, header))
    y = header + body_height + 8
    draw = ImageDraw.Draw(page)
    draw.text((28, y), ("Max GT-IoU query: q{} | IoU={:.4f} | detector={:.4f} | CRS={:.6f} | cluster C{}" if all_queries else
              "Best retained: q{} | IoU={:.4f} | detector={:.4f} | CRS={:.6f} | cluster C{}").format(
        record["retained_best_query"], record["retained_best_iou"], record["retained_best_detector_score"],
        record["retained_best_crs_score"],
        int(labels[best_local]) + 1), font=_font(27), fill="#152638")
    global_note = ("retained" if record["global_best_retained"] else
                   "REMOVED by higher-score overlapping q{}".format(record["global_best_suppressed_by"]))
    second_footer = ("No deduplication, NMS, detector threshold or Top-K filtering. All 256 queries enter feature clustering." if all_queries else
                     "Best of all 256: q{} | IoU={:.4f} | {}".format(
                         record["global_best_query"], record["global_best_iou"], global_note))
    draw.text((28, y + 40), second_footer, font=_font(26), fill="#152638")
    visible_text = " | ".join("Top-{}: q{}={:.5f}".format(r["rank"], r["query_id"], r["score"])
                              for r in visible_top5.values()) or "none retained"
    draw.text((28, y + 80), ("Native CRS Top-5 (all 256, no filtering): " if all_queries else
              "Native CRS Top-5 (retained only, no backfill): ") + visible_text,
              font=_font(23), fill="#152638")
    draw.text((28, y + 125), "Per-image clusters: same color across the two panels ONLY; not semantic classes or spatial clusters. IDs are zero-based.",
              font=_font(23), fill="#425466")
    relative = "images/{:04d}_{}.jpg".format(record["index"], Path(record["filename"]).stem)
    page.save(output / relative, quality=94, subsampling=0)
    return relative


def render(args):
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    import sklearn
    from threadpoolctl import threadpool_limits

    output = Path(args.output).resolve()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    records = json.loads((output / "samples.json").read_text(encoding="utf-8"))
    (output / "images").mkdir(exist_ok=True)
    (output / "clusters").mkdir(exist_ok=True)
    query_rows, sample_rows, cards, cluster_rows, top5_rows = [], [], [], [], []
    with threadpool_limits(limits=1):
        for record in tqdm(records, desc="K-Means + t-SNE + figures"):
            with np.load(output / record["cache"], allow_pickle=False) as cached:
                kept = cached["kept"]
                if max_pair_intersection(cached["boxes_xyxy"][kept]) != 0:
                    raise RuntimeError("Cached retained boxes overlap")
                np.testing.assert_array_equal(kept, disjoint_by_confidence(cached["boxes_xyxy"], cached["foreground"])[0])
                cluster_path = output / "clusters" / "{:04d}.npz".format(record["index"])
                if cluster_path.exists():
                    with np.load(cluster_path, allow_pickle=False) as old_clusters:
                        np.testing.assert_array_equal(old_clusters["query_ids"], kept)
                        np.testing.assert_allclose(old_clusters["normalized_features"], l2_features(cached["features"][kept]))
                        labels, projection = old_clusters["clusters"].copy(), old_clusters["tsne_xy"].copy()
                        distances = old_clusters["centroid_squared_distances"].copy()
                        stats = dict(perplexity=min(15.0, max(2.0, (len(kept) - 1) / 3.0)),
                                     inertia=float(distances[np.arange(len(kept)), labels].sum()),
                                     centroid_squared_distances=distances)
                else:
                    labels, projection, stats = cluster_features(cached["features"][kept], metadata["seed"], clusters=4)
                if "crs_scores" in cached:
                    crs_scores = cached["crs_scores"]
                else:
                    with np.load(output / "scores" / "{:04d}.npz".format(record["index"]), allow_pickle=False) as score_cache:
                        np.testing.assert_array_equal(score_cache["query_ids"], np.arange(256))
                        crs_scores = score_cache["crs_scores"].copy()
                record["cluster_crs_statistics"] = cluster_score_statistics(labels, crs_scores[kept])
                record["retained_best_crs_score"] = float(crs_scores[record["retained_best_query"]])
                record["crs_top5"] = crs_topk_annotations(crs_scores, kept)
                record["visible_crs_top5_count"] = sum(row["retained"] for row in record["crs_top5"])
                top5_ranks = {row["query_id"]: row["rank"] for row in record["crs_top5"]}
                for row in record["crs_top5"]:
                    top5_rows.append(dict(sample_index=record["index"], **row))
                for cluster_stats in record["cluster_crs_statistics"]:
                    cluster_rows.append(dict(sample_index=record["index"], **cluster_stats))
                record["visualization"] = render_one(record, cached, labels, projection, output)
                record["cluster_counts"] = np.bincount(labels, minlength=4).tolist()
                record["tsne_perplexity"] = stats["perplexity"]
                record["kmeans_inertia"] = stats["inertia"]
                record["cluster_cache"] = "clusters/{:04d}.npz".format(record["index"])
                if not cluster_path.exists():
                    np.savez_compressed(output / record["cluster_cache"], query_ids=kept, clusters=labels,
                                        tsne_xy=projection, normalized_features=l2_features(cached["features"][kept]),
                                        centroid_squared_distances=stats["centroid_squared_distances"])
                for local, query in enumerate(kept):
                    query_rows.append(dict(sample_index=record["index"], query_id=int(query),
                                           cluster=int(labels[local]) + 1, detector_score=float(cached["foreground"][query]),
                                           crs_score=float(crs_scores[query]),
                                           crs_top5_rank=top5_ranks.get(int(query), 0),
                                           gt_iou=float(cached["ious"][query]),
                                           best_retained=int(query) == record["retained_best_query"],
                                           best_global=int(query) == record["global_best_query"],
                                           x1=float(cached["boxes_xyxy"][query, 0]), y1=float(cached["boxes_xyxy"][query, 1]),
                                           x2=float(cached["boxes_xyxy"][query, 2]), y2=float(cached["boxes_xyxy"][query, 3]),
                                           tsne_1=float(projection[local, 0]), tsne_2=float(projection[local, 1])))
            sample_rows.append({k: record[k] for k in ("index", "filename", "retained_queries", "global_best_query",
                                                     "global_best_iou", "global_best_retained", "global_best_suppressed_by",
                                                     "retained_best_query", "retained_best_iou", "retained_best_crs_score", "visible_crs_top5_count", "visualization")})
            sample_rows[-1].update({"C{}_mean_crs".format(s["cluster"]): s["mean"] for s in record["cluster_crs_statistics"]})
            image_name = html.escape(record["visualization"], quote=True)
            cards.append('<article><h2>val #{:04d}: {} retained | best IoU {:.3f}</h2><p>{}</p>'
                         '<a href="{}"><img loading="lazy" src="{}"></a></article>'.format(
                             record["index"], record["retained_queries"], record["retained_best_iou"],
                             html.escape(record["expression"]), image_name, image_name))
    for name, rows in (("per_query.csv", query_rows), ("per_sample.csv", sample_rows),
                       ("per_cluster.csv", cluster_rows), ("crs_top5.csv", top5_rows)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(metadata, sklearn_version=sklearn.__version__, feature_normalization="Per-query L2 across 256 channels",
                   clustering="Per image, Euclidean K-Means on normalized 256-D features; K=4, n_init=30",
                   projection="Per image t-SNE on the SAME normalized 256-D features; not used for clustering",
                   crs_mean_definition="Unweighted arithmetic mean of native final CRS sigmoid scores among retained queries of each image/cluster; no softmax or rescoring",
                   top5_definition="Original full-256 CRS ranking; annotate only retained slots, no backfill; ties use ascending query ID",
                   visible_crs_top5_total=sum(r["visible_crs_top5_count"] for r in records),
                   retained_min=min(r["retained_queries"] for r in records),
                   retained_max=max(r["retained_queries"] for r in records),
                   retained_mean=float(np.mean([r["retained_queries"] for r in records])),
                   global_best_removed=sum(not r["global_best_retained"] for r in records),
                   before_dedup_gt_coverage_at_05=sum(r["global_best_iou"] >= .5 for r in records),
                   after_dedup_gt_coverage_at_05=sum(r["retained_best_iou"] >= .5 for r in records),
                   visualized_samples=len(records), all_samples_four_clusters=all(len(r["cluster_counts"]) == 4 and min(r["cluster_counts"]) > 0 for r in records))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "clustered_samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    document = ('<!doctype html><meta charset="utf-8"><title>Decoder feature clusters</title>'
                '<style>body{font:18px sans-serif;max-width:2200px;margin:24px auto;background:#edf1f6}'
                'article{background:white;padding:16px;margin:24px 0}img{width:100%;height:auto}p{line-height:1.6}</style>'
                '<h1>Decoder 后、CRS 前：严格无重叠候选与特征聚类</h1>'
                '<p>固定随机抽取 val 30 条。检测得分降序，任何正面积重叠均剔除；不设置信度阈值、不先取 Top-K。'
                '4 类 K-Means 在原始 256 维 query 特征经逐 query L2 归一化后完成；右侧 t-SNE 仅展示。'
                '星号为保留框中与 GT IoU 最大的 query，白虚线为 GT。每图单独聚类，颜色不代表跨图语义类别。'
                '图例 mean CRS 为各簇保留 query 的最终 CRS 概率算术平均，不是检测置信度，也没有去重后重跑 CRS。'
                'Top-1 至 Top-5 是原始 256 个 query 的 CRS 排名，仅标注去重后仍保留的 query，已过滤的名次不补位。'
                'GT 不参与筛选、聚类或推理。<a href="README.md">方法说明</a> | <a href="per_sample.csv">样本统计</a>'
                ' | <a href="per_cluster.csv">每簇 CRS 统计</a></p>'
                + "".join(cards))
    (output / "index.html").write_text(document, encoding="utf-8")
    sheet = Image.new("RGB", (4 * 720, math.ceil(len(records) / 4) * 470), "#edf1f6")
    for i, record in enumerate(records):
        with Image.open(output / record["visualization"]) as handle:
            thumb = handle.convert("RGB")
        thumb.thumbnail((712, 438), Image.Resampling.LANCZOS)
        left, top = (i % 4) * 720, (i // 4) * 470
        sheet.paste(thumb, (left + 4, top + 28))
        ImageDraw.Draw(sheet).text((left + 10, top + 3), "val #{:04d} | {} kept".format(
            record["index"], record["retained_queries"]), font=_font(19), fill="#152638")
    sheet.save(output / "contact_sheet.jpg", quality=92)
    report = """# Decoder query 特征聚类可视化

## 模型与数据

- 实验：`{experiment}`。
- 权重：`{checkpoint}`（第 {checkpoint_epoch} epoch）。严格使用当次归档 `propvg/` 和配置加载，不修改训练代码/配置/参数。
- 数据：val 中固定随机、无放回抽取 {visualized_samples} 条，种子 {seed}；不依赖 GT、预测正确性或聚类效果挑选样本，不替换不理想的图。
- 特征：最后一层 Deformable Decoder 的 `inter_states[-1]`，每图 `[256,256]`；尚未经过 `refer_adapter`、BCEM 或 CRS。该 BASE 的 Grab、BCEM、CRV 等均关闭。

## 每一步操作

1. 单卡 batch=1 正常前向，通过只读 hook 缓存 Decoder 特征、末层分类置信度及 query 对齐的预测框。GT 从模型输入中移除，仅离线分析。
2. 检测得分使用末层分类 logits 的 softmax 前景概率（排除 no-object），不是 CRS 定位分数。不先取 Top-64，也不加置信度阈值。
3. 框按原评估方式缩放到有效图像并裁边，剔除退化框；依检测得分从高到低贪心保留，只要与任一已保留框有正面积交集就丢弃。等价于 IoU 阈值为 0 的严格去重，允许边界相接；每图检测所有保留框两两交集确为 0。
4. 以保留 query 的 **256 维 Decoder 特征**逐 query 做 L2 归一化，然后每图独立 K-Means 聚成 4 类（k-means++，30 次初始化，固定种子）。不输入框位置、GT 或文本标签，不在 t-SNE 二维结果上聚类。
5. 同一组归一化特征做 t-SNE，perplexity 为 `min(15, max(2, (N-1)/3))`，PCA 初始化，1500 次迭代。左右同色同 query；编号为原始 0-based query ID。
6. 图中白色虚线为 GT，星号/加粗边框为 **保留集合中**与 GT IoU 最大的 query。图下注明全部 256 个 query 的最大 IoU query 是否被删、若被删则是哪一个高检测分 query 抑制了它，避免误把保留集合最佳当成原始全体最佳。
7. 图例中的 `mean CRS` 是当前图片该簇所有保留 query 的最终 CRS sigmoid 概率的**等权算术平均**，范围为 0–1，不是检测分、不做簇内 softmax、不按检测得分加权。CRS 仍在原始全部 256 个候选的正常交互上下文中前向；只在离线统计时取出去重保留 query 的得分，没有将去重后的 query 重新送入 CRS。星号 query 的单独 CRS 得分也标在图上。
8. 原始全部 256 个 query 按最终 CRS 得分降序取前五名，左右图只标其中去重后仍保留的框/点，并写明原始名次、query ID 和得分。被过滤的候选不画，Top-6 及以后不补位；同分时按原 query ID 排序（本次样本的前五名及边界无同分）。这个 Top-1 是 CRS 最高分候选，不是最大 GT IoU 的星号，两者可能不同。

已有聚类缓存时直接复用原有四类标签、颜色对应和 t-SNE 坐标，不因补充分数而重新聚类。旧缓存不包含 CRS 分数时，只补跑原有 30 个样本并核对图像、文本、特征、框、检测分和保留索引与旧缓存一致；原始缓存不覆盖，补充分数保存在 `scores/`。

## 当前结果

- 每图保留 {retained_min}–{retained_max} 个框，平均 {retained_mean:.2f} 个。
- 原始最大 IoU query 被严格去重删除：{global_best_removed}/{visualized_samples}。
- GT 覆盖（存在 IoU≥0.5 的框）：去重前 {before_dedup_gt_coverage_at_05}/{visualized_samples}，去重后 {after_dedup_gt_coverage_at_05}/{visualized_samples}。这里只是 30 条展示样本，不是完整 val 指标。
- 全部样本均获得 4 个非空特征簇：{all_samples_four_clusters}。

## 如何解读

特征已经经过图文骨干、Decoder 位置编码和候选交互，并非纯外观，也不是纯空间表示。不同颜色只代表当前图内特征相似性，不保证对应同一车道、同一物体类别或同一空间群体；这是需要观察的现象，不能先认定聚类就是物理分组。

K=4 是本次指定的展示约束，不代表每图真实有四类。t-SNE 的簇间距离和坐标轴不等于物理距离，局部二维形态也不能替代高维定量检验。不同图片的 C1–C4 不具有共同语义。

严格 IoU=0 会把部分真实相邻物体一起剔除；保留下来的框也不保证一定是前景。为忠实展示本次规则，没有按 GT、框面积或检测阈值追加筛选。

## 输出

- `index.html`：左右对照图库；`contact_sheet.jpg`：30 图总览；`images/`：完整分辨率图。
- `per_sample.csv`：保留数量、全体/保留集合最大 IoU 与删除情况。
- `per_query.csv`：保留 query ID、检测分数、最终 CRS 得分、GT IoU、类别编号、像素框、t-SNE 坐标。
- `per_cluster.csv`：30 张图各 4 类，共 120 行；每簇数量、CRS 均值/标准差/最小值/最大值。颜色编号仅在单图内有效，不将不同图片的 C1 当成同一语义类做总体平均。
- `crs_top5.csv`：原始 Top-5 的 query、CRS 得分和去重后是否保留；被删除的只在表中留记录，不在图上标注、不补位。
- `cache/`：每图全部 256 个原始 Decoder 特征、框、检测分数、保留索引、抑制关系；不必再次推理即可换聚类/绘图方案。
- `clusters/`：当前 4 类标签、归一化特征、t-SNE 坐标与到聚类中心的距离。
- `metadata.json` / `summary.json`：模型来源、样本索引、随机种子、检查结果和汇总。

propvg 环境没有 sklearn，因此只在该环境执行前向；聚类绘图使用已有的绘图 Python 环境，不安装或改变训练环境。
""".format(**summary)
    (output / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("visualized_samples", "retained_min", "retained_max", "retained_mean",
                                             "global_best_removed", "before_dedup_gt_coverage_at_05",
                                             "after_dedup_gt_coverage_at_05", "all_samples_four_clusters")},
                     ensure_ascii=False, indent=2), flush=True)
    print("Gallery: " + str(output / "index.html"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "add-scores", "render"), required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="top1_best.pth")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.stage in ("extract", "add-scores"):
        if not args.experiment:
            parser.error("--experiment is required for extraction")
        extract(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
