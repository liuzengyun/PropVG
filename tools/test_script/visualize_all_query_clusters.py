"""Cluster all 256 cached decoder queries, without altering model or old figures.

Run with the existing plotting Python (sklearn required), no CUDA or torch.
Scores come from the previous native full-model forward, not from clustering.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from visualize_decoder_clusters import (
    _font, cluster_features, cluster_score_statistics, crs_topk_annotations,
    l2_features, render_one,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_digests(source):
    """Include all old plots/caches/metadata to verify this run is additive only."""
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Previous 30-sample feature/score cache")
    parser.add_argument("--output", required=True, help="Fresh output folder, never the source")
    parser.add_argument("--clusters", type=int, default=8, choices=(4, 8))
    parser.add_argument("--reuse-clusters", action="store_true", help="Redraw existing output without changing clustering or t-SNE")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate sibling output folder")
    if args.reuse_clusters:
        previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if previous["clusters_per_image"] != args.clusters or Path(previous["source_cache_directory"]) != source:
            raise ValueError("Existing results use a different source or cluster count")
    elif output.exists():
        raise FileExistsError("Refusing to replace existing results: " + str(output))
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    records = json.loads((source / "samples.json").read_text(encoding="utf-8"))
    if len(records) != 30 or [r["index"] for r in records] != metadata["indices"]:
        raise ValueError("Expected the exact previous 30 samples in original order")
    original_digests = source_digests(source)
    output.mkdir(parents=True, exist_ok=args.reuse_clusters)
    (output / "images").mkdir(exist_ok=args.reuse_clusters)
    (output / "clusters").mkdir(exist_ok=args.reuse_clusters)
    queries, clusters, top5_rows, sample_rows, rendered, cards = [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for old_record in tqdm(records, desc="All 256 queries: K-Means + t-SNE"):
            record = dict(old_record)
            with np.load(source / record["cache"], allow_pickle=False) as handle:
                cached = {name: handle[name].copy() for name in handle.files}
            if "crs_scores" not in cached:
                with np.load(source / "scores" / "{:04d}.npz".format(record["index"]), allow_pickle=False) as scores:
                    np.testing.assert_array_equal(scores["query_ids"], np.arange(256))
                    cached["crs_scores"] = scores["crs_scores"].copy()
            if cached["features"].shape != (256, 256) or cached["crs_scores"].shape != (256,):
                raise ValueError("Expected all 256 decoder query features and native CRS scores")
            if not np.isfinite(cached["features"]).all():
                raise ValueError("Non-finite decoder features")
            # Deliberately ignore old selection/validity arrays for clustering.
            # No detector filtering, score sorting/truncation, or deduplication.
            cached["kept"] = np.arange(256, dtype=np.int64)
            if args.reuse_clusters:
                with np.load(output / "clusters" / "{:04d}.npz".format(record["index"]), allow_pickle=False) as old:
                    np.testing.assert_array_equal(old["query_ids"], np.arange(256))
                    np.testing.assert_allclose(old["normalized_features"], l2_features(cached["features"]))
                    np.testing.assert_array_equal(old["crs_scores"], cached["crs_scores"])
                    labels, projection = old["clusters"].copy(), old["tsne_xy"].copy()
                    distances = old["centroid_squared_distances"].copy()
                    stats = dict(perplexity=15.0, inertia=float(distances[np.arange(256), labels].sum()),
                                 centroid_squared_distances=distances)
            else:
                labels, projection, stats = cluster_features(cached["features"], metadata["seed"], args.clusters)
            best = int(np.argmax(cached["ious"]))
            record.update(all_queries=True, retained_queries=256, retained_query_ids=list(range(256)),
                          global_best_query=best, global_best_iou=float(cached["ious"][best]),
                          global_best_retained=True, global_best_suppressed_by=-1,
                          retained_best_query=best, retained_best_iou=float(cached["ious"][best]),
                          retained_best_detector_score=float(cached["foreground"][best]),
                          retained_best_crs_score=float(cached["crs_scores"][best]),
                          cluster_counts=np.bincount(labels, minlength=args.clusters).tolist(),
                          cluster_crs_statistics=cluster_score_statistics(labels, cached["crs_scores"], args.clusters),
                          crs_top5=crs_topk_annotations(cached["crs_scores"], cached["kept"]),
                          visible_crs_top5_count=5, tsne_perplexity=stats["perplexity"],
                          kmeans_inertia=stats["inertia"], source_cache=str(source / old_record["cache"]),
                          cluster_cache="clusters/{:04d}.npz".format(record["index"]))
            record.pop("max_retained_pair_intersection", None)
            record.pop("cache", None)
            for row in record["crs_top5"]:
                row["cluster"] = int(labels[row["query_id"]]) + 1
                row["gt_iou"] = float(cached["ious"][row["query_id"]])
                top5_rows.append(dict(sample_index=record["index"], **row))
            top_ranks = {row["query_id"]: row["rank"] for row in record["crs_top5"]}
            for group in record["cluster_crs_statistics"]:
                mask = labels == group["cluster"] - 1
                clusters.append(dict(sample_index=record["index"], **group,
                                     mean_detector_score=float(cached["foreground"][mask].astype(float).mean()),
                                     gt_iou_ge_05_count=int((cached["ious"][mask] >= .5).sum()),
                                     contains_max_gt_iou_query=bool(mask[best])))
            for query in range(256):
                queries.append(dict(sample_index=record["index"], query_id=query,
                                    cluster=int(labels[query]) + 1,
                                    detector_score=float(cached["foreground"][query]),
                                    crs_score=float(cached["crs_scores"][query]),
                                    crs_top5_rank=top_ranks.get(query, 0), gt_iou=float(cached["ious"][query]),
                                    max_gt_iou_query=query == best,
                                    x1=float(cached["boxes_xyxy"][query, 0]), y1=float(cached["boxes_xyxy"][query, 1]),
                                    x2=float(cached["boxes_xyxy"][query, 2]), y2=float(cached["boxes_xyxy"][query, 3]),
                                    tsne_1=float(projection[query, 0]), tsne_2=float(projection[query, 1])))
            if not args.reuse_clusters:
                np.savez_compressed(output / record["cluster_cache"], query_ids=cached["kept"],
                                    clusters=labels, tsne_xy=projection,
                                    normalized_features=l2_features(cached["features"]),
                                    centroid_squared_distances=stats["centroid_squared_distances"],
                                    crs_scores=cached["crs_scores"], foreground=cached["foreground"],
                                    boxes_xyxy=cached["boxes_xyxy"], ious=cached["ious"])
            record["visualization"] = render_one(record, cached, labels, projection, output)
            sample_rows.append(dict(index=record["index"], filename=record["filename"], queries=256,
                                    max_gt_iou_query=best, max_gt_iou=record["global_best_iou"],
                                    max_gt_iou_query_cluster=int(labels[best]) + 1,
                                    max_mean_crs_cluster=int(np.argmax([g["mean"] for g in record["cluster_crs_statistics"]])) + 1,
                                    **{"C{}_mean_crs".format(g["cluster"]): g["mean"]
                                       for g in record["cluster_crs_statistics"]},
                                    visualization=record["visualization"]))
            rendered.append(record)
            picture = html.escape(record["visualization"], quote=True)
            cards.append('<article><h2>val #{:04d} | 256 queries | {} clusters</h2><p>{}</p>'
                         '<a href="{}"><img loading="lazy" src="{}"></a></article>'.format(
                             record["index"], args.clusters, html.escape(record["expression"]), picture, picture))
    for name, rows in (("per_query.csv", queries), ("per_cluster.csv", clusters),
                       ("crs_top5.csv", top5_rows), ("per_sample.csv", sample_rows)):
        write_csv(output / name, rows)
    # Source is outside output: any difference would be an unintended mutation.
    if source_digests(source) != original_digests:
        raise RuntimeError("Original source results changed during this run")
    summary = dict(experiment=metadata["experiment"], checkpoint=metadata["checkpoint"],
                   checkpoint_epoch=metadata["checkpoint_epoch"], source_cache_directory=str(source),
                   split=metadata["split"], seed=metadata["seed"], indices=metadata["indices"], samples=30,
                   queries_per_image=256, clusters_per_image=args.clusters,
                   feature_source=metadata["source_feature"], feature_normalization="Per-query L2 across 256 channels",
                   clustering="Per image K-Means in normalized 256-D decoder feature space, k-means++, n_init=30",
                   projection="t-SNE on same 256 features; perplexity=15, PCA init, 1500 iterations",
                   score_source="Original full-256 query CRS sigmoid output, unchanged; never fed into K-Means/t-SNE",
                   score_statistic="Unweighted mean final CRS probability per image/cluster, not detector score",
                   filtering="NONE: all 256 query slots, no clipping-validity exclusion, NMS, threshold or dedup",
                   image_display="All 256 boxes faint, CRS Top-5 and GT emphasized; all 256 points in t-SNE",
                   plotted_top5=150, per_query_rows=len(queries), per_cluster_rows=len(clusters),
                   source_unchanged=True, additional_model_inference=False, used_gpu=False,
                   all_clusters_nonempty=all(min(r["cluster_counts"]) > 0 for r in rendered))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "clustered_samples.json").write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    document = ('<!doctype html><meta charset="utf-8"><title>All 256 query clusters</title>'
                '<style>body{font:18px sans-serif;max-width:2200px;margin:24px auto;background:#edf1f6}'
                'article{background:white;padding:16px;margin:24px 0}img{width:100%;height:auto}p{line-height:1.7}</style>'
                '<h1>原始 256 个 query：8 类特征聚类与 CRS Top-5</h1>'
                '<p>沿用上一组完全相同的 30 个 val 样本和原始 CRS 前向分数。不去重、不筛选，每图全部 256 个 '
                'Decoder query 特征进行 L2 归一化后聚成 8 类；不使用 CRS 分数或 GT 参与聚类。'
                '右侧 t-SNE 包含所有 256 个点，并标注 Top-1～5 的原始排名、query ID 与 CRS 分数。'
                '左侧淡色显示所有框，加粗 Top-5，GT 白色虚线，最大 GT-IoU query 用星号标记。'
                '每簇图例含数量和平均 CRS 得分。颜色编号仅在单图内有意义。'
                '<a href="README.md">方法说明</a> | <a href="per_cluster.csv">每簇统计</a> | '
                '<a href="crs_top5.csv">Top-5 明细</a></p>' + "".join(cards))
    (output / "index.html").write_text(document, encoding="utf-8")
    sheet = Image.new("RGB", (4 * 720, math.ceil(len(rendered) / 4) * 470), "#edf1f6")
    for i, record in enumerate(rendered):
        with Image.open(output / record["visualization"]) as handle:
            thumbnail = handle.convert("RGB")
        thumbnail.thumbnail((712, 438), Image.Resampling.LANCZOS)
        left, top = (i % 4) * 720, (i // 4) * 470
        sheet.paste(thumbnail, (left + 4, top + 28))
        ImageDraw.Draw(sheet).text((left + 10, top + 3), "val #{:04d} | 256 queries | K={}".format(
            record["index"], args.clusters), font=_font(19), fill="#152638")
    sheet.save(output / "contact_sheet.jpg", quality=92)
    report = """# 全部 256 个 query 的 8 类聚类

沿用上一组相同的 30 个 val 样本，BASE `20261105_105035` 的 `top1_best.pth`（epoch 28）。
复用原来严格加载归档源码及权重得到的特征和最终 CRS 概率，本次没有模型前向、训练或 GPU 使用。

## 数据与操作

1. 每图原始 Decoder 末层特征 `[256,256]`，位置在 `refer_adapter` 和 CRS 之前。
2. 原始 CRS 概率已通过全部 256 个 query 的正常图文评分得到。分数仅用于排名、标注及聚类后的统计；不作为聚类特征，也不参与 t-SNE。
3. 本次 **不做任何去重、NMS、检测阈值、Top-K 或退化框筛选**，所有 256 个 query 槽位均参与聚类。
4. 每个 query 的 256 维特征先 L2 归一化，再每图独立 K-Means（K=8、k-means++、30 次初始化、固定种子）。没有在二维 t-SNE 坐标上聚类。
5. 同一组特征做 t-SNE（perplexity=15、PCA 初始化、1500 迭代），所有 256 个点均显示。重点标注原始 CRS Top-1～5 的 query 编号、概率和排名；星号为全体 query 中与 GT IoU 最大者。
6. 左侧所有 256 个框以淡色显示，突出 Top-5 与 GT，避免 256 个文字标签遮挡图像。没有移动预测框坐标。
7. 每簇显示 query 数和最终 CRS 概率算术平均；CSV 另有标准差、最小/最大值、检测平均分。GT 只离线统计和标注。

这是**Decoder 特征的聚类**，不是 CRS 输出特征或 CRS 分数的一维分组。先取得 CRS 得分并不会改变被聚类的特征阶段。
每张图的 C1～C8 不代表固定语义类别，颜色也不跨图对齐；t-SNE 距离不是物理距离。固定 8 类是展示设定，不代表真实类别数。

## 输出

- `index.html` / `images/` / `contact_sheet.jpg`：图库、30 张原图与总览。
- `per_query.csv`：30×256=7680 行，全部 query 的类别、CRS/检测分、GT IoU、框与 t-SNE 坐标。
- `per_cluster.csv`：30×8=240 行，每图每簇统计。
- `crs_top5.csv`：30×5=150 行，全部 Top-5 均标注，不再过滤。
- `clusters/`：全部 query 的归一化特征、类别、t-SNE、原始 CRS 概率与预测框。
- `summary.json`：完整来源、样本索引与设置；`clustered_samples.json`：图像级明细。

本结果另存于独立目录，旧的严格去重 4 类图、特征、坐标和分数均未修改，并进行了源文件 SHA-256 一致性检查。
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Gallery: " + str(output / "index.html"), flush=True)


if __name__ == "__main__":
    main()
