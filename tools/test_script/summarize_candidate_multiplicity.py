"""Offline interpretation of frozen-model context-multiplicity diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_candidate_multiplicity import score_order


def filename_cluster_interval(delta, filenames, seed=20261109, repetitions=5000):
    """Resample filenames with replacement, carrying ALL expressions per image.

    This handles repeated filenames, not latent dependence across video frames.
    """
    _, assignment = np.unique(filenames, return_inverse=True)
    counts = np.bincount(assignment)
    differences = np.bincount(assignment, weights=delta)
    generator = np.random.default_rng(seed)
    estimates = []
    for offset in range(0, repetitions, 100):
        sampled = generator.integers(0, len(counts), size=(min(100, repetitions - offset), len(counts)))
        estimates.extend((100 * differences[sampled].sum(1) / counts[sampled].sum(1)).tolist())
    return [float(value) for value in np.quantile(estimates, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis")
    args = parser.parse_args()
    output = Path(args.analysis).resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cache = np.load(output / "candidate_statistics.npz")
    ious, valid, scores = cache["ious"], cache["valid"], cache["scores"]
    groups, foreground = cache["groups"], cache["foreground"]
    modes = cache["mode_names"].tolist()
    with (output / "per_sample.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    native = [row for row in rows if row["mode"] == "native"]
    top1 = np.asarray([int(row["top1"]) for row in native])
    if not np.array_equal([int(row["index"]) for row in native], cache["indices"]):
        raise RuntimeError("Candidate cache/sample CSV are not aligned")
    sample = np.arange(len(ious))
    counts = np.stack([np.bincount(group_ids)[group_ids] for group_ids in groups])
    correct = ious[sample, top1] >= .5
    target_exists = (valid & (ious >= .5)).any(1)
    wrong_reachable = ~correct & target_exists
    best_target = np.argmax(np.where(valid & (ious >= .5), scores, -np.inf), axis=1)
    predicted_size, target_size = counts[sample, top1], counts[sample, best_target]
    crs_topk, detector_topk = {}, {}
    order = [score_order(s, v) for s, v in zip(scores, valid)]
    detector_order = [np.flatnonzero(v)[np.argsort(-s[v], kind="stable")] for s, v in zip(foreground, valid)]
    for k in (1, 5, 8, 16, 32, 64, 128, 256):
        crs_topk[str(k)] = sum(bool(ious[n, indices[:k]].max() >= .5) for n, indices in enumerate(order))
        detector_topk[str(k)] = sum(bool(ious[n, indices[:k]].max() >= .5) for n, indices in enumerate(detector_order))
    ranking_errors = []
    for n, ranked in enumerate(order):
        positives = np.flatnonzero(ious[n, ranked] >= .5)
        if correct[n] or not len(positives) or positives[0] >= 8:
            continue
        target = int(ranked[positives[0]])
        ranking_errors.append(dict(index=int(cache["indices"][n]), filename=native[n]["filename"],
            expression=native[n]["expression"], top1_query=int(top1[n]), target_query=target,
            first_target_rank=int(positives[0]) + 1, top1_iou=float(ious[n, top1[n]]),
            target_iou=float(ious[n, target]), top1_score=float(scores[n, top1[n]]),
            target_score=float(scores[n, target]), top1_fg=float(foreground[n, top1[n]]),
            target_fg=float(foreground[n, target]), top1_group_size=int(counts[n, top1[n]]),
            target_group_size=int(counts[n, target])))
    if ranking_errors:
        for filename, selected in (("ranking_errors_target_in_top8.csv", ranking_errors),
                ("ranking_error_audit100.csv", [ranking_errors[i] for i in np.sort(
                    np.random.default_rng(20261109).choice(len(ranking_errors), min(100, len(ranking_errors)), replace=False))])):
            with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
                writer.writeheader()
                writer.writerows(selected)
    control_checks = {mode: summary["modes"][mode] for mode in ("sham", "uniform_x8")}
    union_group_flips = np.zeros(len(ious), dtype=bool)
    for mode in ("top1_slot_x8", "rival_slot_x8", "top1_group_x8", "rival_group_x8", "detector_top1_x8"):
        selected = [row for row in rows if row["mode"] == mode]
        union_group_flips |= np.asarray([int(row["group_flip"]) for row in selected], dtype=bool)
    details = dict(samples=len(ious), distinct_filenames=len(set(row["filename"] for row in native)),
        oracle_reachable=int(target_exists.sum()), oracle_unreachable=int((~target_exists).sum()),
        correct=int(correct.sum()), wrong=int((~correct).sum()),
        wrong_target_already_in_top8=len(ranking_errors),
        wrong_reachable=int(wrong_reachable.sum()),
        wrong_reachable_predicted_group_more_duplicates=int((wrong_reachable & (predicted_size > target_size)).sum()),
        wrong_reachable_equal_duplicates=int((wrong_reachable & (predicted_size == target_size)).sum()),
        wrong_reachable_target_group_more_duplicates=int((wrong_reachable & (predicted_size < target_size)).sum()),
        wrong_reachable_predicted_group_mean_size=float(predicted_size[wrong_reachable].mean()),
        wrong_reachable_target_group_mean_size=float(target_size[wrong_reachable].mean()),
        top1_group_size_distribution={str(k): int((predicted_size == k).sum()) for k in np.unique(predicted_size)},
        mean_group_count=float(np.mean([len(np.unique(g)) for g in groups])),
        mean_duplicate_slots=float(np.mean([len(g) - len(np.unique(g)) for g in groups])),
        union_group_flips_x8=int(union_group_flips.sum()),
        correct_affected_x8=int((union_group_flips & correct).sum()),
        wrong_affected_x8=int((union_group_flips & ~correct).sum()),
        crs_topk_hits=crs_topk, detector_topk_hits=detector_topk,
        crs_topk_pct={k: 100 * v / len(ious) for k, v in crs_topk.items()},
        detector_topk_pct={k: 100 * v / len(ious) for k, v in detector_topk.items()},
        controls_no_ranking_flip=all(not v["slot_flips"] for v in control_checks.values()),
        exact_same_boxes_for_all_interventions=True,
        duplicate_group_caveat="IoU groups are a geometric proxy, not physical-instance GT.")
    # The exploratory stratum is labeled, not used to choose an operating point.
    details["exploratory_balance_strata"] = {}
    balanced = [row for row in rows if row["mode"] == "balance_iou0.8"]
    for name, mask in (("original_correct", correct), ("original_wrong", ~correct),
                       ("top1_singleton", predicted_size == 1), ("top1_group_ge3", predicted_size >= 3)):
        selected = [row for row, keep in zip(balanced, mask) if keep]
        details["exploratory_balance_strata"][name] = dict(samples=len(selected),
            corrected=sum(int(row["corrected"]) for row in selected),
            harmed=sum(int(row["harmed"]) for row in selected))
    details["filename_cluster_bootstrap_95ci_pp"] = {}
    for mode in ("balance_iou0.8", "dedup_context_iou0.8", "top1_slot_x8"):
        selected = [row for row in rows if row["mode"] == mode]
        delta = np.asarray([int(row["corrected"]) - int(row["harmed"]) for row in selected])
        details["filename_cluster_bootstrap_95ci_pp"][mode] = filename_cluster_interval(
            delta, [row["filename"] for row in native])
    details["copied_candidate_own_logit_changes"] = {}
    for mode in ("top1_slot_x2", "top1_slot_x4", "top1_slot_x8", "top1_group_x8"):
        selected = [row for row in rows if row["mode"] == mode]
        change = np.asarray([float(row["native_top1_logit_change"]) for row in selected])
        details["copied_candidate_own_logit_changes"][mode] = dict(
            mean=float(change.mean()), increased=int((change > 1e-4).sum()),
            decreased=int((change < -1e-4).sum()), negligible=int((np.abs(change) <= 1e-4).sum()))
    (output / "mechanism_checks.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    native_metric = summary["modes"]["native"]
    balance = summary["modes"]["balance_iou0.8"]
    dedup = summary["modes"]["dedup_context_iou0.8"]
    copy = summary["modes"]["top1_slot_x8"]
    lines = ["# 诊断结论与下一步", "", "## 结论边界", "",
        "本次仅诊断归档的 20261015_215412 BASE，不训练新模块。{}条{}表达式，{}。".format(
            len(ious), summary["split"], "完整 split" if summary["complete_split"] else "固定随机子集"),
        "验证集只用于诊断；没有按这些结果改参数或报告新的测试集性能。", "",
        "## 1. 候选数量敏感性成立，但不等于已找到主要性能瓶颈", "",
        "复制原 Top-1 的同一个 K/V 表示至总计8份，{}条发生槽位变化（{:.2f}%），{}条跨 IoU 组变化。".format(
            copy["slot_flips"], 100 * copy["slot_flips"] / len(ious), copy["group_flips"]),
        "图像和文本没有增加信息，所有框及输出槽位不变。sham 与全体均匀复制的排名变化均为0：{}。".format(details["controls_no_ranking_flip"]),
        "这证明 frozen CRS 会受表示计数影响；不能据此直接断言自然错误主要由重复候选导致。",
        "复制原Top-1至8份后，该候选自身logit上升/下降/基本不变：{}/{}/{}条，平均变化{:.3f}。".format(
            details["copied_candidate_own_logit_changes"]["top1_slot_x8"]["increased"],
            details["copied_candidate_own_logit_changes"]["top1_slot_x8"]["decreased"],
            details["copied_candidate_own_logit_changes"]["top1_slot_x8"]["negligible"],
            details["copied_candidate_own_logit_changes"]["top1_slot_x8"]["mean"]),
        "因此不能把‘重复表示更多’直接叙述成‘该物体获得更高定位分数’；注意力质量与最终定位分数不是同一个量。", "",
        "## 2. 不训练时，直接消除重复权重的结果", "",
        "|方法|Top-1|Top-5|mIoU|改对|改错|净变化（百分点）|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in (("原始 BASE", native_metric), ("按组均衡（IoU=0.8）", balance), ("仅上下文去重（IoU=0.8）", dedup)):
        lines.append("|{}|{:.3f}|{:.3f}|{:.3f}|{}|{}|{:+.3f}|".format(name, metric["top1"], metric["top5"],
            metric["miou"], metric["corrected"], metric["harmed"], metric["net_pp"]))
    lines += ["", "均衡的配对净变化，按文件名聚类bootstrap的95%区间：[{:.3f}, {:.3f}]个百分点。".format(
            *details["filename_cluster_bootstrap_95ci_pp"]["balance_iou0.8"]),
        "5000次有放回图像重采样，每张图的表达式一起抽取；仍不能排除不同文件之间的场景/视频相关性。", "",
        "## 3. 自然错误是否符合‘错误物体靠更多重复候选胜出’", "",
        "原 Top-1 错误但256框内存在正确框的{}条样本中：".format(details["wrong_reachable"]), "",
        "- 错误候选组更多重复：{}条。".format(details["wrong_reachable_predicted_group_more_duplicates"]),
        "- 两组数量相同：{}条。".format(details["wrong_reachable_equal_duplicates"]),
        "- 正确候选组更多重复：{}条。".format(details["wrong_reachable_target_group_more_duplicates"]),
        "- 错误组/正确组平均数量：{:.3f}/{:.3f}。正确组取 refer GT IoU≥0.5 中原 CRS 得分最高的框所在组。".format(
            details["wrong_reachable_predicted_group_mean_size"], details["wrong_reachable_target_group_mean_size"]), "",
        "这里的分组只是 IoU 代理；统计关系不是物理因果证明。", "", "## 4. 下一步应做什么", ""]
    if details["filename_cluster_bootstrap_95ci_pp"]["balance_iou0.8"][0] > 0:
        lines += ["值得做最小训练对照，但还不能直接扩展成复杂三阶段模块：", "",
            "1. 保持256个输出槽位、检测/BCEM/文本路径不变，只在 CRS 自注意力中加入组计数校正。",
            "2. 同预算联合训练原始 BASE、计数校正、普通上下文去重三个对照；至少3个种子。",
            "3. 同时汇报原始精度、改对/改错、复制扰动稳定性，证明收益不是简单去重或随机正则化。"]
    else:
        lines += ["当前证据不足以把 IBCI 当作确定有效的第二创新点，不建议马上写复杂三阶段网络或投入完整重训练。", "",
            "首选的下一步：审查已导出的 ranking_error_audit100.csv。这些样本原Top-1错误，但Top-8已有正确框；逐项区分外观混淆、参考物缺失/关系判断、标注歧义。当前自动统计不能直接判断这些语义原因。", "",
            "如果仍要检验这个方向，下一步只做一个可证伪的小规模训练对照：", "",
            "1. 结构仍用原 CRS，图像/文本/256个目标槽位不变；图像编码与检测前向只做一次，候选特征走共享 CRS 的原始与次数扰动两分支，用同一 refer GT 监督。",
            "2. 定位损失取 $L_{refer}=0.5(L_{original}+L_{perturbed})$，检测损失等原样保留，避免仅因总定位损失翻倍造成伪收益。",
            "3. 与同预算原始训练、随机上下文 dropout 做对照，区分‘针对数量偏置’和普通正则化效果。联合训练，不需要增加训练阶段。",
            "4. 预先规定成功条件：未扰动验证集 Top-1 稳定提升，改对多于改错，且复制扰动的跨组变化下降；不能只追求鲁棒性数字。",
            "5. 若该最小实验仍无稳定净收益，停止该方向；回到原 CRS，避免再叠加门控、重排序器和新评分头。"]
    lines += ["", "这仍不保证训练有效；零训练修正失败也不证明任何训练方案都不可能成功。", "",
        "## 5. 原始排序的覆盖上限", "",
        "下表只描述现有排序，不代表一个新训练的外观阶段会有同样召回。", "",
        "|K|检测置信度Top-K目标召回（%）|原CRS Top-K命中（%）|", "|---|---:|---:|"]
    for k in crs_topk:
        lines.append("|{}|{:.3f}|{:.3f}|".format(k, details["detector_topk_pct"][k], details["crs_topk_pct"][k]))
    lines += ["", "## 文件", "",
        "- report.md：全部干预结果和方法。",
        "- per_sample.csv / changed_examples.csv：逐条配对数据。",
        "- candidate_statistics.npz：未过滤槽位的框、置信度、GT IoU、分组及各分支 logits，可离线复核。",
        "- mechanism_checks.json：自然重复数量、原始 Top-K 覆盖和分层统计。",
        "- ranking_errors_target_in_top8.csv / ranking_error_audit100.csv：优先审查的排序错误及固定随机100条子集。", ""]
    (output / "诊断结论与下一步.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
