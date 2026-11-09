"""GT-only offline checks of captured, GT-free two-stage relation predictions."""
import numpy as np
import torch


def sample_record(detail, target, index, filename="", expression=""):
    boxes = detail["boxes"].float().cpu()
    gt = torch.as_tensor(target, dtype=torch.float32).cpu().reshape(4)
    inter = (torch.minimum(boxes[:, 2:], gt[2:]) - torch.maximum(boxes[:, :2], gt[:2])).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(-1)
    gt_area = (gt[2:] - gt[:2]).clamp(min=0).prod()
    ious = inter / (area + gt_area - inter).clamp(min=1e-12)
    record = dict(dataset_index=int(index), filename=str(filename), expression=str(expression))
    for mode, field in (("crs", "crs_order"), ("crs_logit", "logit_order"),
                        ("relation", "relation_order"), ("final", "final_order")):
        order = detail[field].long().cpu()
        ranked = ious[order]
        record[mode + "_top1_query"] = int(order[0]) if len(order) else -1
        record[mode + "_top1_iou"] = float(ranked[0]) if len(order) else 0.
        record[mode + "_top1_correct"] = bool(len(order) and ranked[0] >= .5)
        for k in (3, 5, 8, 16, 32, 64, 128, 256):
            record[mode + "_top{}_hit".format(k)] = bool(len(order) and ranked[:k].max() >= .5)
    selected = detail["target_indices"][detail["target_valid"]].long()
    old, new = record["crs_top1_correct"], record["final_top1_correct"]
    groups = detail["group_valid"]
    group_logits = detail["crs_logits"][detail["target_indices"][detail["group_slots"][groups]]].float()
    p = group_logits.softmax(0)
    entropy = float(-(p * p.clamp(min=1e-12).log()).sum() / np.log(len(p))) if len(p) > 1 else 0.
    record.update(
        corrected=not old and new, harmed=old and not new,
        rank_changed=record["crs_top1_query"] != record["final_top1_query"],
        target_pool_reachable=bool(len(selected) and (ious[selected] >= .5).any()),
        null_weight=float(detail["null_weight"]), strength=float(detail["strength"]),
        reference_count=int(detail["reference_valid"].sum()), group_count=int(groups.sum()),
        mean_abs_correction=float(detail["correction"][detail["target_valid"]].abs().mean()) if len(selected) else 0.,
        crs_group_entropy=entropy,
    )
    return record


def summarize(records):
    if not records:
        raise ValueError("No relation-refinement records to summarize")
    n = len(records)
    result = dict(samples=n)
    for mode in ("crs", "crs_logit", "relation", "final"):
        result[mode] = dict(
            top1_acc=100 * np.mean([row[mode + "_top1_correct"] for row in records]),
            top5_acc=100 * np.mean([row[mode + "_top5_hit"] for row in records]),
            miou=100 * np.mean([row[mode + "_top1_iou"] for row in records]),
            topk={str(k): 100 * np.mean([row[mode + "_top{}_hit".format(k)] for row in records])
                  for k in (3, 5, 8, 16, 32, 64, 128, 256)},
        )
    corrected, harmed = sum(row["corrected"] for row in records), sum(row["harmed"] for row in records)
    result.update(corrected=corrected, harmed=harmed, net=corrected-harmed,
                  net_accuracy_points=100 * (corrected-harmed)/n,
                  rank_changed=sum(row["rank_changed"] for row in records),
                  reachable=sum(row["target_pool_reachable"] for row in records),
                  mean_null_weight=float(np.mean([row["null_weight"] for row in records])),
                  mean_abs_correction=float(np.mean([row["mean_abs_correction"] for row in records])),
                  mean_reference_count=float(np.mean([row["reference_count"] for row in records])),
                  mean_strength=float(np.mean([row["strength"] for row in records])))
    result["entropy_groups"] = []
    for lo, hi in ((0., .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.000001)):
        rows = [r for r in records if lo <= r["crs_group_entropy"] < hi]
        result["entropy_groups"].append(dict(
            lower=lo, upper=min(hi, 1.), samples=len(rows),
            corrected=sum(r["corrected"] for r in rows), harmed=sum(r["harmed"] for r in rows)))
    result["notes"] = [
        "CRS is the first stage of THIS jointly trained checkpoint, not the original independent baseline.",
        "crs_logit isolates probability-tie/precision effects inside the same shortlist.",
        "relation ranks the shortlist by signed correction only; it is a diagnostic, not a standalone grounding model.",
        "Empty/null reference weight is NOT a supervised probability that the sentence has no reference object.",
        "All GT is used offline, after prediction; there is no reference-object GT recall claim.",
    ]
    return result
