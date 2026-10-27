"""Grounding correctness discrimination/calibration; GT is evaluator-only."""
import numpy as np
import torch


def correctness_record(prediction, target):
    if "top1_correct_probability" not in prediction:
        return None
    boxes = prediction["boxes"].detach().cpu()
    ranking = prediction.get("ranking_logits", prediction["scores"]).detach().reshape(-1).cpu()
    order = ranking.argsort(descending=True, stable="ranking_logits" in prediction)
    overlap = 0.
    if len(order):
        box = boxes[order[0]]
        gt = torch.as_tensor(target, dtype=boxes.dtype).cpu().reshape(-1, 4)[0]
        intersection = (torch.minimum(box[2:], gt[2:]) - torch.maximum(box[:2], gt[:2])).clamp(min=0).prod()
        union = (box[2:] - box[:2]).clamp(min=0).prod() + (gt[2:] - gt[:2]).clamp(min=0).prod() - intersection
        overlap = float(intersection / union.clamp(min=1e-6))
    return dict(probability=float(prediction["top1_correct_probability"]), correct=int(overlap >= .5),
                logit=float(prediction["top1_correct_logit"]), entropy=float(prediction["top1_grouped_entropy"]))


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = scores.argsort(kind="stable")
    ranked = scores[order]
    start, rank_sum = 0, 0.
    while start < len(order):
        end = start + 1
        while end < len(order) and ranked[end] == ranked[start]:
            end += 1
        rank_sum += labels[order[start:end]].sum() * (start + 1 + end) / 2
        start = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def summarize_correctness(records, threshold=.5):
    if not records:
        return None
    probability = np.array([record["probability"] for record in records])
    correct = np.array([record["correct"] for record in records], dtype=bool)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Correctness probabilities must be finite and in [0,1]")
    predicted_wrong, wrong = probability < threshold, ~correct
    detected = int((predicted_wrong & wrong).sum())
    false_alarms = int((predicted_wrong & correct).sum())
    ratio = lambda numerator, denominator: float(100 * numerator / denominator) if denominator else None
    ece = 0.
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        chosen = (probability >= lower) & ((probability <= upper) if upper == 1 else (probability < upper))
        if chosen.any():
            ece += chosen.mean() * abs(probability[chosen].mean() - correct[chosen].mean())
    return dict(samples=len(records), accuracy=ratio(int(((~predicted_wrong) == correct).sum()), len(records)),
                majority_accuracy=ratio(max(int(correct.sum()), int(wrong.sum())), len(records)),
                error_precision=ratio(detected, int(predicted_wrong.sum())),
                error_recall=ratio(detected, int(wrong.sum())), correct_fpr=ratio(false_alarms, int(correct.sum())),
                auroc=binary_auc(wrong, [-record["logit"] for record in records]),
                entropy_auroc=binary_auc(wrong, [record["entropy"] for record in records]),
                brier=float(((probability - correct) ** 2).mean()), ece=float(ece),
                predicted_wrong=int(predicted_wrong.sum()), detected_wrong=detected, false_alarms=false_alarms)


def log_correctness(metrics, logger, prefix, track=None):
    if metrics is None:
        return
    percent = lambda key: "N/A" if metrics[key] is None else f"{metrics[key]:.2f}%"
    decimal = lambda key: "N/A" if metrics[key] is None else f"{metrics[key]:.4f}"
    logger.info(f"{prefix} - Top-1 correctness head: Acc={percent('accuracy')}, "
                f"majority baseline={percent('majority_accuracy')}, error precision={percent('error_precision')}, "
                f"error recall={percent('error_recall')}, correct FPR={percent('correct_fpr')}, "
                f"AUROC={decimal('auroc')}, entropy AUROC={decimal('entropy_auroc')}, "
                f"Brier={decimal('brier')}, ECE={decimal('ece')}, N={metrics['samples']}")
    if track is not None:
        namespace = "test_epoch" if prefix.startswith("test_epoch_") else "test_best" if prefix == "test" else prefix
        track({namespace + "/correctness/" + key: value for key, value in metrics.items() if value is not None})
