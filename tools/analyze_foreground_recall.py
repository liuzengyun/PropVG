"""Read-only GT recall analysis of DETR foreground confidence, not referring scores.

All metrics are computed offline, after GT is removed from model inputs. No
model filtering, weights, training configuration or neighbor policy is changed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/propvg_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/propvg_cache')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mmcv import Config
from tqdm import tqdm

from propvg.datasets import build_dataset, build_dataloader, extract_data
from propvg.models import build_model
from propvg.utils import load_checkpoint


def pixel_boxes(normalized_cxcywh, meta):
    """Use the evaluator's valid-image clipping BEFORE original-image rescale."""
    boxes = normalized_cxcywh.detach().float().cpu()
    xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)
    height, width = meta['img_shape'][:2]
    xyxy *= xyxy.new_tensor([width, height, width, height])
    xyxy[:, 0::2].clamp_(min=0, max=width)
    xyxy[:, 1::2].clamp_(min=0, max=height)
    valid = (xyxy[:, 2:] > xyxy[:, :2]).all(-1)
    scale = torch.as_tensor(meta['scale_factor'], dtype=xyxy.dtype).reshape(-1)
    if scale.numel() == 1:
        scale = scale.repeat(4)
    if scale.numel() != 4 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Expected positive scalar/four-element scale_factor')
    return xyxy / scale, valid


def proposal_ious(boxes, gt):
    boxes, gt = np.asarray(boxes, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    inter = np.maximum(np.minimum(boxes[..., 2:], gt[:, None, 2:]) -
                       np.maximum(boxes[..., :2], gt[:, None, :2]), 0).prod(-1)
    area = np.maximum(boxes[..., 2:] - boxes[..., :2], 0).prod(-1)
    gt_area = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(-1)
    return inter / np.maximum(area + gt_area[:, None] - inter, 1e-12)


def recall_curves(scores, ious, valid, thresholds, iou_levels=(.5, .75)):
    """Per-expression recall: ANY retained valid box must match its refer GT.

    Top-K ranks are one-based and deterministic; invalid boxes never consume a
    retained slot. Conditional retention uses only samples reachable before
    filtering; total recall always includes unreachable samples in denominator.
    """
    scores, ious, valid = np.asarray(scores), np.asarray(ious), np.asarray(valid, dtype=bool)
    if scores.ndim != 2 or ious.shape != scores.shape or valid.shape != scores.shape or not len(scores):
        raise ValueError('Expected nonempty aligned M,N scores, IoUs and validity')
    valid = valid & np.isfinite(scores) & np.isfinite(ious)
    m, n = scores.shape
    order = np.argsort(-np.where(valid, scores, -np.inf), axis=1, kind='stable')
    valid_counts = valid.sum(1)
    by_threshold = [dict(threshold=float(t), mean_kept=float((valid & (scores >= t)).sum(1).mean()),
                         empty_samples=int((~(valid & (scores >= t)).any(1)).sum())) for t in thresholds]
    by_topk = [dict(topk=k, mean_kept=float(np.minimum(valid_counts, k).mean())) for k in range(1, n + 1)]
    info, sample_fields = {}, {}
    for iou in iou_levels:
        suffix = str(iou).replace('.', '_')
        matched = valid & (ious >= iou)
        covered = matched.any(1)
        reachable = int(covered.sum())
        max_conf = np.max(np.where(matched, scores, -np.inf), axis=1)
        ranked_match = np.take_along_axis(matched, order, axis=1)
        cumulative = np.maximum.accumulate(ranked_match, axis=1)
        ranks = np.where(covered, cumulative.argmax(1) + 1, -1)
        info[str(iou)] = dict(reachable=reachable, unreachable=m - reachable,
                             oracle_recall_pct=100. * reachable / m)
        sample_fields['target_rank_iou_' + suffix] = ranks
        sample_fields['best_target_fg_iou_' + suffix] = max_conf
        for row in by_threshold:
            hits = int((covered & (max_conf >= row['threshold'])).sum())
            row.update({
                'hits_iou_' + suffix: hits,
                'recall_iou_' + suffix: 100. * hits / m,
                'retention_iou_' + suffix: 100. * hits / reachable if reachable else None,
                'lost_reachable_iou_' + suffix: reachable - hits,
            })
        for row in by_topk:
            hits = int(cumulative[:, row['topk'] - 1].sum())
            row.update({
                'hits_iou_' + suffix: hits,
                'recall_iou_' + suffix: 100. * hits / m,
                'retention_iou_' + suffix: 100. * hits / reachable if reachable else None,
                'lost_reachable_iou_' + suffix: reachable - hits,
            })
    return by_threshold, by_topk, info, sample_fields


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_curves(output, thresholds, topk, oracle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for level, style, color in [(.5, '-', '#1677ff'), (.75, '--', '#d97706')]:
        key = 'recall_iou_' + str(level).replace('.', '_')
        axes[0, 0].plot([r['threshold'] for r in thresholds], [r[key] for r in thresholds],
                        style, color=color, label='IoU >= {}'.format(level))
        axes[0, 1].plot([r['topk'] for r in topk], [r[key] for r in topk],
                        style, color=color, label='IoU >= {}'.format(level))
        for ax in axes[0]:
            ax.axhline(oracle[str(level)]['oracle_recall_pct'], color=color, linestyle=':', alpha=.5)
    axes[1, 0].plot([r['threshold'] for r in thresholds], [r['mean_kept'] for r in thresholds], color='#169447')
    axes[1, 1].plot([r['mean_kept'] for r in topk], [r['recall_iou_0_5'] for r in topk], color='#1677ff')
    axes[0, 0].set(title='Foreground confidence threshold vs refer-target recall', xlabel='Detector foreground threshold')
    axes[0, 1].set(title='Foreground-confidence Top-K vs refer-target recall', xlabel='K (log scale)', xscale='log')
    axes[1, 0].set(title='Threshold vs average retained candidate count', xlabel='Detector foreground threshold', ylabel='Mean retained candidates')
    axes[1, 1].set(title='Candidate budget vs recall at IoU >= 0.5', xlabel='Mean retained candidates (log scale)',
                   ylabel='Recall over ALL test expressions (%)', xscale='log')
    for ax in axes[0]:
        ax.set(ylabel='Recall over ALL test expressions (%)', ylim=(0, 101))
        ax.legend()
    axes[1, 1].set_ylim(0, 101)
    for ax in axes.flat:
        ax.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(output / 'foreground_recall_curves.png', dpi=170)
    plt.close(figure)


def prediction_rank_priority(prediction, valid):
    """Encode the evaluator's exact order, including legacy probability ties."""
    kept = valid.nonzero(as_tuple=True)[0]
    scores = prediction.get('ranking_logits', prediction['scores']).detach().reshape(-1).cpu()
    if scores.numel() != len(kept):
        raise RuntimeError('Post-clipping ranking scores and original query slots disagree')
    order = scores.argsort(descending=True, stable='ranking_logits' in prediction)
    priority = np.zeros(len(valid), dtype=np.float32)
    priority[kept[order].numpy()] = np.arange(len(kept), 0, -1)
    return priority


def save_report(output, meta, scores, boxes, gt, ious, valid, top1_queries, sample_metadata,
                referring_priority=None):
    thresholds = sorted(set(np.linspace(0., 1., 101).round(6).tolist() +
                            [1e-5, 1e-4, .0005, .001, .002, .005, .025, .075]))
    threshold_rows, topk_rows, oracle, fields = recall_curves(scores, ious, valid, thresholds)
    write_csv(output / 'threshold_recall.csv', threshold_rows)
    write_csv(output / 'topk_recall.csv', topk_rows)
    count = len(scores)
    top1_ious = ious[np.arange(count), top1_queries]
    top1_conf = scores[np.arange(count), top1_queries]
    per_sample = []
    for index, row in enumerate(sample_metadata):
        item = dict(row, oracle_max_iou=float(np.where(valid[index], ious[index], 0).max()),
                    top1_query=int(top1_queries[index]), top1_iou=float(top1_ious[index]),
                    top1_foreground=float(top1_conf[index]), valid_candidates=int(valid[index].sum()))
        for key, values in fields.items():
            value = values[index]
            item[key] = float(value) if np.isfinite(value) else None
            if key.startswith('target_rank'):
                item[key] = int(value) if value > 0 else None
        per_sample.append(item)
    write_csv(output / 'per_sample.csv', per_sample)
    milestones = {}
    for desired in [90., 95., 98., 99., 99.5, 99.9, 100.]:
        eligible_k = [row for row in topk_rows if row['retention_iou_0_5'] is not None and row['retention_iou_0_5'] >= desired - 1e-9]
        eligible_t = [row for row in threshold_rows if row['retention_iou_0_5'] is not None and row['retention_iou_0_5'] >= desired - 1e-9]
        milestones[str(desired)] = dict(
            meaning='Retention among samples reachable at IoU>=0.5 using all valid proposals',
            minimum_topk=eligible_k[0] if eligible_k else None,
            highest_threshold_on_tested_grid=eligible_t[-1] if eligible_t else None)
    absolute_milestones = {}
    for desired in [90., 95., 98., 99.]:
        ks = [row for row in topk_rows if row['recall_iou_0_5'] >= desired - 1e-9]
        ts = [row for row in threshold_rows if row['recall_iou_0_5'] >= desired - 1e-9]
        absolute_milestones[str(desired)] = dict(minimum_topk=ks[0] if ks else None,
                                               highest_threshold_on_tested_grid=ts[-1] if ts else None)
    summary = dict(meta, samples=count, oracle=oracle,
                   grounding_top1_accuracy=100. * float((top1_ious >= .5).mean()),
                   grounding_top1_miou=100. * float(top1_ious.mean()),
                   conditional_retention_milestones=milestones,
                   absolute_recall_milestones=absolute_milestones,
                   note='Proposal recall, NOT grounding Top-1 accuracy. No NMS, neighbor-IoU rules or confidence filters are applied inside the model. Test curves are descriptive, not a validation-selected operating point.')
    extra_arrays = {}
    referring_rows = []
    if referring_priority is not None:
        _, referring_rows, _, _ = recall_curves(referring_priority, ious, valid, [])
        write_csv(output / 'referring_topk_recall.csv', referring_rows)
        summary['grounding_top5_accuracy'] = referring_rows[min(5, len(referring_rows)) - 1]['recall_iou_0_5']
        summary['referring_topk'] = [row for row in referring_rows if row['topk'] in [1, 3, 5, 8, 16, 32, 64, 128, 256]]
        extra_arrays['referring_rank_priority'] = referring_priority
    np.savez_compressed(output / 'proposal_statistics.npz', foreground_scores=scores, boxes_xyxy=boxes,
                        gt_xyxy=gt, ious=ious, valid=valid, top1_queries=top1_queries, **extra_arrays)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_curves(output, threshold_rows, topk_rows, oracle)
    selected_t = [r for r in threshold_rows if r['threshold'] in [0, .001, .005, .01, .02, .05, .1, .2, .3, .4, .5]]
    selected_k = [r for r in topk_rows if r['topk'] in [1, 3, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]]
    lines = ['# 检测前景置信度与目标召回', '',
             '完整{}集，共{}个表达式；checkpoint：`{}`。'.format(meta['split'], count, meta['checkpoint']), '',
             '命中定义：保留候选中至少有一个与原始refer GT的IoU≥0.5；不是最终定位准确率。另列IoU≥0.75作更严格框覆盖检查。', '',
             '256框召回上限：IoU≥0.5为{:.4f}%（{}个），IoU≥0.75为{:.4f}%（{}个）。'.format(
                 oracle['0.5']['oracle_recall_pct'], oracle['0.5']['reachable'], oracle['0.75']['oracle_recall_pct'], oracle['0.75']['reachable']), '',
             '前景分数取DETR二分类softmax的第0类，不用referring score。GT在推理前移除，只用于离线统计；没有进行NMS或强制零重叠选邻。', '',
             '## 置信度阈值', '', '| p_fg阈值 | 平均保留框数 | 目标召回率@0.5 | 原可覆盖目标保留率 | 过滤损失样本数 |', '|---:|---:|---:|---:|---:|']
    for row in selected_t:
        lines.append('| {threshold:g} | {mean_kept:.2f} | {recall_iou_0_5:.4f}% | {retention_iou_0_5:.4f}% | {lost_reachable_iou_0_5} |'.format(**row))
    lines += ['', '## 按前景置信度保留Top-K', '', '| K | 目标召回率@0.5 | 原可覆盖目标保留率 | 目标召回率@0.75 | 过滤损失样本数 |', '|---:|---:|---:|---:|---:|']
    for row in selected_k:
        lines.append('| {topk} | {recall_iou_0_5:.4f}% | {retention_iou_0_5:.4f}% | {recall_iou_0_75:.4f}% | {lost_reachable_iou_0_5} |'.format(**row))
    if referring_rows:
        lines += ['', '## 按文本定位分数排序的Top-K（与前景筛选不是同一指标）', '',
                  '使用当前模型正式评估的排序规则，包括原CRS概率并列时的处理，不用检测前景分数。', '',
                  '| K | 定位Top-K命中率@0.5 | 命中样本数 |', '|---:|---:|---:|']
        for row in summary['referring_topk']:
            lines.append('| {topk} | {recall_iou_0_5:.4f}% | {hits_iou_0_5} |'.format(**row))
    lines += ['', '## 保留率说明与使用限制', '',
              '总召回率的分母是全部样本；条件保留率的分母是原256框已经覆盖目标的样本。原候选没有覆盖的目标，不可能靠降低置信度或增大K召回。', '',
              '这是指定test集的诊断结果，不应直接据此反复调参并宣称无偏test提升。正式前景阈值或Top-K应在val集确定后，再在test上验证。', '',
              '目标保留率不能代表参考物召回率，也不能证明留下的框是完整物体；参考物标注不完备，不能从这条统计推导六个邻居均可靠。', '',
              '完整阈值网格和K=1…256见CSV；`proposal_statistics.npz`保存原始预测统计，可不重复GPU推理重新分析。', '',
              '[曲线图](foreground_recall_curves.png) · [完整阈值CSV](threshold_recall.csv) · [完整Top-K CSV](topk_recall.csv) · [逐样本统计](per_sample.csv)']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('ORACLE:', json.dumps(oracle, ensure_ascii=False))
    print('THRESHOLDS:', json.dumps(selected_t, ensure_ascii=False))
    print('TOPK:', json.dumps(selected_k, ensure_ascii=False))
    return summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite {}'.format(output))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device access is required')
    torch.cuda.set_device(torch.device(args.device)); torch.set_num_threads(4)
    torch.manual_seed(6666); torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.config)
    if (cfg.model.head.get('enable_grab', False) or cfg.model.head.get('enable_crv', False)
            or cfg.model.head.get('num_queries') != 256):
        raise ValueError('This analysis expects all 256 original proposals with Grab and CRV off')
    if cfg.model.post_params.get('with_nms', False):
        raise ValueError('Proposal recall analysis requires the original non-NMS evaluation')
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    cfg.data.samples_per_gpu, cfg.data.workers_per_gpu = args.batch_size, args.workers
    cfg.model.vis_enc.pretrain = None
    dataset = build_dataset(getattr(cfg.data, args.split))
    loader = build_dataloader(cfg, dataset)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token).to(args.device)
    _, _, _, strict = load_checkpoint(model, load_from=args.checkpoint, dataset=cfg.dataset)
    if not strict:
        raise RuntimeError('Checkpoint must load strictly')
    model.eval()
    captured = {}
    original_forward = model.head.forward_test
    def capture(*inputs, **kwargs):
        outputs = original_forward(*inputs, **kwargs)
        raw, _ = outputs
        captured.clear()
        captured.update(boxes=raw['pred_bbox_allobj'].detach().float().cpu().clone(),
                        foreground=raw['pred_class_allobj'].detach().float().softmax(-1)[..., 0].cpu())
        return outputs
    # Observe the actual CRS head result. No
    # parameters/features/scores are changed, and the method is restored below.
    # A plain wrapper is important here: Mock/patch side_effect records each
    # call's GPU feature tensors and can exhaust VRAM over a full split.
    model.head.forward_test = capture
    output.mkdir(parents=True, exist_ok=True)
    all_boxes, all_gt, all_scores, all_valid, top1_queries, sample_metadata = [], [], [], [], [], []
    all_referring_priority = []
    try:
        for batch in tqdm(loader, desc='Foreground threshold/Top-K target recall', dynamic_ncols=True):
            metas = batch['img_metas'].data[0]
            for key in ['gt_bbox', 'gt_mask_rle', 'is_crowd']:
                batch.pop(key, None)
            captured.clear()
            predictions = model(**extract_data(batch), return_loss=False, with_bbox=True, with_mask=False, rescale=True)
            for bi, meta in enumerate(metas):
                index = len(sample_metadata)
                sample = dataset.samples[index]
                if Path(sample['filename']).name != Path(meta['filename']).name:
                    raise RuntimeError('Sequential test-loader and annotation indexing disagree')
                boxes, valid = pixel_boxes(captured['boxes'][bi], meta)
                if len(boxes) != 256:
                    raise RuntimeError('Expected 256 original proposal slots')
                torch.testing.assert_close(boxes[valid], predictions['pred_bboxes'][bi]['boxes'].detach().float().cpu(), rtol=1e-5, atol=2e-3)
                priority = prediction_rank_priority(predictions['pred_bboxes'][bi], valid)
                if not valid.any():
                    raise RuntimeError('No valid proposals in this expression')
                query = int(priority.argmax())
                gt_index = int(np.asarray(sample['refer_target_index']).reshape(-1)[0])
                all_boxes.append(boxes.numpy()); all_valid.append(valid.numpy())
                all_scores.append(captured['foreground'][bi].numpy())
                all_referring_priority.append(priority)
                all_gt.append(sample['boxes'][gt_index]); top1_queries.append(query)
                sample_metadata.append(dict(dataset_index=index, filename=sample['filename'], expression=sample['expression']))
    finally:
        model.head.forward_test = original_forward
    scores, boxes = np.asarray(all_scores), np.asarray(all_boxes)
    gt, valid, queries = np.asarray(all_gt), np.asarray(all_valid), np.asarray(top1_queries)
    ious = proposal_ious(boxes, gt)
    summary = save_report(output, dict(split=args.split, checkpoint=str(Path(args.checkpoint).resolve()),
                                      config=str(Path(args.config).resolve()), batch_size=args.batch_size,
                                      scoring_head='CRS'),
                          scores, boxes, gt, ious, valid, queries, sample_metadata,
                          referring_priority=np.asarray(all_referring_priority))
    print('Grounding checks:', summary['grounding_top1_accuracy'], summary['grounding_top5_accuracy'], summary['grounding_top1_miou'])
    print('Report:', output / 'report.md')


if __name__ == '__main__':
    main()
