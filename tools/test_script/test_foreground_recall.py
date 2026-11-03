"""Toy-data checks for foreground proposal recall (not grounding accuracy)."""
if __package__:
    from .. import _bootstrap  # noqa: F401
else:
    import tools._bootstrap as _bootstrap  # noqa: F401

import unittest

import numpy as np
import torch

from analyze_foreground_recall import proposal_ious, recall_curves, prediction_rank_priority


class ForegroundRecallTests(unittest.TestCase):
    def test_grounding_ranking_matches_legacy_and_logit_evaluator_with_filtered_slots(self):
        valid = torch.ones(40, dtype=torch.bool)
        valid[[3, 15]] = False
        scores = torch.ones(int(valid.sum()))
        for use_logits in [False, True]:
            prediction = dict(scores=scores.clone())
            if use_logits:
                prediction['ranking_logits'] = scores.clone()
            priority = prediction_rank_priority(prediction, valid)
            expected = valid.nonzero(as_tuple=True)[0][scores.argsort(descending=True, stable=use_logits)]
            actual = np.argsort(-priority, kind='stable')[:int(valid.sum())]
            np.testing.assert_array_equal(actual, expected.numpy())
            self.assertTrue((priority[~valid.numpy()] == 0).all())

    def test_threshold_topk_oracle_and_unreachable_denominators(self):
        scores = np.asarray([[.9, .8, .2, .1], [.9, .8, .7, .01], [.99, .9, .8, .7]])
        ious = np.asarray([[.1, .8, .7, .1], [.1, .1, .6, .3], [0, 0, 0, 0]])
        thresholds, topk, oracle, fields = recall_curves(scores, ious, np.ones_like(scores, dtype=bool), [0, .7, .8, .9])
        self.assertEqual(oracle['0.5']['reachable'], 2)
        self.assertEqual(oracle['0.5']['unreachable'], 1)
        self.assertAlmostEqual(oracle['0.5']['oracle_recall_pct'], 200 / 3)
        self.assertEqual([r['hits_iou_0_5'] for r in thresholds], [2, 2, 1, 0])
        self.assertEqual([r['hits_iou_0_5'] for r in topk], [0, 1, 2, 2])
        self.assertEqual(thresholds[2]['retention_iou_0_5'], 50.)
        self.assertEqual(thresholds[2]['lost_reachable_iou_0_5'], 1)
        self.assertEqual(fields['target_rank_iou_0_5'].tolist(), [2, 3, -1])
        np.testing.assert_allclose(fields['best_target_fg_iou_0_5'][:2], [.8, .7])
        self.assertEqual(oracle['0.75']['reachable'], 1)
        self.assertEqual(topk[-1]['hits_iou_0_5'], thresholds[0]['hits_iou_0_5'])

    def test_invalid_box_does_not_count_as_target_or_consume_topk(self):
        scores = np.asarray([[.999, .9, .1]])
        ious = np.asarray([[1., 0., .8]])
        valid = np.asarray([[False, True, True]])
        threshold, topk, _, fields = recall_curves(scores, ious, valid, [0, .5])
        self.assertEqual(topk[0]['hits_iou_0_5'], 0)
        self.assertEqual(topk[1]['hits_iou_0_5'], 1)
        self.assertEqual(topk[-1]['mean_kept'], 2.)
        self.assertEqual(threshold[0]['mean_kept'], 2.)
        self.assertEqual(threshold[1]['hits_iou_0_5'], 0)
        self.assertEqual(fields['target_rank_iou_0_5'].tolist(), [2])

    def test_confidence_ties_keep_original_query_order(self):
        _, topk, _, fields = recall_curves([[.9, .9]], [[.1, .8]], [[True, True]], [.9])
        self.assertEqual(topk[0]['hits_iou_0_5'], 0)
        self.assertEqual(topk[1]['hits_iou_0_5'], 1)
        self.assertEqual(fields['target_rank_iou_0_5'].tolist(), [2])

    def test_zero_coverage_is_explicit_not_division_by_zero(self):
        thresholds, topk, oracle, fields = recall_curves([[.8, .2]], [[.1, .2]], [[True, True]], [0, .9])
        self.assertEqual(oracle['0.5']['reachable'], 0)
        self.assertIsNone(thresholds[0]['retention_iou_0_5'])
        self.assertEqual(thresholds[1]['empty_samples'], 1)
        self.assertIsNone(topk[0]['retention_iou_0_5'])
        self.assertEqual(fields['target_rank_iou_0_5'].tolist(), [-1])

    def test_iou_uses_one_original_gt_per_expression(self):
        boxes = np.asarray([[[0, 0, 10, 10], [0, 0, 5, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]]])
        gt = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]])
        np.testing.assert_allclose(proposal_ious(boxes, gt), [[1, .5], [0, 1]])


if __name__ == '__main__':
    unittest.main()
