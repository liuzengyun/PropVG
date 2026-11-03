"""Regression tests for foreground Top-K, strict KNN, aligned labels and DDP loss."""
if __package__:
    from .. import _bootstrap  # noqa: F401
else:
    import tools._bootstrap as _bootstrap  # noqa: F401

import copy
import unittest
from unittest.mock import patch

import torch
from mmcv import Config

from propvg.models import build_head
from propvg.models.heads.candidate_relation_scoring import CandidateRelationScorer
from propvg.apis.train import train_model
import test_relation_scoring as relation_fixtures
from visualize_disjoint_neighbors import select_disjoint_neighbors
from visualize_relation_scoring import pixel_boxes


class RelationForegroundPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, **changes):
        options = relation_fixtures.RelationScoringTests().head_options()
        options['relation_scoring'].update(foreground_topk=3, strict_disjoint_neighbors=True)
        options.update(changes)
        return options

    def inputs(self):
        return relation_fixtures.RelationScoringTests().head_inputs()

    def test_foreground_uses_detector_not_refer_scores_and_keeps_indices_stable(self):
        scorer = CandidateRelationScorer(foreground_topk=3)
        boxes = torch.tensor([[[1.2, .5, .1, .1], [.1, .1, .1, .1], [.2, .2, .1, .1], [.3, .3, .1, .1]]])
        classes = torch.tensor([[[100., 0.], [5., 0.], [2., 0.], [3., 0.]]], requires_grad=True)
        indices = scorer.select_foreground_indices(boxes, classes)
        self.assertEqual(indices.tolist(), [[1, 3, 2]])
        self.assertFalse(indices.requires_grad)
        classes = torch.zeros_like(classes)
        self.assertEqual(scorer.select_foreground_indices(boxes, classes).tolist(), [[1, 2, 3]])
        scorer.foreground_topk = 64
        self.assertEqual(scorer.select_foreground_indices(boxes, classes).shape, (1, 4))
        scorer.foreground_topk = None
        self.assertIsNone(scorer.select_foreground_indices(boxes, classes))

    def test_strict_neighbors_match_visualization_policy_for_every_candidate(self):
        scorer = CandidateRelationScorer(strict_disjoint_neighbors=True)
        torch.manual_seed(122)
        boxes, classes = torch.rand(2, 32, 4), torch.randn(2, 32, 2)
        metas = [dict(img_shape=(80, 128), pad_shape=(128, 128), scale_factor=[.5] * 4),
                 dict(img_shape=(96, 128), pad_shape=(128, 128), scale_factor=[.4] * 4)]
        result = scorer.select_neighbors(boxes, classes, metas)
        for bi in range(2):
            pixel, valid = pixel_boxes(boxes[bi], metas[bi])
            for target in range(32):
                expected, _, _ = select_disjoint_neighbors(pixel.numpy(), valid.numpy(), target,
                                                           metric_scale=metas[bi]['scale_factor'])
                got = result['neighbor_indices'][bi, target][result['neighbor_valid'][bi, target]].tolist()
                self.assertEqual(got, expected)

    def test_strict_rules_reject_tiny_overlap_and_neighbor_neighbor_overlap(self):
        pixels = torch.tensor([[40., 40., 60., 60.], [59.999, 40., 70., 60.], [61., 42., 69., 58.],
                               [63., 42., 71., 58.], [40., 61., 60., 69.], [10., 40., 20., 60.],
                               [80., 40., 90., 60.], [40., 10., 60., 20.], [40., 80., 60., 90.]])
        boxes = torch.cat(((pixels[:, :2] + pixels[:, 2:]) / 2, pixels[:, 2:] - pixels[:, :2]), -1)[None] / 100
        scorer = CandidateRelationScorer(strict_disjoint_neighbors=True)
        result = scorer.select_neighbors(boxes, torch.zeros(1, 9, 2), [dict(img_shape=(100, 100))])
        # Nominally tied distances can differ by FP32 box reconstruction.
        self.assertCountEqual(result['neighbor_indices'][0, 0].tolist(), [2, 4, 5, 6, 7, 8])
        # Fully overlapping proposals produce an explicit no-neighbor fallback.
        boxes[:] = boxes[:, :1]
        inputs = [torch.randn(1, 9, 256), boxes, torch.randn(1, 4, 256), torch.zeros(1, 4, dtype=torch.bool),
                  torch.zeros(1, 9, 2), [dict(img_shape=(100, 100))]]
        _, logits, _, details = scorer(*inputs)
        self.assertTrue(details['relation_fallback'].all())
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(details['relation_neighbor_valid'].any())

    def test_gather_keeps_feature_gradients_and_composes_original_detector_indices(self):
        head = build_head(self.options())
        queries = torch.randn(1, 4, 256, requires_grad=True)
        boxes = torch.tensor([[[.1, .1, .1, .1], [.3, .3, .1, .1], [.5, .5, .1, .1], [.8, .8, .1, .1]]])
        classes = torch.tensor([[[1., 0.], [4., 0.], [2., 0.], [3., 0.]]])
        details = {}
        out, selected_boxes, selected_classes, indices = head._select_relation_candidates(
            queries, boxes, classes, torch.tensor([[7, 2, 6, 5]]), details)
        self.assertEqual(indices.tolist(), [[1, 3, 2]])
        self.assertEqual(details['relation_detector_indices'].tolist(), [[2, 5, 6]])
        torch.testing.assert_close(selected_boxes, boxes[:, [1, 3, 2]])
        torch.testing.assert_close(selected_classes, classes[:, [1, 3, 2]])
        out.sum().backward()
        self.assertEqual(queries.grad[0, 0].count_nonzero(), 0)
        self.assertTrue((queries.grad[0, [1, 3, 2]] == 1).all())

    def test_filtered_referent_skips_bce_but_true_empty_expression_stays_negative(self):
        head = build_head(self.options())
        logits = torch.tensor([[[.4], [-.2]], [[1.], [.5]]], requires_grad=True)
        labels = torch.tensor([[1., 0.], [0., 0.]])
        original = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
        loss, metrics = head._pruned_relation_loss(logits, labels, original)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :, 0], labels[0]) * head.loss_weight['refer']
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics['metric_relation_refer_valid_samples'].item(), 1)
        self.assertEqual(metrics['metric_relation_refer_skipped'].item(), .5)
        loss.backward()
        self.assertEqual(logits.grad[1].count_nonzero(), 0)
        original[1] = 0
        loss, metrics = head._pruned_relation_loss(logits, labels, original)
        self.assertEqual(metrics['metric_relation_refer_valid_samples'].item(), 2)
        torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels) * head.loss_weight['refer'])

    def test_all_filtered_or_invalid_samples_produce_finite_connected_zero(self):
        head = build_head(self.options())
        logits = torch.randn(2, 3, 1, requires_grad=True)
        loss, metrics = head._pruned_relation_loss(logits, torch.zeros(2, 3), torch.ones(2, 8))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.count_nonzero(), 0)
        _, metrics = head._pruned_relation_loss(logits, torch.ones(2, 3), torch.ones(2, 8), torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(metrics['metric_relation_refer_valid_samples'].item(), 0)

    def test_distributed_loss_normalizes_by_global_valid_samples(self):
        head = build_head(self.options())
        logits = torch.randn(2, 3, 1, requires_grad=True)
        labels = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
        original = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]])
        with patch('torch.distributed.is_available', return_value=True), \
                patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=2), \
                patch('torch.distributed.all_reduce', side_effect=lambda x: x.fill_(3)):
            loss, _ = head._pruned_relation_loss(logits, labels, original)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :, 0], labels[0]) * 2 / 3
        torch.testing.assert_close(loss, expected * head.loss_weight['refer'])

    def test_full_head_keeps_raw_detection_supervision_and_gathers_refer_labels(self):
        options = self.options()
        torch.manual_seed(81); head = build_head(options)
        old_options = copy.deepcopy(options); old_options['relation_scoring']['foreground_topk'] = None
        torch.manual_seed(81); unpruned = build_head(old_options)
        x, targets, args = self.inputs()
        chosen = torch.tensor([[7, 2, 5]])
        full_labels = torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0.]])
        with patch.object(head.relation_scorer, 'select_foreground_indices', return_value=chosen), \
                patch.object(head, 'prepare_referring_targets', return_value=full_labels), \
                patch.object(head.detrloss, 'forward', wraps=head.detrloss.forward) as detector:
            torch.manual_seed(82); losses, pred, details = head.forward_train(x, targets, **args)
        torch.manual_seed(82); old_loss, old_pred, _ = unpruned.forward_train(x, targets, **args)
        self.assertEqual(detector.call_args.args[0].shape, (3, 1, 8, 2))
        torch.testing.assert_close(losses['loss_det'], old_loss['loss_det'], rtol=0, atol=0)
        torch.testing.assert_close(pred['pred_bbox_allobj'], old_pred['pred_bbox_allobj'][:, [7, 2, 5]], rtol=0, atol=0)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(pred['ranking_priority'].squeeze(-1), full_labels[:, [7, 2, 5]])
        torch.testing.assert_close(losses['loss_refer'], expected * head.loss_weight['refer'])
        self.assertEqual(details['relation_detector_indices'].tolist(), [[7, 2, 5]])
        sum(value for key, value in losses.items() if key.startswith('loss_')).backward()
        self.assertGreater(head.refer_adapter.weight.grad.abs().sum().item(), 0)
        head.relation_scorer.mark_optimizer_step(); head.eval()
        with torch.no_grad(), patch.object(head, 'prepare_referring_targets', side_effect=AssertionError('No GT labels in inference')):
            prediction, _ = head.forward_test(x, targets=dict(img_metas=targets['img_metas']), with_mask=False, **args)
        self.assertEqual(prediction['pred_bbox_allobj'].shape, (1, 3, 4))

    def test_real_256_to_64_shape_and_joint_grab_bcem_correctness(self):
        for grab in (False, True):
            options = self.options(num_queries=256 if not grab else 8,
                                   enable_grab=grab, enable_bcem=grab, enable_correctness=grab)
            options['relation_scoring']['foreground_topk'] = 64 if not grab else 3
            if grab:
                options['bcem'] = dict(context_dim=16, num_heads=4, sampling_mode='ray', num_rays=4,
                                       radial_samples=4, min_valid_points=2, chunk_size=2, checkpoint_chunks=False)
            head = build_head(options); x, targets, args = self.inputs()
            losses, pred, details = head.forward_train(x, targets, **args)
            count = 64 if not grab else 3
            self.assertEqual(pred['pred_bbox_allobj'].shape, (1, count, 4))
            self.assertEqual(details['relation_neighbor_indices'].shape, (1, count, 6))
            self.assertEqual(details['relation_detector_indices'].shape, (1, count))
            self.assertEqual(losses['metric_relation_candidates'].item(), count)
            if grab:
                self.assertIn('loss_correctness', losses)
                self.assertIn('bcem_direction_weights', details)

    def test_configuration_and_legacy_constructor_defaults(self):
        legacy = CandidateRelationScorer()
        self.assertIsNone(legacy.foreground_topk)
        self.assertFalse(legacy.strict_disjoint_neighbors)
        for filename in ['configs/aerialvg/PropVG-aerialvg.py', 'configs/aerialsense/PropVG-aerialsense.py']:
            config = Config.fromfile(filename)
            self.assertEqual(config.model.head.relation_scoring.foreground_topk, 64)
            self.assertTrue(config.model.head.relation_scoring.strict_disjoint_neighbors)
        for bad in [dict(foreground_topk=0), dict(foreground_topk=True), dict(strict_disjoint_neighbors=1)]:
            with self.assertRaises(ValueError):
                CandidateRelationScorer(**bad)

    def test_all_missing_training_batch_does_not_mark_scorer_as_trained(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Module()
                self.head.relation_scorer = CandidateRelationScorer(hidden_channels=16, model_dim=16,
                                                                    num_heads=4, geometry_dim=8)
            def forward(self, **unused):
                weight = self.head.relation_scorer.score[0].weight
                return {'loss_refer': weight.square().sum() * 0,
                        'metric_relation_refer_valid_samples': weight.new_tensor(0.),
                        'metric_relation_refer_skipped': weight.new_tensor(1.)}, {}
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        cfg = Config(dict(distributed=False, use_fp16=False, grad_norm_clip=.15, log_interval=1,
                          world_size=1, data=dict(samples_per_gpu=1)))
        with patch('propvg.apis.train.extract_data', side_effect=lambda value: value), \
                patch('propvg.apis.train.is_main', return_value=False):
            train_model(0, cfg, model, optimizer, [{}])
        self.assertEqual(model.head.relation_scorer.trained_steps.item(), 0)


if __name__ == '__main__':
    unittest.main()
