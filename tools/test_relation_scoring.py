"""CPU regressions: pair scoring, geometry, gradients and legacy isolation."""
import copy
import math
import os
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from mmcv import Config

from propvg.models import build_head
from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
from propvg.models.heads.candidate_relation_scoring import CandidateRelationScorer, PairTextLayer
from propvg.utils.checkpoint import _validate_relation_checkpoint
from propvg.apis.train import train_model


def _copy_based_attention_reference(layer, pair, memory, text_valid):
    """Pre-optimization math, intentionally copying text for every pair.

    Adapt B,P,D inputs to the original flattened implementation for numerical
    and gradient regressions; this slow reference is never used in training.
    """
    b, count, dim = pair.shape
    owners = torch.arange(b, device=pair.device).repeat_interleave(count)
    flat = pair.reshape(-1, dim)
    q = layer.query(flat).reshape(-1, layer.heads, layer.head_dim)
    key, value = (item[owners] for item in memory)
    valid = text_valid[owners]
    logits = torch.einsum('shd,shld->shl', q, key) / math.sqrt(layer.head_dim)
    logits = logits.float().masked_fill(~valid[:, None], -torch.inf)
    logits = torch.where(valid.any(1)[:, None, None], logits, torch.zeros_like(logits))
    weights = logits.softmax(-1).to(pair.dtype) * valid[:, None].to(pair.dtype)
    context = torch.einsum('shl,shld->shd', layer.attention_dropout(weights), value).flatten(1)
    flat = layer.norm1(flat + layer.output_dropout(layer.output(context)))
    return layer.norm2(flat + layer.ffn(flat)).reshape_as(pair), weights.mean(1).reshape(b, count, -1)


class RelationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self, **options):
        defaults = dict(hidden_channels=16, model_dim=16, geometry_dim=8, num_heads=4,
                        num_layers=2, ffn_dim=32, dropout=0, chunk_size=11)
        return CandidateRelationScorer(**dict(defaults, **options))

    def inputs(self, n=8, batch=2):
        torch.manual_seed(131)
        query = torch.randn(batch, n, 16, requires_grad=True)
        boxes = torch.rand(batch, n, 4, requires_grad=True)
        text = torch.randn(batch, 7, 16, requires_grad=True)
        mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1]] * batch)
        classes = torch.randn(batch, n, 2, requires_grad=True)
        metas = [dict(img_shape=(80, 100), pad_shape=(100, 100)) for _ in range(batch)]
        return [query, boxes, text, mask, classes, metas]

    def test_selection_deduplicates_context_not_target_hypotheses(self):
        model = self.module()
        boxes = torch.tensor([[[.5, .5, .1, .1], [.55, .5, .1, .1], [.55, .5, .1, .1],
                               [.4, .5, .1, .1], [.5, .7, .1, .1], [1.2, .5, .1, .1],
                               [.1, .1, .1, .1], [.5, .5, .1, .1]]])
        classes = torch.zeros(1, 8, 2)
        classes[0, :, 0] = torch.tensor([1., 4., 2., 3., 3., 9., 3., 5.])
        result = model.select_neighbors(boxes, classes, [dict(img_shape=(100, 100))])
        self.assertEqual(result['neighbor_indices'][0, 0].tolist(), [1, 3, 4, 6, -1, -1])
        self.assertFalse(result['context_kept'][0, 0])  # Higher-confidence duplicate represents it.
        self.assertTrue(result['candidate_valid'][0, 0])  # Still a scoring target.
        self.assertFalse(result['context_kept'][0, 2])
        self.assertFalse(result['candidate_valid'][0, 5])
        self.assertEqual(result['neighbor_indices'].shape, (1, 8, 6))

    def test_geometry_preserves_direction_aspect_ratio_and_size_ratio(self):
        model = self.module(num_neighbors=1)
        boxes = torch.tensor([[[.25, .5, .1, .2], [.5, .5, .2, .1]]])
        classes = torch.zeros(1, 2, 2)
        result = model.select_neighbors(boxes, classes, [dict(img_shape=(100, 200))])
        diagonal = (100 ** 2 + 200 ** 2) ** .5
        eps = model.geometry_eps
        expected = torch.tensor([50 / diagonal, 0., 50 / diagonal,
                                 torch.log(torch.tensor((40 / diagonal + eps) / (20 / diagonal + eps))),
                                 torch.log(torch.tensor((10 / diagonal + eps) / (20 / diagonal + eps)))])
        torch.testing.assert_close(result['geometry'][0, 0, 0], expected)
        inverse = result['geometry'][0, 1, 0]
        torch.testing.assert_close(inverse[[0, 1, 3, 4]], -expected[[0, 1, 3, 4]])
        torch.testing.assert_close(inverse[2], expected[2])

    def test_geometry_ignores_padding_and_is_scale_invariant(self):
        model = self.module()
        args = self.inputs()
        a = model.select_neighbors(args[1], args[4], args[5])
        b = model.select_neighbors(args[1], args[4], [dict(img_shape=(160, 200), pad_shape=(999, 999))] * 2)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_six_scores_are_adaptively_aggregated_logits(self):
        model = self.module()
        probability, logits, attention, details = model(*self.inputs())
        self.assertEqual(logits.shape, (2, 8, 1))
        self.assertEqual(details['relation_pair_logits'].shape, (2, 8, 6))
        expected = (details['relation_weights'] * details['relation_pair_logits']).sum(-1, keepdim=True)
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(probability, logits.sigmoid())
        torch.testing.assert_close(details['relation_weights'].sum(-1), torch.ones(2, 8))
        effective = details['relation_effective_mask']
        torch.testing.assert_close(details['relation_weights'], effective / effective.sum(-1, keepdim=True))
        self.assertEqual(attention.shape, (2, 8, 7))
        self.assertEqual(attention[..., 4:].count_nonzero(), 0)

    def test_single_candidate_has_one_finite_zero_context_fallback(self):
        model = self.module(num_neighbors=6)
        probability, logits, _, details = model(*self.inputs(n=1))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(details['relation_fallback'].all())
        self.assertFalse(details['relation_neighbor_valid'].any())
        self.assertEqual(details['relation_neighbor_indices'].unique().tolist(), [-1])
        torch.testing.assert_close(details['relation_weights'][..., 0], torch.ones(2, 1))
        self.assertEqual(details['relation_weights'][..., 1:].count_nonzero(), 0)
        probability.sum().backward()

    def test_empty_candidate_set(self):
        result = self.module()(*self.inputs(n=0))
        self.assertEqual(result[0].shape, (2, 0, 1))
        self.assertEqual(result[2].shape, (2, 0, 7))

    def test_empty_text_and_integer_masks_remain_finite(self):
        model = self.module().eval()
        args = self.inputs()
        args[3] = torch.ones_like(args[3])
        result = model(*args)
        self.assertTrue(torch.isfinite(result[1]).all())
        self.assertEqual(result[2].count_nonzero(), 0)
        args[2], args[3] = args[2][:, :0], args[3][:, :0]
        self.assertTrue(torch.isfinite(model(*args)[1]).all())

    def test_padding_values_and_mask_dtype_cannot_change_predictions(self):
        model = self.module().eval()
        args = self.inputs()
        first = model(*args)
        args[2] = args[2].detach().clone()
        args[2][args[3].bool()] = 1e6
        args[3] = args[3].bool()
        second = model(*args)
        for a, b in zip(first[:3], second[:3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_gradients_reach_features_text_geometry_and_both_heads_not_boxes(self):
        model = self.module()
        args = self.inputs()
        original = [x.detach().clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in args]
        logits = model(*args)[1]
        labels = torch.zeros_like(logits); labels[:, 0] = 1
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
        for parameter in (args[0], args[2], model.geometry[0].weight, model.score[-1].weight, model.weight[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(args[1].grad)
        self.assertIsNone(args[4].grad)
        for first, second in zip(args, original):
            if torch.is_tensor(first):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_chunking_preserves_outputs_and_gradients(self):
        left = self.module(chunk_size=1)
        right = copy.deepcopy(left); right.chunk_size = 1000
        args = self.inputs()
        a, b = left(*args), right(*args)
        for x, y in zip(a[:3], b[:3]):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
        a[1].square().mean().backward(); b[1].square().mean().backward()
        for (name, x), (_, y) in zip(left.named_parameters(), right.named_parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_shared_kv_matches_copy_based_scores_and_all_gradients(self):
        optimized = self.module(chunk_size=11)
        # Non-uniform learned adoption weights must also preserve equivalence.
        torch.nn.init.normal_(optimized.weight[-1].weight, std=.1)
        reference = copy.deepcopy(optimized)
        for layer in reference.layers:
            layer.forward = MethodType(_copy_based_attention_reference, layer)
        first = self.inputs(n=9, batch=3)
        first[3][1] = 1  # Mix normal text with one all-padding image.
        second = [x.detach().clone().requires_grad_(x.requires_grad) if torch.is_tensor(x)
                  else copy.deepcopy(x) for x in first]
        a, b = optimized(*first), reference(*second)
        for x, y in zip(a[:3], b[:3]):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
        for key in a[3]:
            torch.testing.assert_close(a[3][key], b[3][key], rtol=2e-5, atol=2e-6, msg=key)
        labels = torch.zeros_like(a[1]); labels[:, 0] = 1
        torch.nn.functional.binary_cross_entropy_with_logits(a[1], labels).backward()
        torch.nn.functional.binary_cross_entropy_with_logits(b[1], labels).backward()
        for (name, x), (_, y) in zip(optimized.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=3e-4, atol=3e-6, msg=name)
        for index in (0, 2):
            torch.testing.assert_close(first[index].grad, second[index].grad, rtol=3e-4, atol=3e-6)

    def test_batched_attention_never_mixes_pairs_or_images(self):
        torch.manual_seed(812)
        layer = PairTextLayer(16, 4, 32, 0).eval()
        pair, text = torch.randn(2, 9, 16), torch.randn(2, 7, 16)
        valid, position = torch.ones(2, 7, dtype=torch.bool), torch.zeros(1, 7, 16)
        original = layer(pair, layer.prepare_memory(text, position), valid)[0]
        modified = pair.clone(); modified[0, 3, 0] += 10
        result = layer(modified, layer.prepare_memory(text, position), valid)[0]
        keep = torch.ones(2, 9, dtype=torch.bool); keep[0, 3] = False
        torch.testing.assert_close(result[keep], original[keep], rtol=0, atol=0)
        modified_text = text.clone(); modified_text[0] += torch.randn_like(text[0])
        result = layer(pair, layer.prepare_memory(modified_text, position), valid)[0]
        torch.testing.assert_close(result[1], original[1], rtol=0, atol=0)

    def test_chunks_keep_batch_axis_and_share_same_unexpanded_memory(self):
        model = self.module(chunk_size=11)
        args = self.inputs(batch=3)
        calls, hooks = [], []
        for index, layer in enumerate(model.layers):
            def record(_, inputs, index=index):
                pair, (key, value), valid = inputs
                calls.append((index, pair.shape, key.shape, value.shape,
                              key.data_ptr(), value.data_ptr(), valid.shape))
            hooks.append(layer.register_forward_pre_hook(record))
        model(*args)
        for hook in hooks: hook.remove()
        chunks = math.ceil(8 * 6 / 11)
        self.assertEqual(len(calls), chunks * 2)  # Not ceil(B*N*K/chunk_size)*2.
        for index in range(2):
            layer_calls = [row for row in calls if row[0] == index]
            self.assertEqual([row[1] for row in layer_calls],
                             [torch.Size([3, 11, 16])] * 4 + [torch.Size([3, 4, 16])])
            self.assertEqual(len({row[4:6] for row in layer_calls}), 1)
            for row in layer_calls:
                self.assertEqual(row[2], torch.Size([3, 4, 7, 4]))
                self.assertEqual(row[3], row[2])
                self.assertEqual(row[6], torch.Size([3, 7]))

    def test_backward_does_not_save_pair_expanded_text_key_value(self):
        layer = PairTextLayer(16, 4, 32, 0)
        pair = torch.randn(2, 31, 16, requires_grad=True)
        text = torch.randn(2, 7, 16, requires_grad=True)
        saved_shapes = []
        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out, _ = layer(pair, layer.prepare_memory(text, torch.zeros(1, 7, 16)),
                           torch.ones(2, 7, dtype=torch.bool))
            out.square().mean().backward()
        self.assertNotIn((2 * 31 * 4, 4, 7), saved_shapes)
        self.assertNotIn((2 * 31 * 4, 7, 4), saved_shapes)
        self.assertIn((2 * 4, 4, 7), saved_shapes)
        self.assertIn((2 * 4, 7, 4), saved_shapes)

    def test_neighbour_permutation_keeps_aggregate_unchanged(self):
        model = self.module().eval()
        args = self.inputs(); first = model(*args)
        selection = model.select_neighbors(args[1], args[4], args[5])
        order = torch.tensor([5, 3, 1, 4, 2, 0])
        for key in ('neighbor_indices', 'neighbor_valid', 'geometry'):
            selection[key] = selection[key][:, :, order]
        with patch.object(model, 'select_neighbors', return_value=selection):
            second = model(*args)
        torch.testing.assert_close(first[1], second[1], rtol=1e-5, atol=1e-6)

    def test_candidate_permutation_is_equivariant(self):
        model = self.module().eval(); args = self.inputs()
        first = model(*args)
        order = torch.tensor([5, 3, 1, 4, 2, 0, 7, 6])
        for index in (0, 1, 4):
            args[index] = args[index][:, order]
        second = model(*args)
        torch.testing.assert_close(first[1][:, order], second[1], rtol=1e-5, atol=1e-6)

    def test_real_dimensions_and_text_projections_are_not_repeated_per_pair(self):
        model = CandidateRelationScorer(dropout=0, chunk_size=256).eval()
        args = self.inputs(n=256, batch=1)
        args[0] = torch.randn(1, 256, 256); args[2] = torch.randn(1, 64, 256)
        args[3] = torch.zeros(1, 64, dtype=torch.bool)
        hooks, counts = [], []
        for layer in model.layers:
            hooks.append(layer.key.register_forward_pre_hook(lambda _, inputs: counts.append(inputs[0].shape)))
        with torch.no_grad():
            result = model(*args)
        for hook in hooks: hook.remove()
        self.assertEqual(result[0].shape, (1, 256, 1))
        self.assertEqual(result[3]['relation_pair_logits'].shape, (1, 256, 6))
        self.assertEqual(counts, [torch.Size([1, 64, 256])] * 2)

    def head_options(self):
        options = copy.deepcopy(Config.fromfile('configs/aerialvg/PropVG-aerialvg.py').model.head)
        options.update(num_queries=8, enable_grab=False, enable_bcem=False, enable_crv=False,
                       enable_correctness=False, enable_relation_scoring=True)
        options['relation_scoring'] = dict(model_dim=16, geometry_dim=8, num_heads=4, ffn_dim=32,
                                           dropout=0, chunk_size=8, num_neighbors=6)
        return options

    def head_inputs(self):
        return (torch.randn(1, 768, 8, 8),
                dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.], [5., 5., 20., 20.]])],
                     img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])]),
                dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                     lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128)))

    def test_disabled_and_missing_flags_keep_legacy_rng_state_and_predictions(self):
        options = self.head_options(); options.pop('enable_relation_scoring'); options.pop('relation_scoring')
        torch.manual_seed(917); old = build_head(options); old_rng = torch.get_rng_state()
        torch.manual_seed(917); new = build_head(dict(options, enable_relation_scoring=False))
        torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        x, targets, args = self.head_inputs()
        torch.manual_seed(918); first = old.forward_train(x, targets, **args)
        torch.manual_seed(918); second = new.forward_train(x, targets, **args)
        for key in first[0]: torch.testing.assert_close(first[0][key], second[0][key], rtol=0, atol=0)
        torch.testing.assert_close(first[1]['refer_label'], second[1]['refer_label'], rtol=0, atol=0)
        old.eval(); new.eval()
        with torch.no_grad():
            a = old.forward_test(x, targets=targets, with_mask=False, **args)
            b = new.forward_test(x, targets=targets, with_mask=False, **args)
        torch.testing.assert_close(a[0]['refer_label'], b[0]['refer_label'], rtol=0, atol=0)

    def test_full_head_replaces_crs_preserves_boxes_and_detection_and_uses_original_bce(self):
        options = self.head_options()
        torch.manual_seed(71); relation = build_head(options)
        torch.manual_seed(71); original = build_head(dict(options, enable_relation_scoring=False))
        for name, value in original.state_dict().items():
            if not name.startswith('mrsm.'):
                torch.testing.assert_close(value, relation.state_dict()[name], rtol=0, atol=0)
        self.assertIsNone(relation.mrsm)
        self.assertFalse(any(name.startswith('mrsm.') for name, _ in relation.named_parameters()))
        x, targets, args = self.head_inputs()
        torch.manual_seed(72); a = original.forward_train(x, targets, **args)
        with patch.object(relation, 'prepare_referring_targets', wraps=relation.prepare_referring_targets) as labels:
            torch.manual_seed(72); b = relation.forward_train(x, targets, **args)
        torch.testing.assert_close(a[1]['pred_bbox_allobj'], b[1]['pred_bbox_allobj'], rtol=0, atol=0)
        torch.testing.assert_close(a[0]['loss_det'], b[0]['loss_det'], rtol=0, atol=0)
        self.assertEqual(b[0].keys(), a[0].keys())
        target = relation.prepare_referring_targets(*labels.call_args.args)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(b[1]['ranking_priority'].reshape(-1), target.reshape(-1))
        torch.testing.assert_close(b[0]['loss_refer'], expected * relation.loss_weight['refer'])
        sum(b[0].values()).backward()
        self.assertGreater(relation.relation_scorer.score[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(relation.refer_adapter.weight.grad.abs().sum().item(), 0)
        relation.relation_scorer.mark_optimizer_step(); relation.eval()
        with torch.no_grad(), patch.object(relation, '_crv_refer_gt', side_effect=AssertionError('No GT in inference')):
            prediction, details = relation.forward_test(x, targets=dict(img_metas=targets['img_metas']), with_mask=False, **args)
        self.assertEqual(prediction['ranking_priority'].shape, (1, 8, 1))
        self.assertEqual(details['relation_weights'].shape, (1, 8, 6))

    def test_grab_bcem_and_correctness_can_be_combined(self):
        options = self.head_options()
        options.update(enable_grab=True, enable_bcem=True, enable_correctness=True)
        options['bcem'] = dict(context_dim=16, num_heads=4, sampling_mode='ray', num_rays=4,
                               radial_samples=4, min_valid_points=2, chunk_size=2, checkpoint_chunks=False)
        head = build_head(options); x, targets, args = self.head_inputs()
        losses, prediction, details = head.forward_train(x, targets, **args)
        self.assertIn('loss_correctness', losses)
        self.assertIn('bcem_direction_weights', details)
        self.assertEqual(prediction['refer_label'].shape[1], 4)  # Grab retained 50%; no hardcoded 256.
        sum(losses.values()).backward()
        self.assertTrue(any(p.grad is not None for p in head.bcem.parameters()))
        head.correctness.mark_optimizer_step(); head.relation_scorer.mark_optimizer_step(); head.eval()
        with torch.no_grad():
            pred, _ = head.forward_test(x, targets=dict(img_metas=targets['img_metas']), with_mask=False, **args)
        obs = head.correctness.observations(pred['pred_bbox_allobj'], pred['refer_label'], pred['ranking_priority'],
                                            pred['pred_class_allobj'], targets['img_metas'], pred['ranking_priority'])
        self.assertEqual(pred['top1_correct_probability'].shape, (1,))
        self.assertEqual(obs[0].shape, (1,))
        # Exercise real clipping/postprocessing and its observer/ranking check.
        wrapper = MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
        torch.nn.Module.__init__(wrapper)
        wrapper.with_nms = False
        converted = wrapper.get_predictions_parts(pred, targets['img_metas'], with_bbox=True, with_mask=False)
        torch.testing.assert_close(converted['pred_bboxes'][0]['ranking_logits'],
                                   pred['ranking_priority'][0, :, 0])

    def test_checkpoint_guard_accepts_old_models_and_rejects_wrong_heads_or_untrained_weights(self):
        model = torch.nn.Module(); model.head = torch.nn.Module(); model.head.relation_scorer = self.module()
        with self.assertRaisesRegex(ValueError, 'complete trained'):
            _validate_relation_checkpoint(model, {})
        with self.assertRaisesRegex(ValueError, 'complete trained'):
            _validate_relation_checkpoint(model, model.state_dict())
        model.head.relation_scorer.mark_optimizer_step(); state = copy.deepcopy(model.state_dict())
        _validate_relation_checkpoint(model, state)
        _validate_relation_checkpoint(model, {'module.' + k: v for k, v in state.items()})
        state['head.relation_scorer.schema_version'].fill_(100)
        with self.assertRaises(ValueError): _validate_relation_checkpoint(model, state)
        model.head.relation_scorer = None
        _validate_relation_checkpoint(model, {'head.mrsm.fake_old_parameter': torch.ones(1)})
        with self.assertRaisesRegex(ValueError, 'Checkpoint uses relation'):
            _validate_relation_checkpoint(model, state)

    def test_training_loop_marks_new_scorer_after_optimizer_step(self):
        class Model(torch.nn.Module):
            def __init__(inner):
                super().__init__(); inner.head = torch.nn.Module(); inner.head.relation_scorer = self.module()
            def forward(inner, **unused):
                return {'loss_refer': inner.head.relation_scorer.score[0].weight.square().sum()}, {}
        model = Model(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        config = Config(dict(distributed=False, use_fp16=False, grad_norm_clip=.15, log_interval=1,
                             world_size=1, data=dict(samples_per_gpu=1)))
        with patch('propvg.apis.train.extract_data', side_effect=lambda x: x), \
                patch('propvg.apis.train.is_main', return_value=False):
            train_model(0, config, model, optimizer, [{}])
        self.assertEqual(model.head.relation_scorer.trained_steps.item(), 1)

    def test_switches_and_hyperparameter_validation(self):
        for bad in (dict(num_neighbors=0), dict(num_neighbors=True), dict(duplicate_iou=0),
                    dict(model_dim=15), dict(chunk_size=0), dict(dropout=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError): self.module(**bad)
        options = self.head_options(); options['enable_crv'] = True
        with self.assertRaisesRegex(ValueError, 'set enable_crv=False'): build_head(options)
        for filename in ('configs/aerialvg/PropVG-aerialvg.py', 'configs/aerialsense/PropVG-aerialsense.py'):
            cfg = Config.fromfile(filename)
            # The user's active experiment may enable this switch.
            self.assertIsInstance(cfg.model.head.enable_relation_scoring, bool)
            self.assertEqual(cfg.model.head.relation_scoring.num_neighbors, 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
