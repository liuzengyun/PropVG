"""Configurable pair aggregation: legacy equivalence, hard max and integration."""
import copy
import unittest
from unittest.mock import patch

import torch
from mmcv import Config

from propvg.models import build_head
from propvg.utils.checkpoint import _validate_relation_checkpoint
import test_relation_scoring as fixtures


class RelationAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.fixture = fixtures.RelationScoringTests()

    def test_default_equals_explicit_adaptive_including_rng_outputs_and_gradients(self):
        torch.manual_seed(521)
        old = self.fixture.module()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(521)
        explicit = self.fixture.module(aggregation='adaptive')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(old.aggregation, 'adaptive')
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        # Exercise nonuniform learned weights, not only their uniform init.
        with torch.no_grad():
            old.weight[-1].weight.normal_()
        explicit.load_state_dict(old.state_dict(), strict=True)
        inputs = self.fixture.inputs()
        for model in (old, explicit):
            model.eval()
        a, b = old(*inputs), explicit(*inputs)
        for first, second in zip(a[:3], b[:3]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for key in a[3]:
            torch.testing.assert_close(a[3][key], b[3][key], rtol=0, atol=0)
        a[1].square().mean().backward()
        b[1].square().mean().backward()
        for (_, left), (_, right) in zip(old.named_parameters(), explicit.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_max_excludes_invalid_pairs_and_routes_gradients_only_to_winner(self):
        model = self.fixture.module(aggregation='max', chunk_size=128)
        inputs = self.fixture.inputs(n=4)
        selection = model.select_neighbors(inputs[1], inputs[4], inputs[5])
        effective = selection['neighbor_valid'].clone()
        effective[..., 0] |= ~effective.any(-1)
        self.assertTrue((~effective).any())
        raw = torch.arange(6, dtype=torch.float32).expand(2, 4, 6).clone()
        raw[~effective] = 1e5  # Padded pairs would win without masking.
        raw.requires_grad_()
        with patch.object(model.score, 'forward', return_value=raw.reshape(2, 24, 1)):
            probability, logits, _, details = model(*inputs)
        expected, winner = raw.detach().masked_fill(~effective, -torch.inf).max(-1, keepdim=True)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(probability, expected.sigmoid(), rtol=0, atol=0)
        one_hot = torch.zeros_like(raw).scatter_(-1, winner, 1)
        torch.testing.assert_close(details['relation_weights'], one_hot, rtol=0, atol=0)
        logits.sum().backward()
        torch.testing.assert_close(raw.grad, one_hot, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.weight.parameters()))

    def test_max_attention_uses_winning_pair_not_adaptive_weights(self):
        model = self.fixture.module(aggregation='max').eval()
        collected = []
        hook = model.layers[-1].register_forward_hook(lambda _, args, out: collected.append(out[1].detach()))
        try:
            _, _, attention, details = model(*self.fixture.inputs())
        finally:
            hook.remove()
        pair_attention = torch.cat(collected, dim=1).reshape(2, 8, 6, 7)
        expected = (pair_attention * details['relation_weights'][..., None]).sum(2)
        torch.testing.assert_close(attention, expected, rtol=0, atol=0)
        self.assertEqual(attention[..., 4:].count_nonzero(), 0)

    def test_max_fallback_empty_candidates_empty_text_and_ties(self):
        model = self.fixture.module(aggregation='max')
        for count in (0, 1, 8):
            with self.subTest(candidates=count):
                inputs = self.fixture.inputs(n=count)
                inputs[2], inputs[3] = inputs[2][:, :0], inputs[3][:, :0]
                probability, logits, attention, details = model(*inputs)
                self.assertEqual(probability.shape, (2, count, 1))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(attention).all())
                self.assertEqual(attention.count_nonzero(), 0)
                if count == 1:
                    self.assertTrue(details['relation_fallback'].all())
                    self.assertTrue((details['relation_weights'][..., 0] == 1).all())
                    probability.sum().backward()
        with torch.no_grad():
            model.score[-1].weight.zero_()
            model.score[-1].bias.zero_()
        _, _, _, details = model(*self.fixture.inputs())
        first_valid = details['relation_effective_mask'].long().argmax(-1)
        self.assertTrue(torch.equal(details['relation_weights'].argmax(-1), first_valid))

    def test_max_feature_gradients_but_not_coordinates_detector_or_weight_head(self):
        model = self.fixture.module(aggregation='max')
        inputs = self.fixture.inputs()
        model(*inputs)[1].square().mean().backward()
        for value in (inputs[0], inputs[2], model.score[-1].weight):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertIsNone(inputs[1].grad)
        self.assertIsNone(inputs[4].grad)
        self.assertTrue(all(p.grad is None for p in model.weight.parameters()))

    def test_max_chunks_preserve_outputs_and_gradients(self):
        first = self.fixture.module(aggregation='max', chunk_size=11)
        second = self.fixture.module(aggregation='max', chunk_size=128)
        second.load_state_dict(first.state_dict())
        inputs = self.fixture.inputs()
        a, b = first(*inputs), second(*inputs)
        torch.testing.assert_close(a[1], b[1], rtol=1e-5, atol=1e-6)
        a[1].sum().backward()
        b[1].sum().backward()
        for (_, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)

    def test_state_optimizer_layout_and_checkpoint_guard_are_compatible(self):
        adaptive = self.fixture.module()
        maximum = self.fixture.module(aggregation='max')
        adaptive.mark_optimizer_step()
        maximum.load_state_dict(adaptive.state_dict(), strict=True)
        self.assertEqual(list(adaptive.state_dict()), list(maximum.state_dict()))
        self.assertEqual([(k, p.shape) for k, p in adaptive.named_parameters()],
                         [(k, p.shape) for k, p in maximum.named_parameters()])
        optimizer = torch.optim.AdamW(adaptive.parameters())
        adaptive(*self.fixture.inputs())[1].sum().backward()
        optimizer.step()
        resumed = torch.optim.AdamW(maximum.parameters())
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        maximum(*self.fixture.inputs())[1].sum().backward()
        resumed.step()
        wrapper = torch.nn.Module()
        wrapper.head = torch.nn.Module()
        wrapper.head.relation_scorer = maximum
        _validate_relation_checkpoint(wrapper, wrapper.state_dict())
        self.assertEqual(maximum.schema_version.item(), 1)

    def test_max_full_head_training_and_inference_with_optional_modules(self):
        for optional in (False, True):
            with self.subTest(optional_modules=optional):
                options = self.fixture.head_options()
                options['relation_scoring'].update(aggregation='max', foreground_topk=6,
                                                    strict_disjoint_neighbors=True)
                options.update(enable_grab=optional, enable_bcem=optional, enable_correctness=optional)
                if optional:
                    options['bcem'] = dict(context_dim=16, num_heads=4, sampling_mode='ray', num_rays=4,
                                           radial_samples=4, min_valid_points=2, chunk_size=2,
                                           checkpoint_chunks=False)
                head = build_head(options)
                x, targets, args = self.fixture.head_inputs()
                losses, prediction, _ = head.forward_train(x, targets, **args)
                self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
                sum(losses.values()).backward()
                self.assertIsNotNone(head.relation_scorer.score[-1].weight.grad)
                self.assertTrue(all(p.grad is None for p in head.relation_scorer.weight.parameters()))
                head.relation_scorer.mark_optimizer_step()
                if optional:
                    head.correctness.mark_optimizer_step()
                head.eval()
                with torch.no_grad():
                    pred, details = head.forward_test(x, targets=dict(img_metas=targets['img_metas']),
                                                     with_mask=False, **args)
                expected = details['relation_pair_logits'].masked_fill(
                    ~details['relation_effective_mask'], -torch.inf).max(-1, keepdim=True).values
                torch.testing.assert_close(pred['ranking_priority'], expected)

    def test_modes_validated_and_both_configs_keep_adaptive_default(self):
        for bad in ('mean', 'MAX', None, 0):
            with self.subTest(invalid=bad), self.assertRaisesRegex(ValueError, 'aggregation'):
                self.fixture.module(aggregation=bad)
        for path in ('configs/aerialvg/PropVG-aerialvg.py', 'configs/aerialsense/PropVG-aerialsense.py'):
            config = Config.fromfile(path)
            self.assertEqual(config.relation_aggregation, 'adaptive')
            self.assertEqual(config.model.head.relation_scoring.aggregation, 'adaptive')


if __name__ == '__main__':
    unittest.main()
