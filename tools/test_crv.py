"""CRV geometry, supervision, ordering, gradients and legacy isolation tests."""
import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from mmcv import Config
from propvg.models import build_head
from propvg.models.heads.counterfactual_refinement import CounterfactualRefinement, pairwise_iou
from propvg.apis.train import clip_training_gradients
from propvg.utils.checkpoint import _validate_crv_checkpoint


class CRVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self, **kwargs):
        options = dict(hidden_channels=16, scorer_dim=16, scorer_heads=4, scorer_layers=2,
                       grid_size=6, topk=5)
        options.update(kwargs)
        return CounterfactualRefinement(**options)

    def inputs(self, count=8):
        torch.manual_seed(73)
        boxes = torch.tensor([[[.12 + .1 * i, .5, .06, .1] for i in range(count)]])
        scores = torch.linspace(.65, .50, count)[None, :, None]
        return [torch.randn(1, 16, 24, 24), torch.zeros(1, 24, 24, dtype=torch.bool), boxes,
                torch.randn(1, 6, 16), torch.tensor([[0, 0, 0, 0, 1, 1]]), scores,
                torch.logit(scores), [dict(img_shape=(80, 100), pad_shape=(100, 100))]]

    def test_uncertainty_distinguishes_examples_and_duplicate_boxes(self):
        model = self.module()
        boxes = torch.tensor([[.02 + i * .18, .2, .1 + i * .18, .3] for i in range(5)])
        a = torch.tensor([.95, .3, .3, .2, .16])
        b = torch.tensor([.65, .61, .55, .52, .51])
        ua, ga = model.uncertainty(torch.logit(a), boxes, a.argsort(descending=True))
        ub, gb = model.uncertainty(torch.logit(b), boxes, b.argsort(descending=True))
        self.assertLess(ua.item(), ub.item())
        self.assertLess(ua.item(), model.uncertainty_threshold)
        self.assertGreater(ub.item(), model.uncertainty_threshold)
        self.assertEqual(ga.item(), 5)
        self.assertEqual(gb.item(), 5)
        repeated_entropy, repeated = model.uncertainty(torch.logit(b), boxes[:1].expand(5, -1), b.argsort(descending=True))
        self.assertEqual(repeated.item(), 1)
        self.assertEqual(repeated_entropy.item(), 0)

    def test_mask_geometry_padding_and_per_channel_mean(self):
        model = self.module(grid_size=24)
        args = self.inputs()
        valid = torch.ones(24, 24, dtype=torch.bool)
        valid[19:] = False
        feature = torch.arange(24 * 24.).reshape(1, 1, 24, 24).expand(1, 16, -1, -1).clone()
        feature[:, 1] = feature[:, 1] * 2 + 50
        meta = dict(img_shape=(80, 100), pad_shape=(100, 100))
        boxes = torch.tensor([[.4, .4, .6, .6]])
        mask = model.intervention_masks(boxes, feature[0], meta, valid)
        xx, yy, _, _ = model.spatial_grid(feature[0], meta)
        inner = (xx - .5).abs().le(.15) & (yy - .5).abs().le(.15) & valid
        outer = ((xx - .5).abs().ge(.2) | (yy - .5).abs().ge(.2))
        torch.testing.assert_close(mask[0][inner], torch.ones_like(mask[0][inner]))
        self.assertEqual(mask[0][outer].count_nonzero(), 0)
        self.assertEqual(mask[0][~valid].count_nonzero(), 0)
        scene, _ = model.pooled_scenes(feature, valid, mask)
        expected = (feature * valid).sum((-1, -2)) / valid.sum()
        for channel in (0, 1):
            torch.testing.assert_close(scene[0, channel][inner], expected[0, channel].expand_as(scene[0, channel][inner]))
        self.assertNotEqual(expected[0, 0].item(), expected[0, 1].item())

    def test_projection_commutes_with_full_resolution_mean_replacement(self):
        model = self.module()
        visual, mask, boxes, *_ = self.inputs()
        valid = ~mask[0]
        masks = model.intervention_masks(model.clipped_boxes(boxes[0, :2]), visual[0],
                                         dict(img_shape=(80, 100), pad_shape=(100, 100)), valid)
        projected = model.scorer.visual(visual)
        actual, _ = model.pooled_scenes(projected, valid, masks)
        mean = (visual * valid).sum((-1, -2), keepdim=True) / valid.sum()
        modified = visual * (1 - masks[:, None]) + mean * masks[:, None]
        expected = model.scorer.visual(modified)
        expected = torch.nn.functional.adaptive_avg_pool2d(expected, 6)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_expanded_negative_covering_gt_is_not_a_negative(self):
        model = self.module()
        visual = torch.zeros(16, 100, 100)
        gt = torch.tensor([.5, .4, .6, .6])
        candidate = torch.tensor([[.35, .4, .49, .6]])
        self.assertEqual(pairwise_iou(candidate, gt[None]).item(), 0)
        masks = model.intervention_masks(candidate, visual, dict(img_shape=(100, 100), pad_shape=(100, 100)),
                                         torch.ones(100, 100, dtype=torch.bool))
        coverage = model.gt_coverage(masks, gt, visual, dict(img_shape=(100, 100), pad_shape=(100, 100)))
        self.assertGreater(coverage.item(), model.negative_gt_coverage)
        _, negative = model.counterfactual_loss(torch.tensor([3.], requires_grad=True), torch.tensor([0.]), coverage)
        self.assertFalse(negative.item())

    def test_counterfactual_labels_and_final_multiple_positives(self):
        model = self.module()
        loss, negative = model.counterfactual_loss(torch.tensor([2., 0., 99., 99.]),
                                                  torch.tensor([.6, 0., .2, .5]), torch.zeros(4))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(negative.tolist(), [False, True, False, False])
        # Switching between two metric-correct proposals is not a harmful flip.
        base = torch.tensor([2., 1., 0.])
        fused = torch.tensor([1., 3., 0.])
        _, _, safety, _ = model.fusion_loss(base, fused, torch.tensor([.6, .8, 0.]))
        self.assertEqual(safety.item(), 0)
        loss, *_ = model.fusion_loss(base, fused, torch.zeros(3))
        self.assertEqual(loss.item(), 0)

    def test_gate_learns_to_repair_and_to_avoid_harm(self):
        for correct in (False, True):
            model = self.module()
            base = torch.tensor([.2, .1])
            drops = torch.tensor([-1., 1.], requires_grad=True)
            fused, alpha, branches = model.fusion(base, drops, torch.tensor(1.))
            ious = torch.tensor([.9, 0.]) if correct else torch.tensor([0., .9])
            loss, *_ = model.fusion_loss(branches["base_logp"], fused, ious)
            loss.backward()
            gradient = model.gate.bias.grad.item()
            self.assertGreater(gradient, 0) if correct else self.assertLess(gradient, 0)
            self.assertGreater(drops.grad.abs().sum().item(), 0)
            self.assertGreater(alpha.item(), 0)

    def test_distribution_fusion_stays_normalized_and_temperatures_are_fixed(self):
        model = self.module(gate_initial_probability=.99)
        base = torch.tensor([8., 0.])
        drops = torch.tensor([0., 8.], requires_grad=True)
        fused, alpha, branches = model.fusion(base, drops, torch.tensor(.8))
        self.assertGreater(alpha.item(), .5)
        self.assertEqual(fused.argmax().item(), 1)
        expected = (1 - alpha) * branches["base_logp"].exp() + alpha * branches["cf_logp"].exp()
        torch.testing.assert_close(fused.exp(), expected)
        torch.testing.assert_close(expected.sum(), torch.tensor(1.))
        loss, *_ = model.fusion_loss(branches["base_logp"], fused, torch.tensor([0., .9]))
        loss.backward()
        self.assertGreater(drops.grad.abs().sum().item(), 0)
        self.assertFalse(model.base_temperature.requires_grad)
        self.assertFalse(model.cf_temperature.requires_grad)
        self.assertFalse(any("temperature" in name for name, _ in model.named_parameters()))

    def test_extreme_scores_and_saturated_gates_stay_finite(self):
        for bias in (-1000., 1000.):
            model = self.module()
            model.gate.bias.data.fill_(bias)
            fused, alpha, branches = model.fusion(torch.tensor([1000., -1000.]), torch.tensor([-1000., 1000.]),
                                                  torch.tensor(1.))
            self.assertTrue(torch.isfinite(fused).all())
            torch.testing.assert_close(fused.exp().sum(), torch.tensor(1.))
            expected = branches["base_logp"] if bias < 0 else branches["cf_logp"]
            # Dominant branch wins even though the log-mixture retains tiny tails.
            self.assertEqual(fused.argmax().item(), expected.argmax().item())
        self.assertGreater(model.base_temperature.item(), 0)

    def test_gate_uses_only_detached_entropy(self):
        model = self.module()
        base, drops = torch.tensor([2., 1., 0.]), torch.tensor([0., 1., 2.])
        entropy = torch.tensor(.8, requires_grad=True)
        _, alpha_a, a = model.fusion(base, drops, entropy)
        _, alpha_b, b = model.fusion(base + 10, drops * 10, entropy)
        stats = a["gate_statistics"]
        self.assertEqual(stats.shape, (1,))
        self.assertFalse(stats.requires_grad)
        torch.testing.assert_close(stats, entropy.detach().reshape(1))
        torch.testing.assert_close(alpha_a, alpha_b)
        torch.testing.assert_close(a["base_logp"], b["base_logp"])
        self.assertEqual(sum(p.numel() for p in model.gate.parameters()), 2)
        alpha_a.backward()
        self.assertIsNone(entropy.grad)

    def test_threshold_is_strict_and_inactive_fusion_is_exact_base(self):
        model = self.module(gate_initial_probability=.99)
        for entropy in (.0, .59, .6):
            fused, alpha, branches = model.fusion(torch.tensor([2., 1.]), torch.tensor([0., 100.]),
                                                  torch.tensor(entropy))
            self.assertEqual(alpha.item(), 0)
            torch.testing.assert_close(fused, branches["base_logp"], rtol=0, atol=0)
        _, alpha, _ = model.fusion(torch.tensor([2., 1.]), torch.tensor([0., 100.]), torch.tensor(.61))
        self.assertGreater(alpha.item(), 0)

    def test_safety_reference_is_detached(self):
        model = self.module()
        base = torch.tensor([.1, 0.], requires_grad=True)
        fused = torch.tensor([-.2, 0.], requires_grad=True)
        _, _, safety, _ = model.fusion_loss(base, fused, torch.tensor([.8, 0.]))
        safety.backward()
        self.assertIsNone(base.grad)
        self.assertGreater(fused.grad.abs().sum().item(), 0)

    def test_independent_gradients_and_deterministic_forward(self):
        model = self.module().train()
        args = self.inputs()
        for index in (0, 2, 3, 5, 6):
            args[index].requires_grad_(True)
        gt = [torch.tensor([.09, .45, .15, .55])]
        rng = torch.get_rng_state()
        probabilities, priority, losses, info = model(*args, refer_gt=gt)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        sum(losses.values()).backward()
        for index in (0, 2, 3, 5, 6):
            self.assertIsNone(args[index].grad)
        self.assertGreater(model.scorer.visual.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))
        self.assertEqual(priority.shape, probabilities.shape)
        self.assertEqual(info["crv_active"].item(), 1.)

    def test_rank_changes_are_confined_to_original_topk(self):
        model = self.module().eval()
        args = self.inputs()
        original = args[5][0, :, 0].argsort(descending=True)
        def reverse(base, *unused):
            logp = torch.arange(len(base), dtype=base.dtype).log_softmax(0)
            return logp, base.new_tensor(.5), dict(base_logp=base.log_softmax(0), cf_logp=logp,
                                                gate_statistics=torch.zeros(1))
        with patch.object(model, "fusion", side_effect=reverse):
            _, priority, _, _ = model(*args)
        result = priority[0, :, 0].argsort(descending=True)
        self.assertEqual(result[:5].tolist(), original[:5].flip(0).tolist())
        self.assertEqual(result[5:].tolist(), original[5:].tolist())
        self.assertEqual(set(result[:5].tolist()), set(original[:5].tolist()))

    def test_confident_inference_skips_scorer_and_ignores_gt(self):
        model = self.module().eval()
        args = self.inputs()
        args[5] = torch.tensor([[[.95], [.3], [.29], [.2], [.16], [.1], [.09], [.08]]])
        args[6] = torch.logit(args[5])
        with patch.object(model.scorer, "forward", wraps=model.scorer.forward) as score:
            probabilities, _, _, info = model(*args, refer_gt=[torch.ones(4) * 999])
        self.assertEqual(score.call_count, 0)
        self.assertEqual(info["crv_alpha"].item(), 0)
        self.assertEqual(info["crv_active"].item(), 0)
        torch.testing.assert_close(probabilities, args[5], rtol=0, atol=0)
        self.assertTrue(info["crv_samples"][0]["skipped"])
        ambiguous = self.inputs()
        a = model(*ambiguous, refer_gt=[torch.zeros(4)])
        b = model(*ambiguous, refer_gt=[torch.ones(4)])
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)

    def test_low_entropy_training_only_trains_counterfactual_scorer(self):
        model = self.module().train()
        args = self.inputs()
        args[5] = torch.tensor([[[.99], [.1], [.09], [.08], [.07], [.06], [.05], [.04]]])
        args[6] = torch.logit(args[5])
        probabilities, _, losses, info = model(*args, refer_gt=[torch.tensor([.09, .45, .15, .55])])
        self.assertEqual(info["crv_active"].item(), 0)
        self.assertEqual(info["crv_alpha"].item(), 0)
        self.assertEqual(losses["loss_crv_fusion"].item(), 0)
        self.assertGreater(losses["loss_crv_cf"].item(), 0)
        torch.testing.assert_close(probabilities, args[5], rtol=0, atol=0)
        sum(losses.values()).backward()
        self.assertGreater(model.scorer.visual.weight.grad.abs().sum().item(), 0)
        for parameter in model.gate.parameters():
            self.assertTrue(parameter.grad is None or parameter.grad.count_nonzero() == 0)

    def test_padding_text_and_visual_values_cannot_change_scores(self):
        model = self.module().eval()
        args = self.inputs()
        args[1][:, 18:] = True
        first = model(*args)
        changed = list(args)
        changed[0] = args[0].masked_fill(args[1][:, None], 10000.)
        changed[3] = args[3].masked_fill(args[4].bool()[..., None], -10000.)
        second = model(*changed)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)

    def test_empty_text_or_visual_returns_original_order_and_zero_losses(self):
        for empty_text in (False, True):
            model = self.module().train()
            args = self.inputs()
            args[4][:] = 1 if empty_text else args[4]
            args[1][:] = not empty_text
            probability, _, losses, info = model(*args, refer_gt=[torch.tensor([.1, .4, .2, .6])])
            torch.testing.assert_close(probability, args[5], rtol=0, atol=0)
            self.assertEqual(sum(value.item() for value in losses.values()), 0)
            self.assertEqual(info["crv_alpha"].item(), 0)

    def test_full_head_base_rng_losses_gradients_and_updates_are_unchanged(self):
        options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        options["num_queries"] = 8
        options["crv"] = dict(scorer_dim=16, scorer_heads=4, grid_size=4, topk=5)
        for with_context in (False, True):
            options["enable_bcem"] = with_context
            options["enable_grab"] = with_context
            options["bcem"] = dict(context_dim=16, num_heads=4, sampling_mode="ray", num_rays=4,
                                   radial_samples=4, min_valid_points=2, chunk_size=2, checkpoint_chunks=False)
            torch.manual_seed(925)
            original = build_head(dict(options, enable_crv=False))
            rng = torch.get_rng_state()
            torch.manual_seed(925)
            refined = build_head(dict(options, enable_crv=True))
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(refined.state_dict()[name], value, rtol=0, atol=0)

            x = torch.randn(1, 768, 8, 8)
            args = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                        lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128))
            targets = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]])],
                           img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
            torch.manual_seed(926)
            losses_a, _, _ = original.forward_train(x, targets, **args)
            next_rng = torch.get_rng_state()
            sum(losses_a.values()).backward()
            clip_training_gradients(original, .15)
            torch.manual_seed(926)
            losses_b, _, _ = refined.forward_train(x, targets, **args)
            torch.testing.assert_close(torch.get_rng_state(), next_rng, rtol=0, atol=0)
            for key in losses_a:
                torch.testing.assert_close(losses_a[key], losses_b[key], rtol=0, atol=0)
            loss = sum(value for key, value in losses_b.items() if key.startswith("loss_"))
            loss.backward()
            clip_training_gradients(refined, .15)
            parameters = dict(refined.named_parameters())
            for name, parameter in original.named_parameters():
                if parameter.grad is None:
                    self.assertIsNone(parameters[name].grad)
                else:
                    torch.testing.assert_close(parameters[name].grad, parameter.grad, rtol=0, atol=0)
            a = torch.optim.Adam(original.parameters(), lr=1e-4)
            b = torch.optim.Adam(refined.parameters(), lr=1e-4)
            a.step()
            b.step()
            for name, value in original.state_dict().items():
                torch.testing.assert_close(refined.state_dict()[name], value, rtol=0, atol=0)

            # Exercise enabled inference too: only metadata is provided, no GT.
            original.eval()
            refined.eval()
            with torch.no_grad(), patch.object(refined, "_crv_refer_gt", side_effect=AssertionError("No inference GT")):
                prediction_a, _ = original.forward_test(x, targets=dict(img_metas=targets["img_metas"]),
                                                         with_mask=False, **args)
                prediction_b, info = refined.forward_test(x, targets=dict(img_metas=targets["img_metas"]),
                                                          with_mask=False, **args)
            torch.testing.assert_close(prediction_a["pred_bbox_allobj"], prediction_b["pred_bbox_allobj"], rtol=0, atol=0)
            torch.testing.assert_close(prediction_a["refer_label"], info["crv_crs_probabilities"][..., None], rtol=0, atol=0)
            base_order = prediction_a["refer_label"][0, :, 0].argsort(descending=True, stable=True)
            final_order = prediction_b["ranking_priority"][0, :, 0].argsort(descending=True)
            self.assertEqual(set(base_order[:5].tolist()), set(final_order[:5].tolist()))

    def test_checkpoint_does_not_silently_enable_untrained_crv(self):
        model = torch.nn.Module()
        model.head = torch.nn.Module()
        model.head.crv = self.module()
        state = model.state_dict()
        _validate_crv_checkpoint(model, state)
        _validate_crv_checkpoint(model, {"module." + key: value for key, value in state.items()})
        with self.assertRaisesRegex(ValueError, "compatible CRV checkpoint"):
            _validate_crv_checkpoint(model, {})
        for version in (1, 2):
            with self.assertRaisesRegex(ValueError, "compatible CRV checkpoint"):
                _validate_crv_checkpoint(model, dict(state, **{"head.crv.schema_version": torch.tensor(version)}))
        model.head.crv = None
        _validate_crv_checkpoint(model, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
