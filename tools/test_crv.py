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
        ready = kwargs.pop("ready", True)
        options = dict(hidden_channels=16, scorer_dim=16, scorer_heads=4, scorer_layers=2,
                       grid_size=6, topk=5)
        options.update(kwargs)
        model = CounterfactualRefinement(**options)
        if ready:
            model.last_training_epoch.fill_(model.start_epoch)
        return model

    def inputs(self, count=8):
        torch.manual_seed(73)
        boxes = torch.tensor([[[.12 + .1 * i, .5, .06, .1] for i in range(count)]])
        scores = torch.linspace(.65, .50, count)[None, :, None]
        return [torch.randn(1, 16, 24, 24), torch.zeros(1, 24, 24, dtype=torch.bool), boxes,
                torch.randn(1, 6, 16), torch.tensor([[0, 0, 0, 0, 1, 1]]), scores,
                torch.logit(scores), [dict(img_shape=(80, 100), pad_shape=(100, 100))]]

    def pooled_scenes(self, model, projected, valid, masks):
        original, denominator, mean = model.pooled_original(projected, valid[None])
        masked = model.pooled_interventions(projected, valid[None], masks, None, original, denominator, mean)
        return original.expand(len(masks), -1, -1, -1), masked, denominator.expand(len(masks), -1, -1, -1)

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
        _, scene, _ = self.pooled_scenes(model, feature, valid, mask)
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
        _, actual, _ = self.pooled_scenes(model, projected, valid, masks)
        mean = (visual * valid).sum((-1, -2), keepdim=True) / valid.sum()
        modified = visual * (1 - masks[:, None]) + mean * masks[:, None]
        expected = model.scorer.visual(modified)
        expected = torch.nn.functional.adaptive_avg_pool2d(expected, 6)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_full_scene_pooling_matches_full_resolution_mean_intervention(self):
        model = self.module(grid_size=24)
        visual = torch.arange(2 * 144 * 144.).reshape(1, 2, 144, 144)
        valid = torch.ones(144, 144, dtype=torch.bool)
        valid[110:] = False
        masks = torch.zeros(2, 144, 144)
        masks[0, 42:54, 52:64] = 1
        masks[1, 90:100, 80:92] = .4
        original, masked, support = self.pooled_scenes(model, visual, valid, masks)
        mean = (visual * valid).sum((-1, -2), keepdim=True) / valid.sum()
        footprint = masks * valid
        full = visual * (1 - footprint[:, None]) + mean * footprint[:, None]
        denominator = torch.nn.functional.adaptive_avg_pool2d(valid[None, None].float(), 24)
        expected_original = torch.nn.functional.adaptive_avg_pool2d(visual * valid, 24) / denominator.clamp(min=1e-8)
        expected_masked = torch.nn.functional.adaptive_avg_pool2d(full * valid, 24) / denominator.clamp(min=1e-8)
        torch.testing.assert_close(original, expected_original.expand_as(original))
        torch.testing.assert_close(masked, expected_masked)
        torch.testing.assert_close(support, denominator.expand_as(support))
        self.assertEqual(original.shape, (2, 2, 24, 24))
        self.assertFalse(torch.allclose(mean[0, :, 0, 0], visual[:, :, 24:72, 24:72].mean((-1, -2))[0]))

    def test_intervention_and_context_are_not_truncated_to_48_cells(self):
        model = self.module(grid_size=24)
        feature = torch.arange(2 * 144 * 144.).reshape(1, 2, 144, 144)
        meta = dict(img_shape=(100, 100), pad_shape=(100, 100))
        valid = torch.ones(144, 144, dtype=torch.bool)
        boxes = torch.tensor([[.1, .1, .9, .9], [0., 0., 0., 0.]])
        masks = model.intervention_masks(boxes, feature[0], meta, valid)
        self.assertEqual(masks[0, 10, 10].item(), 1)
        self.assertEqual(masks[1].count_nonzero().item(), 0)
        original, masked, pooled_valid = self.pooled_scenes(model, feature, valid, masks)
        mean = feature.mean((-1, -2), keepdim=True)
        torch.testing.assert_close(masked[:1], mean.expand(1, 2, 24, 24))
        torch.testing.assert_close(masked[1:], original[1:])
        self.assertEqual(pooled_valid[1].count_nonzero().item(), 24 * 24)
        # Evidence far outside the former candidate crop belongs to the scene.
        changed = feature.clone()
        changed[:, :, :24] = 10000
        other, _, _ = self.pooled_scenes(model, changed, valid, torch.zeros(2, 144, 144))
        self.assertFalse(torch.allclose(other[0], original[0]))
        torch.testing.assert_close(other[0], other[1])
        # Coverage accounts for the complete expanded intervention, not a crop.
        gt = torch.tensor([.8, .5, .9, .6])
        coverage = model.gt_coverage(masks[:1], gt, feature[0], meta)
        torch.testing.assert_close(coverage, torch.ones(1))

    def test_common_original_is_shared_across_all_interventions(self):
        model = self.module()
        original = torch.randn((), requires_grad=True)
        masked = torch.randn(4, requires_grad=True)
        actual = model.support_drop(original, masked)
        torch.testing.assert_close(actual, original - masked)
        zero = model.support_drop(original, original)
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=1e-6)
        actual[0].backward()
        self.assertGreater(original.grad.abs().sum().item(), 0)
        self.assertEqual(masked.grad[1:].count_nonzero().item(), 0)

    def test_full_scene_pooling_matches_nondivisible_reference_and_gradients(self):
        for height, width in ((25, 29), (19, 23)):
            model = self.module(grid_size=6)
            visual = torch.randn(2, 3, height, width, requires_grad=True)
            valid = torch.ones(2, height, width, dtype=torch.bool)
            valid[0, -4:] = False
            valid[1, :, -3:] = False
            owners = torch.tensor([0, 1, 0, 1])
            masks = torch.rand(4, height, width) * valid[owners]
            original, denominator, mean = model.pooled_original(visual, valid)
            actual = model.pooled_interventions(visual, valid, masks, owners, original, denominator, mean)
            full = visual[owners] + (mean[owners] - visual[owners]) * masks[:, None]
            expected = torch.nn.functional.adaptive_avg_pool2d(full * valid[owners, None], 6) / denominator[owners].clamp(min=1e-8)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            actual_grad, = torch.autograd.grad(actual.square().sum(), visual, retain_graph=True)
            expected_grad, = torch.autograd.grad(expected.square().sum(), visual)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    def test_batched_grouping_matches_original_greedy_algorithm(self):
        model = self.module()
        args = self.inputs()
        clipped = model.clipped_boxes(args[2]).expand(4, -1, -1).clone()
        clipped[1, 1:4] = clipped[1, :1]
        clipped[2] = clipped[2, :1]
        clipped[3, 2:4] = 0
        logits = args[6].squeeze(-1).expand(4, -1)
        order = logits.argsort(1, descending=True)
        entropy, counts = model.batch_uncertainty(logits, clipped, order)
        for bi in range(len(logits)):
            kept = []
            for index in order[bi, :model.uncertainty_topn].tolist():
                if not kept or pairwise_iou(clipped[bi, index:index + 1], clipped[bi, kept]).max() < model.duplicate_iou:
                    kept.append(index)
            probability = logits[bi, kept].softmax(0)
            expected = probability.new_zeros(()) if len(kept) < 2 else (
                -(probability * probability.log()).sum() / torch.tensor(float(len(kept))).log())
            torch.testing.assert_close(entropy[bi], expected)
            self.assertEqual(counts[bi].item(), len(kept))

    def test_cross_image_batching_caches_text_and_scores_one_original_per_image(self):
        model = self.module(chunk_size=8).train()
        args = self.inputs()
        args = [value.repeat(3, *([1] * (value.ndim - 1))) if torch.is_tensor(value)
                else value * 3 for value in args]
        args[0][1] += .7
        args[5][2] = torch.tensor([[.99], [.1], [.09], [.08], [.07], [.06], [.05], [.04]])
        args[6] = torch.logit(args[5])
        gt = [torch.tensor([.09, .45, .15, .55])] * 3
        encoded = []
        hook = model.scorer.register_forward_pre_hook(lambda _, inputs: encoded.append(inputs[0].shape[0]))
        try:
            with patch.object(model.scorer, "prepare_text", wraps=model.scorer.prepare_text) as text_cache, \
                 patch.object(model.scorer, "spatial_positions", wraps=model.scorer.spatial_positions) as positions, \
                 patch.object(model.scorer.visual, "forward", wraps=model.scorer.visual.forward) as projection:
                probabilities, priority, losses, info = model(*args, refer_gt=gt)
        finally:
            hook.remove()
        self.assertEqual(encoded, [8, 6])  # Two routed images, each with 1+5+1 scenes.
        self.assertEqual(text_cache.call_count, 1)
        self.assertEqual(positions.call_count, 1)
        self.assertEqual(projection.call_count, 1)
        self.assertEqual(projection.call_args.args[0].shape[0], 2)
        torch.testing.assert_close(info["crv_active"], torch.tensor(2 / 3))
        torch.testing.assert_close(probabilities[2], args[5][2], rtol=0, atol=0)
        self.assertTrue(info["crv_samples"][2]["skipped"])
        per_image_losses = []
        for bi in range(2):
            single = [value[bi:bi + 1] for value in args]
            p, rank, loss, details = model(*single, refer_gt=gt[bi:bi + 1])
            torch.testing.assert_close(p[0], probabilities[bi])
            torch.testing.assert_close(rank[0], priority[bi])
            torch.testing.assert_close(details["crv_samples"][0]["drops"], info["crv_samples"][bi]["drops"], atol=1e-6, rtol=1e-5)
            per_image_losses.append(loss)
        for key, value in losses.items():
            torch.testing.assert_close(value, sum(loss[key] for loss in per_image_losses) / 3)

    def test_chunk_size_changes_neither_outputs_nor_parameter_gradients(self):
        first = self.module(chunk_size=1).train()
        second = copy.deepcopy(first)
        second.chunk_size = 8
        args = self.inputs()
        gt = [torch.tensor([.09, .45, .15, .55])]
        a = first(*args, refer_gt=gt)
        b = second(*args, refer_gt=gt)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])
        for key in a[2]:
            torch.testing.assert_close(a[2][key], b[2][key])
        sum(a[2].values()).backward()
        sum(b[2].values()).backward()
        for (name, parameter), (other_name, other) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=2e-6)

    def test_legacy_local_crop_setting_is_rejected(self):
        for value in (0, -1, True, 3, 48, 48.5):
            with self.assertRaisesRegex(ValueError, "crop_size"):
                self.module(crop_size=value)

    def test_configs_use_48_full_scene_grid(self):
        for filename in ("configs/aerialvg/PropVG-aerialvg.py", "configs/aerialsense/PropVG-aerialsense.py"):
            config = Config.fromfile(filename)
            self.assertEqual(config.crv_grid_size, 48)
            self.assertEqual(config.model.head.crv.grid_size, 48)
            self.assertNotIn("crop_size", config.model.head.crv)

    def test_144_feature_forward_scores_48_grids_with_one_original_and_six_interventions(self):
        model = self.module(hidden_channels=256, scorer_dim=128, grid_size=48, chunk_size=2).train()
        args = self.inputs()
        args[0] = torch.randn(1, 256, 144, 144, requires_grad=True)
        args[1] = torch.zeros(1, 144, 144, dtype=torch.bool)
        args[1][:, 116:] = True
        args[3] = torch.randn(1, 64, 256, requires_grad=True)
        args[4] = torch.zeros(1, 64, dtype=torch.bool)
        args[4][:, 18:] = True
        gt = torch.tensor([.09, .45, .15, .55])
        encoded = []
        hook = model.scorer.register_forward_pre_hook(lambda _, inputs: encoded.append(inputs[0].shape))
        try:
            with patch.object(model, "pooled_interventions", wraps=model.pooled_interventions) as pooling:
                probabilities, _, losses, info = model(*args, refer_gt=[gt])
        finally:
            hook.remove()
        # One full-scene original, five predicted-box masks and one exact GT.
        self.assertEqual(pooling.call_count, 3)
        self.assertEqual(encoded, [torch.Size([2, 128, 48, 48])] * 3 + [torch.Size([1, 128, 48, 48])])
        masks = torch.cat([call.args[2] for call in pooling.call_args_list])
        boxes = torch.cat((model.clipped_boxes(args[2][0, :5]), gt[None]))
        torch.testing.assert_close(masks, model.intervention_masks(boxes, args[0][0], args[7][0], ~args[1][0]))
        details = info["crv_samples"][0]
        self.assertNotIn("crop_origins", details)
        self.assertEqual(details["original_support"].shape, (5,))
        torch.testing.assert_close(details["original_support"], details["original_support"][:1].expand(5))
        torch.testing.assert_close(details["drops"], details["original_support"] - details["masked_support"])
        expected = details["base_logits"] + info["crv_alpha"] * details["drops"]
        torch.testing.assert_close(details["fused_logits"], expected)
        torch.testing.assert_close(probabilities[0, details["indices"], 0], expected.sigmoid())
        outside = torch.ones(args[2].shape[1], dtype=torch.bool)
        outside[details["indices"]] = False
        torch.testing.assert_close(probabilities[0, outside], args[5][0, outside], rtol=0, atol=0)
        sum(losses.values()).backward()
        self.assertIsNone(args[0].grad)
        self.assertIsNone(args[3].grad)
        self.assertGreater(model.scorer.visual.weight.grad.abs().sum().item(), 0)

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
        importance, ious = torch.tensor([2., 0., 99., 99.]), torch.tensor([.6, 0., .2, .5])
        loss, negative = model.counterfactual_loss(importance, ious, torch.zeros(4))
        torch.testing.assert_close(loss, model.localization_loss(importance, ious))
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
            loss, *_ = model.fusion_loss(branches["base_logits"], fused, ious)
            loss.backward()
            gradient = model.gate.bias.grad.item()
            self.assertGreater(gradient, 0) if correct else self.assertLess(gradient, 0)
            self.assertGreater(drops.grad.abs().sum().item(), 0)
            self.assertGreater(alpha.item(), 0)

    def test_residual_fusion_uses_original_logits_not_probability_mixture(self):
        model = self.module(gate_initial_probability=.99)
        base = torch.tensor([8., 0.])
        drops = torch.tensor([0., 10.], requires_grad=True)
        fused, alpha, branches = model.fusion(base, drops, torch.tensor(.8))
        self.assertGreater(alpha.item(), .5)
        self.assertEqual(fused.argmax().item(), 1)
        torch.testing.assert_close(fused, base + alpha * drops)
        zero_drop, _, _ = model.fusion(base, torch.zeros_like(drops), torch.tensor(.8))
        torch.testing.assert_close(zero_drop, base, rtol=0, atol=0)
        # Independent sigmoid scores need not sum to one; CRS is never replaced.
        self.assertGreater(fused.sigmoid().sum().item(), 1.9)
        mixture = (1-alpha) * base.softmax(0) + alpha * drops.softmax(0)
        self.assertFalse(torch.allclose(fused.sigmoid(), mixture))
        loss, *_ = model.fusion_loss(branches["base_logits"], fused, torch.tensor([0., .9]))
        loss.backward()
        self.assertGreater(drops.grad.abs().sum().item(), 0)
        self.assertFalse(model.cf_temperature.requires_grad)
        self.assertFalse(hasattr(model, "base_temperature"))
        self.assertFalse(hasattr(model, "comparator"))
        self.assertFalse(any("temperature" in name for name, _ in model.named_parameters()))

    def test_zero_intervention_end_to_end_preserves_crs_logits_and_order(self):
        model = self.module().eval()
        args = self.inputs()
        def no_mask(boxes, feature, *unused):
            return feature.new_zeros((len(boxes),) + feature.shape[-2:])
        with patch.object(model, "intervention_masks", side_effect=no_mask), torch.no_grad():
            probabilities, priority, _, info = model(*args)
        details = info["crv_samples"][0]
        torch.testing.assert_close(details["drops"], torch.zeros_like(details["drops"]), rtol=0, atol=0)
        torch.testing.assert_close(details["fused_logits"], details["base_logits"], rtol=0, atol=0)
        torch.testing.assert_close(probabilities, args[5])
        self.assertEqual(priority[0, :, 0].argsort(descending=True).tolist(),
                         args[5][0, :, 0].argsort(descending=True, stable=True).tolist())

    def test_extreme_scores_and_saturated_gates_stay_finite(self):
        for bias in (-1000., 1000.):
            model = self.module()
            model.gate.bias.data.fill_(bias)
            base, drops = torch.tensor([1000., -1000.]), torch.tensor([-2000., 2000.])
            fused, alpha, branches = model.fusion(base, drops,
                                                  torch.tensor(1.))
            self.assertTrue(torch.isfinite(fused).all())
            torch.testing.assert_close(fused, base + alpha * drops)
            self.assertTrue(torch.isfinite(fused.sigmoid()).all())
            self.assertEqual(fused.argmax().item(), 0 if bias < 0 else 1)
        self.assertGreater(model.cf_temperature.item(), 0)

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
        torch.testing.assert_close(b["base_logits"], a["base_logits"] + 10)
        self.assertEqual(sum(p.numel() for p in model.gate.parameters()), 2)
        alpha_a.backward()
        self.assertIsNone(entropy.grad)

    def test_threshold_is_strict_and_inactive_fusion_is_exact_base(self):
        model = self.module(gate_initial_probability=.99)
        for entropy in (.0, .59, .6):
            fused, alpha, branches = model.fusion(torch.tensor([2., 1.]), torch.tensor([0., 100.]),
                                                  torch.tensor(entropy))
            self.assertEqual(alpha.item(), 0)
            torch.testing.assert_close(fused, branches["base_logits"], rtol=0, atol=0)
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

    def test_shared_scorer_outputs_support_and_drop_has_correct_sign(self):
        model = self.module()
        scenes = torch.randn(2, 16, 6, 6)
        valid = torch.ones(2, 1, 6, 6)
        text = torch.randn(1, 6, 16).expand(2, -1, -1)
        text_mask = torch.zeros(2, 6, dtype=torch.bool)
        support = model.scorer(scenes, valid, text, text_mask)
        self.assertEqual(support.shape, (2,))
        repeated = model.scorer(scenes[:1].expand_as(scenes), valid, text, text_mask)
        torch.testing.assert_close(model.support_drop(repeated[:1], repeated[1:]), torch.zeros(1))
        torch.testing.assert_close(model.support_drop(torch.tensor([2., 2.]), torch.tensor([0., 1.5])),
                                   torch.tensor([2., .5]))
        original = torch.tensor([1., 1.], requires_grad=True)
        masked = torch.tensor([.8, .9], requires_grad=True)
        loss, _ = model.counterfactual_loss(model.support_drop(original, masked), torch.tensor([.8, 0.]),
                                           torch.zeros(2))
        loss.backward()
        self.assertLess(original.grad[0].item(), 0)
        self.assertGreater(masked.grad[0].item(), 0)

    def test_common_original_cancels_in_ranking_but_receives_anchor_gradients(self):
        model = self.module()
        masked = torch.randn(4, requires_grad=True)
        original = torch.randn((), requires_grad=True)
        loss = model.localization_loss(model.support_drop(original, masked), torch.tensor([.8, 0., 0., .2]))
        loss.backward()
        torch.testing.assert_close(original.grad, torch.zeros_like(original), rtol=0, atol=1e-6)
        original = torch.tensor(0., requires_grad=True)
        loss, _ = model.counterfactual_loss(model.support_drop(original, torch.zeros(2)),
                                           torch.tensor([.8, 0.]), torch.zeros(2))
        loss.backward()
        self.assertLess(original.grad.item(), 0)

    def test_direct_localization_uses_multiple_positives_and_ignores_unreachable_samples(self):
        model = self.module()
        scores = torch.tensor([.1, .2, -.2, .3], requires_grad=True)
        ious = torch.tensor([.5, .8, .2, 0.])
        expected = -torch.log(scores.softmax(0)[:2].sum())
        torch.testing.assert_close(model.localization_loss(scores, ious), expected)
        self.assertEqual(model.localization_loss(scores, torch.zeros(4)).item(), 0)
        self.assertEqual(model.localization_loss(scores, torch.ones(4)).item(), 0)
        model.localization_loss(scores, ious).backward()
        self.assertTrue((scores.grad[:2] < 0).all())
        self.assertTrue((scores.grad[2:] > 0).all())

    def test_direct_localization_gradient_does_not_depend_on_fusion_gate(self):
        model = self.module()
        model.gate.bias.data.fill_(-1000.)
        scenes = torch.randn(6, 16, 6, 6)
        support = model.scorer(scenes, torch.ones(6, 1, 6, 6), torch.randn(6, 6, 16),
                               torch.zeros(6, 6, dtype=torch.bool))
        importance = model.support_drop(support[:3], support[3:])
        loss = model.localization_loss(importance, torch.tensor([.8, 0., .3]))
        loss.backward()
        self.assertGreater(model.scorer.score[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.gate.parameters()))

    def test_warmup_is_exact_crs_and_adamw_cannot_update_crv(self):
        model = self.module(start_epoch=3, ready=False).train()
        args = self.inputs()
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        anchor = torch.nn.Parameter(torch.tensor(1.))
        for epoch in (0, 1):
            with patch.object(model.scorer, "forward", side_effect=AssertionError("Before start")), \
                 patch.object(model, "intervention_masks", side_effect=AssertionError("Before start")):
                probabilities, priority, losses, info = model(*args, epoch=epoch, refer_gt=[torch.ones(4)])
            torch.testing.assert_close(probabilities, args[5], rtol=0, atol=0)
            self.assertIsNone(priority)
            self.assertEqual(sum(loss.item() for loss in losses.values()), 0)
            self.assertEqual(info["crv_ready"].item(), 0)
            optimizer.zero_grad(set_to_none=True)
            (anchor.square() + sum(losses.values())).backward()
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            optimizer.step()
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
            model.eval()
            _, priority, _, info = model(*args)
            self.assertIsNone(priority)
            self.assertEqual(info["crv_ready"].item(), 0)
            model.train()
        _, priority, losses, info = model(*args, epoch=2, refer_gt=[torch.tensor([.09, .45, .15, .55])])
        self.assertIsNotNone(priority)
        self.assertEqual(info["crv_ready"].item(), 1)
        sum(losses.values()).backward()
        self.assertGreater(model.scorer.visual.weight.grad.abs().sum().item(), 0)

    def test_start_epoch_validation_and_checkpointed_inference_state(self):
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.module(start_epoch=value)
        model = self.module(start_epoch=3, ready=False)
        with self.assertRaisesRegex(ValueError, "requires the zero-based training epoch"):
            model(*self.inputs())
        self.assertFalse(model.stage_ready(1))
        restored = self.module(start_epoch=3, ready=False).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(restored.stage_ready())
        self.assertTrue(model.stage_ready(2))
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(restored.stage_ready())

    def test_head_forwards_epoch_and_validation_waits_for_crv_start(self):
        options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        options.update(num_queries=8, enable_crv=True, enable_bcem=False, enable_grab=False,
                       crv=dict(scorer_dim=16, scorer_heads=4, grid_size=4, topk=5, start_epoch=3))
        head = build_head(options)
        x = torch.randn(1, 768, 8, 8)
        args = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                    lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128))
        targets = dict(epoch=1, mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]])],
                       img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
        with patch.object(head.crv.scorer, "forward", side_effect=AssertionError("Before start")):
            losses, prediction, info = head.forward_train(x, targets, **args)
            self.assertEqual(info["crv_ready"].item(), 0)
            self.assertNotIn("ranking_priority", prediction)
            head.eval()
            with torch.no_grad():
                prediction, info = head.forward_test(x, targets=dict(img_metas=targets["img_metas"]),
                                                     with_mask=False, **args)
            self.assertNotIn("ranking_priority", prediction)
            self.assertEqual(info["crv_ready"].item(), 0)
        head.train()
        targets["epoch"] = 2
        losses, prediction, info = head.forward_train(x, targets, **args)
        self.assertEqual(info["crv_ready"].item(), 1)
        self.assertEqual(head.crv.last_training_epoch.item(), 3)
        self.assertIn("ranking_priority", prediction)
        self.assertIn("metric_crv_ready", losses)

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
            scores = torch.arange(len(base), dtype=base.dtype)
            return scores, base.new_tensor(.5), dict(base_logits=base, gate_statistics=torch.zeros(1))
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

    def test_low_entropy_training_skips_crv_and_adamw_does_not_update_it(self):
        model = self.module().train()
        args = self.inputs()
        args[5] = torch.tensor([[[.99], [.1], [.09], [.08], [.07], [.06], [.05], [.04]]])
        args[6] = torch.logit(args[5])
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.1)
        with patch.object(model.scorer, "forward", side_effect=AssertionError("Low entropy")), \
             patch.object(model.scorer.visual, "forward", side_effect=AssertionError("Low entropy")), \
             patch.object(model, "intervention_masks", side_effect=AssertionError("Low entropy")):
            probabilities, _, losses, info = model(*args, refer_gt=[torch.tensor([.09, .45, .15, .55])])
        self.assertEqual(info["crv_active"].item(), 0)
        self.assertEqual(info["crv_alpha"].item(), 0)
        self.assertEqual(losses["loss_crv_fusion"].item(), 0)
        self.assertEqual(losses["loss_crv_cf"].item(), 0)
        torch.testing.assert_close(probabilities, args[5], rtol=0, atol=0)
        anchor = torch.nn.Parameter(torch.tensor(1.))
        (anchor.square() + sum(losses.values())).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        optimizer.step()
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)

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
        for version in (1, 2, 3, 4, 5, 6, 7, 8):
            with self.assertRaisesRegex(ValueError, "compatible CRV checkpoint"):
                _validate_crv_checkpoint(model, dict(state, **{"head.crv.schema_version": torch.tensor(version)}))
        model.head.crv = None
        _validate_crv_checkpoint(model, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
