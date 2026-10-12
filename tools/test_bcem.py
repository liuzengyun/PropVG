"""CPU regressions for sector selection, nonlinear fusion and CRS integration.

Run from the repository root: PYTHONPATH=. python tools/test_bcem.py
"""
import copy
import math
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")

import torch
from mmcv import Config

from propvg.models import build_head
from propvg.models.heads.background_context import BackgroundContextEnhancement


class BCEMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)

    def module(self, **kwargs):
        options = dict(hidden_channels=16, context_dim=8, chunk_size=2)
        options.update(kwargs)
        return BackgroundContextEnhancement(**options)

    def inputs(self, batch=2, count=5):
        visual = torch.randn(batch, 16, 20, 20)
        boxes = torch.tensor([[[.5, .5, .12, .1]] * count] * batch)
        queries = torch.randn(batch, count, 16)
        text = torch.randn(batch, 7, 16)
        mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1]] * batch)
        metas = [dict(img_shape=(80, 100), pad_shape=(100, 100)) for _ in range(batch)]
        return [visual, boxes, queries, text, mask, metas]

    def test_sectors_span_from_box_boundary_to_image_edge(self):
        model = self.module()
        boxes = torch.tensor([[[.5, .5, .2, .2]]], requires_grad=True)
        points, area, _, _, bounds = model._sampling_geometry(boxes, torch.tensor([[100., 200.]]))
        # Center (100,50): top boundary y=40 to y=0; right x=120 to x=200.
        # The middle sub-ray samples radial cell centers, including near context.
        torch.testing.assert_close(points[0, 0, 0, 10:15],
                                   torch.tensor([[100., 36.], [100., 28.], [100., 20.],
                                                 [100., 12.], [100., 4.]]))
        torch.testing.assert_close(points[0, 0, 3, 10:15],
                                   torch.tensor([[128., 50.], [144., 50.], [160., 50.],
                                                 [176., 50.], [192., 50.]]))
        inside = ((points >= bounds[:, :, None, None, :2]) &
                  (points <= bounds[:, :, None, None, 2:])).all(-1)
        self.assertFalse(inside.any())
        self.assertEqual(tuple(points.shape), (1, 1, 8, 25, 2))
        self.assertTrue((area > 0).all())
        self.assertFalse(points.requires_grad)

    def test_every_subray_reaches_an_image_side_on_a_rectangular_offcenter_image(self):
        model = self.module()
        boxes = torch.tensor([[[.3, .65, .1, .2]]])
        points, _, _, _, bounds = model._sampling_geometry(boxes, torch.tensor([[100., 300.]]))
        cells = points.reshape(1, 1, 8, 5, 5, 2)
        step = cells[..., 1, :] - cells[..., 0, :]
        starts = cells[..., 0, :] - step / 2
        ends = cells[..., -1, :] + step / 2
        box_sides = torch.cat(((starts - bounds[:, :, None, None, :2]).abs(),
                               (starts - bounds[:, :, None, None, 2:]).abs()), -1)
        self.assertLess(box_sides.min(-1).values.max().item(), 1e-4)
        image_sides = torch.cat((ends.abs(), (ends - torch.tensor([300., 100.])).abs()), -1)
        self.assertLess(image_sides.min(-1).values.max().item(), 1e-4)
        self.assertTrue((points >= 0).all())
        self.assertTrue((points < torch.tensor([300., 100.])).all())

    def test_area_weights_include_the_different_radial_spans(self):
        model = self.module()
        points, area, _, _, _ = model._sampling_geometry(
            torch.tensor([[[.5, .5, .2, .2]]]), torch.tensor([[100., 200.]]))
        # Central top/right rays have spans 40/80 and first radii 14/28.
        torch.testing.assert_close(area[0, 0, 0, 10], torch.tensor(14. * 40.))
        torch.testing.assert_close(area[0, 0, 3, 10], torch.tensor(28. * 80.))
        # Right-hand area is four times the top-hand area in this geometry.
        torch.testing.assert_close(area[0, 0, 3, 10] / area[0, 0, 0, 10], torch.tensor(4.))

    def test_box_covering_image_has_no_exterior_area_and_returns_identity(self):
        args = self.inputs(batch=1, count=1)
        args[1] = torch.tensor([[[.5, .5, 1., 1.]]])
        output, info = self.module()(*args)
        self.assertFalse(info["bcem_valid_sectors"].any())
        self.assertEqual(info["bcem_context"].count_nonzero(), 0)
        torch.testing.assert_close(output, args[2], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())

    def test_area_pooling_matches_affine_field_at_the_valid_centroid(self):
        model = self.module()
        model.visual_norm = torch.nn.Identity()
        yy, xx = torch.meshgrid(torch.arange(50.) * 2 + 1, torch.arange(100.) * 2 + 1, indexing="ij")
        visual = torch.stack([((i + 1) * xx + i * yy) / 100 for i in range(8)])[None]
        boxes = torch.tensor([[[.5, .5, .2, .2]]])
        sizes = torch.tensor([[100., 200.]])
        features, position, valid, coverage = model._sample_sectors(
            visual, boxes, sizes, sizes, torch.zeros(1, 50, 100, dtype=torch.bool))
        centroid = position[0, 0, :, 2:] * 100 + torch.tensor([100., 50.])
        expected = torch.stack([((i + 1) * centroid[:, 0] + i * centroid[:, 1]) / 100 for i in range(8)], -1)
        torch.testing.assert_close(features[0, 0], expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(valid.all())
        torch.testing.assert_close(coverage, torch.ones_like(coverage))

    def test_padding_canvas_does_not_change_physical_evidence(self):
        model = self.module().eval()
        yy, xx = torch.meshgrid(torch.arange(10.), torch.arange(20.), indexing="ij")
        visual = torch.stack([(xx + 1) / 20 + (i + 1) * (yy + 1) / 100 for i in range(16)])[None]
        padded = torch.full((1, 16, 20, 40), 1e5)
        padded[:, :, :10, :20] = visual
        boxes = torch.tensor([[[.5, .5, .1, .1], [.35, .4, .1, .1]]])
        queries, text = torch.randn(1, 2, 16), torch.randn(1, 5, 16)
        mask = torch.tensor([[0, 0, 0, 1, 1]])
        with torch.no_grad():
            expected, a = model(visual, boxes, queries, text, mask,
                                [dict(img_shape=(100, 200), pad_shape=(100, 200))])
            actual, b = model(padded, boxes, queries, text, mask,
                              [dict(img_shape=(100, 200), pad_shape=(200, 400))])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b["bcem_context"], a["bcem_context"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b["bcem_valid_sectors"], a["bcem_valid_sectors"])
        torch.testing.assert_close(b["bcem_direction_weights"], a["bcem_direction_weights"], rtol=1e-5, atol=1e-6)

    def test_invalid_environment_is_exact_identity_even_with_learned_biases(self):
        model = self.module().eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("bias"):
                    parameter.fill_(2.)
        args = self.inputs()
        output, info = model(*args, visual_mask=torch.ones(2, 20, 20, dtype=torch.bool))
        torch.testing.assert_close(output, args[2], rtol=0, atol=0)
        for key in ("bcem_context", "bcem_residual_ratio", "bcem_direction_weights", "bcem_background_weight",
                    "bcem_text_sector_attention"):
            self.assertEqual(info[key].count_nonzero(), 0)
        self.assertNotIn("bcem_candidate_weight", info)
        self.assertFalse(info["bcem_valid_sectors"].any())

    def test_image_edge_and_empty_boxes_are_masked(self):
        args = self.inputs(batch=1, count=2)
        args[1] = torch.tensor([[[.02, .02, .1, .1], [.5, .5, 0., .1]]])
        output, info = self.module()(*args)
        valid = info["bcem_valid_sectors"]
        self.assertFalse(valid[0, 0, 0])  # Up outside image.
        self.assertFalse(valid[0, 0, 2])  # Left outside image.
        self.assertFalse(valid[0, 1].any())
        torch.testing.assert_close(output[0, 1], args[2][0, 1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())

    def test_text_padding_is_ignored_for_integer_and_bool_masks(self):
        model = self.module().eval()
        args = self.inputs()
        expected, a = model(*args)
        args[3] = args[3].masked_fill(args[4].bool()[..., None], 1e6)
        actual, b = model(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(a["bcem_direction_weights"], b["bcem_direction_weights"], rtol=0, atol=0)
        args[4] = args[4].bool()
        boolean, _ = model(*args)
        torch.testing.assert_close(boolean, actual, rtol=0, atol=0)

    def test_valid_text_changes_background_usage(self):
        model = self.module().eval()
        args = self.inputs()
        expected, a = model(*args)
        args[3] = args[3] + 2
        actual, b = model(*args)
        self.assertGreater((a["bcem_direction_weights"] - b["bcem_direction_weights"]).abs().max().item(), 1e-5)
        self.assertGreater((expected - actual).abs().max().item(), 1e-5)

    def test_all_text_padding_produces_zero_context(self):
        args = self.inputs()
        args[4] = torch.ones_like(args[4])
        output, info = self.module()(*args)
        torch.testing.assert_close(output, args[2], rtol=0, atol=0)
        self.assertEqual(info["bcem_context"].count_nonzero(), 0)
        self.assertEqual(info["bcem_direction_weights"].count_nonzero(), 0)
        self.assertEqual(info["bcem_text_sector_attention"].count_nonzero(), 0)

    def test_direction_weights_normalize_valid_sectors_and_handle_empty_sectors(self):
        model = self.module(use_position=False)
        features = torch.ones(1, 2, 8, 8)
        position = torch.zeros(1, 2, 8, 4)
        valid = torch.ones(1, 2, 8, dtype=torch.bool)
        valid[0, 0, 4:] = False
        valid[0, 1] = False
        tokens, text_valid = torch.zeros(1, 1, 8), torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad():
            context, weights, _ = model._read_sectors(features, position, valid, tokens, text_valid)
        torch.testing.assert_close(weights[0, 0, :4], torch.full((4,), .25))
        one_valid = valid.clone()
        one_valid[0, 0, 1:] = False
        single, _, _ = model._read_sectors(features, position, one_valid, tokens, text_valid)
        torch.testing.assert_close(context[0, 0], single[0, 0])
        self.assertEqual(weights[0, 0, 4:].count_nonzero(), 0)
        self.assertEqual(context[0, 1].count_nonzero(), 0)
        self.assertEqual(weights[0, 1].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(context).all())

    def test_sparse_selection_can_use_all_one_two_or_other_numbers_of_sectors(self):
        relevance = torch.tensor([[.125] * 8,
                                  [.7] + [.3 / 7] * 7,
                                  [.45, .45] + [.1 / 6] * 6,
                                  [1 / 3] * 3 + [0.] * 5,
                                  [.25] * 4 + [0.] * 4])
        weights = self.module()._sparse_sector_weights(relevance, torch.ones_like(relevance, dtype=torch.bool))
        torch.testing.assert_close((weights > 0).sum(-1), torch.tensor([8, 1, 2, 3, 4]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(5))
        torch.testing.assert_close(weights[0], relevance[0])
        torch.testing.assert_close(weights[1], torch.tensor([1.] + [0.] * 7))
        torch.testing.assert_close(weights[2], torch.tensor([.5, .5] + [0.] * 6))

    def test_sparse_selection_excludes_invalid_sectors_and_empty_rows(self):
        relevance = torch.tensor([[.5, .5] + [0.] * 6, [0.] * 8])
        selectable = torch.tensor([[True, True] + [False] * 6, [False] * 8])
        weights = self.module()._sparse_sector_weights(relevance, selectable)
        torch.testing.assert_close(weights[0], relevance[0])
        self.assertEqual(weights[1].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(weights).all())

    def test_sparse_selection_gradients_match_finite_differences(self):
        # Stay away from support boundaries while keeping three active sectors.
        logits = torch.tensor([[.4, .35, .20] + [.01] * 5], dtype=torch.double).log().requires_grad_()
        valid = torch.ones_like(logits, dtype=torch.bool)
        selector = self.module()._sparse_sector_weights
        self.assertTrue(torch.autograd.gradcheck(lambda x: selector(x.softmax(-1), valid), (logits,)))
        (selector(logits.softmax(-1), valid) * torch.arange(8., dtype=torch.double)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_excluded_sector_values_cannot_change_sparse_background(self):
        model = self.module(use_position=False).eval()
        model.sector_fusion = torch.nn.Identity()
        model.sector_value = torch.nn.Identity()
        query = torch.nn.functional.layer_norm(torch.tensor([1., -1.] + [0.] * 6), (8,))
        keys = -query.expand(1, 1, 8, 8).clone()
        keys[..., 0, :] = query

        class FixedKeys(torch.nn.Module):
            def forward(self, features):
                return keys

        # Isolate the Value path so selection stays identical across the pair.
        model.sector_key = FixedKeys()
        features, position = torch.randn(1, 1, 8, 8), torch.zeros(1, 1, 8, 4)
        valid, text_valid = torch.ones(1, 1, 8, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)
        tokens = query[None, None]
        first, weights, _ = model._read_sectors(features, position, valid, tokens, text_valid)
        changed = features.clone()
        changed[..., 1:, :] = torch.randn_like(changed[..., 1:, :]) * 100
        other, _, _ = model._read_sectors(changed, position, valid, tokens, text_valid)
        torch.testing.assert_close(weights, torch.tensor([[[1.] + [0.] * 7]]))
        torch.testing.assert_close(first, other, rtol=0, atol=0)
        model.use_sparse_selection = False
        dense_first, _, _ = model._read_sectors(features, position, valid, tokens, text_valid)
        dense_other, _, _ = model._read_sectors(changed, position, valid, tokens, text_valid)
        self.assertGreater((dense_first - dense_other).abs().max().item(), 1e-6)

    def test_sparse_diagnostics_report_the_sectors_actually_used(self):
        _, info = self.module()(*self.inputs())
        weights, valid = info["bcem_direction_weights"], info["bcem_valid_sectors"]
        torch.testing.assert_close(info["bcem_selected_sectors"], weights > 0)
        torch.testing.assert_close(info["bcem_selected_sector_count"], (weights > 0).sum(-1))
        self.assertFalse((info["bcem_selected_sectors"] & ~valid).any())
        torch.testing.assert_close(info["bcem_direction_weights_dense"].sum(-1), torch.ones(2, 5))

    def test_candidate_conditions_final_fusion_after_text_reads_sectors(self):
        model = self.module().eval()
        args = self.inputs()
        _, before = model(*args)
        args[2] = torch.randn_like(args[2])
        _, after = model(*args)
        for key in ("bcem_context", "bcem_direction_weights", "bcem_text_sector_attention"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertGreater((before["bcem_background_weight"] - after["bcem_background_weight"]).abs().max().item(), 1e-6)

    def test_position_selects_directions_with_identical_visual_text_and_candidate_features(self):
        model = self.module(use_position=True).eval()
        features = torch.ones(1, 1, 8, 8)
        tokens, text_valid = torch.randn(1, 1, 8), torch.ones(1, 1, dtype=torch.bool)
        valid = torch.ones(1, 1, 8, dtype=torch.bool)
        position = torch.zeros(1, 1, 8, 4)
        _, uniform, _ = model._read_sectors(features, position, valid, tokens, text_valid)
        torch.testing.assert_close(uniform, torch.full((1, 1, 8), 1 / 8))
        position[0, 0, :, :2] = model.directions[:, [1, 0]]
        _, directed, _ = model._read_sectors(features, position, valid, tokens, text_valid)
        self.assertGreater((directed - uniform).abs().max().item(), 1e-6)
        permutation = torch.tensor([4, 2, 0, 3, 1, 7, 6, 5])
        _, reordered, _ = model._read_sectors(features, position[:, :, permutation], valid, tokens, text_valid)
        torch.testing.assert_close(reordered, directed[:, :, permutation])

    def test_default_selects_visual_content_without_explicit_sector_position(self):
        model = self.module().eval()
        self.assertFalse(model.use_position)
        self.assertIsNone(model.position_mlp)
        self.assertFalse(any(name.startswith("position_mlp") for name, _ in model.named_parameters()))
        features, position = torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 4)
        valid, text_valid = torch.ones(1, 1, 8, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)
        tokens = torch.randn(1, 3, 8)
        first, weights, attention = model._read_sectors(features, position, valid, tokens, text_valid)
        other, other_weights, other_attention = model._read_sectors(
            features, position * -1000, valid, tokens, text_valid)
        for before, after in ((first, other), (weights, other_weights), (attention, other_attention)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        permutation = torch.tensor([4, 2, 0, 3, 1, 7, 6, 5])
        reordered, reordered_weights, _ = model._read_sectors(
            features[:, :, permutation], position, valid, tokens, text_valid)
        torch.testing.assert_close(reordered_weights, weights[:, :, permutation])
        torch.testing.assert_close(reordered, first)

    def test_text_queries_select_matching_visual_sectors_and_mask_padding(self):
        model = self.module(use_position=False).eval()
        model.sector_fusion = torch.nn.Identity()
        model.sector_key = torch.nn.Identity()
        model.sector_value = torch.nn.Identity()
        vector = torch.tensor([1., -1., 0., 0., 0., 0., 0., 0.])
        features = torch.zeros(1, 1, 8, 8)
        features[0, 0, 0], features[0, 0, 1] = vector, -vector
        query = torch.nn.functional.layer_norm(vector, (8,))
        tokens = torch.stack([query, -query, query * 1e6])[None]
        valid = torch.zeros(1, 1, 8, dtype=torch.bool)
        valid[..., :2] = True
        context, weights, attention = model._read_sectors(
            features, torch.zeros(1, 1, 8, 4), valid, tokens, torch.tensor([[True, True, False]]))
        self.assertGreater(attention[0, 0, 0, 0].item(), .99)
        self.assertGreater(attention[0, 0, 1, 1].item(), .99)
        self.assertEqual(attention[..., 2:].count_nonzero(), 0)
        self.assertEqual(attention[:, :, 2].count_nonzero(), 0)
        torch.testing.assert_close(attention.sum(-1), torch.tensor([[[1., 1., 0.]]]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1))
        self.assertTrue(torch.isfinite(context).all())

    def test_uniform_attention_keeps_visual_context_but_text_conditions_final_gate(self):
        model = self.module(use_text_attention=False).eval()
        args = self.inputs()
        _, before = model(*args)
        args[3] = args[3] + 2
        _, after = model(*args)
        for key in ("bcem_direction_weights", "bcem_context", "bcem_text_sector_attention"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertGreater((before["bcem_background_weight"] - after["bcem_background_weight"]).abs().max().item(), 1e-6)

    def test_single_valid_sector_returns_visual_values_without_text_gain(self):
        model = self.module(use_position=False).eval()
        features, position = torch.randn(1, 1, 8, 8), torch.zeros(1, 1, 8, 4)
        valid = torch.zeros(1, 1, 8, dtype=torch.bool)
        valid[..., 0] = True
        text_valid = torch.tensor([[True, True, False]])
        first, weights, attention = model._read_sectors(features, position, valid,
                                                       torch.randn(1, 3, 8), text_valid)
        other, _, _ = model._read_sectors(features, position, valid,
                                         torch.randn(1, 3, 8) * 100, text_valid)
        torch.testing.assert_close(first, other, rtol=0, atol=0)
        torch.testing.assert_close(weights[..., 0], torch.ones(1, 1))
        torch.testing.assert_close(attention[..., 0], text_valid[:, None].float())

    def test_background_input_gate_initialization_has_no_complementary_candidate_weight(self):
        args = self.inputs()
        _, info = self.module()(*args)
        gate = info["bcem_background_weight"]
        self.assertLess((gate - .1).abs().max().item(), .005)
        self.assertNotIn("bcem_candidate_weight", info)
        torch.testing.assert_close(info["bcem_direction_weights"].sum(-1), torch.ones(2, 5))

    def test_concatenation_preserves_candidate_input_and_gates_only_background(self):
        model = self.module()
        context, queries, sentence = torch.randn(1, 2, 8), torch.randn(1, 2, 16), torch.randn(1, 8)
        usable = torch.tensor([[True, False]])
        raw = model.environment_projection(context)
        rms = queries.float().norm(dim=-1, keepdim=True) / math.sqrt(16)
        background = torch.nn.functional.layer_norm(raw, (16,)) * rms
        self.assertLess((background.square().mean(-1).sqrt() - rms.squeeze(-1)).abs().max().item(), 1e-3)
        supplied = []
        handle = model.feature_fusion.register_forward_pre_hook(
            lambda module, args: supplied.append(args[0].detach().clone()))
        outputs = []
        with torch.no_grad():
            for parameter in model.fusion_gate.parameters():
                parameter.zero_()
            for weight in (.2, .8):
                model.fusion_gate[-1].bias.fill_(math.log(weight / (1 - weight)))
                output, gate = model._fuse_context(context, queries, sentence, usable)
                # Increasing background use never multiplies the candidate by 1-g.
                torch.testing.assert_close(supplied[-1][..., :16], queries, rtol=0, atol=0)
                torch.testing.assert_close(supplied[-1][0, 0, 16:], weight * background[0, 0])
                self.assertEqual(supplied[-1][0, 1, 16:].count_nonzero(), 0)
                torch.testing.assert_close(output[0, 1], queries[0, 1], rtol=0, atol=0)
                torch.testing.assert_close(gate, torch.tensor([[[weight], [0.]]]))
                self.assertEqual(output.shape, queries.shape)
                outputs.append(output)
        handle.remove()
        self.assertGreater((outputs[0][0, 0] - outputs[1][0, 0]).abs().max().item(), 1e-6)

    def test_global_gate_can_ignore_background_with_normalized_direction_weights(self):
        model = self.module().eval()
        args = self.inputs()
        with torch.no_grad():
            for parameter in model.fusion_gate.parameters():
                parameter.zero_()
            model.fusion_gate[-1].bias.fill_(-1000)
            output, info = model(*args)
            sentence = model.text_projection(args[3][:, :4]).mean(1)
            changed, _ = model._fuse_context(torch.randn(2, 5, 8) * 100, args[2], sentence,
                                             info["bcem_valid_sectors"].any(-1))
        torch.testing.assert_close(info["bcem_direction_weights"].sum(-1), torch.ones(2, 5))
        self.assertLess(info["bcem_background_weight"].max().item(), 1e-12)
        # A zero background gate eliminates dependence on context; the candidate
        # still passes through the learned MLP on geometrically valid rows.
        torch.testing.assert_close(output, changed, rtol=0, atol=0)

    def test_chunk_size_and_candidate_permutation_preserve_results(self):
        model = self.module().eval()
        args = self.inputs()
        expected, a = model(*args)
        model.chunk_size = 64
        actual, b = model(*args)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b["bcem_context"], a["bcem_context"], rtol=1e-5, atol=1e-6)
        permutation = torch.tensor([4, 2, 0, 3, 1])
        args[1], args[2] = args[1][:, permutation], args[2][:, permutation]
        permuted, _ = model(*args)
        torch.testing.assert_close(permuted, actual[:, permutation], rtol=1e-5, atol=1e-6)

    def test_gradients_reach_visual_text_attention_and_fusion_not_sampling_boxes(self):
        model = self.module()
        args = self.inputs()
        for index in (0, 1, 2, 3):
            args[index].requires_grad_()
        output, info = model(*args)
        output.square().mean().backward()
        for index in (0, 2, 3):
            self.assertTrue(torch.isfinite(args[index].grad).all())
            self.assertGreater(args[index].grad.abs().sum().item(), 0)
        self.assertIsNone(args[1].grad)
        for name in ("visual_projection.weight",
                     "text_projection.weight", "sector_key.weight", "sector_value.weight", "sector_fusion.0.weight",
                     "fusion_gate.0.weight", "fusion_gate.2.weight", "environment_projection.weight",
                     "feature_fusion.0.weight", "feature_fusion.2.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertTrue(all(not tensor.requires_grad for tensor in info.values()))
        self.assertTrue(((info["bcem_direction_weights"] >= 0) & (info["bcem_direction_weights"] <= 1)).all())

    def test_ablations_remove_inactive_parameters(self):
        args = self.inputs()
        for options in (dict(use_position=True), dict(use_position=False),
                        dict(use_text_attention=False), dict(use_sparse_selection=False),
                        dict(use_learnable_fusion=False),
                        dict(use_text_attention=False, use_learnable_fusion=False)):
            with self.subTest(options=options):
                model = self.module(**options)
                output, info = model(*args)
                self.assertTrue(torch.isfinite(output).all())
                if options.get("use_position") is False:
                    self.assertIsNone(model.position_mlp)
                if options.get("use_position") is True:
                    self.assertIsNotNone(model.position_mlp)
                if options.get("use_text_attention") is False and options.get("use_learnable_fusion") is False:
                    self.assertIsNone(model.text_projection)
                    args_copy = list(args)
                    args_copy[3] = args[3] + 100
                    other, _ = model(*args_copy)
                    torch.testing.assert_close(output, other, rtol=0, atol=0)
                if options.get("use_text_attention") is False:
                    self.assertIsNone(model.sector_key)
                    weights = info["bcem_valid_sectors"].float()
                    torch.testing.assert_close(info["bcem_direction_weights"], weights / weights.sum(-1, keepdim=True))
                if options.get("use_learnable_fusion") is False:
                    self.assertIsNone(model.fusion_gate)
                    self.assertIsNotNone(model.feature_fusion)
                    torch.testing.assert_close(info["bcem_background_weight"], torch.full((2, 5, 1), .1))

    def test_cpu_autocast_is_finite(self):
        model = self.module()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output, _ = model(*self.inputs())
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_empty_candidate_axis(self):
        args = self.inputs()
        args[1], args[2] = args[1][:, :0], args[2][:, :0]
        output, info = self.module()(*args)
        self.assertEqual(output.shape, args[2].shape)
        self.assertEqual(info["bcem_valid_sectors"].shape, (2, 0, 8))
        self.assertEqual(info["bcem_background_weight"].shape, (2, 0, 1))
        self.assertEqual(info["bcem_text_sector_attention"].shape, (2, 0, 7, 8))


class BCEMHeadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        # Integration checks exercise BCEM even when the user's experiment config
        # currently selects a baseline run.
        cls.options["enable_bcem"] = True
        cls.options["num_queries"] = 8
        cls.options["bcem"].update(context_dim=16, chunk_size=3)

    def inputs(self):
        torch.manual_seed(572)
        x = torch.randn(1, 768, 8, 8)
        targets = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]])],
                       img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
        kwargs = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                      lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128))
        return x, targets, kwargs

    def test_enabled_branch_trains_and_reaches_crs_in_train_and_test(self):
        torch.manual_seed(103)
        head = build_head(self.options)
        self.assertFalse(head.bcem.use_position)
        self.assertIsNone(head.bcem.position_mlp)
        x, targets, kwargs = self.inputs()
        seen = []
        handle = head.mrsm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        losses, train_predictions, train_info = head.forward_train(x, targets, **kwargs)
        sum(losses.values()).backward()
        gradient = head.bcem.visual_projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        for name in ("text_projection.weight", "sector_key.weight", "sector_value.weight", "fusion_gate.0.weight",
                     "feature_fusion.0.weight", "feature_fusion.2.weight"):
            gradient = dict(head.bcem.named_parameters())[name].grad
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertEqual(train_info["bcem_direction_weights"].shape, (1, 8, 8))
        self.assertEqual(train_info["bcem_background_weight"].shape, (1, 8, 1))
        head.eval()
        with torch.no_grad():
            predictions, test_info = head.forward_test(x, targets=targets, with_mask=False, **kwargs)
        handle.remove()
        self.assertEqual(len(seen), 2)
        self.assertEqual(predictions["refer_label"].shape, (1, 8, 1))
        self.assertEqual(test_info["bcem_valid_sectors"].shape, (1, 8, 8))
        self.assertTrue(torch.isfinite(train_predictions["refer_label"]).all())

    def test_background_enhancement_preserves_proposal_boxes_and_selection_indices(self):
        for grab in (False, True):
            with self.subTest(grab=grab):
                options = copy.deepcopy(self.options)
                options["enable_grab"] = grab
                torch.manual_seed(109)
                head = build_head(options).eval()
                x, targets, kwargs = self.inputs()
                with torch.no_grad():
                    enhanced, info = head.forward_test(x, targets=targets, with_mask=False, **kwargs)
                    head.enable_bcem = False
                    baseline, _ = head.forward_test(x, targets=targets, with_mask=False, **kwargs)
                torch.testing.assert_close(enhanced["pred_bbox_allobj"], baseline["pred_bbox_allobj"], rtol=0, atol=0)
                torch.testing.assert_close(enhanced["pred_class_allobj"], baseline["pred_class_allobj"], rtol=0, atol=0)
                self.assertEqual(info["bcem_direction_weights"].shape[1], enhanced["pred_bbox_allobj"].shape[1])
                self.assertFalse(torch.equal(enhanced["refer_label"], baseline["refer_label"]))

    def test_disabled_branch_preserves_baseline_parameters_and_has_no_weights(self):
        options = copy.deepcopy(self.options)
        options["enable_bcem"] = False
        torch.manual_seed(117)
        baseline = build_head(options)
        baseline_rng = torch.get_rng_state()
        options.pop("bcem")
        options.pop("enable_bcem")
        torch.manual_seed(117)
        default = build_head(options)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        self.assertIsNone(default.bcem)
        self.assertFalse(any("bcem" in name for name, _ in default.named_parameters()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(default.state_dict()[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
