"""CPU checks for sector geometry, radial recurrence, sparse/null selection and CRS.

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
        options = dict(hidden_channels=16, context_dim=8, chunk_size=2, checkpoint_chunks=False)
        options.update(kwargs)
        return BackgroundContextEnhancement(**options)

    def inputs(self, batch=2, count=5):
        return [torch.randn(batch, 16, 20, 20),
                torch.tensor([[[.5, .5, .12, .1]] * count] * batch),
                torch.randn(batch, count, 16), torch.randn(batch, 7, 16),
                torch.tensor([[0, 0, 0, 0, 1, 1, 1]] * batch),
                [dict(img_shape=(80, 100), pad_shape=(100, 100)) for _ in range(batch)]]

    def evidence(self):
        return (torch.randn(1, 2, 8, 7, 8), torch.ones(1, 2, 8, 7, dtype=torch.bool),
                torch.randn(1, 3, 8), torch.tensor([[True, True, False]]))

    def test_radial_region_geometry_from_box_to_image_edge(self):
        model = self.module()
        boxes = torch.tensor([[[.5, .5, .2, .2]]], requires_grad=True)
        points, area, _, _, bounds, widths = model._sampling_geometry(boxes, torch.tensor([[100., 200.]]))
        torch.testing.assert_close(points[0, 0, 0, :, 4:6, 1],
                                   torch.tensor([[39., 37.], [34.5, 31.5], [27.5, 22.5], [15., 5.]]))
        torch.testing.assert_close(points[0, 0, 3, :, 4:6, 0],
                                   torch.tensor([[122., 126.], [131., 137.], [145., 155.], [170., 190.]]))
        inside = ((points >= bounds[:, :, None, None, None, :2]) &
                  (points <= bounds[:, :, None, None, None, 2:])).all(-1)
        self.assertFalse(inside.any())
        self.assertEqual(tuple(points.shape), (1, 1, 8, 4, 10, 2))
        self.assertTrue((area > 0).all())
        self.assertFalse(points.requires_grad)
        torch.testing.assert_close(widths[0, 0, 0, :, 4], torch.tensor([4., 6., 10., 20.]))
        torch.testing.assert_close(area[0, 0, 0, 0, 4], torch.tensor(11. * 4.))
        torch.testing.assert_close(area[0, 0, 3, 0, 4] / area[0, 0, 0, 0, 4], torch.tensor(4.))

    def test_default_eight_preserves_historical_angles_order_and_state_dict(self):
        torch.manual_seed(29)
        default = self.module().eval()
        torch.manual_seed(29)
        explicit = self.module(num_sectors=8).eval()
        angles = torch.tensor([-math.pi / 2, math.pi / 2, math.pi, 0,
                               -3 * math.pi / 4, -math.pi / 4, 3 * math.pi / 4, math.pi / 4])
        offsets = ((torch.arange(5) + .5) / 5 - .5) * (math.pi / 4)
        rays = torch.stack(((angles[:, None] + offsets).cos(), (angles[:, None] + offsets).sin()), -1)
        torch.testing.assert_close(default.sector_rays, rays, rtol=0, atol=0)
        self.assertEqual(default.direction_names, ("up", "down", "left", "right", "upper_left",
                                                   "upper_right", "lower_left", "lower_right"))
        for key in default.state_dict():
            torch.testing.assert_close(default.state_dict()[key], explicit.state_dict()[key], rtol=0, atol=0)
        self.assertNotIn("sector_angles", default.state_dict())
        self.assertNotIn("sector_rays", default.state_dict())
        explicit.load_state_dict(default.state_dict(), strict=True)
        args = self.inputs()
        with torch.no_grad():
            expected, a = default(*args)
            actual, b = explicit(*args)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_configurable_sector_geometry_shapes_gradients_and_empty_candidates(self):
        args = self.inputs(batch=1, count=3)
        old_state = self.module().state_dict()
        for count in (1, 2, 3, 4, 6, 12, 16):
            with self.subTest(num_sectors=count):
                model = self.module(num_sectors=count)
                # Parameters are shared across sectors; their names/shapes stay fixed.
                model.load_state_dict(old_state, strict=True)
                self.assertEqual(model.num_directions, count)
                self.assertEqual(model.num_nodes, count * 4)
                self.assertEqual(model.num_transitions, count * 3)
                self.assertEqual(len(model.direction_names), count)
                expected_angles = torch.tensor([-math.pi / 2 + 2 * math.pi * i / count for i in range(count)])
                torch.testing.assert_close(model.sector_angles, expected_angles, rtol=0, atol=0)
                offset = ((torch.arange(5) + .5) / 5 - .5) * (2 * math.pi / count)
                rays = torch.stack(((expected_angles[:, None] + offset).cos(),
                                    (expected_angles[:, None] + offset).sin()), -1)
                torch.testing.assert_close(model.sector_rays, rays, rtol=0, atol=0)
                points, *_ = model._sampling_geometry(args[1], torch.tensor([[80., 100.]]))
                self.assertEqual(points.shape, (1, 3, count, 4, 10, 2))
                self.assertTrue(((points >= 0) & (points < torch.tensor([100., 80.]))).all())
                output, info = model(*args)
                self.assertEqual(info['bcem_direction_weights'].shape, (1, 3, count))
                self.assertEqual(info['bcem_valid_regions'].shape, (1, 3, count, 4))
                self.assertEqual(info['bcem_valid_transitions'].shape, (1, 3, count, 3))
                self.assertEqual(info['bcem_text_sector_attention'].shape, (1, 3, 7, count))
                torch.testing.assert_close(info['bcem_direction_weights'].sum(-1, keepdim=True)
                                           + info['bcem_null_weight'], torch.ones(1, 3, 1))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for name in ('visual_projection.weight', 'text_projection.weight', 'transition_gru.weight_ih',
                             'gate.0.weight', 'environment_projection.weight'):
                    grad = dict(model.named_parameters())[name].grad
                    self.assertIsNotNone(grad, name)
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)
                empty = list(args)
                empty[1], empty[2] = args[1][:, :0], args[2][:, :0]
                _, info = model(*empty)
                self.assertEqual(info['bcem_direction_weights'].shape, (1, 0, count))
                self.assertEqual(info['bcem_valid_regions'].shape, (1, 0, count, 4))
                self.assertEqual(info['bcem_text_sector_attention'].shape, (1, 0, 7, count))

    def test_nondefault_sector_ablations_masking_and_checkpointing(self):
        args = self.inputs(batch=1, count=3)
        for options in (dict(use_transitions=False), dict(use_gru=False), dict(use_position=False),
                        dict(use_text_attention=False), dict(checkpoint_chunks=True)):
            model = self.module(num_sectors=4, **options)
            result, _ = model(*args)
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(result).all())
        masked = list(args)
        masked[4] = torch.ones_like(args[4])
        result, info = self.module(num_sectors=12)(*masked)
        torch.testing.assert_close(result, args[2], rtol=0, atol=0)
        self.assertEqual(info['bcem_direction_weights'].count_nonzero(), 0)

    def test_subrays_contiguously_cover_their_own_rectangular_image_intersection(self):
        model = self.module()
        points, _, _, _, bounds, _ = model._sampling_geometry(
            torch.tensor([[[.3, .65, .1, .2]]]), torch.tensor([[100., 300.]]))
        cells = points.reshape(1, 1, 8, 4, 5, 2, 2)
        step = cells[..., 1, :] - cells[..., 0, :]
        starts, ends = cells[..., 0, :] - step / 2, cells[..., 1, :] + step / 2
        torch.testing.assert_close(ends[..., :-1, :, :], starts[..., 1:, :, :], rtol=1e-5, atol=3e-5)
        first = starts[..., 0, :, :]
        box_sides = torch.cat(((first - bounds[:, :, None, None, :2]).abs(),
                               (first - bounds[:, :, None, None, 2:]).abs()), -1)
        self.assertLess(box_sides.min(-1).values.max().item(), 1e-4)
        last = ends[..., -1, :, :]
        image_sides = torch.cat((last.abs(), (last - torch.tensor([300., 100.])).abs()), -1)
        self.assertLess(image_sides.min(-1).values.max().item(), 1e-4)

    def test_region_area_pooling_matches_affine_field_at_centroid(self):
        model = self.module()
        model.visual_norm = torch.nn.Identity()
        yy, xx = torch.meshgrid(torch.arange(50.) * 2 + 1, torch.arange(100.) * 2 + 1, indexing="ij")
        visual = torch.stack([((i + 1) * xx + i * yy) / 100 for i in range(8)])[None]
        sizes = torch.tensor([[100., 200.]])
        features, position, valid, coverage = model._sample_sectors(
            visual, torch.tensor([[[.5, .5, .2, .2]]]), sizes, sizes,
            torch.zeros(1, 50, 100, dtype=torch.bool))
        centroid = position[0, 0, ..., 4:6] * torch.tensor([200., 100.])
        expected = torch.stack([((i + 1) * centroid[..., 0] + i * centroid[..., 1]) / 100 for i in range(8)], -1)
        torch.testing.assert_close(features[0, 0], expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(valid.all())
        torch.testing.assert_close(coverage, torch.ones_like(coverage))
        # Physical distance grows with radial index, and differs across directions.
        self.assertTrue((position[..., 1:, 2] > position[..., :-1, 2]).all())
        self.assertGreater(position[0, 0, 3, 0, 2].item(), position[0, 0, 0, 0, 2].item())

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
        for key in ("bcem_context", "bcem_direction_weights", "bcem_valid_regions"):
            torch.testing.assert_close(b[key], a[key], rtol=1e-5, atol=1e-6)

    def test_empty_exterior_and_masked_image_are_exact_identity_with_learned_biases(self):
        for cover_image in (False, True):
            model = self.module().eval()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith("bias"):
                        parameter.fill_(2.)
            args = self.inputs()
            if cover_image:
                args[1][..., 2:] = 1.
                mask = None
            else:
                mask = torch.ones(2, 20, 20, dtype=torch.bool)
            output, info = model(*args, visual_mask=mask)
            torch.testing.assert_close(output, args[2], rtol=0, atol=0)
            for key in ("bcem_context", "bcem_residual_ratio", "bcem_direction_weights", "bcem_background_weight"):
                self.assertEqual(info[key].count_nonzero(), 0)
            torch.testing.assert_close(info["bcem_null_weight"], torch.ones(2, 5, 1))
            self.assertFalse(info["bcem_valid_sectors"].any())

    def test_border_and_empty_boxes_are_masked(self):
        args = self.inputs(batch=1, count=2)
        args[1] = torch.tensor([[[.02, .02, .1, .1], [.5, .5, 0., .1]]])
        output, info = self.module()(*args)
        self.assertFalse(info["bcem_valid_sectors"][0, 0, 0])
        self.assertFalse(info["bcem_valid_sectors"][0, 0, 2])
        self.assertFalse(info["bcem_valid_sectors"][0, 1].any())
        torch.testing.assert_close(output[0, 1], args[2][0, 1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())

    def test_text_padding_and_integer_bool_masks_are_equivalent(self):
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

    def test_all_padding_text_is_identity_and_has_no_attention(self):
        args = self.inputs()
        args[4] = torch.ones_like(args[4])
        output, info = self.module()(*args)
        torch.testing.assert_close(output, args[2], rtol=0, atol=0)
        for key in ("bcem_context", "bcem_direction_weights", "bcem_null_weight",
                    "bcem_text_sector_attention", "bcem_text_null_attention"):
            self.assertEqual(info[key].count_nonzero(), 0)

    def test_original_text_changes_selection_but_candidate_only_conditions_gate(self):
        model = self.module().eval()
        args = self.inputs()
        first, a = model(*args)
        args[2] = args[2] + 10
        _, b = model(*args)
        for key in ("bcem_context", "bcem_direction_weights", "bcem_text_sector_attention"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertGreater((a["bcem_background_weight"] - b["bcem_background_weight"]).abs().max().item(), 1e-5)
        args[3] = args[3] + 2
        other, c = model(*args)
        self.assertGreater((b["bcem_direction_weights"] - c["bcem_direction_weights"]).abs().max().item(), 1e-5)
        self.assertTrue(torch.isfinite(first).all() and torch.isfinite(other).all())

    def test_regions_and_all_transition_states_are_retained(self):
        model = self.module()
        features = torch.randn(1, 2, 8, 4, 8)
        position = torch.randn(1, 2, 8, 4, 9)
        valid = torch.ones(1, 2, 8, 4, dtype=torch.bool)
        evidence, allowed, transitions = model._encode_evidence(features, position, valid)
        self.assertEqual(evidence.shape, (1, 2, 8, 7, 8))
        self.assertEqual(allowed.shape, (1, 2, 8, 7))
        self.assertEqual(transitions.shape, (1, 2, 8, 3))
        self.assertEqual(model.num_nodes, 32)
        self.assertEqual(model.num_transitions, 24)

    def test_gru_propagates_near_evidence_only_within_the_same_sector(self):
        model = self.module(use_position=False)
        features = torch.randn(1, 1, 8, 4, 8)
        position = torch.zeros(1, 1, 8, 4, 9)
        valid = torch.ones(1, 1, 8, 4, dtype=torch.bool)
        changed = features.clone()
        changed[:, :, 0, 0] += 3
        first, _, _ = model._encode_evidence(features, position, valid)
        other, _, _ = model._encode_evidence(changed, position, valid)
        self.assertGreater((first[:, :, 0, -1] - other[:, :, 0, -1]).abs().max().item(), 1e-6)
        torch.testing.assert_close(first[:, :, 1:], other[:, :, 1:], rtol=0, atol=0)
        model.use_gru = False
        first, _, _ = model._encode_evidence(features, position, valid)
        other, _, _ = model._encode_evidence(changed, position, valid)
        torch.testing.assert_close(first[..., -1, :], other[..., -1, :], rtol=0, atol=0)

    def test_invalid_radial_observation_breaks_recurrence_and_isolated_regions_survive(self):
        model = self.module(use_position=False, radial_edges=(0., .1, .25, .5, .75, 1.))
        features = torch.randn(1, 1, 8, 5, 8)
        position = torch.zeros(1, 1, 8, 5, 9)
        valid = torch.ones(1, 1, 8, 5, dtype=torch.bool)
        valid[..., 2] = False
        changed = features.clone()
        changed[..., 0, :] += 100
        first, allowed, transitions = model._encode_evidence(features, position, valid)
        other, _, _ = model._encode_evidence(changed, position, valid)
        torch.testing.assert_close(first[..., -1, :], other[..., -1, :], rtol=0, atol=0)
        self.assertFalse(transitions[..., 1:3].any())
        self.assertEqual(first.masked_select(~allowed[..., None].expand_as(first)).count_nonzero(), 0)
        valid[:] = False
        valid[:, :, 0, 1] = True
        first, allowed, transitions = model._encode_evidence(features, position, valid)
        self.assertEqual(allowed.sum(), 1)
        self.assertFalse(transitions.any())
        self.assertGreater(first[0, 0, 0, 1].abs().sum().item(), 0)

    def test_geometry_encoding_distinguishes_equal_visual_regions_and_can_be_ablated(self):
        model = self.module()
        features = torch.ones(1, 1, 8, 4, 8)
        position = torch.zeros(1, 1, 8, 4, 9)
        valid = torch.ones(1, 1, 8, 4, dtype=torch.bool)
        first, _, _ = model._encode_evidence(features, position, valid)
        changed = position.clone()
        changed[:, :, 0, :, 1] = -1
        other, _, _ = model._encode_evidence(features, changed, valid)
        self.assertGreater((first[:, :, 0] - other[:, :, 0]).abs().max().item(), 1e-5)
        model = self.module(use_position=False)
        first, _, _ = model._encode_evidence(features, position, valid)
        other, _, _ = model._encode_evidence(features, changed, valid)
        torch.testing.assert_close(first, other, rtol=0, atol=0)

    def test_sparse_selection_can_use_zero_one_two_four_or_all_eight_sectors(self):
        relevance = torch.tensor([[.125] * 8 + [0.], [.9] + [.1 / 7] * 7 + [0.],
                                  [.45, .45] + [.1 / 6] * 6 + [0.],
                                  [.23] * 4 + [.02] * 4 + [0.], [0.] * 8 + [1.]])
        weights = self.module()._sparse_sector_weights(relevance, torch.ones_like(relevance, dtype=torch.bool))
        torch.testing.assert_close(weights.sum(-1), torch.ones(5))
        torch.testing.assert_close((weights[:, :8] > 0).sum(-1), torch.tensor([8, 1, 2, 4, 0]))
        masked = self.module()._sparse_sector_weights(relevance, torch.zeros_like(relevance, dtype=torch.bool))
        self.assertEqual(masked.count_nonzero(), 0)

    def test_sparse_selection_masks_invalid_sectors_and_has_correct_gradients(self):
        model = self.module()
        relevance = torch.tensor([[.1] * 9, [.1] * 9])
        selectable = torch.zeros_like(relevance, dtype=torch.bool)
        selectable[0, 3] = True
        weights = model._sparse_sector_weights(relevance, selectable)
        self.assertEqual(weights[0, 3], 1.)
        self.assertEqual(weights[1].count_nonzero(), 0)
        inputs = torch.tensor([[.4, .35, .05, .04, .04, .03, .03, .03, .03]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: model._sparse_sector_weights(x, torch.ones_like(x, dtype=torch.bool)),
                                                (inputs,), eps=1e-6, atol=1e-5))

    def test_null_option_can_exclude_all_geometrically_valid_sectors(self):
        model = self.module()
        with torch.no_grad():
            model.evidence_key.weight.zero_()
            model.null_key.fill_(100)
        evidence, valid, _, _ = self.evidence()
        tokens, text_valid = torch.ones(1, 1, 8), torch.ones(1, 1, dtype=torch.bool)
        context, weights, _, _, null, _, _ = model._read_evidence(evidence, valid, tokens, text_valid)
        self.assertEqual(weights.count_nonzero(), 0)
        self.assertEqual(context.count_nonzero(), 0)
        torch.testing.assert_close(null, torch.ones_like(null))
        queries = torch.randn(1, 2, 16)
        enhanced, gate = model._fuse_context(context, queries, torch.randn(1, 16), weights.sum(-1) > 0)
        torch.testing.assert_close(enhanced, queries, rtol=0, atol=0)
        self.assertEqual(gate.count_nonzero(), 0)

    def test_excluded_sector_values_have_no_effect_or_gradient(self):
        model = self.module()
        class FixedKeys(torch.nn.Module):
            def forward(self, x):
                key = torch.zeros_like(x)
                key[:, :, 0, :, ::2] = 1
                return key
        model.evidence_key = FixedKeys()
        model.evidence_value = torch.nn.Identity()
        with torch.no_grad():
            model.null_key.fill_(-10)
        evidence, valid, _, _ = self.evidence()
        evidence.requires_grad_()
        tokens = torch.zeros(1, 1, 8)
        tokens[..., ::2] = 4 * math.sqrt(2)
        text_valid = torch.ones(1, 1, dtype=torch.bool)
        first, weights, _, _, _, _, _ = model._read_evidence(evidence, valid, tokens, text_valid)
        self.assertEqual((weights > 0).sum(-1).max(), 1)
        changed = evidence.detach().clone()
        changed[:, :, 1:] += 100
        other, _, _, _, _, _, _ = model._read_evidence(changed, valid, tokens, text_valid)
        torch.testing.assert_close(first, other, rtol=0, atol=0)
        first.square().sum().backward()
        self.assertEqual(evidence.grad[:, :, 1:].count_nonzero(), 0)
        model.use_sparse_selection = False
        dense_first = model._read_evidence(evidence, valid, tokens, text_valid)[0]
        dense_other = model._read_evidence(changed, valid, tokens, text_valid)[0]
        self.assertGreater((dense_first - dense_other).abs().max().item(), 1e-3)

    def test_dense_reader_matches_ordinary_multihead_text_read_without_double_weighting(self):
        model = self.module(use_sparse_selection=False)
        evidence, valid, tokens, text_valid = self.evidence()
        valid[0, 0, 3, 4:] = False
        context, weights, dense, token_sector, null, _, token_null = model._read_evidence(evidence, valid, tokens, text_valid)
        clean = evidence.masked_fill(~valid[..., None], 0)
        b, n, d, k, c = clean.shape
        h, hd = model.num_heads, c // model.num_heads
        keys = model.evidence_key(clean).reshape(b, n, d * k, h, hd).permute(0, 1, 3, 2, 4)
        keys = torch.cat((keys, model.null_key.reshape(1, 1, h, 1, hd).expand(b, n, -1, -1, -1)), -2)
        values = model.evidence_value(clean).reshape(b, n, d * k, h, hd).permute(0, 1, 3, 2, 4)
        values = torch.cat((values, values.new_zeros(b, n, h, 1, hd)), -2)
        queries = tokens.reshape(b, -1, h, hd).permute(0, 2, 1, 3)
        allowed = torch.cat((valid.reshape(b, n, d * k), valid.new_ones(b, n, 1)), -1)
        attn = (torch.einsum('bhlc,bnhec->bnhle', queries, keys) / math.sqrt(hd)).masked_fill(~allowed[:, :, None, None], -float('inf')).softmax(-1)
        read = torch.einsum('bnhle,bnhec->bnhlc', attn, values)
        read = read.permute(0, 1, 3, 2, 4).reshape(b, n, tokens.shape[1], c)
        expected = (model.reader_output(read) * text_valid[:, None, :, None]).sum(2) / text_valid.sum(-1)[:, None, None]
        torch.testing.assert_close(context, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(weights, dense)
        torch.testing.assert_close(weights.sum(-1, keepdim=True) + null, torch.ones(b, n, 1))
        torch.testing.assert_close(token_sector.sum(-1, keepdim=True) + token_null,
                                   text_valid[:, None, :, None].expand(b, n, -1, -1).float())

    def test_sector_permutation_preserves_read_and_permutes_weights(self):
        model = self.module()
        evidence, valid, tokens, text_valid = self.evidence()
        first, weights, *_ = model._read_evidence(evidence, valid, tokens, text_valid)
        order = torch.tensor([4, 2, 0, 3, 1, 7, 5, 6])
        other, changed, *_ = model._read_evidence(evidence[:, :, order], valid[:, :, order], tokens, text_valid)
        torch.testing.assert_close(first, other, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(changed, weights[:, :, order], rtol=1e-5, atol=1e-6)

    def test_diagnostics_are_actual_sector_usage_and_null_mass(self):
        _, info = self.module()(*self.inputs())
        weights, valid = info['bcem_direction_weights'], info['bcem_valid_sectors']
        torch.testing.assert_close(info['bcem_selected_sectors'], weights > 0)
        torch.testing.assert_close(info['bcem_selected_sector_count'], (weights > 0).sum(-1))
        self.assertFalse((info['bcem_selected_sectors'] & ~valid).any())
        torch.testing.assert_close(weights.sum(-1, keepdim=True) + info['bcem_null_weight'], torch.ones(2, 5, 1))
        torch.testing.assert_close(info['bcem_direction_weights_dense'].sum(-1, keepdim=True)
                                   + info['bcem_null_weight_dense'], torch.ones(2, 5, 1))
        self.assertTrue(all(not x.requires_grad for x in info.values()))

    def test_residual_gate_scales_only_the_added_background(self):
        model = self.module().eval()
        self.assertFalse(hasattr(model, 'feature_fusion'))
        self.assertIsNone(model.environment_projection.bias)
        context, queries, sentence = torch.randn(1, 2, 8), torch.randn(1, 2, 16), torch.randn(1, 16)
        usable = torch.tensor([[True, False]])
        background = model.environment_projection(context)
        for weight in (.01, .5, .99):
            with torch.no_grad():
                for parameter in model.gate.parameters():
                    parameter.zero_()
                model.gate[-1].bias.fill_(math.log(weight / (1 - weight)))
            output, gate = model._fuse_context(context, queries, sentence, usable)
            torch.testing.assert_close(output[0, 0], queries[0, 0] + .1 * weight * background[0, 0])
            torch.testing.assert_close(output[0, 1], queries[0, 1], rtol=0, atol=0)
            torch.testing.assert_close(gate, torch.tensor([[[weight], [0.]]]))
        with torch.no_grad():
            model.gate[-1].bias.fill_(-1000)
        output, gate = model._fuse_context(context, queries, sentence, usable)
        torch.testing.assert_close(output, queries, rtol=0, atol=0)
        self.assertEqual(gate.count_nonzero(), 0)

    def test_chunking_and_checkpointing_preserve_output_and_gradients(self):
        model = self.module()
        other = copy.deepcopy(model)
        other.chunk_size = 64
        other.checkpoint_chunks = True
        args = self.inputs()
        clones = [x.detach().clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in args]
        first, a = model(*args)
        actual, b = other(*clones)
        torch.testing.assert_close(actual, first, rtol=1e-5, atol=1e-6)
        first.square().mean().backward()
        actual.square().mean().backward()
        for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-6)
        torch.testing.assert_close(a['bcem_context'], b['bcem_context'], rtol=1e-5, atol=1e-6)

    def test_gradients_reach_visual_text_gate_and_gru_but_not_box_geometry(self):
        model = self.module()
        args = self.inputs()
        for index in (0, 1, 2, 3):
            args[index].requires_grad_()
        output, _ = model(*args)
        output.square().mean().backward()
        for index in (0, 2, 3):
            self.assertTrue(torch.isfinite(args[index].grad).all())
            self.assertGreater(args[index].grad.abs().sum().item(), 0)
        self.assertIsNone(args[1].grad)
        for name in ('visual_projection.weight', 'text_projection.weight', 'evidence_key.weight',
                     'evidence_value.weight', 'position_mlp.0.weight', 'region_fusion.0.weight',
                     'pair_position.0.weight', 'transition_mlp.0.weight', 'transition_gru.weight_ih',
                     'transition_gru.weight_hh', 'gate.0.weight', 'gate.2.weight', 'environment_projection.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_ablations_remove_inactive_parameters_and_keep_uniform_read_well_defined(self):
        args = self.inputs()
        for options in (dict(use_position=False), dict(use_transitions=False), dict(use_gru=False),
                        dict(use_sparse_selection=False), dict(use_text_attention=False),
                        dict(learnable_gate=False), dict(use_text_attention=False, learnable_gate=False)):
            with self.subTest(options=options):
                model = self.module(**options)
                output, info = model(*args)
                self.assertTrue(torch.isfinite(output).all())
                if options.get('use_position') is False:
                    self.assertIsNone(model.position_mlp)
                    self.assertIsNone(model.pair_position)
                if options.get('use_transitions') is False:
                    self.assertIsNone(model.transition_mlp)
                    self.assertIsNone(model.transition_gru)
                    self.assertEqual(info['bcem_valid_transitions'].shape[-1], 0)
                if options.get('use_gru') is False:
                    self.assertIsNone(model.transition_gru)
                if options.get('learnable_gate') is False:
                    self.assertIsNone(model.gate)
                    torch.testing.assert_close(info['bcem_background_weight'], torch.ones(2, 5, 1))
                if options.get('use_text_attention') is False:
                    self.assertIsNone(model.evidence_key)
                    self.assertIsNone(model.text_projection)
                    self.assertIsNone(model.null_key)
                    weights = info['bcem_valid_sectors'].float()
                    torch.testing.assert_close(info['bcem_direction_weights'], weights / weights.sum(-1, keepdim=True))
                    if options.get('learnable_gate') is False:
                        changed = list(args)
                        changed[3] = args[3] + 100
                        other, _ = model(*changed)
                        torch.testing.assert_close(output, other, rtol=0, atol=0)

    def test_uniform_reader_balances_sectors_despite_different_evidence_counts(self):
        model = self.module(use_text_attention=False)
        evidence, valid, tokens, text_valid = self.evidence()
        valid[:, :, 1:] = False
        valid[:, :, 1, 0] = True
        _, weights, _, _, null, _, _ = model._read_evidence(evidence, valid, tokens, text_valid)
        torch.testing.assert_close(weights[..., :2], torch.full((1, 2, 2), .5))
        self.assertEqual(weights[..., 2:].count_nonzero(), 0)
        self.assertEqual(null.count_nonzero(), 0)

    def test_zero_context_strength_and_empty_candidate_axis(self):
        args = self.inputs()
        output, _ = self.module(context_strength=0)(*args)
        torch.testing.assert_close(output, args[2], rtol=0, atol=0)
        args[1], args[2] = args[1][:, :0], args[2][:, :0]
        output, info = self.module()(*args)
        self.assertEqual(output.shape, (2, 0, 16))
        self.assertEqual(info['bcem_valid_regions'].shape, (2, 0, 8, 4))
        self.assertEqual(info['bcem_valid_transitions'].shape, (2, 0, 8, 3))
        self.assertEqual(info['bcem_text_sector_attention'].shape, (2, 0, 7, 8))

    def test_cpu_autocast_forward_and_backward_are_finite(self):
        model = self.module(checkpoint_chunks=True)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output, _ = model(*self.inputs())
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_settings_fail_early(self):
        for options in (dict(radial_edges=(0., .2, .1, 1.)), dict(radial_edges=(.1, .5, 1.)),
                        dict(radial_edges=(0., .5, .9)), dict(num_heads=3), dict(angular_samples=True),
                        dict(min_valid_points=11), dict(context_strength=-1), dict(num_sectors=0),
                        dict(num_sectors=-4), dict(num_sectors=True), dict(num_sectors=4.5)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module(**options)


class BCEMHeadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        # Integration checks exercise BCEM even when the user's experiment config
        # currently selects a baseline run.
        cls.options["enable_bcem"] = True
        cls.options["num_queries"] = 8
        # Legacy regression checks must not inherit the user's ablation count.
        cls.options["bcem"].update(context_dim=16, chunk_size=3, num_sectors=8)

    def inputs(self):
        torch.manual_seed(572)
        x = torch.randn(1, 768, 8, 8)
        targets = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]])],
                       img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
        kwargs = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                      lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128))
        return x, targets, kwargs

    def test_configured_sector_count_reaches_head_training_and_inference(self):
        for count in (4, 12):
            with self.subTest(num_sectors=count):
                options = copy.deepcopy(self.options)
                options['bcem']['num_sectors'] = count
                head = build_head(options)
                x, targets, kwargs = self.inputs()
                losses, _, train_info = head.forward_train(x, targets, **kwargs)
                sum(losses.values()).backward()
                self.assertEqual(train_info['bcem_direction_weights'].shape, (1, 8, count))
                self.assertTrue(torch.isfinite(head.bcem.transition_gru.weight_ih.grad).all())
                head.eval()
                with torch.no_grad():
                    prediction, info = head.forward_test(x, targets=targets, with_mask=False, **kwargs)
                self.assertEqual(prediction['refer_label'].shape, (1, 8, 1))
                self.assertEqual(info['bcem_valid_regions'].shape, (1, 8, count, 4))

    def test_enabled_branch_trains_and_reaches_crs_in_train_and_test(self):
        torch.manual_seed(103)
        head = build_head(self.options)
        self.assertTrue(head.bcem.use_position)
        self.assertIsNotNone(head.bcem.position_mlp)
        x, targets, kwargs = self.inputs()
        seen = []
        handle = head.mrsm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        losses, train_predictions, train_info = head.forward_train(x, targets, **kwargs)
        sum(losses.values()).backward()
        gradient = head.bcem.visual_projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        for name in ("text_projection.weight", "evidence_key.weight", "evidence_value.weight", "gate.0.weight",
                     "transition_gru.weight_ih", "transition_gru.weight_hh"):
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
