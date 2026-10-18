"""CPU checks for four on-ray points and legacy sampling compatibility."""
import copy
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")

import torch
from mmcv import Config
from propvg.models import build_head
from propvg.models.heads.background_context import BackgroundContextEnhancement


class BCEMRayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)

    def module(self, **kwargs):
        options = dict(hidden_channels=16, context_dim=8, chunk_size=2,
                       checkpoint_chunks=False, sampling_mode="ray")
        options.update(kwargs)
        return BackgroundContextEnhancement(**options)

    def inputs(self, count=3):
        return [torch.randn(1, 16, 20, 20), torch.tensor([[[.5, .5, .12, .1]] * count]),
                torch.randn(1, count, 16), torch.randn(1, 7, 16),
                torch.tensor([[0, 0, 0, 0, 1, 1, 1]]),
                [dict(img_shape=(80, 100), pad_shape=(100, 100))]]

    def test_four_points_are_distinct_midpoints_directly_on_the_ray(self):
        model = self.module(num_rays=4)
        sizes = torch.tensor([[100., 200.]])
        boxes = torch.tensor([[[.5, .5, .2, .2]]], requires_grad=True)
        points, weights, _, _, _, widths = model._sampling_geometry(boxes, sizes, sizes, (20, 40))
        self.assertEqual(points.shape, (1, 1, 4, 4, 4, 2))
        torch.testing.assert_close(points[0, 0, 0, 0], torch.tensor([[100., 39.5], [100., 38.5],
                                                                  [100., 37.5], [100., 36.5]]))
        torch.testing.assert_close(points[0, 0, 1, 0], torch.tensor([[121., 50.], [123., 50.],
                                                                  [125., 50.], [127., 50.]]))
        torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)
        torch.testing.assert_close(widths[0, 0, 0, :, 0], torch.tensor([4., 6., 10., 20.]))
        self.assertFalse(points.requires_grad)
        self.assertEqual(model.min_valid_points, 2)
        self.assertEqual(model.direction_names, ("up", "right", "down", "left"))

    def test_zero_lateral_displacement_and_increasing_radial_spacing(self):
        model = self.module(num_rays=16)
        sizes = torch.tensor([[100., 200.]])
        points = model._sampling_geometry(torch.tensor([[[.5, .5, .2, .2]]]), sizes, sizes, (20, 40))[0]
        rays = model.sector_rays[:, 0]
        normal = torch.stack((-rays[:, 1], rays[:, 0]), -1)
        lateral = ((points - torch.tensor([100., 50.])) * normal[None, None, :, None, None]).sum(-1)
        torch.testing.assert_close(lateral, torch.zeros_like(lateral), rtol=0, atol=2e-5)
        axial = points[0, 0, 0].reshape(16, 2)
        spacing = (axial[1:] - axial[:-1]).norm(dim=-1)
        self.assertTrue((spacing[1:] + 1e-5 >= spacing[:-1]).all())
        self.assertGreater(spacing[-1], spacing[0])
        self.assertEqual(model.radial_samples, 4)
        self.assertEqual(model.ray_lateral_offset, 0)

    def test_archived_positive_offset_preserves_legacy_neighborhood(self):
        model = self.module(num_rays=4, ray_lateral_offset=.5, radial_samples=2)
        sizes = torch.tensor([[100., 200.]])
        points = model._sampling_geometry(torch.tensor([[[.5, .5, .2, .2]]]), sizes, sizes, (20, 40))[0]
        torch.testing.assert_close(points[0, 0, 0, 0], torch.tensor([[97.5, 39.], [102.5, 39.],
                                                                  [97.5, 37.], [102.5, 37.]]))

    def test_padding_and_feature_canvas_preserve_physical_sampling(self):
        model = self.module(num_rays=8)
        boxes, sizes = torch.tensor([[[.5, .5, .1, .1]]]), torch.tensor([[100., 200.]])
        a = model._sampling_geometry(boxes, sizes, sizes, (20, 40))[0]
        b = model._sampling_geometry(boxes, sizes, sizes * 2, (40, 80))[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_uniform_pooling_matches_four_anchor_centroid_on_affine_field(self):
        model = self.module(num_rays=4)
        model.visual_norm = torch.nn.Identity()
        yy, xx = torch.meshgrid(torch.arange(50.) * 2 + 1, torch.arange(100.) * 2 + 1, indexing="ij")
        visual = torch.stack([((i + 1) * xx + i * yy) / 100 for i in range(8)])[None]
        sizes = torch.tensor([[100., 200.]])
        boxes = torch.tensor([[[.5, .5, .2, .2]]])
        features, position, valid, coverage = model._sample_sectors(
            visual, boxes, sizes, sizes, torch.zeros(1, 50, 100, dtype=torch.bool))
        points = model._sampling_geometry(boxes, sizes, sizes, (50, 100))[0]
        centroid = points.mean(-2)
        torch.testing.assert_close(position[..., 4:6] * torch.tensor([200., 100.]), centroid)
        expected = torch.stack([((i + 1) * centroid[..., 0] + i * centroid[..., 1]) / 100 for i in range(8)], -1)
        torch.testing.assert_close(features, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(valid.all())
        torch.testing.assert_close(coverage, torch.ones_like(coverage))

    def test_border_points_are_masked_without_clamping_and_threshold_is_respected(self):
        args = self.inputs(count=1)
        args[1] = torch.tensor([[[.5, .027, .12, .001]]])
        args[5] = [dict(img_shape=(100, 100), pad_shape=(100, 100))]
        model = self.module(num_rays=4)
        _, info = model(*args)
        torch.testing.assert_close(info['bcem_region_coverage'][0, 0, 0, 0], torch.tensor(.5))
        self.assertTrue(info['bcem_valid_regions'][0, 0, 0, 0])
        self.assertFalse(info['bcem_valid_regions'][0, 0, 0, 1:].any())
        strict = self.module(num_rays=4, min_valid_points=3)
        strict.load_state_dict(model.state_dict(), strict=True)
        _, info = strict(*args)
        self.assertFalse(info['bcem_valid_regions'][0, 0, 0].any())

    def test_counts_gradients_evidence_and_state_dict_across_ray_counts(self):
        old = BackgroundContextEnhancement(hidden_channels=16, context_dim=8).state_dict()
        for count in (4, 8, 16):
            with self.subTest(num_rays=count):
                model = self.module(num_rays=count)
                model.load_state_dict(old, strict=True)
                self.assertEqual(model.num_nodes, count * 4)
                self.assertEqual(model.num_transitions, count * 3)
                self.assertEqual(model.num_points, 4)
                self.assertEqual(model.sector_rays.shape, (count, 1, 2))
                args = self.inputs()
                for index in (0, 1, 2, 3):
                    args[index].requires_grad_()
                output, info = model(*args)
                self.assertEqual(info['bcem_direction_weights'].shape, (1, 3, count))
                self.assertEqual(info['bcem_valid_regions'].shape, (1, 3, count, 4))
                self.assertEqual(info['bcem_valid_transitions'].shape, (1, 3, count, 3))
                torch.testing.assert_close(info['bcem_direction_weights'].sum(-1, keepdim=True)
                                           + info['bcem_null_weight'], torch.ones(1, 3, 1))
                output.square().mean().backward()
                self.assertIsNone(args[1].grad)
                for index in (0, 2, 3):
                    self.assertGreater(args[index].grad.abs().sum().item(), 0)
                for name in ('position_mlp.0.weight', 'region_fusion.0.weight', 'pair_position.0.weight',
                             'transition_gru.weight_ih', 'text_projection.weight', 'gate.0.weight'):
                    grad = dict(model.named_parameters())[name].grad
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_invalid_and_empty_inputs_preserve_identity(self):
        model = self.module()
        for kind in ('text', 'visual', 'box', 'empty'):
            args = self.inputs()
            extra = {}
            if kind == 'text':
                args[4].fill_(1)
            elif kind == 'visual':
                extra['visual_mask'] = torch.ones(1, 20, 20, dtype=torch.bool)
            elif kind == 'box':
                args[1][..., 2:] = 0
            else:
                args[1], args[2] = args[1][:, :0], args[2][:, :0]
            output, info = model(*args, **extra)
            torch.testing.assert_close(output, args[2], rtol=0, atol=0)
            self.assertEqual(info['bcem_context'].count_nonzero(), 0)
            self.assertEqual(info['bcem_valid_regions'].shape, (1, args[2].shape[1], 16, 4))

    def test_mixed_image_batch_and_cpu_autocast(self):
        model = self.module().eval()
        args = self.inputs(count=2)
        args[:5] = [torch.cat((value, value), 0) for value in args[:5]]
        args[5] = [dict(img_shape=(80, 100), pad_shape=(100, 100)),
                   dict(img_shape=(100, 60), pad_shape=(100, 100))]
        with torch.no_grad():
            actual, info = model(*args)
            for index in range(2):
                one = [value[index:index+1] for value in args[:5]] + [[args[5][index]]]
                expected, details = model(*one)
                torch.testing.assert_close(actual[index:index+1], expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(info['bcem_direction_weights'][index:index+1],
                                           details['bcem_direction_weights'], rtol=1e-5, atol=1e-6)
        model.train()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output, _ = model(*args)
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_chunk_checkpointing_and_optional_branches(self):
        model = self.module()
        other = copy.deepcopy(model)
        other.chunk_size, other.checkpoint_chunks = 1, True
        args = self.inputs()
        first, _ = model(*args)
        actual, _ = other(*args)
        torch.testing.assert_close(first, actual, rtol=1e-5, atol=1e-6)
        first.square().mean().backward()
        actual.square().mean().backward()
        for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-6, msg=name)
        for options in (dict(use_transitions=False), dict(use_gru=False), dict(use_position=False),
                        dict(use_text_attention=False), dict(learnable_gate=False)):
            result, _ = self.module(**options)(*args)
            self.assertTrue(torch.isfinite(result).all())

    def test_ray_settings_fail_early(self):
        for options in (dict(num_rays=0), dict(num_rays=True), dict(radial_samples=3),
                        dict(radial_samples=2), dict(ray_lateral_offset=-1),
                        dict(ray_lateral_offset=float('nan')), dict(min_valid_points=5),
                        dict(sampling_mode='unknown')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module(**options)

    def test_current_config_reaches_crs_in_training_and_inference(self):
        options = copy.deepcopy(Config.fromfile('configs/aerialvg/PropVG-aerialvg.py').model.head)
        options['enable_crv'] = False  # This test covers BCEM, not post-CRS refinement.
        options.update(enable_bcem=True, num_queries=8)
        options['bcem'].update(sampling_mode='ray', num_rays=16, min_valid_points=2, context_dim=16)
        head = build_head(options)
        self.assertEqual(head.bcem.sampling_mode, 'ray')
        x = torch.randn(1, 768, 8, 8)
        targets = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]])],
                       img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
        kwargs = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                      lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128))
        losses, _, info = head.forward_train(x, targets, **kwargs)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(head.bcem.transition_gru.weight_ih.grad).all())
        self.assertEqual(info['bcem_direction_weights'].shape, (1, 8, 16))
        head.eval()
        with torch.no_grad():
            prediction, info = head.forward_test(x, targets=targets, with_mask=False, **kwargs)
        self.assertEqual(prediction['refer_label'].shape, (1, 8, 1))
        self.assertEqual(info['bcem_valid_regions'].shape, (1, 8, 16, 4))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
