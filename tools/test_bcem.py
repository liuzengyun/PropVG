"""CPU regressions for BCEM geometry, masking, gradients and head integration.

Run: PYTHONPATH=. python tools/test_bcem.py
"""
import copy
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
        options = dict(hidden_channels=16, context_dim=8, num_heads=2,
                       chunk_size=2, checkpoint_chunks=False)
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

    def test_scales_and_boundary_relative_coordinates(self):
        model = self.module()
        torch.testing.assert_close(model.scales, torch.tensor([.1, .15, .2, .25]))
        boxes = torch.tensor([[[.5, .5, .2, .2]]])
        points, nominal, _, _, bounds, distance = model._sampling_geometry(boxes, torch.tensor([[100., 200.]]))
        torch.testing.assert_close(distance, torch.tensor([[10., 15., 20., 25.]]))
        # Up starts at y=40, not at the center y=50; right starts at x=120.
        torch.testing.assert_close(nominal[0, 0, 0], torch.tensor([[100., 30.], [100., 25.], [100., 20.], [100., 15.]]))
        torch.testing.assert_close(nominal[0, 0, 3], torch.tensor([[130., 50.], [135., 50.], [140., 50.], [145., 50.]]))
        inside = ((points >= bounds[:, :, None, None, None, :2]) &
                  (points <= bounds[:, :, None, None, None, 2:])).all(-1)
        self.assertFalse(inside.any())
        self.assertEqual(tuple(points.shape), (1, 1, 8, 4, 9, 2))

    def test_padding_canvas_does_not_change_physical_evidence(self):
        model = self.module().eval()
        # Identical physical feature grid, placed on two different canvases.
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
        torch.testing.assert_close(b["bcem_valid_nodes"], a["bcem_valid_nodes"])

    def test_invalid_environment_is_exact_identity_even_with_learned_biases(self):
        model = self.module().eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("bias"):
                    parameter.fill_(2.)
        args = self.inputs()
        visual_mask = torch.ones(2, 20, 20, dtype=torch.bool)
        out, info = model(*args, visual_mask=visual_mask)
        torch.testing.assert_close(out, args[2], rtol=0, atol=0)
        self.assertEqual(info["bcem_context"].count_nonzero(), 0)
        self.assertEqual(info["bcem_residual_ratio"].count_nonzero(), 0)
        self.assertFalse(info["bcem_valid_nodes"].any())
        self.assertFalse(info["bcem_valid_transitions"].any())

    def test_image_edge_and_empty_boxes_are_masked(self):
        model = self.module().eval()
        args = self.inputs(batch=1, count=2)
        args[1] = torch.tensor([[[.02, .02, .1, .1], [.5, .5, 0., .1]]])
        out, info = model(*args)
        valid = info["bcem_valid_nodes"].reshape(1, 2, 8, 4)
        self.assertFalse(valid[0, 0, 0].any())  # Up outside image.
        self.assertFalse(valid[0, 0, 2].any())  # Left outside image.
        self.assertFalse(valid[0, 1].any())
        torch.testing.assert_close(out[0, 1], args[2][0, 1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(out).all())

    def test_text_padding_is_ignored_for_integer_and_bool_masks(self):
        model = self.module().eval()
        args = self.inputs()
        expected, a = model(*args)
        args[3] = args[3].masked_fill(args[4].bool()[..., None], 1e6)
        actual, b = model(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(a["bcem_gate"], b["bcem_gate"], rtol=0, atol=0)
        args[4] = args[4].bool()
        boolean, _ = model(*args)
        torch.testing.assert_close(boolean, actual, rtol=0, atol=0)

    def test_all_text_padding_produces_zero_context(self):
        args = self.inputs()
        args[4] = torch.ones_like(args[4])
        output, info = self.module()(*args)
        torch.testing.assert_close(output, args[2], rtol=0, atol=0)
        self.assertEqual(info["bcem_context"].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(info["bcem_gate"]).all())

    def test_gru_does_not_bridge_missing_observations(self):
        model = self.module(scales=(.1, .15, .2, .25, .3)).eval()
        features, position = torch.randn(1, 1, 8, 5, 8), torch.randn(1, 1, 8, 5, 11)
        valid = torch.ones(1, 1, 8, 5, dtype=torch.bool)
        valid[:, :, :, 2] = False
        a, _, transition_valid = model._encode_evidence(features, position, valid)
        altered = features.clone()
        altered[:, :, :, :2] += 100
        b, _, _ = model._encode_evidence(altered, position, valid)
        # With five scales: valid pair, two missing pairs, valid pair.
        transitions_a = a[:, :, model.num_nodes:].reshape(1, 1, 8, 4, 8)
        transitions_b = b[:, :, model.num_nodes:].reshape(1, 1, 8, 4, 8)
        torch.testing.assert_close(transitions_a[:, :, :, -1], transitions_b[:, :, :, -1], rtol=0, atol=0)
        self.assertFalse(torch.equal(transitions_a[:, :, :, 0], transitions_b[:, :, :, 0]))
        torch.testing.assert_close(transition_valid.reshape(1, 1, 8, 4)[0, 0, 0],
                                   torch.tensor([True, False, False, True]))

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

    def test_gradients_reach_background_text_gate_and_queries_not_sampling_boxes(self):
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
        for name in ("visual_projection.weight", "transition_mlp.0.weight",
                     "transition_gru.weight_hh", "text_query.weight", "gate.0.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertTrue(all(not tensor.requires_grad for tensor in info.values()))
        self.assertTrue(((info["bcem_gate"] >= 0) & (info["bcem_gate"] <= 1)).all())

    def test_checkpointing_preserves_outputs_and_gradients(self):
        direct = self.module()
        saved = copy.deepcopy(direct)
        saved.checkpoint_chunks = True
        args = self.inputs()
        a, _ = direct(*args)
        b, _ = saved(*args)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for (name, pa), (_, pb) in zip(direct.named_parameters(), saved.named_parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_ablations_do_not_allocate_inactive_parameters(self):
        args = self.inputs()
        for options in (dict(use_transitions=False), dict(use_gru=False), dict(learnable_gate=False)):
            with self.subTest(options=options):
                model = self.module(**options)
                output, info = model(*args)
                self.assertTrue(torch.isfinite(output).all())
                if options.get("use_transitions") is False:
                    self.assertEqual(info["bcem_valid_transitions"].shape[-1], 0)
                    self.assertFalse(any("transition" in name or "pair_position" in name for name, _ in model.named_parameters()))
                if options.get("use_gru") is False:
                    self.assertIsNone(model.transition_gru)
                if options.get("learnable_gate") is False:
                    torch.testing.assert_close(info["bcem_gate"], torch.ones_like(info["bcem_gate"]))
                    self.assertIsNone(model.gate)

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
        self.assertEqual(info["bcem_valid_nodes"].shape, (2, 0, 32))


class BCEMHeadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        cls.options["num_queries"] = 8
        cls.options["bcem"].update(context_dim=16, num_heads=2, chunk_size=3)

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
        x, targets, kwargs = self.inputs()
        seen = []
        handle = head.mrsm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        losses, train_predictions, train_info = head.forward_train(x, targets, **kwargs)
        sum(losses.values()).backward()
        gradient = head.bcem.visual_projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertEqual(train_info["bcem_gate"].shape, (1, 8, 1))
        head.eval()
        with torch.no_grad():
            predictions, test_info = head.forward_test(x, targets=targets, with_mask=False, **kwargs)
        handle.remove()
        self.assertEqual(len(seen), 2)
        self.assertEqual(predictions["refer_label"].shape, (1, 8, 1))
        self.assertEqual(test_info["bcem_valid_nodes"].shape, (1, 8, 32))
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
                self.assertEqual(info["bcem_gate"].shape[1], enhanced["pred_bbox_allobj"].shape[1])
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
