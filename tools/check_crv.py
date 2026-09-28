"""Contract checks for the independent CRV, not an accuracy benchmark."""

import argparse
import ast
import copy
import importlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")

import torch
from mmcv import Config
from propvg.models import build_head
from propvg.models.heads.REFHead import CRS
from propvg.models.heads.crv import CounterfactualReferentVerifier


def fixture(device="cpu", batch=2):
    torch.manual_seed(7109)
    crv = CounterfactualReferentVerifier(hidden_channels=32, verifier_dim=32,
                                       grid_size=8, num_heads=4).to(device)
    boxes = torch.tensor([
        [.25, .25, .125, .125], [.8, .8, .06, .06], [.8, .2, .06, .06],
        [.25, .25, .0625, .125], [.6, .6, .06, .06], [.1, .8, .06, .06],
        [.5, .8, .06, .06], [.8, .5, .06, .06], [.4, .4, .06, .06],
    ], device=device)[None].repeat(batch, 1, 1).requires_grad_()
    visual = torch.randn(batch, 32, 24, 24, device=device, requires_grad=True)
    mask = torch.zeros(batch, 24, 24, dtype=torch.bool, device=device)
    mask[:, 18:] = True
    logits = torch.linspace(2, -2, 9, device=device)[None, :, None].repeat(batch, 1, 1).requires_grad_()
    return crv, dict(
        visual=visual, visual_mask=mask, boxes=boxes,
        refer_scores=logits.sigmoid(), refer_logits=logits,
        text=torch.randn(batch, 7, 32, device=device, requires_grad=True),
        text_mask=torch.tensor([[False] * 5 + [True] * 2], device=device).repeat(batch, 1),
        valid_ratios=torch.tensor([[1., .75]], device=device).repeat(batch, 1),
        target_boxes=boxes[:, 0].detach(),
        target_valid=torch.ones(batch, dtype=torch.bool, device=device),
    )


def activate(crv):
    with torch.no_grad():
        crv.scene_head[-1].weight.normal_(std=.05)


class CRVChecks(unittest.TestCase):
    device = "cpu"
    ddp = False
    backend = "gloo"
    observations = {}

    def setUp(self):
        self.crv, self.args = fixture(self.device)

    def test_shapes_and_exact_zero_start(self):
        output = self.crv(**self.args)
        self.assertEqual(tuple(output["weighted_scores"].shape), (2, 8, 1))
        self.assertEqual(tuple(output["original_support_logits"].shape), (2, 1))
        self.assertEqual(tuple(output["gt_support_drop_logits"].shape), (2, 1))
        torch.testing.assert_close(output["fused_logits"], output["raw_crs_logits"], rtol=0, atol=0)
        torch.testing.assert_close(output["fusion_alpha"], torch.full((2, 1), .1, device=self.device))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.observations["parameter_count_256"] = sum(p.numel() for p in CounterfactualReferentVerifier().parameters())

    def test_loss_gradients_only_train_crv(self):
        activate(self.crv)
        self.crv(**self.args)["loss"].backward()
        for key in ("visual", "text", "boxes", "refer_logits"):
            self.assertIsNone(self.args[key].grad, key)
        self.assertTrue(all(p.grad is not None for p in self.crv.parameters()))
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in self.crv.scene_matcher.parameters()))
        self.assertGreater(float(self.crv.fusion_gate[-1].bias.grad.abs()), 0)

    def test_final_loss_trains_scorer_and_gate(self):
        activate(self.crv)
        self.crv(**self.args)["loss_final"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.crv.scene_head.parameters()))
        self.assertGreater(float(self.crv.fusion_gate[-1].bias.grad.abs()), 0)

    def test_final_loss_trains_original_and_masked_scenes(self):
        activate(self.crv)
        original_method = self.crv._match_scenes
        captured = {}

        def capture(*args):
            result = original_method(*args)
            result.retain_grad()
            captured["scene_logits"] = result
            return result

        with patch.object(self.crv, "_match_scenes", side_effect=capture):
            output = self.crv(**self.args)
        output["loss_final"].backward()
        self.assertGreater(float(captured["scene_logits"].grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(captured["scene_logits"].grad[:, 1:9].abs().sum()), 0)

    def test_gate_rewards_correct_evidence_even_with_positive_common_drop(self):
        positive = torch.tensor([[True, False]], device=self.device)
        negative = ~positive
        raw = torch.tensor([[8., 7.]], device=self.device)
        for drops, expected_sign in (([2., 1.], -1), ([1., 2.], 1)):
            alpha_logit = torch.tensor([[-2.]], device=self.device, requires_grad=True)
            alpha = alpha_logit.sigmoid()
            evidence = self.crv._evidence(torch.tensor([drops], device=self.device))
            loss = self.crv._pairwise_ranking(raw + alpha * evidence, positive, negative, .5)
            loss.backward()
            self.assertGreater(float(alpha_logit.grad) * expected_sign, 0)
        # Pairwise loss is invariant to arbitrary common score offsets.
        alpha = torch.tensor([[.1]], device=self.device)
        evidence = self.crv._evidence(torch.tensor([[2., 1.]], device=self.device))
        before = self.crv._pairwise_ranking(raw + alpha * evidence, positive, negative, .5)
        after = self.crv._pairwise_ranking(raw + 3 + alpha * evidence, positive, negative, .5)
        torch.testing.assert_close(before, after)

    def test_large_drops_keep_gradient_and_aux_checks_actual_evidence(self):
        drops = torch.tensor([[12., 11.]], device=self.device, requires_grad=True)
        positive = torch.tensor([[True, False]], device=self.device)
        negative = ~positive
        evidence = self.crv._evidence(drops)
        self.assertGreater(float(evidence[0, 0] - evidence[0, 1]), 0)
        cf = self.crv._pairwise_ranking(evidence, positive, negative, .5, smooth=False)
        self.assertGreater(float(cf), 0)
        alpha = torch.tensor([[.1]], device=self.device, requires_grad=True)
        fused = torch.tensor([[0., 1.]], device=self.device) + alpha * evidence
        final = self.crv._pairwise_ranking(fused, positive, negative, .5)
        (final + .1 * cf).backward()
        self.assertTrue((drops.grad.abs() > 0).all())
        self.assertLess(float(alpha.grad), 0)
        self.observations["large_drop_evidence_gap"] = float(evidence[0, 0] - evidence[0, 1])
        self.observations["large_drop_final_gradient"] = drops.grad.detach().cpu().tolist()
        self.observations["helpful_evidence_alpha_gradient"] = float(alpha.grad)

    def test_exact_fractional_padding_mean_and_boundary_core(self):
        crv = CounterfactualReferentVerifier(hidden_channels=4, verifier_dim=4,
                                           num_heads=1, grid_size=2).to(self.device)
        visual = torch.zeros(1, 4, 4, 4, device=self.device)
        visual[:, :, 2] = 100
        mask = torch.zeros(1, 4, 4, dtype=torch.bool, device=self.device)
        mask[:, 3] = True
        ratio = torch.tensor([[1., .51]], device=self.device)
        # A full-image intervention fully replaces even the partial last cell.
        scenes, _, masks, neutral = crv._scene_features(
            visual, mask, torch.tensor([[[.5, .5, 1., 1.]]], device=self.device), ratio)
        expected = 100 * .04 / 2.04
        self.assertAlmostEqual(float(neutral[0, 0, 0, 0]), expected, places=4)
        self.assertGreater(float(masks[0, 0, 2, 2]), .999)
        self.assertEqual(float(masks[0, 0, 3, 2]), 0)
        self.assertTrue(torch.isfinite(scenes).all())
        self.observations["fractional_padding_mean"] = float(neutral[0, 0, 0, 0])

    def test_topk_uses_logits_despite_sigmoid_ties(self):
        logits = torch.tensor([12. + i * .001 for i in range(9)],
                              device=self.device)[None, :, None].repeat(2, 1, 1)
        self.assertLess(int(logits.sigmoid().unique().numel()), 9)
        args = dict(self.args, refer_logits=logits, refer_scores=logits.sigmoid())
        output = self.crv(**args)
        expected = torch.arange(8, 0, -1, device=self.device)[None].repeat(2, 1)
        torch.testing.assert_close(output["indices"], expected)
        equal = dict(self.args, refer_logits=torch.ones_like(logits), refer_scores=torch.ones_like(logits).sigmoid())
        output = self.crv(**equal)
        torch.testing.assert_close(output["indices"], torch.arange(8, device=self.device)[None].repeat(2, 1))

    def test_postprocessing_and_metrics_rank_logits_not_probabilities(self):
        from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
        from propvg.apis.test import aerialvg_box_metrics
        model = MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
        torch.nn.Module.__init__(model)
        model.with_nms = False
        rank = torch.tensor([[[18.], [17.001], [17.002]]], device=self.device)
        pred = dict(
            pred_class_allobj=torch.zeros(1, 3, 2, device=self.device),
            pred_bbox_allobj=torch.tensor([[[.2, .2, 0, .1], [.4, .4, .2, .2], [.8, .8, .1, .1]]], device=self.device),
            pred_global_mask=None, refer_label=rank.sigmoid(), refer_logits=rank,
        )
        result = model.get_predictions_parts(pred, [dict(img_shape=(100, 100))], with_bbox=True, with_mask=False)
        prediction = result["pred_bboxes"][0]
        self.assertEqual(tuple(prediction["boxes"].shape), (2, 4))
        self.assertEqual(tuple(prediction["ranking_logits"].shape), (2,))
        target = prediction["boxes"][1:2].clone()
        metrics = aerialvg_box_metrics([prediction], [target])
        self.assertEqual(metrics, (100., 100., 100.))

    def test_rank_pairs_ignore_gray_candidates_and_missing_classes(self):
        logits = torch.tensor([[0., 1., 2.]], device=self.device, requires_grad=True)
        pos = torch.tensor([[True, False, False]], device=self.device)
        neg = torch.tensor([[False, True, False]], device=self.device)
        loss = self.crv._pairwise_ranking(logits, pos, neg, .5)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0)
        empty = self.crv._pairwise_ranking(logits, torch.zeros_like(pos), neg, .5)
        self.assertEqual(float(empty), 0)

    def test_cf_aux_does_not_train_gate(self):
        activate(self.crv)
        self.crv(**self.args)["loss_counterfactual"].backward()
        self.assertTrue(all(p.grad is None for p in self.crv.fusion_gate.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.crv.scene_matcher.parameters()))

    def test_inference_has_no_gt(self):
        self.crv.eval()
        with self.assertRaises(ValueError):
            self.crv(**self.args)
        arguments = {k: v for k, v in self.args.items() if k not in ("target_boxes", "target_valid")}
        output = self.crv(**arguments)
        self.assertIsNone(output["gt_support_drop_logits"])
        self.assertIsNone(output["selected_ious"])
        self.assertEqual(float(output["loss"]), 0)

    def test_padding_and_text_padding_are_ignored(self):
        activate(self.crv)
        before = self.crv(**self.args)
        modified = dict(self.args)
        modified["visual"] = torch.where(self.args["visual_mask"][:, None], 1e5, self.args["visual"].detach())
        modified["text"] = torch.where(self.args["text_mask"][:, :, None], 1e5, self.args["text"].detach())
        after = self.crv(**modified)
        torch.testing.assert_close(before["support_drop_logits"], after["support_drop_logits"], rtol=0, atol=1e-6)
        self.assertTrue(torch.isfinite(after["loss"]))

    def test_unmasked_scenes_have_zero_drop(self):
        activate(self.crv)
        with patch.object(self.crv, "_soft_box_masks", side_effect=lambda b, r, h, w, dtype=None: b.new_zeros(b.shape[0], b.shape[1], h, w)):
            output = self.crv(**self.args)
        torch.testing.assert_close(output["support_drop_logits"], torch.zeros(2, 8, device=self.device), rtol=0, atol=2e-6)

    def test_mask_and_channelwise_valid_mean(self):
        boxes = torch.tensor([[[.4, .4, .3, .3]]], device=self.device)
        ratios = torch.tensor([[1., .75]], device=self.device)
        mask = self.crv._soft_box_masks(boxes, ratios, 100, 100)[0, 0]
        self.assertGreater(float(mask[30, 40]), .999)
        self.assertEqual(float(mask[95, 95]), 0)
        _, _, masks, neutral = self.crv._scene_features(self.args["visual"].detach(), self.args["visual_mask"],
                                                      self.args["boxes"].detach(), self.args["valid_ratios"])
        valid = (~self.args["visual_mask"])[:, None]
        expected = (self.args["visual"].detach() * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True)
        torch.testing.assert_close(neutral, expected)
        core = masks[:, :1] == 1
        filled = self.args["visual"].detach() * (1 - masks[:, :1]) + neutral * masks[:, :1]
        torch.testing.assert_close(filled[core.expand_as(filled)], neutral.expand_as(filled)[core.expand_as(filled)])

    def test_tiny_box_still_has_intervention(self):
        box = torch.tensor([[[.47, .43, .008, .006]]], device=self.device)
        ratios = torch.tensor([[.83, .71]], device=self.device)
        mask = self.crv._soft_box_masks(box, ratios, 144, 144)
        self.assertGreater(float(mask.sum()), 0)
        self.assertTrue(((mask >= 0) & (mask <= 1)).all())

    def test_iou_matches_inference_clipping_at_image_edge(self):
        predicted = torch.tensor([[[.1, .5, .4, .2]]], device=self.device)
        target = torch.tensor([[.15, .5, .3, .2]], device=self.device)
        # Raw prediction extends left to -0.1; clipped box equals the GT.
        torch.testing.assert_close(self.crv._box_iou(predicted, target),
                                   torch.ones(1, 1, device=self.device), rtol=0, atol=1e-6)

    def test_strict_iou_labels_and_safe_negatives(self):
        output = self.crv(**self.args)
        self.assertTrue(output["positive_candidates"][:, 0].all())
        # Fourth box is contained in the target and has exactly IoU=0.5.
        torch.testing.assert_close(output["selected_ious"][:, 3], torch.full((2,), .5, device=self.device))
        self.assertFalse(output["positive_candidates"][:, 3].any())
        self.assertFalse(output["negative_candidates"][:, 3].any())
        self.assertTrue(output["negative_candidates"][:, 1].all())
        # Disjoint original box, but expanded mask deletes the target.
        altered = dict(self.args)
        altered["boxes"] = self.args["boxes"].detach().clone()
        altered["boxes"][:, 1] = torch.tensor([.38, .25, .13, .13], device=self.device)
        altered["boxes"][:, 2] = torch.tensor([.53, .25, .4, .4], device=self.device)
        out = self.crv(**altered)
        self.assertEqual(float(out["selected_ious"][0, 2]), 0)
        self.assertFalse(out["negative_candidates"][:, 2].any())

    def test_exact_gt_aux_when_topk_misses_target(self):
        modified = dict(self.args)
        modified["boxes"] = self.args["boxes"].detach().clone()
        modified["boxes"][:, 0] = torch.tensor([.8, .8, .06, .06], device=self.device)
        modified["boxes"][:, 3] = torch.tensor([.8, .5, .06, .06], device=self.device)
        output = self.crv(**modified)
        self.assertEqual(float(output["loss_final"]), 0)
        self.assertGreater(float(output["loss_counterfactual"]), 0)
        output["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in self.crv.parameters()))

    def test_invalid_gt_is_finite_zero_with_graph(self):
        modified = dict(self.args, target_valid=torch.zeros_like(self.args["target_valid"]),
                        target_boxes=torch.full_like(self.args["target_boxes"], float("nan")))
        output = self.crv(**modified)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(float(output["loss"]), 0)
        output["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in self.crv.parameters()))

    def test_visual_and_text_change_matching(self):
        activate(self.crv)
        before = self.crv(**self.args)
        for key in ("visual", "text"):
            modified = dict(self.args)
            modified[key] = self.args[key].detach().flip(0) * -1
            after = self.crv(**modified)
            self.assertGreater(float((before["support_drop_logits"] - after["support_drop_logits"]).abs().max()), 1e-7)

    def test_checkpoint_version_and_roundtrip(self):
        clone, _ = fixture(self.device)
        clone.load_state_dict(self.crv.state_dict(), strict=True)
        torch.testing.assert_close(clone(**self.args)["weighted_scores"], self.crv(**self.args)["weighted_scores"])
        with self.assertRaisesRegex(RuntimeError, "Old/incompatible CRV"):
            clone.load_state_dict({"gain": torch.tensor(0.)}, strict=False)
        old_version = self.crv.state_dict()
        old_version["independent_version"] = torch.tensor(1)
        with self.assertRaisesRegex(RuntimeError, "Old/incompatible CRV"):
            clone.load_state_dict(old_version, strict=False)
        clone.load_state_dict({}, strict=False)  # baseline-only initialization allowed

    def test_evaluation_checkpoint_guard_requires_trained_v2(self):
        checkpoint = importlib.import_module("propvg.utils.checkpoint")
        holder = torch.nn.Module()
        holder.head = torch.nn.Module()
        holder.head.crv = self.crv
        state = holder.state_dict()
        with patch.object(checkpoint.torch, "load", return_value={"state_dict": state}) as read, \
                patch.object(checkpoint, "is_main", return_value=False):
            result = checkpoint.load_checkpoint(holder, load_from="unused.pth", expected_crv_version=2)
            self.assertTrue(result[-1])
            self.assertEqual(read.call_count, 1)
        for marker in (None, torch.tensor(1)):
            old = dict(state)
            old.pop("head.crv.independent_version")
            if marker is not None:
                old["head.crv.independent_version"] = marker
            with patch.object(checkpoint.torch, "load", return_value={"state_dict": old}), \
                    self.assertRaisesRegex(ValueError, "trained independent-CRV v2"):
                checkpoint.load_checkpoint(holder, load_from="unused.pth", expected_crv_version=2)

    def test_config_baseline_initialization_and_rng(self):
        for filename, p in (("configs/aerialvg/PropVG-aerialvg.py", 256),
                            ("configs/aerialsense/PropVG-aerialsense.py", 128)):
            cfg = Config.fromfile(filename)
            self.assertTrue(cfg.pretty_text)
            self.assertEqual(cfg.model.head.num_queries, p)
            opts = copy.deepcopy(cfg.model.head)
            opts.update(num_queries=8, enable_crv=True)
            torch.manual_seed(135)
            enabled = build_head(opts)
            rng = torch.random.get_rng_state()
            opts["enable_crv"] = False
            torch.manual_seed(135)
            disabled = build_head(opts)
            torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
            base = {k: v for k, v in enabled.state_dict().items() if not k.startswith("crv.")}
            self.assertEqual(set(base), set(disabled.state_dict()))
            for key, value in base.items():
                torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
            self.assertIsNone(disabled.crv)
            for split in ("val", "test"):
                self.assertIsNone(getattr(cfg.data, split).get("teacher_annsfile"))

    def test_head_preserves_baseline_losses_gradients_and_no_gt_inference(self):
        cfg = Config.fromfile("configs/aerialsense/PropVG-aerialsense.py")
        opts = copy.deepcopy(cfg.model.head)
        opts.update(num_queries=12, enable_crv=True, enable_grab=False)
        torch.manual_seed(421)
        enabled = build_head(opts).to(self.device)
        opts["enable_crv"] = False
        torch.manual_seed(421)
        baseline = build_head(opts).to(self.device)
        x = torch.randn(1, 768, 8, 8, device=self.device, requires_grad=True)
        language = torch.randn(1, 8, 768, device=self.device, requires_grad=True)
        args = dict(cls_feat=torch.randn(1, 768, device=self.device), lan_feat=language,
                    lan_mask=torch.zeros(1, 8, dtype=torch.bool, device=self.device),
                    img=torch.zeros(1, 3, 128, 128, device=self.device))
        metadata = [dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])]
        targets = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]], device=self.device)], img_metas=metadata)
        captured = {}
        hook = enabled.crv.register_forward_hook(lambda m, a, kw, out: captured.update(arguments=kw, output=out), with_kwargs=True)
        torch.manual_seed(419)
        old_loss, old_pred, _ = baseline.forward_train(x, targets, **args)
        torch.manual_seed(419)
        loss, pred, _ = enabled.forward_train(x, targets, **args)
        for key in ("loss_det", "loss_refer", "loss_filter", "loss_mask"):
            torch.testing.assert_close(loss[key], old_loss[key], rtol=0, atol=0)
        indices = captured["output"]["indices"]
        torch.testing.assert_close(pred["refer_label"].squeeze(-1), old_pred["refer_label"].squeeze(-1).gather(1, indices))
        loss["loss_crv"].backward(retain_graph=True)
        self.assertIsNone(x.grad)
        self.assertIsNone(language.grad)
        self.assertTrue(all(p.grad is None for name, p in enabled.named_parameters() if not name.startswith("crv.")))
        enabled.zero_grad(set_to_none=True)
        sum(v for key, v in old_loss.items() if key.startswith("loss_")).backward()
        x.grad = language.grad = None
        sum(v for key, v in loss.items() if key.startswith("loss_")).backward()
        max_gradient_delta = 0.
        for key, parameter in enabled.named_parameters():
            if key.startswith("crv."):
                continue
            reference = dict(baseline.named_parameters())[key]
            if reference.grad is None:
                self.assertIsNone(parameter.grad, key)
            else:
                delta = float((parameter.grad - reference.grad).abs().max())
                max_gradient_delta = max(max_gradient_delta, delta)
                # The deformable CUDA extension backward uses atomic adds.
                # CPU must be bit-identical; GPU allows numerical noise only.
                torch.testing.assert_close(parameter.grad, reference.grad,
                                           rtol=1e-5 if self.device.startswith("cuda") else 0,
                                           atol=1e-5 if self.device.startswith("cuda") else 0)
        enabled.eval()
        with torch.no_grad():
            result, _ = enabled.forward_test(x, targets=dict(img_metas=metadata), with_mask=False, **args)
        self.assertNotIn("target_boxes", captured["arguments"])
        self.assertIsNone(captured["output"]["gt_support_drop_logits"])
        self.assertEqual(tuple(result["pred_bbox_allobj"].shape), (1, 8, 4))
        hook.remove()
        self.observations["baseline_gradient_max_delta"] = max_gradient_delta
        self.observations["baseline_loss_max_delta"] = max(
            float((loss[k] - old_loss[k]).abs()) for k in ("loss_det", "loss_refer", "loss_filter", "loss_mask"))
        self.observations["full_head_losses"] = {k: float(v) for k, v in loss.items()}

    def test_disabled_path_matches_git_head(self):
        reference = subprocess.check_output(["git", "show", "HEAD:propvg/models/heads/REFHead.py"], text=True)
        namespace = vars(importlib.import_module("propvg.models.heads.REFHead")).copy()
        classes = []
        for node in ast.parse(reference).body:
            if isinstance(node, ast.ClassDef) and node.name in ("CRS", "REFHead"):
                node.decorator_list = []
                classes.append(node)
        exec(compile(ast.Module(body=classes, type_ignores=[]), "git_HEAD_REFHead.py", "exec"), namespace)
        opts = copy.deepcopy(Config.fromfile("configs/aerialsense/PropVG-aerialsense.py").model.head)
        opts.pop("type")
        opts.update(num_queries=8, enable_crv=False)
        torch.manual_seed(923)
        old = namespace["REFHead"](**opts).to(self.device)
        torch.manual_seed(923)
        current = build_head(dict(type="REFHead", **opts)).to(self.device)
        current.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(1, 768, 8, 8, device=self.device)
        args = dict(cls_feat=torch.zeros(1, 768, device=self.device),
                    lan_feat=torch.randn(1, 8, 768, device=self.device),
                    lan_mask=torch.zeros(1, 8, dtype=torch.bool, device=self.device),
                    img=torch.zeros(1, 3, 128, 128, device=self.device))
        target = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.]], device=self.device)],
                      img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
        with patch.object(torch.Tensor, "cuda", lambda t: t.to(self.device)):
            torch.manual_seed(921)
            previous_loss, previous, _ = old.forward_train(x, target, **args)
            torch.manual_seed(921)
            loss, pred, _ = current.forward_train(x, target, **args)
        for key, value in previous_loss.items():
            # Diagnostics were renamed; only actual baseline losses matter.
            if key.startswith("loss_"):
                torch.testing.assert_close(value, loss[key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(previous["refer_label"], pred["refer_label"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(previous["pred_bbox_allobj"], pred["pred_bbox_allobj"], rtol=1e-5, atol=1e-6)
        self.observations["disabled_git_head_max_loss_delta"] = max(
            float((v - loss[k]).abs()) for k, v in previous_loss.items() if k.startswith("loss_"))

    def test_three_way_metrics_and_net_correction(self):
        from types import SimpleNamespace
        path = os.path.join(os.path.dirname(__file__), "eval_crv_only.py")
        spec = importlib.util.spec_from_file_location("independent_crv_eval_check", path)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)

        class Verifier(torch.nn.Module):
            def forward(self):
                return dict(raw_crs_logits=torch.tensor([[17.002, 17.001, 17.]]),
                            support_drop_logits=torch.tensor([[0., 3., -1.]]))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.crv = Verifier()

            def forward(self, **kwargs):
                assert "gt_bbox" not in kwargs
                self.crv()
                return dict(pred_bboxes=[dict(
                    boxes=torch.tensor([[10., 10., 12., 12.], [0., 0., 2., 2.], [20., 20., 22., 22.]]),
                    scores=torch.ones(3), ranking_logits=torch.tensor([17.001, 17.002, 17.]),
                )])

        model = Model()
        data = dict(img_metas=SimpleNamespace(data=[[dict(refer_target_index=[0])]]),
                    gt_bbox=SimpleNamespace(data=[[torch.tensor([[0., 0., 2., 2.]])]]))
        with patch.object(evaluator, "extract_data", lambda value: value):
            result = evaluator.evaluate(model, [data], model.crv)
        self.assertEqual(result["crs_only"], (0., 100., 0.))
        self.assertEqual(result["fused"], (100., 100., 100.))
        self.assertEqual(result["crv_only"], (100., 100., 100.))
        self.assertEqual(result["crs_wrong_fused_correct"], 1)
        self.assertEqual(result["crs_correct_fused_wrong"], 0)

    def test_two_rank_training(self):
        if not self.ddp:
            self.skipTest("enable with --ddp")
        with tempfile.TemporaryDirectory(prefix="independent_crv_ddp_") as directory:
            torch.multiprocessing.spawn(ddp_worker, args=(directory, self.backend), nprocs=2, join=True)
            reports = []
            for rank in range(2):
                with open(os.path.join(directory, "rank%d.json" % rank), encoding="utf-8") as handle:
                    reports.append(json.load(handle))
            self.assertEqual(reports[0]["parameters"], reports[1]["parameters"])
            self.observations["ddp"] = dict(backend=self.backend, ranks=reports)


class DistributedFixture(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.crv, _ = fixture(device, batch=1)
        self.mrsm = CRS(32).to(device)

    def forward(self, arguments):
        queries = arguments["visual"].flatten(2).transpose(1, 2)[:, :9]
        probability, _, logits = self.mrsm(queries, arguments["text"], arguments["text_mask"], return_logits=True)
        arguments = dict(arguments, refer_scores=probability, refer_logits=logits)
        result = self.crv(**arguments)
        labels = torch.zeros_like(probability)
        labels[:, 0] = 1
        return result["loss"] + torch.nn.functional.binary_cross_entropy(probability, labels)


def ddp_worker(rank, directory, backend):
    torch.set_num_threads(1)
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        torch.cuda.set_per_process_memory_fraction(.08, rank)
        device = "cuda:%d" % rank
    torch.distributed.init_process_group(backend, init_method="file://" + os.path.join(directory, "rendezvous"),
                                        rank=rank, world_size=2)
    try:
        model = torch.nn.parallel.DistributedDataParallel(DistributedFixture(device),
                                                         device_ids=[rank] if backend == "nccl" else None)
        _, args = fixture(device, batch=1)
        args["visual"] = args["visual"].detach() + rank * .01
        args["text"] = args["text"].detach()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(args)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None for p in model.parameters())
            optimizer.step()
        params = [float(p.detach().sum()) for p in model.parameters()]
        with open(os.path.join(directory, "rank%d.json" % rank), "w", encoding="utf-8") as handle:
            json.dump(dict(rank=rank, loss=float(loss), parameters=params), handle)
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--report", default="reports/independent_crv_checks.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(.08, args.device)
    CRVChecks.device, CRVChecks.ddp, CRVChecks.backend = args.device, args.ddp, args.backend
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CRVChecks))
    report = dict(device=args.device, tests=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), skipped=len(result.skipped), passed=result.wasSuccessful(),
                  observations=CRVChecks.observations)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
