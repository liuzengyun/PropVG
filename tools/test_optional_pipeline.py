"""CPU regressions for optional branches and candidate/score alignment."""
import copy
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import numpy as np
import pycocotools.mask as mask_utils
import torch
from mmcv import Config
from propvg.models import build_head
from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
from propvg.apis.test import aerialvg_box_metrics
from propvg.utils.checkpoint import _load_optimizer_state, _optimizer_parameter_names


class OptionalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        cls.options["enable_crv"] = False  # Keep baseline regressions independent of the user's run switch.
        # These regressions isolate legacy Grab/mask behavior from BCEM.
        cls.options["enable_bcem"] = False
        cls.options["num_queries"] = 8

    def head(self, grab, mask, legacy=False):
        options = dict(self.options, enable_grab=True if legacy else grab, enable_mask=True if legacy else mask)
        torch.manual_seed(713)
        head = build_head(options)
        # The historical model allocated every branch, even when not used.
        head.enable_grab = grab
        head.enable_mask = mask
        return head

    def inputs(self, mask=False):
        torch.manual_seed(512)
        x = torch.randn(1, 768, 8, 8)
        metas = [dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])]
        rles = None
        if mask:
            binary = np.zeros((128, 128), dtype=np.uint8)
            binary[30:50, 40:64] = 1
            rles = [mask_utils.encode(np.asfortranarray(binary))]
        target = dict(mask=rles, bbox=[torch.tensor([[40., 30., 64., 50.], [5., 5., 20., 20.]])], img_metas=metas)
        args = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                    lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), img=torch.zeros(1, 3, 128, 128))
        return x, target, args

    def test_initialization_rng_and_forward_backward_are_identical(self):
        for grab in (False, True):
            for mask in (False, True):
                with self.subTest(grab=grab, mask=mask):
                    old = self.head(grab, mask, legacy=True)
                    old_rng = torch.get_rng_state()
                    new = self.head(grab, mask)
                    torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
                    for name, value in new.state_dict().items():
                        torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)
                    self.assertEqual(new.img_embedding.out_channels, 512)
                    self.assertEqual(new.grab_binding is None, not grab)
                    self.assertEqual(new.unet_decoder is None, not mask)
                    self.assertEqual(new.seg_branch is None, not mask)
                    x, target, args = self.inputs(mask)
                    torch.manual_seed(817)
                    old_loss, old_pred, _ = old.forward_train(x, target, **args)
                    sum(old_loss.values()).backward()
                    torch.manual_seed(817)
                    new_loss, new_pred, _ = new.forward_train(x, target, **args)
                    sum(new_loss.values()).backward()
                    for key in new_loss:
                        torch.testing.assert_close(new_loss[key], old_loss[key], rtol=0, atol=0)
                    for key in ("pred_bbox_allobj", "refer_label", "pred_global_mask"):
                        if new_pred[key] is not None:
                            torch.testing.assert_close(new_pred[key], old_pred[key], rtol=0, atol=0)
                    old_parameters = dict(old.named_parameters())
                    for name, parameter in new.named_parameters():
                        expected = old_parameters[name].grad
                        if expected is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
                    old.eval()
                    new.eval()
                    with torch.no_grad():
                        expected, _ = old.forward_test(x, targets=target, with_mask=mask, **args)
                        actual, _ = new.forward_test(x, targets=target, with_mask=mask, **args)
                    for key in ("pred_bbox_allobj", "refer_label", "pred_global_mask"):
                        if actual[key] is not None:
                            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_disabled_mask_rejects_mask_requests(self):
        head = self.head(False, False)
        x, target, args = self.inputs(True)
        with self.assertRaisesRegex(ValueError, "enable_mask=True"):
            head.forward_train(x, target, **args)
        with self.assertRaisesRegex(ValueError, "enable_mask=True"):
            head.forward_test(x, targets=target, with_mask=True, **args)

    def test_old_and_named_optimizer_resume_preserve_next_update(self):
        x, target, args = self.inputs()
        old = self.head(False, False, legacy=True)
        optimizer = torch.optim.Adam(old.parameters(), lr=1e-4, amsgrad=True)
        torch.manual_seed(823)
        losses, _, _ = old.forward_train(x, target, **args)
        sum(losses.values()).backward()
        torch.nn.utils.clip_grad_norm_(old.parameters(), .15)
        optimizer.step()
        checkpoint = dict(state_dict=copy.deepcopy(old.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()))
        old_names = _optimizer_parameter_names(old, optimizer)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(824)
        losses, _, _ = old.forward_train(x, target, **args)
        sum(losses.values()).backward()
        torch.nn.utils.clip_grad_norm_(old.parameters(), .15)
        optimizer.step()
        for named in (False, True):
            with self.subTest(named=named):
                current = self.head(False, False)
                current.load_state_dict(checkpoint["state_dict"], strict=False)
                current_optimizer = torch.optim.Adam(current.parameters(), lr=1e-4, amsgrad=True)
                saved = dict(checkpoint)
                if named:
                    saved["optimizer_param_names"] = old_names
                _load_optimizer_state(current, current_optimizer, saved, saved["state_dict"])
                torch.manual_seed(824)
                losses, _, _ = current.forward_train(x, target, **args)
                sum(losses.values()).backward()
                torch.nn.utils.clip_grad_norm_(current.parameters(), .15)
                current_optimizer.step()
                for name, value in current.state_dict().items():
                    torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)

    def predictions(self, empty=False, logits=False, nms=False, clipped=False):
        model = MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
        torch.nn.Module.__init__(model)
        model.with_nms = nms
        boxes = [[.25, .25, .2, .2], [1.2, .8, .1, .1] if clipped else [.8, .8, 0., .1], [.6, .6, .2, .2]]
        if empty:
            boxes = [[.5, .5, 0., .1]] * 3
        prediction = dict(pred_class_allobj=torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]]),
                          pred_bbox_allobj=torch.tensor([boxes]), refer_label=torch.tensor([[[.2], [.9], [.3]]]))
        if logits:
            prediction["refer_logits"] = torch.tensor([[[-2.], [5.], [1.]]])
        return model.get_predictions_parts(prediction, [dict(img_shape=(100, 100), scale_factor=[2.] * 4)],
                                           rescale=True, with_bbox=True)["pred_bboxes"][0]

    def test_box_filter_keeps_probabilities_logits_and_nms_aligned(self):
        for logits in (False, True):
            for nms in (False, True):
                for clipped in (False, True):
                    with self.subTest(logits=logits, nms=nms, clipped=clipped):
                        prediction = self.predictions(logits=logits, nms=nms, clipped=clipped)
                        self.assertEqual(len(prediction["boxes"]), 2)
                        torch.testing.assert_close(prediction["scores"], torch.tensor([.2, .3]))
                        if logits:
                            torch.testing.assert_close(prediction["ranking_logits"], torch.tensor([-2., 1.]))
                        metrics = aerialvg_box_metrics([prediction], [torch.tensor([[25., 25., 35., 35.]])])
                        self.assertEqual(metrics[:2], (100., 100.))
                        self.assertAlmostEqual(metrics[2], 100., delta=1e-4)
                        if nms:
                            self.assertEqual(len(prediction["filtered_boxes"]), len(prediction["filtered_scores"]))

    def test_all_boxes_removed_counts_as_failure(self):
        prediction = self.predictions(empty=True)
        self.assertEqual(len(prediction["scores"]), 0)
        self.assertEqual(aerialvg_box_metrics([prediction], [torch.tensor([[25., 25., 35., 35.]])]), (0., 0., 0.))

    def test_three_optimizer_groups_and_ddp_checkpoint_names(self):
        def wrapper(legacy):
            model = torch.nn.Module()
            model.vis_enc = torch.nn.Linear(2, 2)
            model.lan_enc = torch.nn.Linear(2, 2)
            model.head = self.head(False, False, legacy=legacy)
            return model

        def optimizer(model):
            return torch.optim.Adam([
                dict(params=list(model.vis_enc.parameters()), lr=4e-5),
                dict(params=list(model.lan_enc.parameters()), lr=4e-4),
                dict(params=list(model.head.parameters()), lr=4e-4),
            ], amsgrad=True)

        old = wrapper(True)
        old_optimizer = optimizer(old)
        # Distinct states expose any accidental reassignment to another tensor.
        for index, parameter in enumerate(old.parameters()):
            if not parameter.shape or parameter.shape[0] < 4:
                parameter.grad = torch.full_like(parameter, float(index + 1))
        old_optimizer.step()
        checkpoint = dict(optimizer=old_optimizer.state_dict())
        state = {"module." + key: value for key, value in old.state_dict().items()}
        current = wrapper(False)
        current_optimizer = optimizer(current)
        _load_optimizer_state(current, current_optimizer, checkpoint, state)
        old_parameters = dict(old.named_parameters())
        for name, parameter in current.named_parameters():
            expected = old_optimizer.state[old_parameters[name]]
            actual = current_optimizer.state[parameter]
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual([group["lr"] for group in current_optimizer.param_groups], [4e-5, 4e-4, 4e-4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
