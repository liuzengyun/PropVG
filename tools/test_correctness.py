"""CPU regressions for detached correctness estimation and legacy compatibility."""
import copy
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import numpy as np
import torch
from mmcv import Config
from torch import nn

from propvg.models import build_head
from propvg.models.heads.top1_correctness import Top1CorrectnessHead
from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
from propvg.apis.test import aerialvg_box_metrics
from propvg.apis.train import clip_training_gradients, train_model
from propvg.evaluations.correctness import binary_auc, correctness_record, summarize_correctness
from propvg.utils.checkpoint import _validate_correctness_checkpoint, _optimizer_parameter_names, _load_optimizer_state


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(1231)
        query = torch.randn(2, 4, 16, requires_grad=True)
        boxes = torch.tensor([[[.5, .5, .4, .4], [.2, .2, .1, .1], [.51, .5, .4, .4], [.8, .8, .1, .1]]] * 2,
                             requires_grad=True)
        text = torch.randn(2, 6, 16, requires_grad=True)
        mask = torch.tensor([[False, False, False, True, True, True]] * 2)
        logits = torch.tensor([[[3.], [1.], [2.], [-2.]]] * 2, requires_grad=True)
        scores = logits.sigmoid().detach().requires_grad_(True)
        classes = torch.randn(2, 4, 2, requires_grad=True)
        metas = [dict(img_shape=(80, 100), pad_shape=(100, 100), refer_target_index=[0])] * 2
        return query, boxes, text, mask, scores, logits, classes, metas

    def test_detached_all_inputs_true_iou_supervision_and_finite_gradients(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        args = self.inputs()
        observation = head(*args)
        # Second sample's teacher-like box matches the prediction, but the
        # actual refer GT is index 1 and must label this sample as wrong.
        metas = [dict(args[-1][0]), dict(args[-1][1], refer_target_index=[1])]
        targets = [torch.tensor([[30., 24., 70., 56.]], requires_grad=True),
                   torch.tensor([[30., 24., 70., 56.], [0., 0., 10., 10.]], requires_grad=True)]
        loss, metrics = head.loss(observation, targets, metas)
        expected = nn.functional.binary_cross_entropy_with_logits(observation['logit'], torch.tensor([1., 0.]))
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics['valid_samples'].item(), 2)
        loss.backward()
        for value in (args[0], args[1], args[2], args[4], args[5], args[6], *targets):
            self.assertIsNone(value.grad)
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_text_padding_and_forward_does_not_need_gt(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8).eval()
        args = list(self.inputs())
        first = head(*args)
        args[2] = args[2].detach().clone()
        args[2][args[3]] = 10000
        second = head(*args)
        for key in ('probability', 'logit', 'index', 'entropy'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_clipped_degenerate_boxes_priority_and_ties_match_evaluator(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        args = list(self.inputs())
        args[1] = args[1].detach().clone()
        args[1][:, 0] = torch.tensor([1.2, .5, .1, .1])
        args[4] = torch.tensor([[[.99], [.5], [.5], [.1]]] * 2)
        result = head(*args)
        for batch in range(2):
            valid = torch.tensor([1, 2, 3])
            order = args[4][batch, valid, 0].argsort(descending=True, stable=False)
            self.assertEqual(result['index'][batch], valid[order[0]])
        priority = torch.tensor([[[-float('inf')], [0.], [0.], [10.]]] * 2)
        result = head(*args, ranking_priority=priority)
        torch.testing.assert_close(result['index'], torch.tensor([3, 3]))
        for batch in range(2):
            prediction = dict(boxes=result['clipped_boxes'][batch, 1:], scores=args[4][batch, 1:, 0],
                              ranking_logits=priority[batch, 1:, 0], top1_correct_probability=result['probability'][batch],
                              top1_correct_logit=result['logit'][batch], top1_grouped_entropy=result['entropy'][batch])
            gt = result['clipped_boxes'][batch, 3]
            self.assertEqual(correctness_record(prediction, gt)['correct'], 1)
            self.assertEqual(aerialvg_box_metrics([prediction], [gt])[0], 100.)

    def test_empty_predictions_and_missing_refer_gt(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        args = list(self.inputs())
        for index in (0, 1, 4, 5, 6):
            args[index] = args[index][:, :0]
        result = head(*args)
        torch.testing.assert_close(result['index'], torch.tensor([-1, -1]))
        torch.testing.assert_close(result['probability'], torch.zeros(2))
        targets = [torch.tensor([[0., 0., 10., 10.]])] * 2
        loss, metrics = head.loss(result, targets, args[-1])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['accuracy'], 1.)
        metas = [dict(meta, refer_target_index=[]) for meta in args[-1]]
        loss, metrics = head.loss(result, targets, metas)
        self.assertEqual(loss, 0.)
        self.assertEqual(metrics['valid_samples'], 0.)
        loss.backward()

    def test_exact_iou_threshold(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        raw = torch.tensor([0.], requires_grad=True)
        observation = dict(logit=raw, probability=raw.sigmoid(), index=torch.tensor([0]),
                           clipped_boxes=torch.tensor([[[0., 0., 10., 10.]]]))
        loss, _ = head.loss(observation, [torch.tensor([[0., 0., 5., 10.]])], [dict(refer_target_index=[0])])
        loss.backward()
        self.assertLess(raw.grad.item(), 0.)  # IoU==.5 must be positive.

    def test_half_precision_box_clipping_matches_real_postprocessor(self):
        from propvg.core.structure.postprocessing import detector_postprocess
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        boxes = torch.tensor([[[.5332, .2144, .1123, .0879], [.99, .8, .2, .3]]], dtype=torch.float16)
        meta = [dict(img_shape=(787, 1439), pad_shape=(1440, 1440))]
        actual = head.clipped_boxes(boxes, meta)[0]
        wrapper = MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
        nn.Module.__init__(wrapper)
        predictions = wrapper.inference(torch.zeros(1, 2, 2), boxes, [meta[0]['img_shape']])
        expected = detector_postprocess(predictions[0], 787, 1439).pred_boxes.tensor
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_autocast_forward_backward_are_finite(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        args = self.inputs()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            observation = head(*args)
            loss, _ = head.loss(observation, [torch.tensor([[30., 24., 70., 56.]])] * 2, args[-1])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(observation['clipped_boxes'].dtype, torch.float32)
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in head.parameters()))

    def test_global_mean_scaling_with_missing_gt_on_other_rank(self):
        head = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        args = self.inputs()
        result = head(*args)
        targets = [torch.tensor([[30., 24., 70., 56.]])] * 2
        metas = [dict(args[-1][0]), dict(args[-1][1], refer_target_index=[])]
        with patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=2), \
                patch('torch.distributed.all_reduce', side_effect=lambda count: count.add_(2.)):
            loss, metrics = head.loss(result, targets, metas)
        expected = nn.functional.softplus(-result['logit'][0]) * 2 / 3
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics['valid_samples'], 3.)

    def test_auc_ties_and_metrics_are_not_majority_class_accuracy(self):
        labels = np.array([0, 1, 1, 0, 1], dtype=bool)
        scores = np.array([.2, .2, .8, .8, .9])
        expected = np.mean([(a > b) + .5 * (a == b) for a in scores[labels] for b in scores[~labels]])
        self.assertAlmostEqual(binary_auc(labels, scores), expected)
        self.assertIsNone(binary_auc([1, 1], [.2, .8]))
        records = [dict(probability=.99, correct=int(i < 7), logit=math.log(99), entropy=.4) for i in range(10)]
        metrics = summarize_correctness(records)
        self.assertEqual(metrics['accuracy'], 70.)
        self.assertEqual(metrics['majority_accuracy'], 70.)
        self.assertEqual(metrics['error_recall'], 0.)
        self.assertEqual(metrics['correct_fpr'], 0.)
        self.assertIsNone(metrics['error_precision'])
        self.assertEqual(metrics['auroc'], .5)
        self.assertIsNone(summarize_correctness([]))

    def test_checkpoint_guards_disabled_old_trained_and_untrained(self):
        model = nn.Module()
        model.head = nn.Module()
        model.head.correctness = None
        _validate_correctness_checkpoint(model, {})
        model.head.correctness = Top1CorrectnessHead(hidden_channels=16, hidden_dim=8)
        with self.assertRaisesRegex(ValueError, 'enable_correctness=False'):
            _validate_correctness_checkpoint(model, {})
        with self.assertRaisesRegex(RuntimeError, 'untrained'):
            model.head.correctness.require_trained()
        with self.assertRaises(ValueError):
            _validate_correctness_checkpoint(model, model.state_dict())
        model.head.correctness.mark_optimizer_step()
        _validate_correctness_checkpoint(model, model.state_dict())
        _validate_correctness_checkpoint(SimpleNamespace(module=model), {'module.' + k: v for k, v in model.state_dict().items()})
        model.head.correctness.require_trained()
        incomplete = dict(model.state_dict())
        incomplete.pop('head.correctness.classifier.3.weight')
        with self.assertRaises(ValueError):
            _validate_correctness_checkpoint(model, incomplete)


class CorrectnessIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.options = copy.deepcopy(Config.fromfile('configs/aerialvg/PropVG-aerialvg.py').model.head)
        cls.options["enable_birs"] = False
        cls.options.update(num_queries=8, enable_mask=False, enable_crv=False, enable_relation_refinement=False,
                           correctness=dict(hidden_dim=16, topk=5))
        cls.options['bcem'] = dict(context_dim=16, sampling_mode='ray', num_rays=4,
                                   radial_samples=4, min_valid_points=2, chunk_size=2, checkpoint_chunks=False)

    def inputs(self):
        torch.manual_seed(1237)
        x = torch.randn(1, 768, 8, 8)
        target = dict(mask=None, bbox=[torch.tensor([[40., 30., 64., 50.], [5., 5., 20., 20.]])],
                      img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128), refer_target_index=[0])])
        args = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                    lan_mask=torch.tensor([[False] * 4 + [True] * 4]), img=torch.zeros(1, 3, 128, 128))
        return x, target, args

    def test_base_initialization_rng_predictions_gradients_clipping_and_updates_are_unchanged(self):
        for grab, bcem in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(grab=grab, bcem=bcem):
                options = dict(self.options, enable_grab=grab, enable_bcem=bcem)
                torch.manual_seed(1249)
                baseline = build_head(dict(options, enable_correctness=False))
                initial_rng = torch.get_rng_state()
                torch.manual_seed(1249)
                observed = build_head(dict(options, enable_correctness=True))
                torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(observed.state_dict()[name], value, rtol=0, atol=0)
                # Loading legacy states remains strict-compatible when off.
                old_options = dict(options)
                old_options.pop('correctness', None)
                old_options.pop('enable_correctness', None)
                legacy = build_head(old_options)
                legacy.load_state_dict(baseline.state_dict(), strict=True)
                x, target, args = self.inputs()
                torch.manual_seed(1259)
                losses_a, predictions_a, _ = baseline.forward_train(x, target, **args)
                next_rng = torch.get_rng_state()
                torch.manual_seed(1259)
                losses_b, predictions_b, _ = observed.forward_train(x, target, **args)
                torch.testing.assert_close(torch.get_rng_state(), next_rng, rtol=0, atol=0)
                for name, value in losses_a.items():
                    torch.testing.assert_close(losses_b[name], value, rtol=0, atol=0)
                sum(value for name, value in losses_a.items() if name.startswith('loss_')).backward()
                sum(value for name, value in losses_b.items() if name.startswith('loss_')).backward()
                clip_training_gradients(baseline, .15)
                clip_training_gradients(observed, .15)
                parameters = dict(observed.named_parameters())
                for name, parameter in baseline.named_parameters():
                    if parameter.grad is None:
                        self.assertIsNone(parameters[name].grad, name)
                    else:
                        torch.testing.assert_close(parameters[name].grad, parameter.grad, rtol=0, atol=0)
                for name in ('pred_bbox_allobj', 'refer_label'):
                    torch.testing.assert_close(predictions_a[name], predictions_b[name], rtol=0, atol=0)
                a = torch.optim.Adam(baseline.parameters(), lr=1e-4)
                b = torch.optim.Adam(observed.parameters(), lr=1e-4)
                a.step(); b.step()
                observed.correctness.mark_optimizer_step()
                for name, parameter in baseline.named_parameters():
                    torch.testing.assert_close(parameters[name], parameter, rtol=0, atol=0)
                baseline.eval(); observed.eval()
                with torch.no_grad():
                    pred_a, _ = baseline.forward_test(x, targets=target, with_mask=False, **args)
                    pred_b, _ = observed.forward_test(x, targets=target, with_mask=False, **args)
                    poison = dict(target, bbox=[torch.full((2, 4), 100000.)])
                    pred_poison, _ = observed.forward_test(x, targets=poison, with_mask=False, **args)
                for name in pred_a:
                    if pred_a[name] is not None:
                        torch.testing.assert_close(pred_b[name], pred_a[name], rtol=0, atol=0)
                torch.testing.assert_close(pred_b['top1_correct_probability'], pred_poison['top1_correct_probability'], rtol=0, atol=0)
                wrapper = MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
                nn.Module.__init__(wrapper)
                wrapper.with_nms = False
                converted = wrapper.get_predictions_parts(pred_b, target['img_metas'], with_bbox=True, with_mask=False)
                record = correctness_record(converted['pred_bboxes'][0], target['bbox'][0][0])
                metric = aerialvg_box_metrics(converted['pred_bboxes'], [target['bbox'][0][0]])[0]
                self.assertEqual(record['correct'] * 100, metric)

    def test_named_optimizer_resume_can_disable_correctness(self):
        def model(enabled):
            result = nn.Module()
            result.base = nn.Linear(3, 2)
            result.head = nn.Module()
            result.head.correctness = Top1CorrectnessHead(hidden_channels=4, hidden_dim=4) if enabled else None
            return result
        old, current = model(True), model(False)
        original = torch.optim.Adam(old.parameters(), lr=1e-4)
        for parameter in old.parameters():
            parameter.grad = torch.ones_like(parameter)
        original.step()
        saved = dict(optimizer=original.state_dict(), optimizer_param_names=_optimizer_parameter_names(old, original))
        optimizer = torch.optim.Adam(current.parameters(), lr=1e-4)
        _load_optimizer_state(current, optimizer, saved, old.state_dict())
        for name, parameter in current.named_parameters():
            expected = original.state[dict(old.named_parameters())[name]]
            for key, value in optimizer.state[parameter].items():
                torch.testing.assert_close(value, expected[key])

    def test_correctness_coexists_with_crv_without_changing_crv_gradients_or_ranking(self):
        options = dict(self.options, enable_grab=False, enable_bcem=False, enable_crv=True,
                       crv=dict(scorer_dim=16, scorer_heads=4, scorer_layers=1, grid_size=4,
                                topk=5, uncertainty_threshold=0.))
        torch.manual_seed(1301)
        baseline = build_head(dict(options, enable_correctness=False))
        torch.manual_seed(1301)
        observed = build_head(dict(options, enable_correctness=True))
        x, target, args = self.inputs()
        torch.manual_seed(1303)
        losses_a, _, _ = baseline.forward_train(x, target, **args)
        torch.manual_seed(1303)
        losses_b, _, _ = observed.forward_train(x, target, **args)
        for key, value in losses_a.items():
            torch.testing.assert_close(losses_b[key], value, rtol=0, atol=0)
        sum(v for k, v in losses_a.items() if k.startswith('loss_')).backward()
        sum(v for k, v in losses_b.items() if k.startswith('loss_')).backward()
        clip_training_gradients(baseline, .15)
        clip_training_gradients(observed, .15)
        parameters = dict(observed.named_parameters())
        for name, parameter in baseline.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(parameters[name].grad)
            else:
                torch.testing.assert_close(parameters[name].grad, parameter.grad, rtol=0, atol=0)
        observed.correctness.mark_optimizer_step()
        baseline.eval(); observed.eval()
        with torch.no_grad():
            first, _ = baseline.forward_test(x, targets=target, with_mask=False, **args)
            second, _ = observed.forward_test(x, targets=target, with_mask=False, **args)
        for name, value in first.items():
            if value is not None:
                torch.testing.assert_close(second[name], value, rtol=0, atol=0)
        wrapper = MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
        nn.Module.__init__(wrapper)
        wrapper.with_nms = False
        wrapper.get_predictions_parts(second, target['img_metas'], with_bbox=True, with_mask=False)

    def test_real_training_loop_optimizes_head_marks_checkpoint_and_logs_loss(self):
        class Wrapper(nn.Module):
            def __init__(wrapper, options):
                super().__init__()
                wrapper.head = build_head(options)

            def forward(wrapper, x, target, epoch=None, rescale=False, **kwargs):
                losses, _, _ = wrapper.head.forward_train(x, target, **kwargs)
                return losses, None

        model = Wrapper(dict(self.options, enable_grab=False, enable_bcem=False, enable_correctness=True))
        before = {name: parameter.detach().clone() for name, parameter in model.head.correctness.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        x, target, args = self.inputs()
        cfg = SimpleNamespace(distributed=False, use_fp16=False, grad_norm_clip=.15,
                              log_interval=1, data=SimpleNamespace(samples_per_gpu=1))
        with patch('propvg.apis.train.extract_data', side_effect=lambda value: value), \
                patch('propvg.apis.train.get_root_logger'), patch('propvg.apis.train.set_tracking_context'), \
                patch('propvg.apis.train.log_metrics') as tracking:
            train_model(0, cfg, model, optimizer, [dict(x=x, target=target, **args)])
        self.assertEqual(model.head.correctness.trained_steps.item(), 1)
        self.assertTrue(any(not torch.equal(parameter, before[name])
                            for name, parameter in model.head.correctness.named_parameters()))
        observed = tracking.call_args[0][0]
        self.assertIn('train/loss_correctness', observed)
        self.assertIn('correctness_accuracy', observed)
        _validate_correctness_checkpoint(model, model.state_dict())

    def test_both_evaluators_keep_old_returns_and_log_new_head_without_gt_leak(self):
        class Loader:
            dataset = SimpleNamespace(which_set='val')

            def __len__(self):
                return 1

            def __iter__(self):
                meta = dict(img_shape=(50, 50), pad_shape=(50, 50), refer_target_index=[0])
                yield dict(img_metas=SimpleNamespace(data=[[dict(meta) for _ in range(4)]]),
                           gt_bbox=SimpleNamespace(data=[[torch.tensor([[10., 10., 20., 20.]]) for _ in range(4)]]))

        class Model(nn.Module):
            def __init__(self, enabled):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.head = nn.Module()
                self.head.correctness = Top1CorrectnessHead(hidden_channels=4, hidden_dim=4) if enabled else None

            def forward(self, **kwargs):
                assert 'gt_bbox' not in kwargs
                predictions = []
                for index, probability in enumerate((.9, .2, .1, .8)):
                    first = [10., 10., 20., 20.] if index < 2 else [30., 30., 40., 40.]
                    prediction = dict(boxes=torch.tensor([first, [10., 10., 20., 20.]]), scores=torch.tensor([.8, .1]))
                    if self.head.correctness is not None:
                        prediction.update(top1_correct_probability=torch.tensor(probability),
                                          top1_correct_logit=torch.tensor(math.log(probability / (1 - probability))),
                                          top1_grouped_entropy=torch.tensor((.1, .9, .8, .2)[index]))
                    predictions.append(prediction)
                return dict(pred_bboxes=predictions, pred_masks=[])

        from propvg.apis.test import evaluate_model
        spec = importlib.util.spec_from_file_location('correctness_eval_tool', Path(__file__).with_name('eval_aerialvg.py'))
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        cfg = SimpleNamespace(dataset='AerialVG', distributed=False)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = Model(enabled)
                with patch('propvg.apis.test.extract_data', side_effect=lambda value: value), \
                        patch('propvg.apis.test.get_root_logger'), patch('propvg.apis.test.log_evaluation'), \
                        patch('propvg.apis.test.log_metrics') as tracking:
                    self.assertEqual(evaluate_model('val', cfg, model, Loader()), (50., 100., 50.))
                self.assertEqual(tracking.call_count, int(enabled))
                if enabled:
                    metrics = tracking.call_args[0][0]
                    self.assertEqual(metrics['val/correctness/accuracy'], 50.)
                    self.assertEqual(metrics['val/correctness/auroc'], .75)
                with patch.object(evaluator, 'extract_data', side_effect=lambda value: value), \
                        patch.object(evaluator, 'get_root_logger'):
                    self.assertEqual(evaluator.evaluate(model, Loader()), (50., 100., 50.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
