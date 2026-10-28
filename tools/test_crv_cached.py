"""CPU checks for offline CRV, full checkpoints and legacy inference isolation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from mmcv import Config
from mmcv.parallel import DataContainer

from propvg.apis.crv_cached import (
    CACHE_VERSION, CachedCRVDataset, FEATURE_KEYS, assemble_checkpoint, atomic_json_save, atomic_torch_save,
    build_crv, cache_identity, cached_predictions, call_cached_crv, collate_cached,
    cleanup_feature_caches, extract_frozen_inputs, feature_cache_lock, file_sha256, live_feature_batch,
    load_baseline_state, load_configured_baseline, load_frozen_baseline, make_record,
)
from propvg.apis.test import aerialvg_box_metrics
from propvg.apis.bcem_baseline_validation import bcem_config_contract
from propvg.models import build_head
from propvg.models.heads.top1_correctness import Top1CorrectnessHead
from propvg.models.heads.background_context import BackgroundContextEnhancement
from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
from propvg.utils.checkpoint import _validate_correctness_checkpoint, _validate_crv_checkpoint

spec = importlib.util.spec_from_file_location("train_crv_cached_tested", "tools/train_crv_cached.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CachedCRVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def features(self, batch=2):
        torch.manual_seed(17)
        scores = torch.linspace(.65, .50, 6)[None, :, None].expand(batch, -1, -1).clone()
        boxes = torch.tensor([[[.1 + i * .14, .5, .08, .1] for i in range(6)]]).expand(batch, -1, -1).clone()
        features = dict(visual=torch.randn(batch, 16, 24, 24), visual_mask=torch.zeros(batch, 24, 24, dtype=torch.bool),
                        text=torch.randn(batch, 6, 16), text_mask=torch.tensor([[0, 0, 0, 0, 1, 1]]).expand(batch, -1).clone(),
                        boxes=boxes, candidate_features=torch.randn(batch, 6, 16), probabilities=scores,
                        logits=torch.logit(scores))
        meta = dict(img_shape=(80, 100, 3), pad_shape=(100, 100, 3), ori_shape=(80, 100, 3),
                    filename="test.jpg", expression="the target", scale_factor=[1., 1., 1., 1.], refer_target_index=[0])
        inputs = dict(img_metas=[copy.deepcopy(meta) for _ in range(batch)],
                      gt_bbox=[torch.tensor([[6., 36., 14., 44.], [40., 10., 60., 30.]]) for _ in range(batch)])
        return features, inputs

    def options(self):
        return dict(topk=5, uncertainty_topn=5, grid_size=6, scorer_dim=16, scorer_heads=4,
                    scorer_layers=1, chunk_size=4, start_epoch=1, uncertainty_threshold=.6)

    def crv(self):
        return build_crv(Config(dict(model=dict(head=dict(hidden_channels=16, crv=self.options())))))

    def test_record_exactness_gt_and_independent_storage(self):
        features, inputs = self.features()
        record = make_record(features, inputs, 0, 0, "test")
        for key in FEATURE_KEYS:
            torch.testing.assert_close(record[key], features[key][0], atol=0, rtol=0)
            self.assertFalse(record[key].requires_grad)
            self.assertEqual(record[key].untyped_storage().nbytes(), record[key].numel() * record[key].element_size())
        torch.testing.assert_close(record["refer_gt"], torch.tensor([.06, .45, .14, .55]))
        half = make_record(features, inputs, 0, 0, "test", "float16")
        self.assertEqual(half["visual"].dtype, torch.float16)
        for key in ("boxes", "logits", "probabilities", "refer_gt"):
            self.assertEqual(half[key].dtype, torch.float32)
        inputs["gt_bbox"][0] = torch.tensor([[6.123456789, 36.987654321, 14.012345678, 44.123456789]],
                                              dtype=torch.float64)
        double_gt = make_record(features, inputs, 0, 0, "test")
        expected_gt = (inputs["gt_bbox"][0][0] / torch.tensor([100, 80, 100, 80], dtype=torch.float64)).float()
        torch.testing.assert_close(double_gt["refer_gt"], expected_gt, rtol=0, atol=0)

    def test_online_and_cached_crv_are_identical_and_no_gt_in_eval(self):
        features, inputs = self.features()
        batch = collate_cached([make_record(features, inputs, index, index, "test") for index in range(2)])
        left, right = self.crv(), self.crv()
        before = copy.deepcopy(features)
        result = left(**features, img_metas=inputs["img_metas"], refer_gt=batch["refer_gt"], epoch=0)
        cached = call_cached_crv(right, batch, torch.device("cpu"), training=True, epoch=0)
        for index in (0, 1):
            torch.testing.assert_close(result[index], cached[index], rtol=0, atol=0)
        for key in result[2]:
            torch.testing.assert_close(result[2][key], cached[2][key], rtol=0, atol=0)
        sum(result[2].values()).backward()
        sum(cached[2].values()).backward()
        for old, new in zip(left.parameters(), right.parameters()):
            if old.grad is not None:
                torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
        for key in FEATURE_KEYS:
            torch.testing.assert_close(features[key], before[key], rtol=0, atol=0)
        right.eval()
        with patch.object(right, "forward", wraps=right.forward) as forward:
            predicted = call_cached_crv(right, batch, torch.device("cpu"))
            self.assertIsNone(forward.call_args.kwargs["refer_gt"])
        batch["refer_gt"] = [torch.full((4,), float("nan")) for _ in range(2)]
        batch["refer_gt_pixel"] = [torch.full((4,), float("nan")) for _ in range(2)]
        poisoned = call_cached_crv(right, batch, torch.device("cpu"))
        torch.testing.assert_close(predicted[0], poisoned[0], rtol=0, atol=0)
        torch.testing.assert_close(predicted[1], poisoned[1], rtol=0, atol=0)

    def test_live_batches_reuse_feature_storage_and_match_cached_gt(self):
        features, inputs = self.features()
        inputs["gt_bbox"] = [value.double() + .0123456789 for value in inputs["gt_bbox"]]
        live = live_feature_batch(features, inputs)
        cached = collate_cached([make_record(features, inputs, index, index, "test") for index in range(2)])
        for key in FEATURE_KEYS:
            self.assertEqual(live[key].data_ptr(), features[key].data_ptr())
            self.assertFalse(live[key].requires_grad)
            torch.testing.assert_close(live[key], cached[key], rtol=0, atol=0)
        for left, right in zip(live["refer_gt"], cached["refer_gt"]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_cache_validation_and_checkpoint_composition(self):
        with tempfile.TemporaryDirectory() as folder:
            features, inputs = self.features(1)
            directory = Path(folder)
            atomic_json_save(dict(version=CACHE_VERSION, fingerprint="a", count=1, complete=False), directory / "manifest.json")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                CachedCRVDataset(directory)
            atomic_json_save(dict(version=CACHE_VERSION, fingerprint="a", count=1, complete=True), directory / "manifest.json")
            atomic_torch_save(make_record(features, inputs, 0, 0, "a"), directory / "00000000.pt")
            dataset = CachedCRVDataset(directory, "a")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["sample_index"], 0)
            with self.assertRaisesRegex(ValueError, "does not match"):
                CachedCRVDataset(directory, "b")
            atomic_torch_save(make_record(features, inputs, 0, 3, "a"), directory / "00000000.pt")
            with self.assertRaisesRegex(ValueError, "Mismatched"):
                dataset[0]
            base = torch.nn.Module()
            base.vis_enc = torch.nn.Linear(3, 4)
            base.head = torch.nn.Module()
            base.head.crv = self.crv()
            original = {key: value.clone() for key, value in base.state_dict().items() if not key.startswith("head.crv.")}
            optimizer = torch.optim.Adam(base.head.crv.parameters())
            checkpoint = assemble_checkpoint(original, base.head.crv, 2, optimizer)
            _validate_crv_checkpoint(base, checkpoint["state_dict"])
            base.load_state_dict(checkpoint["state_dict"], strict=True)
            for key, value in original.items():
                torch.testing.assert_close(value, checkpoint["state_dict"][key], rtol=0, atol=0)
            atomic_torch_save(checkpoint, directory / "full.pth")
            recovered = load_baseline_state(directory / "full.pth")
            self.assertEqual(set(recovered), set(original))
            disabled = torch.nn.Module()
            disabled.vis_enc = torch.nn.Linear(3, 4)
            load_frozen_baseline(disabled, recovered)
            self.assertFalse(disabled.training)
            self.assertFalse(any(p.requires_grad for p in disabled.parameters()))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_frozen_baseline(disabled, {"bad": torch.ones(1)})

    def test_pretrained_correctness_is_preserved_frozen_and_full_checkpoint_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
            cfg.model.head.enable_correctness = True
            cfg.model.head.num_queries = 8
            runner.validate_config(cfg)  # This is legal, not a CRV-only conflict.
            options = copy.deepcopy(dict(cfg.model.head))
            options["enable_crv"] = False
            head = build_head(options)
            head.correctness.trained_steps.fill_(9)
            original = {"head." + key: value.clone() for key, value in head.state_dict().items()}
            path = Path(folder) / "pretrained_with_correctness.pth"
            torch.save(dict(state_dict={"module." + key: value for key, value in original.items()}), path)
            cfg.offline_crv.baseline_checkpoint = str(path)
            before = torch.get_rng_state()
            state = load_configured_baseline(cfg)
            torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
            self.assertEqual(set(state), set(original))
            wrapper = torch.nn.Module()
            wrapper.head = head
            load_frozen_baseline(wrapper, state)
            self.assertFalse(wrapper.training)
            self.assertFalse(any(p.requires_grad for p in wrapper.parameters()))
            x = torch.randn(1, 768, 8, 8)
            kwargs = dict(cls_feat=torch.zeros(1, 768), lan_feat=torch.randn(1, 8, 768),
                          lan_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
                          img=torch.zeros(1, 3, 128, 128),
                          targets=dict(img_metas=[dict(img_shape=(96, 128), pad_shape=(128, 128))]),
                          with_mask=False)
            with torch.no_grad():
                prediction, _ = head.forward_test(x, **kwargs)
                features, _ = head.forward_test(x, return_crv_inputs=True, **kwargs)
            self.assertIn("top1_correct_probability", prediction)
            torch.testing.assert_close(prediction["refer_label"], features["probabilities"], rtol=0, atol=0)
            crv = build_crv(cfg)
            crv.last_training_epoch.fill_(1)
            checkpoint = assemble_checkpoint(state, crv, 0)
            head.crv, head.enable_crv = crv, True
            wrapper.eval()
            _validate_crv_checkpoint(wrapper, checkpoint["state_dict"])
            _validate_correctness_checkpoint(wrapper, checkpoint["state_dict"])
            wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
            with torch.no_grad():
                fused, _ = head.forward_test(x, **kwargs)
            self.assertIn("top1_correct_probability", fused)
            for name, value in original.items():
                torch.testing.assert_close(value, checkpoint["state_dict"][name], rtol=0, atol=0)
            self.assertEqual(int(head.correctness.trained_steps), 9)
            cfg.model.head.enable_correctness = False
            disabled_state = load_configured_baseline(cfg)
            self.assertFalse(any(key.startswith("head.correctness.") for key in disabled_state))

    def test_enabled_correctness_rejects_missing_untrained_partial_or_wrong_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
            cfg.model.head.enable_correctness = True
            cfg.model.head.hidden_channels = 16
            cfg.offline_crv.baseline_checkpoint = str(Path(folder) / "source.pth")
            observer = Top1CorrectnessHead(hidden_channels=16, **cfg.model.head.correctness)
            observer.trained_steps.fill_(7)
            good = {"head.correctness." + key: value.clone() for key, value in observer.state_dict().items()}
            for error in ("missing", "untrained", "partial", "wrong_shape", "wrong_version"):
                with self.subTest(error=error):
                    state = copy.deepcopy(good)
                    if error == "missing":
                        state = {}
                    elif error == "untrained":
                        state["head.correctness.trained_steps"].zero_()
                    elif error == "partial":
                        state.pop("head.correctness.classifier.1.weight")
                    elif error == "wrong_shape":
                        state["head.correctness.visual.1.weight"] = torch.zeros(128, 15)
                    else:
                        state["head.correctness.schema_version"].fill_(100)
                    torch.save(dict(state_dict=state), cfg.offline_crv.baseline_checkpoint)
                    with self.assertRaisesRegex(ValueError, "trained, shape-compatible"):
                        load_configured_baseline(cfg)

    def test_real_head_extraction_does_not_change_inference(self):
        options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-crv-cached.py").model.head)
        options.update(num_queries=8, enable_crv=False, enable_bcem=True)
        head = build_head(options).eval().requires_grad_(False)
        torch.manual_seed(29)
        x = torch.randn(1, 768, 8, 8)
        text = torch.randn(1, 8, 768)
        text_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        meta = [dict(img_shape=(96, 128, 3), pad_shape=(128, 128, 3), refer_target_index=[0])]
        kwargs = dict(cls_feat=torch.zeros(1, 768), lan_feat=text, lan_mask=text_mask,
                      img=torch.zeros(1, 3, 128, 128), targets=dict(img_metas=meta), with_mask=False)
        before_rng = torch.get_rng_state()
        with torch.no_grad():
            original, _ = head.forward_test(x, **kwargs)
            features, _ = head.forward_test(x, return_crv_inputs=True, **kwargs)
            repeated, _ = head.forward_test(x, **kwargs)
        self.assertEqual(set(features), set(FEATURE_KEYS))
        torch.testing.assert_close(features["boxes"], original["pred_bbox_allobj"], rtol=0, atol=0)
        torch.testing.assert_close(features["probabilities"], original["refer_label"], rtol=0, atol=0)
        torch.testing.assert_close(features["logits"].sigmoid(), original["refer_label"], rtol=0, atol=0)
        torch.testing.assert_close(repeated["refer_label"], original["refer_label"], rtol=0, atol=0)
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)
        fake = torch.nn.Module()
        fake.head, fake.patch_size = head, 16
        fake.eval().requires_grad_(False)
        calls = []
        def extract(*args):
            calls.append(1)
            return x.flatten(2).transpose(1, 2), text, kwargs["cls_feat"]
        fake.extract_visual_language = extract
        with patch.object(head, "forward_test", wraps=head.forward_test) as forward:
            cached = extract_frozen_inputs(fake, dict(img=kwargs["img"], ref_expr_inds=torch.ones(1, 8).long(),
                                                    text_attention_mask=text_mask, img_metas=meta,
                                                    gt_bbox=[torch.full((1, 4), float("nan"))]))
            self.assertEqual(len(calls), 1)
            self.assertNotIn("bbox", forward.call_args.kwargs["targets"])
        torch.testing.assert_close(features["visual"], cached["visual"], rtol=0, atol=0)
        proxy = object.__new__(MIXRefUniModel_OMG)
        proxy.with_nms = False
        actual = proxy.get_predictions_parts(original, meta, with_bbox=True, with_mask=False)["pred_bboxes"]
        batch = dict(boxes=features["boxes"], img_metas=meta)
        expected = cached_predictions(batch, features["probabilities"])
        torch.testing.assert_close(actual[0]["boxes"], expected[0]["boxes"], rtol=0, atol=0)
        torch.testing.assert_close(actual[0]["scores"], expected[0]["scores"], rtol=0, atol=0)

    def test_real_postprocess_degenerate_clipping_and_priority(self):
        boxes = torch.tensor([[[0., .5, 0., .2], [.98, .5, .2, .4], [.5, .5, .2, .2]]])
        scores = torch.tensor([[[.99], [.7], [.6]]])
        priority = torch.tensor([[[1.], [2.], [3.]]])
        meta = [dict(img_shape=(80, 100, 3))]
        prediction = dict(pred_bbox_allobj=boxes, pred_class_allobj=torch.zeros(1, 3, 2),
                          refer_label=scores, ranking_priority=priority)
        proxy = object.__new__(MIXRefUniModel_OMG)
        proxy.with_nms = False
        expected = proxy.get_predictions_parts(prediction, meta, with_bbox=True, with_mask=False)["pred_bboxes"]
        actual = cached_predictions(dict(boxes=boxes, img_metas=meta), scores, priority)
        for key in ("boxes", "scores", "ranking_logits"):
            torch.testing.assert_close(expected[0][key], actual[0][key], rtol=0, atol=0)
        self.assertEqual(aerialvg_box_metrics(actual, [torch.tensor([40., 32., 60., 48.])])[0], 100.)

    def test_independent_config_and_cache_fingerprint(self):
        cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
        runner.validate_config(cfg)
        self.assertFalse(cfg.model.head.enable_grab)
        self.assertFalse(cfg.model.head.enable_correctness)
        self.assertNotEqual(cfg.data.train.annsfile, cfg.data.val.annsfile)
        with tempfile.TemporaryDirectory() as folder:
            annotation = Path(folder) / "ann.jsonl"
            annotation.write_text("{}\n")
            cfg.data.train.annsfile = str(annotation)
            before, _ = cache_identity(cfg, "train", "weights-a")
            cfg.model.head.crv.topk = 8
            same, _ = cache_identity(cfg, "train", "weights-a")
            self.assertEqual(before, same)  # Trainable CRV does not invalidate BASE features.
            changed, _ = cache_identity(cfg, "train", "weights-b")
            self.assertNotEqual(before, changed)
            cfg.data.train.pipeline[1].img_scale = (512, 512)
            resized, _ = cache_identity(cfg, "train", "weights-a")
            self.assertNotEqual(before, resized)

    def cleanup_fixture(self, folder):
        directory = Path(folder)
        cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
        cfg.offline_crv.cache_dir = str(directory / "features")
        result = directory / "model"
        result.mkdir()
        for filename in ("latest.pth", "top1_best.pth", "config.py"):
            (result / filename).write_text("synthetic saved artifact")
        for split in ("train", "val"):
            annotation = directory / (split + ".jsonl")
            annotation.write_text("{}\n")
            cfg.data[split].annsfile = str(annotation)
            cache = Path(cfg.offline_crv.cache_dir) / split
            cache.mkdir(parents=True)
            fingerprint, _ = cache_identity(cfg, split, "baseline-hash")
            atomic_json_save(dict(version=CACHE_VERSION, fingerprint=fingerprint, count=1, complete=True),
                             cache / "manifest.json")
            atomic_torch_save(dict(synthetic=True), cache / "00000000.pt")
        return cfg, result

    def test_cleanup_removes_only_used_validated_cache_and_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, result = self.cleanup_fixture(folder)
            root = Path(cfg.offline_crv.cache_dir)
            unrelated = root / "another_experiment.txt"
            unrelated.write_text("keep")
            untested = root / "test"
            untested.mkdir()
            (untested / "unrelated.txt").write_text("keep")
            audit = cleanup_feature_caches(cfg, "baseline-hash", ["train", "val"], result, Mock())
            self.assertEqual(audit["status"], "deleted")
            self.assertFalse((root / "train").exists())
            self.assertFalse((root / "val").exists())
            self.assertTrue(unrelated.is_file())
            self.assertTrue((untested / "unrelated.txt").is_file())
            self.assertTrue((result / "latest.pth").is_file())
            self.assertTrue((Path(folder) / "train.jsonl").is_file())
            self.assertEqual(json.loads((result / "cache_cleanup.json").read_text())["status"], "deleted")

    def test_cleanup_preflights_all_splits_and_refuses_unsafe_targets(self):
        for problem in ("extra_file", "foreign", "symlink", "no_checkpoint", "broad"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as folder:
                cfg, result = self.cleanup_fixture(folder)
                root = Path(cfg.offline_crv.cache_dir)
                if problem == "extra_file":
                    (root / "val" / "my_notes.txt").write_text("must keep")
                elif problem == "foreign":
                    manifest = root / "val" / "manifest.json"
                    value = json.loads(manifest.read_text())
                    value["fingerprint"] = "another-experiment"
                    atomic_json_save(value, manifest)
                elif problem == "symlink":
                    owned = root / "val" / "00000000.pt"
                    owned.rename(Path(folder) / "original.pt")
                    owned.symlink_to(Path(folder) / "original.pt")
                elif problem == "no_checkpoint":
                    (result / "latest.pth").rename(result / "not_finished.pth")
                else:
                    cfg.offline_crv.cache_dir = str(Path.cwd())
                with self.assertRaises(ValueError):
                    cleanup_feature_caches(cfg, "baseline-hash", ["train", "val"], result, Mock())
                self.assertTrue((root / "train" / "00000000.pt").is_file())
                self.assertTrue((root / "train" / "manifest.json").is_file())

    def test_cache_lock_rejects_concurrent_runs_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "features"
            with feature_cache_lock(root):
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    with feature_cache_lock(root):
                        self.fail("Second writer must not enter")
            with feature_cache_lock(root):
                pass

    def test_cache_stage_runs_only_missing_samples_and_rejects_stale_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
            annotation = directory / "annotations.jsonl"
            annotation.write_text("{}\n")
            cfg.data.train.annsfile = str(annotation)
            cfg.offline_crv.update(cache_dir=str(directory / "cache"), extract_workers=0, extract_batch_size=2)
            base = torch.nn.Module()
            base.vis_enc = torch.nn.Linear(3, 3, bias=False)
            baseline_path = directory / "baseline.pth"
            torch.save(dict(state_dict=base.state_dict()), baseline_path)
            cfg.offline_crv.baseline_checkpoint = str(baseline_path)
            checksum = file_sha256(baseline_path)
            _, inputs = self.features(3)
            class Samples(list):
                word_emb = None
                num_token = 64010
            samples = Samples()
            for index in range(3):
                samples.append(dict(img=DataContainer(torch.zeros(3, 16, 16), stack=True),
                                    ref_expr_inds=DataContainer(torch.ones(6).long(), stack=True, pad_dims=None),
                                    text_attention_mask=DataContainer(torch.zeros(6).long(), stack=True, pad_dims=None),
                                    gt_bbox=DataContainer(inputs["gt_bbox"][index]),
                                    img_metas=DataContainer(inputs["img_metas"][index], cpu_only=True)))
            calls = []
            def extract(model, batch):
                self.assertFalse(model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                count = len(batch["img_metas"])
                calls.append(count)
                return self.features(count)[0]
            with patch.object(runner, "build_dataset", return_value=samples), \
                    patch.object(runner, "build_model", return_value=base) as build, \
                    patch.object(runner, "extract_frozen_inputs", side_effect=extract):
                runner.extract_caches(cfg, ["train"], checksum, torch.device("cpu"), Mock())
                self.assertEqual(calls, [2, 1])
                build.reset_mock()
                runner.extract_caches(cfg, ["train"], checksum, torch.device("cpu"), Mock())
                build.assert_not_called()
                self.assertEqual(calls, [2, 1])
                # Emulate interruption by moving one generated sample aside.
                path = Path(cfg.offline_crv.cache_dir) / "train" / "00000001.pt"
                path.rename(path.with_suffix(".saved"))
                runner.extract_caches(cfg, ["train"], checksum, torch.device("cpu"), Mock())
                self.assertEqual(calls, [2, 1, 1])
                self.assertEqual(len(CachedCRVDataset(path.parent)), 3)
                with self.assertRaisesRegex(ValueError, "different weights/config/data"):
                    runner.extract_caches(cfg, ["train"], "another-checkpoint", torch.device("cpu"), Mock())

    def test_full_training_loop_updates_only_crv_and_saves_full_model(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            annotation = directory / "train.jsonl"
            annotation.write_text("{}\n")
            validation = directory / "val.jsonl"
            validation.write_text("{}\n")
            cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
            cfg.model.head.hidden_channels = 16
            cfg.model.head.crv = self.options()
            cfg.model.head.enable_correctness = True
            cfg.model.head.enable_bcem = True
            cfg.work_dir = str(directory / "runs")
            cfg.swanlab = False
            cfg.offline_crv.update(cache_dir=str(directory / "cache"), workers=0, batch_size=2, max_epochs=2,
                                   test_after_training=False, backup_code=False, rename_with_metrics=False)
            original = {"vis_enc.weight": torch.randn(3, 3), "vis_enc.running_count": torch.tensor(2)}
            observer = Top1CorrectnessHead(hidden_channels=16, **cfg.model.head.correctness)
            observer.trained_steps.fill_(8)
            original.update({"head.correctness." + key: value.clone() for key, value in observer.state_dict().items()})
            bcem = BackgroundContextEnhancement(hidden_channels=16, **cfg.model.head.bcem)
            original.update({"head.bcem." + key: value.clone() for key, value in bcem.state_dict().items()})
            baseline_path = directory / "baseline.pth"
            torch.save(dict(state_dict=original, frozen_bcem_config=bcem_config_contract(cfg.model.head)), baseline_path)
            cfg.offline_crv.baseline_checkpoint = str(baseline_path)
            checksum = file_sha256(baseline_path)
            features, inputs = self.features()
            for split, path in (("train", annotation), ("val", validation)):
                cfg.data[split].annsfile = str(path)
                cache_dir = Path(cfg.offline_crv.cache_dir) / split
                cache_dir.mkdir(parents=True)
                fingerprint, _ = cache_identity(cfg, split, checksum)
                atomic_json_save(dict(version=CACHE_VERSION, fingerprint=fingerprint, count=2, complete=True),
                                 cache_dir / "manifest.json")
                for index in range(2):
                    atomic_torch_save(make_record(features, inputs, index, index, fingerprint),
                                      cache_dir / ("{:08d}.pt".format(index)))
            initial = build_crv(cfg).state_dict()
            failed_cfg = copy.deepcopy(cfg)
            with patch.object(runner, "evaluate_cached", side_effect=RuntimeError("synthetic evaluation failure")), \
                    patch.object(runner, "cleanup_feature_caches", side_effect=AssertionError("No cleanup on error")):
                with self.assertRaisesRegex(RuntimeError, "synthetic evaluation failure"):
                    runner.train_cached(failed_cfg, checksum, torch.device("cpu"))
            self.assertTrue((Path(cfg.offline_crv.cache_dir) / "train" / "00000000.pt").is_file())
            with patch.object(runner, "build_model", side_effect=AssertionError("Grounding model must not run")):
                destination = runner.train_cached(cfg, checksum, torch.device("cpu"))
            saved = torch.load(destination / "latest.pth", map_location="cpu")
            self.assertTrue((destination / "top1_best.pth").is_file())
            self.assertTrue((destination / "config.py").is_file())
            for key, value in original.items():
                torch.testing.assert_close(value, saved["state_dict"][key], rtol=0, atol=0)
            trained = {key[len("head.crv."):]: value for key, value in saved["state_dict"].items() if key.startswith("head.crv.")}
            self.assertTrue(any(not torch.equal(value, trained[key]) for key, value in initial.items()
                                if value.is_floating_point()))
            self.assertEqual(int(trained["last_training_epoch"]), 2)
            self.assertIn("crv_optimizer", saved)
            self.assertEqual(checksum, file_sha256(baseline_path))
            self.assertTrue(saved["frozen_bcem_config"]["enabled"])
            self.assertEqual(saved["frozen_bcem_config"], bcem_config_contract(cfg.model.head))
            self.assertTrue((destination / "baseline_bcem_validation.json").is_file())
            self.assertFalse(Path(cfg.offline_crv.cache_dir).exists())
            self.assertEqual(json.loads((destination / "cache_cleanup.json").read_text())["status"], "deleted")

    def test_online_training_runs_frozen_forward_without_any_cache_io(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
            cfg.model.head.hidden_channels = 16
            cfg.model.head.crv = self.options()
            cfg.model.head.enable_correctness = True
            cfg.model.head.enable_bcem = True
            cfg.work_dir = str(directory / "runs")
            cfg.swanlab = False
            cfg.offline_crv.update(use_cache=False, workers=0, batch_size=2, max_epochs=2,
                                   test_after_training=True, backup_code=False, rename_with_metrics=False,
                                   cache_dir=str(directory / "untouched_cache"))
            base = torch.nn.Module()
            base.vis_enc = torch.nn.Linear(3, 3)
            base.head = torch.nn.Module()
            base.head.correctness = Top1CorrectnessHead(hidden_channels=16, **cfg.model.head.correctness)
            base.head.correctness.trained_steps.fill_(11)
            base.head.bcem = BackgroundContextEnhancement(hidden_channels=16, **cfg.model.head.bcem)
            state = copy.deepcopy(base.state_dict())
            baseline = directory / "baseline.pth"
            torch.save(dict(state_dict=state, frozen_bcem_config=bcem_config_contract(cfg.model.head)), baseline)
            cfg.offline_crv.baseline_checkpoint = str(baseline)
            _, inputs = self.features(2)
            class Samples(list):
                word_emb = None
                num_token = 64010
            samples = Samples()
            for index in range(2):
                samples.append(dict(img=DataContainer(torch.zeros(3, 100, 100), stack=True),
                                    ref_expr_inds=DataContainer(torch.ones(6).long(), stack=True, pad_dims=None),
                                    text_attention_mask=DataContainer(torch.zeros(6).long(), stack=True, pad_dims=None),
                                    gt_bbox=DataContainer(inputs["gt_bbox"][index]),
                                    img_metas=DataContainer(inputs["img_metas"][index], cpu_only=True)))
            calls = []
            # Evaluation wraps the entire loop in no_grad; training does not.
            def checked_extract(model, batch):
                self.assertFalse(model.training)
                self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
                calls.append(1)
                return self.features(len(batch["img_metas"]))[0]
            with patch.object(runner, "build_dataset", return_value=samples), \
                    patch.object(runner, "build_model", return_value=base), \
                    patch.object(runner, "extract_frozen_inputs", side_effect=checked_extract), \
                    patch.object(runner, "cache_loader", side_effect=AssertionError("Must not read cache")), \
                    patch.object(runner, "cleanup_feature_caches", side_effect=AssertionError("Must not delete cache")):
                destination = runner.train_cached(cfg, file_sha256(baseline), torch.device("cpu"))
            self.assertEqual(len(calls), 5)  # Two train + two val + best test.
            self.assertFalse(Path(cfg.offline_crv.cache_dir).exists())
            saved = torch.load(destination / "latest.pth", map_location="cpu")
            self.assertFalse(saved["use_feature_cache"])
            self.assertIsNone(saved["cache_fingerprint"])
            self.assertGreater(saved["crv_total_updates"], 0)
            for key, value in state.items():
                torch.testing.assert_close(value, base.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(value, saved["state_dict"][key], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
            self.assertTrue((destination / "test_best.json").is_file())


if __name__ == "__main__":
    unittest.main()
