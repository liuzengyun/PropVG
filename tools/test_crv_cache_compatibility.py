"""Read-only reuse of audited old feature caches; real input changes still fail."""
import copy
import hashlib
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

from propvg.apis.crv_cached import (
    CACHE_VERSION, LEGACY_FEATURE_ENCODERS, CachedCRVDataset, atomic_json_save, atomic_torch_save,
    cache_identity, cleanup_feature_caches, feature_encoding_sha256, validate_cache_identity,
)

spec = importlib.util.spec_from_file_location("train_crv_compatibility_tested", "tools/train_crv_cached.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def old_hash(details):
    return hashlib.sha256(json.dumps(details, sort_keys=True, default=str).encode()).hexdigest()


class CacheCompatibilityTests(unittest.TestCase):
    def setup_cache(self, folder, enabled_bcem=False):
        cfg = Config.fromfile("configs/aerialvg/PropVG-crv-cached.py")
        cfg.offline_crv.cache_dir = str(Path(folder) / "cache")
        cfg.offline_crv.workers = 0
        cfg.model.head.enable_bcem = enabled_bcem
        annotation = Path(folder) / "train.jsonl"
        annotation.write_text("{}\n")
        cfg.data.train.annsfile = str(annotation)
        current, details = cache_identity(cfg, "train", "same-checkpoint")
        previous = copy.deepcopy(details)
        previous.pop("feature_encoding_sha256")
        previous["extractor_sha256"], previous["runner_sha256"] = next(iter(LEGACY_FEATURE_ENCODERS))
        legacy = old_hash(previous)
        directory = Path(cfg.offline_crv.cache_dir) / "train"
        directory.mkdir(parents=True)
        manifest = dict(version=CACHE_VERSION, count=1, complete=True, fingerprint=legacy, details=previous)
        atomic_json_save(manifest, directory / "manifest.json")
        atomic_torch_save(dict(sample_index=0, fingerprint=legacy, visual=torch.ones(1)), directory / "00000000.pt")
        return cfg, directory, current, details, manifest

    def test_audited_legacy_reuses_read_only_including_disabled_bcem_options(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, directory, _, _, manifest = self.setup_cache(folder)
            cfg.model.head.bcem.num_rays = 16
            expected, details = cache_identity(cfg, "train", "same-checkpoint")
            before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()}
            self.assertEqual(feature_encoding_sha256(), next(iter(LEGACY_FEATURE_ENCODERS.values())))
            accepted = validate_cache_identity(manifest, expected, details)
            self.assertEqual(accepted, manifest["fingerprint"])
            dataset = CachedCRVDataset(directory, expected, details)
            self.assertTrue(dataset.legacy_compatible)
            self.assertEqual(dataset[0]["fingerprint"], manifest["fingerprint"])
            self.assertEqual(len(runner.cache_loader(cfg, "train", "same-checkpoint").dataset), 1)
            with patch.object(runner, "build_dataset", return_value=[None]), \
                    patch.object(runner, "build_frozen_model", side_effect=AssertionError("Do not recompute")):
                runner.extract_caches(cfg, ["train"], "same-checkpoint", torch.device("cpu"), Mock())
            after = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()}
            self.assertEqual(before, after)

    def test_disabled_bcem_is_irrelevant_but_enabled_bcem_changes_are_not(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, _, first, _, _ = self.setup_cache(folder)
            cfg.model.head.bcem.num_rays = 16
            second, _ = cache_identity(cfg, "train", "same-checkpoint")
            self.assertEqual(first, second)
            cfg.model.head.enable_bcem = True
            before, _ = cache_identity(cfg, "train", "same-checkpoint")
            cfg.model.head.bcem.num_rays = 8
            after, _ = cache_identity(cfg, "train", "same-checkpoint")
            self.assertNotEqual(before, after)

    def test_successful_run_can_clean_the_same_compatible_legacy_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, directory, _, _, manifest = self.setup_cache(folder)
            cfg.model.head.bcem.num_rays = 16
            results = Path(folder) / "trained_model"
            results.mkdir()
            for name in ("latest.pth", "top1_best.pth", "config.py"):
                (results / name).write_text("synthetic saved artifact")
            audit = cleanup_feature_caches(cfg, "same-checkpoint", ["train"], results, Mock())
            self.assertEqual(audit["status"], "deleted")
            self.assertEqual(audit["splits"]["train"]["fingerprint"], manifest["fingerprint"])
            self.assertFalse(directory.exists())
            self.assertTrue((results / "top1_best.pth").is_file())
            self.assertTrue((Path(folder) / "train.jsonl").is_file())

    def test_real_input_changes_still_reject_legacy_features(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, expected, details, manifest = self.setup_cache(folder)
            for field in ("checkpoint_sha256", "annotation_sha256", "dtype", "feature_encoding_sha256", "seed"):
                with self.subTest(field=field):
                    changed = copy.deepcopy(details)
                    changed[field] = "different"
                    with self.assertRaisesRegex(ValueError, "changed fields"):
                        validate_cache_identity(manifest, "new-fingerprint", changed)
            for field in ("model", "dataset", "sources"):
                with self.subTest(field=field):
                    changed = copy.deepcopy(details)
                    changed[field]["real_change"] = True
                    with self.assertRaisesRegex(ValueError, "changed fields"):
                        validate_cache_identity(manifest, "new-fingerprint", changed)
            # A corrupted manifest does not become valid just because its fields match.
            broken = copy.deepcopy(manifest)
            broken["fingerprint"] = "invalid"
            with self.assertRaisesRegex(ValueError, "integrity"):
                validate_cache_identity(broken, expected, details)

    def test_unknown_legacy_code_is_not_whitelisted(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, expected, details, manifest = self.setup_cache(folder)
            manifest["details"]["extractor_sha256"] = "unknown-implementation"
            manifest["fingerprint"] = old_hash(manifest["details"])
            with self.assertRaisesRegex(ValueError, "Unknown legacy"):
                validate_cache_identity(manifest, expected, details)

    def test_new_identity_ignores_only_unrelated_file_audit_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, expected, details, _ = self.setup_cache(folder)
            previous = copy.deepcopy(details)
            previous["extractor_sha256"] = "training-helpers-edited"
            previous["runner_sha256"] = "new-training-loop"
            manifest = dict(fingerprint=old_hash(previous), details=previous)
            self.assertEqual(validate_cache_identity(manifest, expected, details), manifest["fingerprint"])


if __name__ == "__main__":
    unittest.main()
