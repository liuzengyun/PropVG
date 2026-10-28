"""BCEM options invisible in state_dict must still match a frozen baseline."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from mmcv import Config

from propvg.apis.bcem_baseline_validation import bcem_config_contract
from propvg.apis.crv_cached import load_configured_baseline
from propvg.models.heads.background_context import BackgroundContextEnhancement


class BCEMBaselineValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self, folder, embedded=False, snapshot=True):
        head = dict(hidden_channels=16, enable_bcem=True,
                    bcem=dict(context_dim=16, num_heads=4, sampling_mode="ray", num_rays=8,
                              num_sectors=4, radial_samples=4, min_valid_points=2))
        cfg = Config(dict(model=dict(head=copy.deepcopy(head)),
                          offline_crv=dict(baseline_checkpoint=str(Path(folder) / "best.pth"), baseline_config=None)))
        module = BackgroundContextEnhancement(hidden_channels=16, **head["bcem"])
        checkpoint = dict(state_dict={"head.bcem." + key: value.clone() for key, value in module.state_dict().items()})
        if embedded:
            checkpoint["frozen_bcem_config"] = bcem_config_contract(head)
        torch.save(checkpoint, cfg.offline_crv.baseline_checkpoint)
        path = Path(folder) / "20261027_120000_PropVG-aerialvg.py"
        if snapshot:
            Config(dict(model=dict(head=head))).dump(str(path))
        return cfg, checkpoint, path

    def test_original_snapshot_matches_and_all_defaults_are_expanded(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, original, snapshot = self.source(folder)
            before = torch.get_rng_state()
            state, report = load_configured_baseline(cfg, return_validation=True)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["sources"], [str(snapshot)])
            arguments = report["contract"]["arguments"]
            self.assertEqual(arguments["num_rays"], 8)
            self.assertEqual(arguments["ray_lateral_offset"], 0.)
            self.assertEqual(arguments["context_strength"], .1)
            for key, value in state.items():
                torch.testing.assert_close(value, original["state_dict"][key], rtol=0, atol=0)
            cfg.model.head.bcem.radial_samples = None
            cfg.model.head.bcem.min_valid_points = None
            cfg.model.head.bcem.radial_edges = [0., .1, .25, .5, 1.]
            load_configured_baseline(cfg)  # Same effective constructor options.

    def test_nonshape_options_are_strictly_compared(self):
        changes = dict(num_rays=16, num_sectors=8, num_heads=8, context_strength=.2,
                       radial_edges=(0., .12, .25, .5, 1.), chunk_size=64,
                       use_sparse_selection=False, checkpoint_chunks=False, use_text_attention=False)
        with tempfile.TemporaryDirectory() as folder:
            cfg, _, _ = self.source(folder)
            for key, value in changes.items():
                with self.subTest(key=key):
                    changed = copy.deepcopy(cfg)
                    changed.model.head.bcem[key] = value
                    with self.assertRaisesRegex(ValueError, "arguments\\." + key):
                        load_configured_baseline(changed)
            cfg.model.head.enable_bcem = False
            with self.assertRaisesRegex(ValueError, "enable_bcem/checkpoint mismatch"):
                load_configured_baseline(cfg)

    def test_no_or_multiple_snapshots_require_explicit_original_config(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, _, snapshot = self.source(folder, snapshot=False)
            with self.assertRaisesRegex(ValueError, "Cannot confirm original BCEM config"):
                load_configured_baseline(cfg)
            Config(dict(model=copy.deepcopy(dict(cfg.model)))).dump(str(snapshot))
            second = Path(folder) / "config.py"
            Config(dict(model=copy.deepcopy(dict(cfg.model)))).dump(str(second))
            with self.assertRaisesRegex(ValueError, "2 config snapshots"):
                load_configured_baseline(cfg)
            cfg.offline_crv.baseline_config = str(snapshot)
            load_configured_baseline(cfg)
            cfg.offline_crv.baseline_config = str(Path(folder) / "missing.py")
            with self.assertRaisesRegex(ValueError, "baseline_config not found"):
                load_configured_baseline(cfg)

    def test_embedded_config_survives_moving_checkpoint_without_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, _, _ = self.source(folder, embedded=True, snapshot=False)
            _, report = load_configured_baseline(cfg, return_validation=True)
            self.assertEqual(report["sources"], ["checkpoint.frozen_bcem_config"])
            cfg.model.head.bcem.num_rays = 16
            with self.assertRaisesRegex(ValueError, "arguments.num_rays"):
                load_configured_baseline(cfg)

    def test_explicit_config_cannot_override_disagreeing_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, _, snapshot = self.source(folder, embedded=True)
            cfg.model.head.bcem.num_rays = 16
            Config(dict(model=copy.deepcopy(dict(cfg.model)))).dump(str(snapshot))
            cfg.offline_crv.baseline_config = str(snapshot)
            with self.assertRaisesRegex(ValueError, "checkpoint.frozen_bcem_config"):
                load_configured_baseline(cfg)

    def test_missing_or_bad_weights_and_unsupported_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, good, _ = self.source(folder, embedded=True, snapshot=False)
            for issue in ("missing", "shape", "version", "no_weights"):
                with self.subTest(issue=issue):
                    checkpoint = copy.deepcopy(good)
                    if issue == "missing":
                        checkpoint["state_dict"].pop("head.bcem.visual_projection.weight")
                    elif issue == "shape":
                        checkpoint["state_dict"]["head.bcem.visual_projection.weight"] = torch.zeros(16, 15, 1, 1)
                    elif issue == "version":
                        checkpoint["frozen_bcem_config"]["version"] = 2
                    else:
                        checkpoint["state_dict"] = {}
                    torch.save(checkpoint, cfg.offline_crv.baseline_checkpoint)
                    with self.assertRaises(ValueError):
                        load_configured_baseline(cfg)

    def test_plain_old_baseline_without_bcem_does_not_require_original_config(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, _, _ = self.source(folder, snapshot=False)
            cfg.model.head.enable_bcem = False
            torch.save(dict(state_dict={"vis_enc.weight": torch.ones(1)}), cfg.offline_crv.baseline_checkpoint)
            _, report = load_configured_baseline(cfg, return_validation=True)
            self.assertEqual(report, dict(contract=dict(version=1, enabled=False), sources=[], status="disabled"))
            cfg.model.head.enable_bcem = True
            with self.assertRaisesRegex(ValueError, "enable_bcem/checkpoint mismatch"):
                load_configured_baseline(cfg)


if __name__ == "__main__":
    unittest.main()
