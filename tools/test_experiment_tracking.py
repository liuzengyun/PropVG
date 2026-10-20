"""CPU-only tracking tests; no account, cloud uploads, GPUs or real training."""
import ast
from enum import Enum
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from mmcv import Config
from propvg.utils import experiment_tracking as tracking

spec = importlib.util.spec_from_file_location("standalone_swanlab_worker_test",
    Path(__file__).resolve().parents[1] / "propvg/utils/swanlab_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class RunState(Enum):
    SUCCESS = 1
    CRASHED = -1


class FakeRun:
    def __init__(self, offline):
        self.mode = "offline" if offline else "cloud"
        self.public = Mock(run_dir="/tmp/fake-swanlab")
        self.log = Mock()
        self.state = RunState.SUCCESS
        self.config = {}


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        os.environ.pop("SWANLAB_MODE", None)
        self.previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.previous = tracking._TRACKER
        tracking._TRACKER = None
        self.temporary = tempfile.TemporaryDirectory(prefix="propvg_tracking_")
        self.root = Path(self.temporary.name)
        self.work = self.root / "experiment" / "20261020_123456"
        self.work.mkdir(parents=True)
        self.cfg = Config(dict(work_dir=str(self.work), timestamp="20261020_123456", tag_name="aerialvg",
                               swanlab=dict(logdir=str(self.root / "swanlog"), finish_wait=0), seed=6666,
                               model=dict(enable_crv=True), api_key="DO_NOT_STORE_THIS"))
        self.logger = logging.getLogger("propvg-tracking-test-" + self.root.name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def tearDown(self):
        tracking.finish_tracking()
        tracking._TRACKER = self.previous
        self.temporary.cleanup()
        logging.disable(self.previous_logging_disable)
        self.environment.stop()

    def start(self):
        with patch.dict(os.environ, {"SWANLAB_MODE": "online"}), \
             patch.object(tracking.subprocess, "Popen", return_value=Mock()) as spawn:
            tracking.initialize_tracking(self.cfg, self.logger)
        return tracking._TRACKER, spawn

    def events(self, tracker):
        return [json.loads(line) for line in (tracker.directory / "events.jsonl").read_text().splitlines()]

    def sdk(self, offline=False, fail_online=False):
        sdk = Mock()
        run = FakeRun(offline)
        def initialize(**kwargs):
            if fail_online and kwargs["mode"] == "online":
                raise ConnectionError("fake outage")
            run.mode = "cloud" if kwargs["mode"] == "online" else kwargs["mode"]
            return run
        sdk.init.side_effect = initialize
        sdk.get_url.return_value = "https://example.invalid/run"
        return sdk, run

    def backend(self, tracker, sdk):
        with patch.dict(os.environ), patch.object(worker.signal, "signal"), patch.object(worker.signal, "alarm"):
            worker.run_backend(tracker.directory, sdk=sdk, poll=0)

    def test_parent_is_non_networking_unique_names_local_journal_and_no_gpu_child(self):
        tracker, spawn = self.start()
        metadata = json.loads((tracker.directory / "metadata.json").read_text())
        self.assertEqual(metadata["name"], "experiment/20261020_123456")
        self.assertEqual(metadata["config"]["api_key"], "[REDACTED]")
        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        imports = [name.name for node in ast.walk(ast.parse(Path(worker.__file__).read_text()))
                   if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any(name.startswith(("torch", "propvg")) for name in imports))
        tracking.set_tracking_context(epoch=25, global_step=100)
        tracking.log_metrics({"train/loss_total": .8, "checkpoint_epoch": 12})
        tracking.log_evaluation("val", "AerialVG", (31., 60., 25.), 2.)
        self.logger.info("text result %s", "recorded")
        self.logger.info("api_key=DO_NOT_STORE_THIS")
        metrics = [e["values"] for e in self.events(tracker) if e["type"] == "metrics"]
        self.assertEqual(metrics[0]["global_step"], 100)
        self.assertEqual(metrics[1]["checkpoint_epoch"], 12)
        self.assertEqual(metrics[1]["val/top1_acc"], 31.)
        self.assertTrue(any("text result recorded" in e.get("text", "") for e in self.events(tracker)))
        self.assertNotIn("DO_NOT_STORE_THIS", (tracker.directory / "events.jsonl").read_text())
        tracking.finish_tracking()
        self.assertEqual(self.events(tracker)[-1]["type"], "finish")

    def test_online_initialization_failure_requires_a_fresh_offline_session(self):
        tracker, _ = self.start()
        tracking.log_metrics({"train/loss_total": 1.})
        tracking.finish_tracking()
        online_sdk, _ = self.sdk(fail_online=True)
        self.backend(tracker, online_sdk)
        online_sdk.init.assert_called_once()
        failed = json.loads((tracker.directory / "status.json").read_text())
        self.assertEqual(failed["mode"], "initialization_failed")
        self.assertIn("fake outage", failed["error"])
        sdk, run = self.sdk(offline=True)
        with patch.dict(os.environ):
            worker.run_backend(tracker.directory, sdk=sdk, poll=0, mode="offline")
        self.assertEqual(sdk.init.call_args.kwargs["mode"], "offline")
        self.assertNotIn("id", sdk.init.call_args.kwargs)
        status = json.loads((tracker.directory / "status.json").read_text())
        self.assertEqual(status["mode"], "offline")
        self.assertIsNone(status["url"])
        self.assertTrue(status["finished"])
        loss_call = next(c for c in run.log.call_args_list if "train/loss_total" in c.args[0])
        self.assertEqual(loss_call.args[0]["train/loss_total"], 1.)
        self.assertEqual(loss_call.kwargs["step"], 0)

    def test_supervisor_uses_distinct_processes_after_initialization_failure(self):
        tracker, _ = self.start()
        tracking.finish_tracking()
        failed = Mock(pid=12345)
        failed.poll.return_value = None
        offline = Mock(pid=12346)
        offline.poll.return_value = None
        offline.wait.return_value = 0
        def spawn(command, **kwargs):
            mode = command[-1]
            state = dict(mode="initialization_failed", error="ApiError: project validation") if mode == "online" else dict(mode="offline")
            (tracker.directory / "status.json").write_text(json.dumps(state))
            return failed if mode == "online" else offline
        with patch.object(worker.subprocess, "Popen", side_effect=spawn) as popen, \
             patch.object(worker.os, "killpg") as kill:
            self.assertEqual(worker.run_supervised(tracker.directory), 0)
        self.assertEqual([c.args[0][-1] for c in popen.call_args_list], ["online", "offline"])
        kill.assert_called_once_with(failed.pid, worker.signal.SIGTERM)
        state = json.loads((tracker.directory / "status.json").read_text())
        self.assertIn("project validation", state["initialization_failures"][0]["error"])

    def test_sdk_error_diagnostics_redact_credentials(self):
        with patch.dict(os.environ, {"SWANLAB_API_KEY": "credential-do-not-log"}):
            detail = worker.error_detail(RuntimeError("api_key=credential-do-not-log secret=private"))
        self.assertNotIn("credential-do-not-log", detail)
        self.assertNotIn("private", detail)

    def test_upload_worker_python_can_be_independent_of_training(self):
        with patch.dict(os.environ, {"PROPVG_SWANLAB_PYTHON": "/tmp/upload-env/bin/python"}):
            tracker, spawn = self.start()
        self.assertEqual(spawn.call_args.args[0][0], "/tmp/upload-env/bin/python")
        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_modern_sdk_uses_new_settings_run_properties_and_finish_state(self):
        tracker, _ = self.start()
        tracking.log_metrics({"train/loss_total": .2})
        tracking.finish_tracking(exit_code=1)
        sdk, run = self.sdk()
        sdk.__version__ = "0.10.2"
        run.id, run.dir, run.url = "modern-run", "/tmp/modern-run", "https://example.invalid/modern-run"
        self.backend(tracker, sdk)
        self.assertIn("name", sdk.init.call_args.kwargs)
        self.assertIn("log_dir", sdk.init.call_args.kwargs)
        self.assertIn("probe", sdk.Settings.call_args.kwargs)
        sdk.get_url.assert_not_called()
        sdk.finish.assert_called_once_with(state="crashed", error="Training exited abnormally")
        state = json.loads((tracker.directory / "status.json").read_text())
        self.assertEqual(state["url"], run.url)
        self.assertTrue(state["finished"])

    def test_online_records_validation_epoch_test_final_test_and_summary(self):
        tracker, _ = self.start()
        tracking.set_tracking_context(epoch=36, global_step=1000)
        tracking.log_evaluation("val", "AerialVG", (31., 60., 25.), 1.)
        tracking.log_evaluation("test_epoch_36", "AerialVG", (30., 59., 24.), 2.)
        tracking.log_metrics({"checkpoint_epoch": 26})
        tracking.log_evaluation("test", "AerialVG", (32., 61., 26.), 3.)
        tracking.log_summary({"best/epoch": 26, "best/val_top1_acc": 31.})
        tracking.finish_tracking()
        sdk, run = self.sdk()
        self.backend(tracker, sdk)
        def metric(key):
            return next(c for c in run.log.call_args_list if key in c.args[0])
        self.assertEqual(metric("test_epoch/top1_acc").args[0]["test_epoch/top1_acc"], 30.)
        self.assertEqual(metric("test_epoch/top1_acc").kwargs["step"], 36)
        self.assertEqual(metric("val/top1_acc").kwargs["step"], 36)
        self.assertEqual(metric("test_best/top1_acc").args[0]["test_best/top1_acc"], 32.)
        self.assertEqual(metric("test_best/top1_acc").kwargs["step"], 26)
        self.assertEqual(run.config["tracking_summary"]["best/epoch"], 26)
        self.assertEqual(run.config["tracking_summary"]["best/val_top1_acc"], 31.)
        self.assertEqual(json.loads((tracker.directory / "summary.json").read_text())["best/epoch"], 26)
        sdk.finish.assert_called_once_with()
        status = json.loads((tracker.directory / "status.json").read_text())
        self.assertEqual(status["mode"], "online")
        self.assertTrue(status["finished"])

    def test_missing_sdk_or_log_error_leaves_complete_local_history(self):
        tracker, _ = self.start()
        tracking.log_metrics({"train/loss_total": 1.})
        tracking.log_metrics({"train/loss_total": 2.})
        tracking.finish_tracking()
        sdk, run = self.sdk()
        run.log.side_effect = RuntimeError("backend failure")
        self.backend(tracker, sdk)
        status = json.loads((tracker.directory / "status.json").read_text())
        self.assertEqual(status["mode"], "local_only")
        self.assertTrue(status["finished"])
        self.assertEqual(len([e for e in self.events(tracker) if e["type"] == "metrics"]), 2)

    def test_no_initialization_and_disabled_configuration_are_safe_noops(self):
        tracking.log_metrics({"x": 1.})
        tracking.log_evaluation("val", "AerialVG", (1., 2., 3.), 0.)
        tracking.log_summary({"x": 1.})
        tracking.set_tracking_context(1, 2)
        tracking.finish_tracking()
        self.cfg.swanlab.mode = "disabled"
        with patch.object(tracking.subprocess, "Popen", side_effect=AssertionError("disabled")):
            tracking.initialize_tracking(self.cfg, self.logger)
        self.assertIsNone(tracking._TRACKER)

    def test_forked_data_workers_cannot_log_or_finish_the_parent_run(self):
        tracker, _ = self.start()
        before = (tracker.directory / "events.jsonl").read_text()
        with patch.object(tracking.os, "getpid", return_value=tracker.owner_pid + 1):
            tracker.log({"loss": 1.})
            tracker.write("text", {"text": "worker"})
            tracker.finish()
        self.assertFalse(tracker.closed)
        self.assertEqual((tracker.directory / "events.jsonl").read_text(), before)

    def test_journal_survives_experiment_rename_and_nonfinite_values(self):
        tracker, _ = self.start()
        self.work.rename(self.work.with_name(self.work.name + "+val_top1=31"))
        tracking.log_metrics({"loss": float("nan"), "lr": np.float32(.01)})
        values = [e["values"] for e in self.events(tracker) if e["type"] == "metrics"][-1]
        self.assertIsNone(values["loss"])
        self.assertIsInstance(values["lr"], float)
        self.assertTrue(tracker.directory.exists())

    def test_tracking_does_not_change_rngs_or_model_optimizer_updates(self):
        from propvg.apis.train import train_model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, sample, **kwargs):
                loss = self.weight.square() + self.weight * sample
                return {"loss_det": loss, "loss_refer": loss * .1}, None
        cfg = Config(dict(distributed=False, use_fp16=False, grad_norm_clip=.15, log_interval=2,
                          world_size=1, data=dict(samples_per_gpu=1)))
        first, second = Model(), Model()
        first_optimizer = torch.optim.Adam(first.parameters(), lr=.01)
        second_optimizer = torch.optim.Adam(second.parameters(), lr=.01)
        loader = [{"sample": torch.tensor(float(i))} for i in range(5)]
        with patch("propvg.apis.train.extract_data", side_effect=lambda v:v):
            train_model(0, cfg, first, first_optimizer, loader)
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            tracker, _ = self.start()
            train_model(0, cfg, second, second_optimizer, loader)
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        a, b = first_optimizer.state_dict(), second_optimizer.state_dict()
        self.assertEqual(a["param_groups"], b["param_groups"])
        for parameter in a["state"]:
            for key in a["state"][parameter]:
                torch.testing.assert_close(a["state"][parameter][key], b["state"][parameter][key], rtol=0, atol=0)
        values = [e["values"] for e in self.events(tracker) if e["type"] == "metrics"]
        self.assertEqual([r["global_step"] for r in values], [2, 4, 5])
        self.assertIn("train/loss_total", values[-1])
        self.assertIn("train_current/loss_det", values[-1])
        self.assertEqual(values[-1]["train_window/loss_det"], values[-1]["train_current/loss_det"])
        self.assertIn("lr/0", values[-1])

    def test_replay_creates_a_new_snapshot_without_modifying_original(self):
        spec = importlib.util.spec_from_file_location("tracking_replay_test", Path(__file__).with_name("replay_swanlab.py"))
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        tracker, _ = self.start()
        tracking.log_metrics({"train/loss_total": 1.})
        tracking.finish_tracking()
        before = (tracker.directory / "events.jsonl").read_bytes()
        with patch.object(replay.Path, "cwd", return_value=self.root):
            destination = replay.prepare_replay(self.work / "swanlab_tracking.json", "online")
        self.assertNotEqual(destination, tracker.directory)
        self.assertEqual((tracker.directory / "events.jsonl").read_bytes(), before)
        old = json.loads((tracker.directory / "metadata.json").read_text())
        new = json.loads((destination / "metadata.json").read_text())
        self.assertNotEqual(old["id"], new["id"])
        self.assertIn("/replay-", new["name"])
        self.assertEqual(new["mode"], "online")
        events = [json.loads(line) for line in (destination / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e["type"] == "finish" for e in events), 1)

    def test_explicit_local_modes_do_not_attempt_cloud(self):
        for mode in ("offline", "local"):
            tracker, _ = self.start()
            tracking.finish_tracking()
            sdk, run = self.sdk()
            with patch.dict(os.environ, {}, clear=False):
                worker.run_backend(tracker.directory, sdk=sdk, poll=0, mode=mode)
            self.assertEqual(sdk.init.call_args.kwargs["mode"], mode)
            sdk.init.assert_called_once()
            self.assertEqual(json.loads((tracker.directory / "status.json").read_text())["mode"], mode)

    def test_abnormal_exit_marks_swanlab_crashed(self):
        tracker, _ = self.start()
        tracking.finish_tracking(exit_code=1)
        sdk, _ = self.sdk()
        self.backend(tracker, sdk)
        sdk.finish.assert_called_once_with(state=RunState.CRASHED, error="Training exited abnormally")

    def test_swanlab_environment_overrides_config(self):
        with patch.dict(os.environ, {"SWANLAB_MODE": "disabled"}), \
             patch.object(tracking.subprocess, "Popen") as spawn:
            tracking.initialize_tracking(self.cfg, self.logger)
        spawn.assert_not_called()
        self.assertIsNone(tracking._TRACKER)
        self.cfg.swanlab.mode = "disabled"
        tracker, _ = self.start()
        self.assertIsNotNone(tracker)
        self.assertEqual(json.loads((tracker.directory / "metadata.json").read_text())["mode"], "online")

    def test_summary_text_and_nonfinite_scalars_are_preserved_locally(self):
        tracker, _ = self.start()
        tracking.log_metrics({"train/loss_total": float("nan"), "train/loss_det": .2})
        tracking.log_summary({"experiment/final_work_dir": "/tmp/final"})
        tracking.finish_tracking()
        sdk, run = self.sdk()
        self.backend(tracker, sdk)
        self.assertFalse(any("train/loss_total" in c.args[0] for c in run.log.call_args_list))
        self.assertEqual(run.config["tracking_summary"]["experiment/final_work_dir"], "/tmp/final")
        self.assertTrue(json.loads((tracker.directory / "status.json").read_text())["finished"])

    def test_segmentation_metric_thresholds_keep_original_definitions(self):
        tracker, _ = self.start()
        tracking.log_evaluation("val", "GRefCOCO", (1., 2., 3., 4., 5., [.7, .8, .9]), 1.)
        values = [e["values"] for e in self.events(tracker) if e["type"] == "metrics"][-1]
        self.assertEqual(values["val/mask_acc_at_0.7"], .7)
        self.assertEqual(values["val/mask_acc_at_0.9"], .9)
        self.assertNotIn("val/mask_acc_at_0.5", values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
