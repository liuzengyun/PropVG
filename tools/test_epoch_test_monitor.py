"""CPU regressions for optional epoch test logging without model selection."""
import ast
import importlib.util
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import numpy as np
import torch
from mmcv import Config
# A vendored detectron2 also exposes a top-level `tools` package.
spec = importlib.util.spec_from_file_location("propvg_training_runner_test", Path(__file__).with_name("train.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RemoveTracking(ast.NodeTransformer):
    """Ignore ONLY observational logging; retain all training/eval decisions."""
    calls = {"initialize_tracking", "set_tracking_context", "log_metrics", "log_summary", "finish_tracking"}

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name) and call.func.id in self.calls:
                return None
            # Normalize tracking calls from the historical Git baseline only.
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) \
                    and call.func.value.id == "wandb" and call.func.attr in {"init", "finish"}:
                return None
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        return node if node.body else None

    def visit_Assign(self, node):
        # Reuse the metadata returned by the SAME checkpoint load, no extra I/O.
        if any(isinstance(t, ast.Name) and t.id == "loaded_best" for t in node.targets):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "load_checkpoint":
                call.keywords = [kw for kw in call.keywords if kw.arg != "return_metadata"]
                return ast.Expr(value=call)
        return self.generic_visit(node)


class EpochTestMonitorTests(unittest.TestCase):
    def config(self, **values):
        options = dict(dataset="AerialVG", start_test=22, distributed=False, rank=0, world_size=1,
                       evaluate_interval=3, data=dict(test=dict(type="FakeDataset")))
        options.update(values)
        return Config(options)

    def model(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
        model.train()
        model[1].eval()  # Preserve per-module modes, not just the top level.
        model.iter = 7
        return model

    def test_boundary_resume_and_disabled_old_configs(self):
        monitor = runner.EpochTestMonitor(self.config())
        self.assertFalse(monitor.due(20))  # Human epoch 21.
        self.assertTrue(monitor.due(21))   # Human epoch 22.
        self.assertTrue(monitor.due(31))   # Resuming at epoch 32 also tests.
        for start in (None, 0, -1):
            disabled = runner.EpochTestMonitor(self.config(start_test=start))
            self.assertFalse(disabled.due(100))
            with patch.object(disabled, "_evaluate", side_effect=AssertionError("disabled")):
                disabled.run(100, self.model())
        old_cfg = self.config()
        old_cfg.pop("start_test")
        self.assertFalse(runner.EpochTestMonitor(old_cfg).due(100))
        self.assertFalse(runner.EpochTestMonitor(self.config(dataset="RefCOCOUNC")).due(100))
        for value in (True, 22.5, "22"):
            with self.assertRaises(ValueError):
                runner.EpochTestMonitor(self.config(start_test=value)).due(21)

    def test_live_model_loader_cache_and_rng_modes_counter_are_preserved(self):
        cfg, model = self.config(), self.model()
        monitor = runner.EpochTestMonitor(cfg)
        modes = [module.training for module in model.modules()]
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        random.seed(22)
        np.random.seed(22)
        torch.manual_seed(22)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        logger = Mock()

        def evaluate(prefix, eval_cfg, eval_model, loader):
            self.assertIn(prefix, ("test_epoch_22", "test_epoch_23"))
            self.assertIs(eval_model, model)
            self.assertFalse(eval_cfg.distributed)
            self.assertEqual((eval_cfg.rank, eval_cfg.world_size), (0, 1))
            eval_model.eval()
            eval_model.iter += 100
            random.random(), np.random.rand(), torch.rand(3)
            return 30., 60., 25.

        with patch.object(runner, "is_main", return_value=True), \
             patch.object(runner, "build_dataset", return_value="dataset") as dataset, \
             patch.object(runner, "build_dataloader", return_value="loader") as loader, \
             patch.object(runner, "evaluate_model", side_effect=evaluate) as evaluator, \
             patch.object(runner, "get_root_logger", return_value=logger), \
             patch.object(runner, "load_checkpoint", side_effect=AssertionError("must use current model")), \
             patch.object(runner, "save_checkpoint", side_effect=AssertionError("must not select test best")), \
             patch.object(runner.os, "rename", side_effect=AssertionError("must not rename")):
            monitor.run(20, model)
            evaluator.assert_not_called()
            monitor.run(21, SimpleNamespace(module=model))
            monitor.run(22, model)  # Independent of the val interval (3).
            self.assertEqual(evaluator.call_count, 2)
            self.assertEqual(dataset.call_count, 1)
            self.assertEqual(loader.call_count, 1)
        self.assertEqual(model.iter, 7)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        self.assertFalse(cfg.distributed)
        self.assertEqual(cfg.world_size, 1)
        self.assertTrue(any(call.args[0].startswith("Epoch [%d] test (current model)") for call in logger.info.call_args_list))

    def test_missing_start_test_is_a_noop_even_without_test_data_or_model(self):
        # A historical config must not suddenly require data.test, GPU state,
        # a different checkpoint, or any new evaluator when the key is absent.
        monitor = runner.EpochTestMonitor(Config(dict(dataset="AerialVG")))
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with patch.object(runner, "build_dataset") as dataset, \
             patch.object(runner, "build_dataloader") as loader, \
             patch.object(runner, "evaluate_model") as evaluate, \
             patch.object(runner, "load_checkpoint") as load, \
             patch.object(runner, "save_checkpoint") as save, \
             patch.object(runner.dist, "barrier") as barrier, \
             patch.object(runner.os, "rename") as rename:
            for epoch in (0, 21, 35, 100):
                monitor.run(epoch, None)
            for function in (dataset, loader, evaluate, load, save, barrier, rename):
                function.assert_not_called()
        self.assertIsNone(monitor.context)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_exception_restores_training_state(self):
        model = self.model()
        monitor = runner.EpochTestMonitor(self.config())
        expected = [module.training for module in model.modules()]
        rng = torch.get_rng_state()

        def fail(*unused):
            model.eval()
            model.iter = 100
            torch.rand(5)
            raise RuntimeError("fake test error")

        with patch.object(runner, "is_main", return_value=True), \
             patch.object(runner, "build_dataset", return_value="dataset"), \
             patch.object(runner, "build_dataloader", return_value="loader"), \
             patch.object(runner, "get_root_logger", return_value=Mock()), \
             patch.object(runner, "evaluate_model", side_effect=fail), \
             self.assertRaisesRegex(RuntimeError, "fake test error"):
            monitor.run(21, model)
        self.assertEqual([module.training for module in model.modules()], expected)
        self.assertEqual(model.iter, 7)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_distributed_config_and_model_wrapper_are_preserved(self):
        cfg = self.config(distributed=True, rank=0, world_size=4)
        monitor = runner.EpochTestMonitor(cfg)
        base = self.model()
        wrapped = torch.nn.Module()
        wrapped.module = base
        modes = [module.training for module in wrapped.modules()]
        def evaluate(prefix, test_cfg, model, loader):
            self.assertTrue(test_cfg.distributed)
            self.assertEqual((test_cfg.rank, test_cfg.world_size), (0, 4))
            self.assertIs(model, wrapped)
            self.assertTrue(cfg.distributed)
            self.assertEqual(cfg.world_size, 4)
            model.eval()
            base.iter += 10
            return 30., 60., 25.
        with patch.object(runner, "is_main", return_value=True), \
             patch.object(runner, "build_dataset", return_value="dataset"), \
             patch.object(runner, "build_dataloader", return_value="loader"), \
             patch.object(runner, "get_root_logger", return_value=Mock()), \
             patch.object(runner, "evaluate_model", side_effect=evaluate):
            monitor._evaluate(21, wrapped)
        self.assertTrue(cfg.distributed)
        self.assertEqual(cfg.world_size, 4)
        self.assertEqual(base.iter, 7)
        self.assertEqual([module.training for module in wrapped.modules()], modes)

    def test_non_main_ranks_load_and_evaluate_their_own_test_shards(self):
        cfg = self.config(distributed=True, rank=1, world_size=2)
        monitor = runner.EpochTestMonitor(cfg)
        model = self.model()
        with patch.object(runner, "is_main", return_value=False), \
             patch.object(runner.dist, "new_group") as create, \
             patch.object(runner.dist, "broadcast_object_list") as broadcast, \
             patch.object(runner, "build_dataset", return_value="dataset") as dataset, \
             patch.object(runner, "build_dataloader", return_value="sharded_loader") as loader, \
             patch.object(runner, "evaluate_model", return_value=(30., 60., 25.)) as evaluate, \
             patch.object(runner, "get_root_logger", side_effect=AssertionError("rank1 must not log")):
            monitor.run(20, model)
            create.assert_not_called()
            evaluate.assert_not_called()
            monitor.run(21, model)
            monitor.run(22, model)
            create.assert_not_called()
            broadcast.assert_not_called()
            self.assertEqual(dataset.call_count, 1)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(evaluate.call_count, 2)
            test_cfg = loader.call_args.args[0]
            self.assertTrue(test_cfg.distributed)
            self.assertEqual((test_cfg.rank, test_cfg.world_size), (1, 2))
            self.assertIs(evaluate.call_args.args[2], model)

    def test_evaluator_errors_propagate_on_all_ranks(self):
        for main in (False, True):
            monitor = runner.EpochTestMonitor(self.config(distributed=True))
            with patch.object(runner, "is_main", return_value=main), \
                 patch.object(monitor, "_evaluate", side_effect=RuntimeError("fake failure")), \
                 self.assertRaisesRegex(RuntimeError, "fake failure"):
                monitor.run(21, None)

    def test_final_best_checkpoint_test_and_rename_are_unchanged_from_git(self):
        path = Path(runner.__file__).resolve()
        previous = subprocess.check_output(["git", "show", "HEAD:tools/train.py"], text=True)
        def final_block(source):
            worker = RemoveTracking().visit(next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main_worker"))
            index = next(i for i, node in enumerate(worker.body) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "main_process" for t in node.targets))
            return [ast.dump(node, include_attributes=False) for node in worker.body[index:]]
        self.assertEqual(final_block(path.read_text()), final_block(previous))

    def test_original_training_and_validation_body_is_unchanged_apart_from_optional_monitor(self):
        previous = subprocess.check_output(["git", "show", "HEAD:tools/train.py"], text=True)
        def worker(source):
            return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main_worker")
        class RemoveOptionalMonitor(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == "epoch_test_monitor" for t in node.targets):
                    return None
                return self.generic_visit(node)
            def visit_Expr(self, node):
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) \
                        and isinstance(value.func.value, ast.Name) and value.func.value.id == "epoch_test_monitor":
                    return None
                return self.generic_visit(node)
        current = RemoveTracking().visit(RemoveOptionalMonitor().visit(worker(Path(runner.__file__).read_text())))
        reference = RemoveTracking().visit(RemoveOptionalMonitor().visit(worker(previous)))
        self.assertEqual(ast.dump(current, include_attributes=False),
                         ast.dump(reference, include_attributes=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
