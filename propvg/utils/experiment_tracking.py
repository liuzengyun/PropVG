"""Non-networking training-side journal; SwanLab runs in a separate CPU process."""
import atexit
import json
import logging
import math
import numbers
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


_TRACKER = None


def clean(value, key=""):
    """Keep config/scalars JSON-safe without reading GPU tensors or secrets."""
    if key.lower().endswith("api_key") or key.lower() in {"password", "access_token", "secret"}:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): clean(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, numbers.Real):
        return clean(float(value))
    return str(value)


class JournalHandler(logging.Handler):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def emit(self, record):
        text = re.sub(r"(?i)(\b(?:[a-z0-9_]*api_key|access_token|password|secret)\s*[:=]\s*)[^\s,)\n]+",
                      r"\1[REDACTED]", self.format(record))
        self.tracker.write("text", {"text": text})


class ExperimentTracker:
    def __init__(self, cfg, logger):
        options = dict(cfg.get("swanlab") or {})
        self.logger, self.closed, self.warned = logger, False, False
        self.owner_pid = os.getpid()
        self.epoch, self.global_step = 0, 0
        self.process = None
        self.notified_mode = None
        self.checkpoint_epoch = None
        self.finish_wait = max(0., float(options.get("finish_wait", 1.)))
        self.work_dir = Path(cfg.work_dir).resolve()
        base = Path(options.get("logdir", Path.cwd() / "swanlog")).resolve()
        run_id = uuid.uuid4().hex[:8]  # OS entropy; never advances training RNGs.
        self.directory = base / ("tracking-" + cfg.timestamp + "-" + run_id)
        self.directory.mkdir(parents=True)
        self.stream = (self.directory / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        group = self.work_dir.parent.name
        name = options.get("experiment_name", group)
        if cfg.timestamp not in name:
            name += "/" + cfg.timestamp
        metadata = dict(backend="swanlab", id=run_id, name=name, group=group,
                        project=options.get("project", "propvg"), workspace=options.get("workspace"),
                        mode=os.environ.get("SWANLAB_MODE", options.get("mode", "online")),
                        init_timeout=float(options.get("init_timeout", 30.)),
                        stats_interval=float(options.get("stats_interval", 30.)),
                        parent_pid=os.getpid(), work_dir=str(self.work_dir),
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        tags=[str(cfg.get("tag_name", group))],
                        config=clean(dict(cfg._cfg_dict)))
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.work_dir / "swanlab_tracking.json").write_text(json.dumps({
            "name": name, "id": run_id, "directory": str(self.directory),
            "status": str(self.directory / "status.json"),
            "events": str(self.directory / "events.jsonl")}, indent=2), encoding="utf-8")
        self.handler = JournalHandler(self)
        logger.addHandler(self.handler)
        self.worker_log = (self.directory / "worker.log").open("a", encoding="utf-8")
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # Executing a file directly avoids importing propvg/torch in child.
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("swanlab_worker.py")), str(self.directory)],
                stdin=subprocess.DEVNULL, stdout=self.worker_log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
        except Exception as error:
            logger.warning("SwanLab backend could not start (%s); local journal remains available.", type(error).__name__)
        logger.info("SwanLab tracking: %s; online/offline backend runs in background. Status: %s",
                    name, self.directory / "status.json")

    def write(self, kind, payload):
        if self.closed or os.getpid() != self.owner_pid:
            return
        try:
            event = dict(type=kind, time=time.time(), **clean(payload))
            self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Never recursively log through our own handler or break training.
            if not self.warned:
                self.warned = True
                sys.stderr.write("SwanLab local journal write failed; original text logging and training continue.\n")

    def log(self, values):
        if self.closed or os.getpid() != self.owner_pid:
            return
        if "checkpoint_epoch" in values:
            self.checkpoint_epoch = values["checkpoint_epoch"]
        payload = dict(epoch=self.epoch, global_step=self.global_step)
        if self.checkpoint_epoch is not None:
            payload["checkpoint_epoch"] = self.checkpoint_epoch
        payload.update(values)
        self.write("metrics", {"values": payload})
        try:
            state = json.loads((self.directory / "status.json").read_text())
            mode = state.get("mode")
            if mode != "initializing" and mode != self.notified_mode:
                self.notified_mode = mode
                if state.get("url"):
                    self.logger.info("SwanLab online: %s", state["url"])
                else:
                    self.logger.info("SwanLab %s: %s (local journal retained)", mode, self.directory)
        except (OSError, ValueError):
            pass

    def finish(self, exit_code=0):
        if self.closed or os.getpid() != self.owner_pid:
            return
        self.logger.removeHandler(self.handler)
        self.write("finish", {"exit_code": exit_code})
        self.closed = True
        self.stream.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=self.finish_wait)
            except subprocess.TimeoutExpired:
                # Network completion is deliberately not a training barrier.
                self.logger.info("SwanLab backend is finishing asynchronously; local events are saved in %s", self.directory)
        self.worker_log.close()


def initialize_tracking(cfg, logger):
    """Call ONLY on training rank zero, after its normal file logger exists."""
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.finish()
        _TRACKER = None
    try:
        options = cfg.get("swanlab", {})
        if options is False or os.environ.get("SWANLAB_MODE", (options or {}).get("mode", "online")) == "disabled":
            return
        _TRACKER = ExperimentTracker(cfg, logger)
        atexit.register(finish_tracking)
    except Exception as error:
        logger.warning("Experiment tracking unavailable (%s); continuing normal training.", type(error).__name__)


def set_tracking_context(epoch=None, global_step=None):
    if _TRACKER is not None:
        if epoch is not None:
            _TRACKER.epoch = int(epoch)
        if global_step is not None:
            _TRACKER.global_step = int(global_step)


def log_metrics(values):
    if _TRACKER is not None:
        _TRACKER.log(values)


def log_summary(values):
    if _TRACKER is not None:
        _TRACKER.write("summary", {"values": values})


def finish_tracking(exit_code=0):
    global _TRACKER
    tracker, _TRACKER = _TRACKER, None
    if tracker is not None:
        try:
            tracker.finish(exit_code)
        except Exception:
            sys.stderr.write("SwanLab finish failed; original training/checkpoints are unaffected.\n")


def log_evaluation(prefix, dataset, values, seconds):
    """Observe existing evaluation outputs only; never rerun or change metrics."""
    if _TRACKER is None:
        return
    names = {
        "AerialVG": ("top1_acc", "top5_acc", "box_miou"),
        "GRefCOCO": ("f1_rec", "nacc_rec", "tacc_rec", "giou", "ciou", "mask_acc"),
        "RefZOM": ("miou", "oiou", "macc", "mask_acc"),
        "RRefCOCO": ("miou", "oiou", "mrr", "riou"),
    }.get(dataset, ("det_acc", "mask_miou", "mask_oiou", "mask_acc", "det_acc_at_iou"))
    namespace = "test_epoch" if prefix.startswith("test_epoch_") else "test_best" if prefix == "test" else prefix
    result = {namespace + "/seconds": seconds}
    for name, value in zip(names, values):
        if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            first_threshold = .7 if dataset == "GRefCOCO" and name == "mask_acc" else .5
            for i, scalar in enumerate(value):
                result[namespace + "/" + name + "_at_" + str(round(first_threshold + .1 * i, 1))] = scalar
        else:
            result[namespace + "/" + name] = value
    _TRACKER.log(result)
