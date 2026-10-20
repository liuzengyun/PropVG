"""Standalone CPU-only SwanLab uploader; imports neither torch nor propvg."""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


EPOCH_PREFIXES = {"val", "test_epoch", "best", "epoch"}


def error_detail(error):
    """Keep useful SDK diagnostics without exposing credentials."""
    detail = str(error)
    for key, value in os.environ.items():
        if value and any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
            detail = detail.replace(value, "[REDACTED]")
    detail = re.sub(r"(?i)(\b(?:[a-z0-9_]*api_key|access_token|password|secret)\s*[:=]\s*)[^\s,)\n]+",
                    r"\1[REDACTED]", detail)
    return type(error).__name__ + ": " + detail[:2000]


def log_scalars(run, values, context):
    """SwanLab tracks steps per metric: train uses batches, evaluation uses epochs."""
    context.update({k: values[k] for k in ("epoch", "global_step", "checkpoint_epoch") if k in values})
    groups = {}
    for key, value in values.items():
        # The journal retains null/nonfinite values; SDK scalar charts cannot.
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        prefix = key.split("/", 1)[0]
        axis = "checkpoint_epoch" if prefix == "test_best" else "epoch" if prefix in EPOCH_PREFIXES else "global_step"
        step = max(0, int(context.get(axis, 0)))
        groups.setdefault(step, {})[key] = value
    for step, data in groups.items():
        run.log(data, step=step)


def run_backend(directory, sdk=None, poll=.1, mode=None):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    state = dict(name=metadata["name"], id=metadata["id"], mode="initializing", url=None,
                 journal=str(directory / "events.jsonl"))

    def status(**values):
        state.update(values)
        temporary = directory / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(directory / "status.json")

    status()
    run = None
    try:
        mode = mode or metadata.get("mode", "online")
        mode = "online" if mode in ("auto", "cloud") else mode
        if mode not in ("online", "offline", "local", "disabled"):
            raise ValueError("Unsupported SwanLab mode")
        os.environ["SWANLAB_MODE"] = mode
        if sdk is None:
            import swanlab as sdk
        version = re.match(r"(\d+)\.(\d+)", str(getattr(sdk, "__version__", "")))
        modern = bool(version and tuple(map(int, version.groups())) >= (0, 10))
        interval = max(5, int(metadata.get("stats_interval", 30)))
        if modern:
            settings = sdk.Settings(interactive=False, core={"skip_store": False},
                                    probe=dict(requirements=False, conda=False, runtime=False,
                                               git=False, monitor_interval=interval))
        else:
            # Legacy SDK remains usable for local-only recovery.
            os.environ["SWANLAB_MODE"] = "cloud" if mode == "online" else mode
            settings = sdk.Settings(hardware_interval=interval, requirements_collect=False,
                                    conda_collect=False, collect_runtime=False, backup=True)
        # Let SwanLab allocate its own run ID. 0.7.x forbids an explicit ID
        # without resume="allow", which also disables its hardware monitor.
        common = dict(project=metadata["project"], workspace=metadata.get("workspace"),
                      group=metadata["group"],
                      tags=metadata.get("tags", []), config=metadata["config"],
                      settings=settings, reinit=True)
        common.update({"name" if modern else "experiment_name": metadata["name"],
                       "log_dir" if modern else "logdir": str(directory / "sdk")})
        run = sdk.init(mode=mode, **common)
        actual = "online" if run.mode in ("online", "cloud") else run.mode
        url = (run.url if modern else sdk.get_url()) if actual == "online" else None
        run_id, run_dir = (run.id, run.dir) if modern else (run.public.run_id, run.public.run_dir)
        status(mode=actual, url=url, sdk_run_id=str(run_id), run_directory=str(run_dir))
        print("SwanLab tracking: mode=%s url=%s" % (actual, url), flush=True)
    except Exception as error:
        detail = error_detail(error)
        status(mode="initialization_failed", fallback_reason=type(error).__name__, error=detail)
        print("SwanLab initialization failed: " + detail, flush=True)
        # A failed SDK init may leave singleton/proxy state behind. The
        # supervisor must retry offline in a fresh process, never this one.
        return 1

    processed, context, summary = 0, {}, {}
    with (directory / "events.jsonl").open() as stream, (directory / "console.log").open("a") as console:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line or not line.endswith("\n"):
                stream.seek(position)
                try:
                    os.kill(metadata["parent_pid"], 0)
                except ProcessLookupError:
                    event = dict(type="finish", exit_code=1)
                else:
                    time.sleep(poll)
                    continue
            else:
                event = json.loads(line)
            processed += 1
            kind = event["type"]
            if kind == "text":
                console.write(event["text"] + "\n")
                console.flush()
                # Printed after init, so SwanLab also captures the console log.
                print(event["text"], flush=True)
            if kind == "summary":
                summary.update(event["values"])
                (directory / "summary.json").write_text(json.dumps(summary, indent=2))
            try:
                if run is not None and kind == "metrics":
                    log_scalars(run, event["values"], context)
                elif run is not None and kind == "summary":
                    # SwanLab 0.7 stores our summary in the run config.
                    run.config.update({"tracking_summary": dict(summary)})
                    numeric = {"summary/" + k: v for k, v in event["values"].items()
                               if isinstance(v, (int, float)) and math.isfinite(v)}
                    if numeric:
                        run.log(numeric)
                elif kind == "finish":
                    if run is not None:
                        # Bound final network flushing independently of training.
                        if hasattr(signal, "SIGALRM"):
                            signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
                            signal.alarm(30)
                        if event.get("exit_code", 0):
                            sdk.finish(state="crashed" if modern else type(run.state).CRASHED,
                                       error="Training exited abnormally")
                        else:
                            sdk.finish()
                        if hasattr(signal, "SIGALRM"):
                            signal.alarm(0)
                    status(finished=True, processed_events=processed)
                    break
            except Exception as error:
                status(mode="local_only", fallback_reason=type(error).__name__, error=error_detail(error), processed_events=processed)
                run = None
                if kind == "finish":
                    status(finished=True)
                    break


def run_supervised(directory):
    """Bound SDK import, initialization and credential/network retries."""
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    timeout = max(1., float(metadata.get("init_timeout", 30.))) + 2.
    modes = [metadata.get("mode", "online")]
    if modes[0] in ("online", "cloud", "auto"):
        modes.append("offline")
    reason = None
    failures = []

    def stop_session(session):
        try:
            os.killpg(session.pid, signal.SIGTERM)
            session.wait(timeout=1.)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(session.pid, signal.SIGKILL)
            session.wait()

    for mode in modes:
        # A fresh process prevents a hung online SDK from affecting fallback.
        command = [sys.executable, str(Path(__file__).resolve()), str(directory), "--session", mode]
        try:
            (directory / "status.json").unlink()
        except FileNotFoundError:
            pass
        env = dict(os.environ, SWANLAB_MODE="online" if mode in ("online", "cloud", "auto") else mode)
        session = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        deadline = time.monotonic() + timeout
        while session.poll() is None:
            try:
                state = json.loads((directory / "status.json").read_text())
            except (OSError, ValueError):
                state = {}
            if state.get("mode") in ("online", "offline", "local", "disabled", "local_only"):
                if failures:
                    state["initialization_failures"] = failures
                    (directory / "status.json").write_text(json.dumps(state, indent=2))
                return session.wait()
            if state.get("mode") == "initialization_failed":
                reason = state.get("error", state.get("fallback_reason"))
                stop_session(session)
                break
            if time.monotonic() >= deadline:
                reason = mode + "_init_deadline"
                stop_session(session)
                break
            time.sleep(.1)
        else:
            try:
                state = json.loads((directory / "status.json").read_text())
            except (OSError, ValueError):
                state = {}
            if state.get("finished"):
                return session.returncode
            reason = state.get("error", "backend_exited_before_initialization")
        failures.append(dict(mode=mode, error=reason))
    (directory / "status.json").write_text(json.dumps(dict(
        name=metadata["name"], id=metadata["id"], mode="local_only", url=None,
        fallback_reason=reason, initialization_failures=failures,
        journal=str(directory / "events.jsonl")), indent=2))
    return 0


if __name__ == "__main__":
    if "--session" in sys.argv:
        raise SystemExit(run_backend(sys.argv[1], mode=sys.argv[3]))
    else:
        raise SystemExit(run_supervised(sys.argv[1]))
