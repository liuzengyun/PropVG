"""Standalone CPU-only SwanLab uploader; imports neither torch nor propvg."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


EPOCH_PREFIXES = {"val", "test_epoch", "best", "epoch"}


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
        # 0.7.x accepts online in init(), but its environment parser uses cloud.
        os.environ["SWANLAB_MODE"] = "cloud" if mode == "online" else mode
        if sdk is None:
            import swanlab as sdk
        settings = sdk.Settings(
            hardware_interval=max(5, int(metadata.get("stats_interval", 30))),
            requirements_collect=False, conda_collect=False,
            collect_runtime=False, backup=True)
        # Let SwanLab allocate its own run ID. 0.7.x forbids an explicit ID
        # without resume="allow", which also disables its hardware monitor.
        common = dict(project=metadata["project"], workspace=metadata.get("workspace"),
                      experiment_name=metadata["name"], group=metadata["group"],
                      tags=metadata.get("tags", []), config=metadata["config"],
                      logdir=str(directory / "sdk"), settings=settings, reinit=True)
        try:
            run = sdk.init(mode=mode, **common)
        except Exception as error:
            if mode != "online":
                raise
            status(fallback_reason="online_init_failed:" + type(error).__name__)
            os.environ["SWANLAB_MODE"] = "offline"
            run = sdk.init(mode="offline", **common)
        actual = "online" if run.mode in ("online", "cloud") else run.mode
        url = sdk.get_url() if actual == "online" else None
        status(mode=actual, url=url, sdk_run_id=str(run.public.run_id), run_directory=str(run.public.run_dir))
        print("SwanLab tracking: mode=%s url=%s" % (actual, url), flush=True)
    except Exception as error:
        status(mode="local_only", fallback_reason=type(error).__name__)
        print("SwanLab unavailable; complete events remain in the local journal.", flush=True)
        run = None

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
                            sdk.finish(state=type(run.state).CRASHED, error="Training exited abnormally")
                        else:
                            sdk.finish()
                        if hasattr(signal, "SIGALRM"):
                            signal.alarm(0)
                    status(finished=True, processed_events=processed)
                    break
            except Exception as error:
                status(mode="local_only", fallback_reason=type(error).__name__, processed_events=processed)
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
    for mode in modes:
        # A fresh process prevents a hung online SDK from affecting fallback.
        command = [sys.executable, str(Path(__file__).resolve()), str(directory), "--session", mode]
        try:
            (directory / "status.json").unlink()
        except FileNotFoundError:
            pass
        env = dict(os.environ, SWANLAB_MODE="cloud" if mode in ("online", "cloud", "auto") else mode)
        session = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        deadline = time.monotonic() + timeout
        while session.poll() is None:
            try:
                state = json.loads((directory / "status.json").read_text())
            except (OSError, ValueError):
                state = {}
            if state.get("mode") in ("online", "offline", "local", "disabled", "local_only"):
                return session.wait()
            if time.monotonic() >= deadline:
                reason = mode + "_init_deadline"
                try:
                    os.killpg(session.pid, signal.SIGTERM)
                    session.wait(timeout=1.)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(session.pid, signal.SIGKILL)
                    session.wait()
                break
            time.sleep(.1)
        else:
            reason = "backend_exited_before_initialization"
    (directory / "status.json").write_text(json.dumps(dict(
        name=metadata["name"], id=metadata["id"], mode="local_only", url=None,
        fallback_reason=reason, journal=str(directory / "events.jsonl")), indent=2))
    return 0


if __name__ == "__main__":
    if "--session" in sys.argv:
        run_backend(sys.argv[1], mode=sys.argv[3])
    else:
        raise SystemExit(run_supervised(sys.argv[1]))
