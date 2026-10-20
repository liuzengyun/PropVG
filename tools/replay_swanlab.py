"""Replay a PropVG tracking journal into a NEW SwanLab run, without training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def prepare_replay(source, mode, project=None, workspace=None):
    source = Path(source).resolve()
    if source.is_file():
        source = Path(json.loads(source.read_text())["directory"])
    metadata = json.loads((source / "metadata.json").read_text())
    records = []
    for line in (source / "events.jsonl").read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # A final incomplete line is not a committed event.
        if record["type"] != "finish":
            records.append(record)
    run_id = uuid.uuid4().hex[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = Path.cwd() / "swanlog"
    destination = base / ("tracking-replay-" + timestamp + "-" + run_id)
    destination.mkdir(parents=True)
    metadata.update(project=project or metadata.get("project", "propvg"),
                    workspace=workspace or metadata.get("workspace"),
                    backend="swanlab", id=run_id, mode=mode, parent_pid=os.getpid(),
                    init_timeout=max(30., float(metadata.get("init_timeout", 30.))),
                    name=metadata["name"] + "/replay-" + timestamp)
    metadata["config"]["tracking_replay_source"] = str(source)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    records.append(dict(type="finish", exit_code=0, time=time.time()))
    (destination / "events.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="swanlab_tracking.json or a SwanLab tracking journal directory")
    parser.add_argument("--mode", choices=("online", "offline", "local", "auto"), default="online")
    parser.add_argument("--project", help="SwanLab project (default: source SwanLab project, or propvg)")
    parser.add_argument("--workspace", help="SwanLab workspace (default: logged-in account)")
    args = parser.parse_args()
    destination = prepare_replay(args.source, args.mode, args.project, args.workspace)
    print("Replaying into a NEW SwanLab run; original journal is unchanged:", destination, flush=True)
    worker = Path(__file__).resolve().parents[1] / "propvg/utils/swanlab_worker.py"
    result = subprocess.run([sys.executable, str(worker), str(destination)], stdin=subprocess.DEVNULL,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
