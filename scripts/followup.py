"""Capture the decoder boundaries and long-context traces for the measured study."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from bench.protocol import FIXTURES, STACK, put_json
from bench.run import preflight, refresh_tracker


def collect(results):
    parent = results / "m4pro-02-a/attention/manifest.json"
    spec = json.loads(parent.read_text())["spec"]
    with (results / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A study is already using this checkout.") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(tempfile.mkdtemp(prefix=f"m4pro-02-followup-{stamp}-", dir=results))
        environment = out / "environment.json"
        subprocess.run([sys.executable, "scripts/environment.py", "--out", str(environment)], check=True)
        _, machine = preflight(environment)
        state = {"parent": spec["session"], "parent_machine": spec["machine"], "machine": machine,
                 "status": "running", "crossovers": {}, "trace_status": "pending"}
        manifest = out / "manifest.json"

        def record_progress():
            put_json(manifest, state)
            refresh_tracker(True)

        record_progress()
        print(f"Writing the follow-up to {out}", flush=True)

        for case in STACK:
            for index in range(FIXTURES):
                seed = spec["seed"] + index
                tag = f"crossover-{case.name}-seed{seed}"
                run = subprocess.run([sys.executable, "-m", "diagnostics.stack_crossover",
                                      "--length", str(case.length), "--seed", str(seed),
                                      "--out", str(out / (tag + ".json"))])
                state["crossovers"][tag] = "complete" if run.returncode == 0 else "failed"
                record_progress()

        trace = subprocess.run([sys.executable, "-m", "scripts.capture_traces",
                                "--out", str(out / "traces"), "--environment", str(environment)])
        state["trace_status"] = "captured" if trace.returncode == 0 else "unavailable or incomplete"
        state["status"] = ("complete" if all(v == "complete" for v in state["crossovers"].values())
                           else "incomplete")
        record_progress()
        print(f"Decoder captures {state['status']}. Trace candidates {state['trace_status']}. See {out}.")
        return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        sys.exit(collect(args.results))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
