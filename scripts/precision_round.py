"""Time the FP32 expert control, then profile the original attention schedules."""
import fcntl
from pathlib import Path
import subprocess
import sys


def main():
    results = Path("results")
    results.mkdir(exist_ok=True)
    study = subprocess.run([sys.executable, "-m", "diagnostics.precision_study",
                            "--out", str(results / "precision-control-01")])
    if study.returncode == 75:
        return 75
    # A numerical failure in the stack does not invalidate an isolated attention trace.
    with (results / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another study is running. Trace capture deferred.", file=sys.stderr)
            return 75
        folder = results / "attention-trace-retry"
        folder.mkdir(exist_ok=True)
        environment = folder / "environment.json"
        probe = subprocess.run([sys.executable, "scripts/environment.py", "--out", str(environment)])
        if probe.returncode:
            return probe.returncode
        trace = subprocess.run([sys.executable, "-m", "scripts.capture_traces",
                                "--out", str(folder), "--environment", str(environment)])
    print(f"Precision study exit {study.returncode}. Attention trace exit {trace.returncode}.")
    return study.returncode or trace.returncode


if __name__ == "__main__":
    sys.exit(main())
