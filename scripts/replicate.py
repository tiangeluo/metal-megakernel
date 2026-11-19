"""Remeasure both published workloads into a new directory on an Apple-silicon Mac."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ATTENTION_SESSIONS = ("m4pro-02-a", "m4pro-02-b")


def collect(label=None, *, traces=False, results=Path("results"), invoke=subprocess.run):
    from diagnostics.precision_study import SESSIONS
    from scripts.study import study_lock

    if label is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", label):
        raise ValueError("Choose a label of up to 64 letters, digits, dots, hyphens or underscores.")

    with study_lock(results) as lock:
        parent = results / "replications"
        parent.mkdir(exist_ok=True)
        if label is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
            root = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
        else:
            root = parent / label
            try:
                root.mkdir()
            except FileExistsError:
                raise ValueError(f"{root} already exists. Choose a new replication label.") from None
        print(f"Writing fresh measurements to {root}", flush=True)

        def run(*args):
            # A child still collecting after its parent exits must retain the checkout lock.
            invoke([sys.executable, *map(str, args)], check=True, pass_fds=(lock.fileno(),))

        for name in ATTENTION_SESSIONS:
            # Session names also seed method order. Match the original acquisition's order.
            environment = root / "attention" / name / "environment.json"
            run("scripts/environment.py", "--out", environment)
            run("-m", "bench.run", "--suite", "attention", "--session", name,
                "--out", root / "attention", "--environment", environment)

        run("-m", "bench.report", "--scope", "attention", "--input", root / "attention",
            "--sessions", *ATTENTION_SESSIONS, "--out", root / "attention-report")
        control = root / "precision-control"
        for name in SESSIONS:
            run("-m", "diagnostics.precision_study", "--out", control, "--session", name)
        run("-m", "diagnostics.precision_study", "--out", control, "--report-only")
        run("-m", "analysis.decoder_performance", "--replication", root)
        print(f"Timing reports and figure are ready in {root}", flush=True)

        if traces:
            from scripts.capture_traces import capture

            # Instruments may launch the target in a different working directory.
            trace_root = (root / "traces").resolve()
            environment = trace_root / "environment.json"
            run("scripts/environment.py", "--out", environment)
            folder = trace_root / "capture"
            folder.mkdir()
            if not capture(folder, environment):
                raise RuntimeError(f"Trace capture was incomplete. See {folder / 'capture.json'}.")
            run("-m", "analysis.inspect_trace", folder, "--save-exports", trace_root / "xml",
                "--out", trace_root / "observations.json")
        return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", help="Create results/replications/LABEL (default: a unique timestamped name)")
    parser.add_argument("--traces", action="store_true", help="Record and inspect four Instruments captures after timing")
    args = parser.parse_args()
    try:
        collect(args.label, traces=args.traces)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
