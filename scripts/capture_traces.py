"""Capture candidate Instruments traces for the registered long-context comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CASE = "attention-b1-l4096"
METHODS = ("single", "split", "compiled", "hybrid")
TEMPLATE = "Metal System Trace"
SIGNPOST_INSTRUMENT = "Points of Interest"


def record_command(method, folder, environment):
    # Preserve the virtualenv path. Resolving the python symlink selects base Python.
    return ["xcrun", "xctrace", "record", "--template", TEMPLATE,
            "--instrument", SIGNPOST_INSTRUMENT, "--no-prompt", "--time-limit", "30s",
            "--output", str(folder / f"{method}.trace"),
            "--target-stdout", str(folder / f"{method}.stdout"),
            "--env", f"PYTHONPATH={ROOT}",
            "--launch", "--", sys.executable, "-m", "bench.trace", "--case", CASE,
            "--method", method, "--environment", str(environment)]


def catalog_has(kind, name):
    result = subprocess.run(["xcrun", "xctrace", "list", kind], capture_output=True,
                            text=True, timeout=45, check=True)
    return name in {line.strip() for line in result.stdout.splitlines()}


def capture(folder, environment):
    report = {"case": CASE, "template": TEMPLATE, "marker": "Decode step", "captures": {},
              "status": "pending", "environment": str(environment)}

    def save():
        tmp = folder / "capture.json.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(folder / "capture.json")

    save()
    try:
        for kind, name in (("templates", TEMPLATE), ("instruments", SIGNPOST_INSTRUMENT)):
            if not catalog_has(kind, name):
                raise RuntimeError(f"{name} is missing from the Instruments {kind} catalog")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status="unavailable", error=str(error))
        save()
        return False

    report["status"] = "recording"
    for method in METHODS:
        trace = folder / f"{method}.trace"
        target_log = folder / f"{method}.stdout"
        tool_log = folder / f"{method}.log"
        row = {"trace": trace.name, "target_log": target_log.name, "tool_log": tool_log.name,
               "status": "recording"}
        report["captures"][method] = row
        save()
        print(f"Capturing {CASE}/{method}", flush=True)
        with tool_log.open("w") as log:
            try:
                proc = subprocess.run(record_command(method, folder, environment),
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                row["xctrace_exit"] = proc.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
        marker = f"Captured candidate interval: {CASE}/{method}"
        row["target_parity_passed"] = (target_log.is_file()
                                       and marker in target_log.read_text(errors="replace").splitlines())
        row["bundle_present"] = trace.is_dir()
        row["status"] = ("awaiting_inspection" if row.get("xctrace_exit") == 0
                         and row["target_parity_passed"] and row["bundle_present"] else "incomplete")
        save()
    ok = all(row["status"] == "awaiting_inspection" for row in report["captures"].values())
    report["status"] = "awaiting_inspection" if ok else "incomplete"
    save()
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environment", type=Path, default=ROOT / "results/environment.json")
    parser.add_argument("--dry-run", action="store_true", help="Print capture commands without starting Instruments.")
    args = parser.parse_args()
    environment, parent = args.environment.resolve(), args.out.resolve()
    if args.dry_run:
        print(json.dumps([record_command(m, parent, environment) for m in METHODS], indent=2))
        return 0
    parent.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    print(f"Trace manifest: {folder / 'capture.json'}", flush=True)
    return 0 if capture(folder, environment) else 1


if __name__ == "__main__":
    sys.exit(main())
