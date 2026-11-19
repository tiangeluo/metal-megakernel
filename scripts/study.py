"""Collect both attention sessions even if connected-decoder parity fails."""
import argparse
import fcntl
import re
import subprocess
import sys
from pathlib import Path

from bench.protocol import FIXTURES, STACK


def collect(name, results=Path("results"), invoke=subprocess.run):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,57}", name):
        raise ValueError("Choose a study label of up to 58 letters, digits, dots, hyphens or underscores.")
    results.mkdir(exist_ok=True, parents=True)
    with (results / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A study is already running in this checkout.") from None

        def run(*args, check=True):
            return invoke([sys.executable, *map(str, args)], check=check)

        environment = results / "environment.json"
        seed = 20261118

        def suite(scope, session):
            run("scripts/environment.py", "--out", environment)
            run("-m", "bench.run", "--suite", scope, "--session", session,
                "--out", results, "--environment", environment, "--seed", seed, "--update-tracker")

        sessions = [name + "-a", name + "-b"]

        def report(scope, folder):
            run("-m", "bench.report", "--scope", scope, "--input", results,
                "--out", results / folder, "--sessions", *sessions)
            run("tracker/render.py")

        run("-m", "pytest", "-q")
        suite("pilot", name + "-pilot")
        diagnostics = results / (name + "-diagnostics")
        diagnostics.mkdir(exist_ok=True)
        failures = []
        for case in STACK:
            for fixture in range(FIXTURES):
                fixture_seed = seed + fixture
                path = diagnostics / f"{case.name}-seed{fixture_seed}.json"
                replay = run("-m", "bench.diagnose_stack", "--environment", environment,
                             "--length", case.length, "--batch", case.batch, "--seed", fixture_seed,
                             "--out", path, check=False)
                if replay.returncode:
                    failures.append(str(path))
                run("tracker/render.py")

        for session in sessions:
            suite("attention", session)
        report("attention", "report-attention")
        if failures:
            raise RuntimeError("Attention report ready. Stack replay failed at " +
                               ", ".join(failures) + ". Review diagnostic output before timing the stack.")
        for session in sessions:
            suite("stack", session)
        report("full", "report")
        print("Study finished. See results/report/performance.md and dist/index.html.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    try:
        collect(args.name)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
