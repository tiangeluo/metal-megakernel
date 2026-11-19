"""Measure the connected FP32-MoE control against its own CPU oracle in two processes."""
from __future__ import annotations

import argparse
from contextlib import closing, ExitStack
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

from bench.diagnose_stack import compare_outputs
from bench.protocol import FIXTURES, MIN_REPEATS, REPO, SCHEMA, SCHEDULES, STACK, equal_stack, finite_time, put_json, source_hash
from bench.report import LABELS, metrics, read_case
from bench.run import now, preflight, randomized_order, save_case
from scripts.environment import snapshot


SEED = 20261118
WARMUPS = 4
SESSIONS = ("a", "b")
FIELDS = {"summary"} | {f"layer{layer}_{field}" for layer in (10, 11) for field in (
    "cache_k", "cache_v", "hist_k", "hist_v", "post_attn", "post_moe", "route_ids"
)}
SPEC_FIELDS = {"schema", "session", "control", "contract", "machine", "control_sha256",
               "seed", "fixtures", "trials", "warmups"}


def control_hash():
    digest = hashlib.sha256()
    for name in ("precision_model.py", "precision_study.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def signature(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_control_case(folder, case, spec):
    """Require all 15 pre/post oracle results plus the original pairing and checksum checks."""
    proof = json.loads((folder / (case.name + ".proof.json")).read_text())
    if proof.get("experiment_signature") != signature(spec):
        raise ValueError(f"Control certificate provenance mismatch in {folder}")
    for phase in ("oracle_max_abs", "post_oracle_max_abs"):
        data = proof.get(phase, {})
        if set(data) != {str(i) for i in range(FIXTURES)}:
            raise ValueError(f"Incomplete {phase} fixtures for {case.name}")
        for methods in data.values():
            if set(methods) != set(case.methods):
                raise ValueError(f"Incomplete {phase} schedules for {case.name}")
            for observations in methods.values():
                if set(observations) != FIELDS or not all(np.isfinite(v) and v >= 0 for v in observations.values()):
                    raise ValueError(f"Incomplete {phase} observables for {case.name}")
    return read_case(folder, case, spec)


def checks(case, fixtures, phase, failures):
    records = {}
    for index, (oracle, runners) in enumerate(fixtures):
        records[index] = {}
        for method, runner in runners.items():
            try:
                runner.reset()
                runner.run()
                actual = runner.output_numpy()
                try:
                    records[index][method] = equal_stack(oracle, actual, case.length - 1)
                except (AssertionError, ValueError) as error:
                    detail = compare_outputs(oracle, actual, case.length - 1)
                    failures.append(dict(phase=phase, seed=SEED+index, method=method,
                                         error=str(error), comparison=detail))
            except (RuntimeError, ValueError, AssertionError, OSError) as error:
                failures.append(dict(phase=phase, seed=SEED+index, method=method, error=str(error)))
    return records


def measure(folder, case, spec, prepare):
    failures = []
    fixtures = []
    with ExitStack() as cleanup:
        for index in range(FIXTURES):
            args = dict(length=case.length, batch=case.batch, seed=SEED + index)
            with closing(prepare("cpu", **args)) as cpu:
                cpu.reset()
                cpu.run()
                oracle = {key: np.array(value, copy=True) for key, value in cpu.output_numpy().items()}
            if set(oracle) != FIELDS:
                raise ValueError("Control oracle omitted registered stack outputs")
            runners = {}
            for method in case.methods:
                try:
                    runner = prepare(method, **args)
                    cleanup.callback(runner.close)
                    runners[method] = runner
                except (RuntimeError, ValueError, AssertionError, OSError, ImportError) as error:
                    failures.append(dict(phase="prepare", seed=SEED+index, method=method, error=str(error)))
            fixtures.append((oracle, runners))
        before = checks(case, fixtures, "before timing", failures)
        if not failures:
            rows = timed_rows(case, spec, fixtures)
            after = checks(case, fixtures, "after timing", failures)
    if failures:
        put_json(folder/(case.name+".failures.json"), dict(case=case.name, spec=spec, failures=failures))
        return False
    # Install the certificate last. An interrupted write may leave an uncertified CSV,
    # which can be measured again without disturbing certificates for the other length.
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        staged = Path(scratch)
        save_case(staged, case, rows, before)
        certificate = staged/(case.name+".proof.json")
        proof = json.loads(certificate.read_text())
        proof.update(experiment_signature=signature(spec), post_oracle_max_abs=after)
        put_json(certificate, proof)
        read_control_case(staged, case, spec)
        (staged/(case.name+".csv")).replace(folder/(case.name+".csv"))
        certificate.replace(folder/certificate.name)
    (folder/(case.name+".failures.json")).unlink(missing_ok=True)
    return True


def timed_rows(case, spec, fixtures):
    # Reset and all host readback are outside the clock. Case.run() synchronizes on the GPU.
    session = spec["session"]
    key = int.from_bytes(hashlib.sha256((session+"/"+case.name).encode()).digest()[:8], "big")
    rng = random.Random(key ^ SEED)
    for _ in range(WARMUPS):
        for _, runners in fixtures:
            for method in randomized_order(case.methods, rng):
                runners[method].reset()
                runners[method].run()
    rows = []
    for first in range(0, MIN_REPEATS, FIXTURES):
        blocks = list(range(first, min(first + FIXTURES, MIN_REPEATS)))
        rng.shuffle(blocks)
        for block in blocks:
            fixture = block % FIXTURES
            runners = fixtures[fixture][1]
            for order, method in enumerate(randomized_order(case.methods, rng)):
                runner = runners[method]
                runner.reset()
                start = time.perf_counter_ns()
                runner.run()
                elapsed = finite_time((time.perf_counter_ns() - start)/1000, "wall latency")
                rows.append(dict(session=session, case=case.name, fixture=fixture, block=block,
                                 order=order, method=method, wall_us=elapsed, native_gpu_us=None,
                                 encoded_dispatches=None))
    return rows


def session_run(out, name):
    from diagnostics.precision_model import CONTROL, PRECISION_CONTRACT, prepare_case
    folder = out/name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"manifest.json"
    environment = folder/"probe-latest.json"
    snapshot(environment)
    env, machine = preflight(environment)
    subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "diagnostics/test_precision_model.py::test_gpu_matvec_irregular_and_batch"], check=True)
    spec = dict(schema=SCHEMA, session=name, control=CONTROL, contract=PRECISION_CONTRACT,
                machine=machine, control_sha256=control_hash(), seed=SEED,
                fixtures=FIXTURES, trials=MIN_REPEATS, warmups=WARMUPS)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["spec"] != spec or manifest["status"] == "source changed":
            raise ValueError("Study folder contains a different acquisition. Use a fresh output folder.")
    else:
        manifest = dict(spec=spec, started=now(), environment=env, attempts=[])
    manifest["attempts"].append(dict(time=now(), environment=env))
    manifest["status"] = "running"
    put_json(path, manifest)
    invalid = []
    try:
        for case in STACK:
            if (folder/(case.name+".proof.json")).exists():
                read_control_case(folder, case, spec)
                continue
            print(f"Session {name}, {case.name}", flush=True)
            if not measure(folder, case, spec, prepare_case):
                invalid.append(case.name)
        if source_hash() != machine["sources_sha256"] or control_hash() != spec["control_sha256"]:
            manifest["status"] = "source changed"
            raise ValueError("Control or shared sources changed during collection")
        manifest["status"] = "incomplete" if invalid else "complete"
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        manifest["updated"] = now()
        put_json(path, manifest)
    if invalid:
        raise ValueError("Control parity failed in " + ", ".join(invalid) + f". See {folder}/*.failures.json.")


def load_verified(out):
    from diagnostics.precision_model import CONTROL
    runs = {}
    cohort = None
    for name in SESSIONS:
        folder = out/name
        manifest = json.loads((folder/"manifest.json").read_text())
        spec = manifest["spec"]
        if (manifest["status"] != "complete" or set(spec) != SPEC_FIELDS or spec["schema"] != SCHEMA
                or spec["session"] != name or spec["control"] != CONTROL or spec["seed"] != SEED
                or spec["fixtures"] != FIXTURES or spec["trials"] != MIN_REPEATS or spec["warmups"] != WARMUPS
                or manifest["environment"]["gpu_probe"]["passed"] is not True):
            raise ValueError(f"Missing a complete independent control session at {folder}")
        comparable = {k: v for k, v in spec.items() if k != "session"}
        if cohort is not None and comparable != cohort:
            raise ValueError("Control sessions differ in source, precision, device or protocol")
        cohort = comparable
        runs[name] = {case.name: read_control_case(folder, case, spec) for case in STACK}
    return runs, cohort


def report(out):
    runs, cohort = load_verified(out)
    rows = metrics(runs, STACK)
    index = {(row["case"], row["method"]): row for row in rows}
    lines = ["# Connected decoder with FP32 expert arithmetic", "", cohort["contract"], "",
             "Median synchronized wall time in µs. Brackets give 95% paired bootstrap intervals within the two sessions.", "",
             "| Visible tokens | " + " | ".join(LABELS[m] for m in SCHEDULES) + " |", "|---:|---:|---:|---:|---:|"]
    for case in STACK:
        cells = [index[case.name, method] for method in SCHEDULES]
        lines.append(f"| {case.length:,} | " + " | ".join(
            f"{r['wall_us']:,.1f} [{r['lo_us']:,.1f}, {r['hi_us']:,.1f}]" for r in cells) + " |")
    lines += ["", "Two processes, two fixed seeds, 100 paired timing blocks per process and length. "
              "Every schedule passed the original 15-observable checks against this control’s CPU equations before and after timing.",
              "", f"Shared benchmark SHA-256 `{cohort['machine']['sources_sha256']}`.",
              f"Control source SHA-256 `{cohort['control_sha256']}`."]
    tmp = out/"measurements.csv.tmp"
    for row in rows:
        row["control"] = cohort["control"]
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out/"measurements.csv")
    tmp = out/"report.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out/"report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", choices=SESSIONS, help=argparse.SUPPRESS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.session:
        session_run(out, args.session)
        return 0
    if args.report_only:
        report(out)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    lock_path = REPO/"results/.study.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another GPU study owns this checkout.", file=sys.stderr)
            return 75
        for name in SESSIONS:
            subprocess.run([sys.executable, "-m", "diagnostics.precision_study", "--out", str(out),
                            "--session", name], check=True, pass_fds=(lock.fileno(),))
        report(out)
    print(out/"report.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
