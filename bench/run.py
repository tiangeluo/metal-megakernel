"""Run synchronized, paired decode trials on a verified Metal device."""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import random
import re
import sys
import time

import numpy as np

from . import adapters
from .protocol import FIELDS, FIXTURES, MIN_REPEATS, SCHEMA, SUITES, finite_time, put_json, source_hash


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(environment):
    # The native probe establishes actual hardware access independently of MLX.
    observed = json.loads(environment.read_text())
    probe = observed.get("gpu_probe", {})
    if probe.get("passed") is not True or not isinstance(probe.get("device"), str) or not probe["device"]:
        raise RuntimeError("Environment report has no successful native Metal compute/readback probe")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This experiment requires native arm64 macOS")
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError("MLX cannot enumerate a Metal device")
    mx.set_default_device(mx.gpu)
    test = mx.arange(4) + 2
    mx.eval(test)
    if test.tolist() != [2, 3, 4, 5]:
        raise RuntimeError("MLX GPU readback failed")
    return observed, {
        "device": probe["device"],
        "arch": platform.machine(),
        "macos": platform.mac_ver()[0],
        "mlx": metadata.version("mlx"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "sources_sha256": source_hash(),
    }


def randomized_order(methods, rng):
    order = list(methods)
    rng.shuffle(order)
    return order


class ParityError(AssertionError):
    def __init__(self, case, seed, fixture, method, phase, actual, error):
        self.context = dict(case=case.name, seed=seed, fixture=fixture, method=method, phase=phase)
        self.readback = None
        if case.scope == "stack":
            # Own the failed readback before native or MLX scratch buffers are released.
            self.readback = {key: np.array(value, copy=True) for key, value in actual.items()}
        super().__init__(f"{case.name} {phase}, fixture {fixture}, seed {seed}, {method}: {error}")


def check_step(case, seed, fixture, method, phase, runner, check):
    runner.reset()
    runner.step()
    actual = runner.read()
    try:
        return check(actual)
    except (AssertionError, ValueError) as error:
        raise ParityError(case, seed, fixture, method, phase, actual, error) from error


def run_case(case, session, seed, trials, warmups):
    rows, parity = [], {}
    sequence = int.from_bytes(hashlib.sha256((session+"/"+case.name).encode()).digest()[:8],"big")
    rng = random.Random(sequence ^ seed)
    # Interleave distinct token states and routes; stack fixtures reuse the same expert banks.
    with ExitStack() as cleanup:
        prepared = []
        for fixture in range(FIXTURES):
            runners, check, close = adapters.prepare(case, seed + fixture)
            cleanup.callback(close)
            if set(runners) != set(case.methods):
                raise ValueError("Missing requested schedule")
            prepared.append((runners, check))
            parity[fixture] = {}
            for method in randomized_order(case.methods, rng):
                parity[fixture][method] = check_step(
                    case, seed + fixture, fixture, method, "before timing", runners[method], check
                )
        for _ in range(warmups):
            for runners, _check in prepared:
                for method in randomized_order(case.methods, rng):
                    runners[method].reset()
                    runners[method].step()
        for first in range(0,trials,FIXTURES):
            blocks=list(range(first,min(first+FIXTURES,trials)))
            rng.shuffle(blocks)
            for block in blocks:
                fixture = block % FIXTURES
                runners, _check = prepared[fixture]
                for order, method in enumerate(randomized_order(case.methods, rng)):
                    runner = runners[method]
                    runner.reset()
                    before = time.perf_counter_ns()
                    gpu, count = runner.step()
                    after = time.perf_counter_ns()
                    wall = finite_time((after-before)/1e3, "synchronized wall time")
                    if gpu is not None:
                        gpu = finite_time(gpu, "GPU time", allow_zero=True)
                    if count is not None and (not isinstance(count, int) or count <= 0):
                        raise ValueError("Invalid encoded dispatch count")
                    rows.append(dict(session=session,case=case.name,fixture=fixture,block=block,
                                     order=order,method=method,wall_us=wall,
                                     native_gpu_us=gpu,encoded_dispatches=count))
        # Detect broken reset semantics, accidental cache growth and late errors.
        for fixture, (runners, check) in enumerate(prepared):
            for method, runner in runners.items():
                check_step(case, seed + fixture, fixture, method, "after timing", runner, check)
    return rows, parity


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_case(directory, case, rows, parity):
    path = directory / (case.name+".csv")
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w",newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    put_json(path.with_suffix(".proof.json"), {
        "schema": SCHEMA, "case": case.name, "row_count": len(rows),
        "sha256": file_digest(path), "oracle_max_abs": parity, "rechecked_after_timing": True,
    })


def certified(directory, case):
    csv_path = directory / (case.name+".csv")
    proof = csv_path.with_suffix(".proof.json")
    if not csv_path.is_file() or not proof.is_file():
        return False
    data = json.loads(proof.read_text())
    return data.get("rechecked_after_timing") is True and data.get("sha256") == file_digest(csv_path)


def refresh_tracker(enabled):
    if not enabled:
        return
    try:
        from tracker.render import render
        render()
    except Exception as error:
        print(f"Could not refresh tracker: {error}", file=sys.stderr)


def measure(case, args, machine, folder):
    try:
        return run_case(case, args.session, args.seed, args.trials, args.warmups)
    except ParityError as exc:
        if exc.readback is None:
            raise
        context, readback, message = exc.context, exc.readback, str(exc)
    # Leaving the handler releases the traceback and the prepared benchmark runners.
    from .diagnose_stack import diagnose
    diagnostic = folder / f"{case.name}-fixture{context['fixture']}.diagnostic.json"
    try:
        diagnose(case, context["seed"], machine, diagnostic, original=(context, readback))
    except Exception as exc:
        raise AssertionError(f"{message}\nDiagnostic replay failed: {exc}") from exc
    raise AssertionError(f"{message}\nDiagnostic replay: {diagnostic}")


def execute(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.session):
        raise ValueError("Invalid session label")
    if args.trials < MIN_REPEATS or args.warmups < 1 or args.seed < 0:
        raise ValueError(f"At least {MIN_REPEATS} trials, one warmup, and a nonnegative seed are required")
    refresh_tracker(args.update_tracker)
    env, machine = preflight(args.environment)
    spec = dict(schema=SCHEMA,suite=args.suite,session=args.session,seed=args.seed,
                trials=args.trials,warmups=args.warmups,fixtures=FIXTURES,machine=machine)
    folder = args.out / args.session / args.suite
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another run owns this session") from e
        path = folder/"manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            if manifest["spec"] != spec:
                raise RuntimeError("Session exists with different inputs, code, or machine; use a new session label")
        else:
            manifest = {"spec": spec, "started": now(), "resumes": [], "environment": env}
        manifest["resumes"].append(now())
        manifest["status"] = "running"
        put_json(path, manifest)
        refresh_tracker(args.update_tracker)
        try:
            for case in SUITES[args.suite]:
                if certified(folder, case):
                    print("Verified existing", case.name, flush=True)
                    continue
                print("Measuring", case.name, flush=True)
                rows, parity = measure(case, args, machine, folder)
                save_case(folder, case, rows, parity)
                refresh_tracker(args.update_tracker)
            if source_hash() != machine["sources_sha256"]:
                raise RuntimeError("Benchmark source changed while timing; discard this session")
        except BaseException as exc:
            manifest.update(status="incomplete",error=f"{type(exc).__name__}: {exc}",updated=now())
            put_json(path, manifest)
            refresh_tracker(args.update_tracker)
            raise
        manifest.pop("error",None)
        manifest.update(status="complete",updated=now())
        put_json(path, manifest)
        refresh_tracker(args.update_tracker)
        print("Verified session:", folder)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",choices=SUITES,default="pilot")
    p.add_argument("--session",required=True,help="Independent session label. Reuse it to resume the same configuration.")
    p.add_argument("--out",type=Path,default=Path("results"))
    p.add_argument("--environment",type=Path,default=Path("results/environment.json"))
    p.add_argument("--trials",type=int,default=MIN_REPEATS)
    p.add_argument("--warmups",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261118)
    p.add_argument("--update-tracker",action="store_true",help="Refresh project HTML as cases finish.")
    args=p.parse_args()
    try:
        execute(args)
    except (OSError, ImportError, ValueError, AssertionError, RuntimeError) as e:
        p.exit(2, f"Benchmark did not complete: {e}\n")


if __name__ == "__main__":
    main()
