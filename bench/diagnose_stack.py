"""Replay one connected decoder fixture and compare every observable, outside timing."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

from .protocol import Case, put_json, stack_regions


def numbers(reference, actual, atol, rtol):
    """Report a region's worst violating coordinate, indexed within this region."""
    exact = atol == 0 and rtol == 0
    if exact:
        bad = reference != actual
        # assert_array_equal treats NaNs in identical locations as equal.
        if reference.dtype.kind == actual.dtype.kind == "f":
            bad &= ~(np.isnan(reference) & np.isnan(actual))
    a = reference.astype(np.float32)
    b = actual.astype(np.float32)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(np.subtract(a, b, where=finite, out=np.zeros_like(a)))
    if not exact:
        bad = ~finite | (diff > atol + rtol * np.abs(a))
    result = {
        "size": int(a.size), "violations": int(np.count_nonzero(bad)),
        "atol": atol, "rtol": rtol, "max_abs_finite": float(diff.max(initial=0)),
    }
    if np.any(bad):
        # Prefer a nonfinite mismatch; otherwise show the largest violating error.
        index = np.unravel_index(np.argmax(np.where(bad, np.where(finite, diff, np.inf), -1)), a.shape)
        def value(x):
            v = x[index].item()
            return v if np.isfinite(v) else str(v)
        result["worst_violation"] = {
            "index": list(map(int, index)), "expected": value(reference), "actual": value(actual),
        }
    return result


def compare_outputs(reference, actual, position):
    details = {}
    failed = []
    for key, expected in reference.items():
        if key not in actual:
            details[key] = {"error": "missing observable"}
            failed.append(key)
            continue
        try:
            regions = {
                label: numbers(a, b, atol, rtol)
                for label, a, b, atol, rtol in stack_regions(key, expected, actual[key], position)
            }
        except ValueError as exc:
            details[key] = {"error": str(exc)}
            failed.append(key)
            continue
        details[key] = {"regions": regions}
        if any(row["violations"] for row in regions.values()):
            failed.append(key)
        if key.endswith("route_ids"):
            details[key].update(expected=expected.tolist(), actual=actual[key].tolist())
    extra = sorted(actual.keys() - reference.keys())
    return {"passed": bool(reference) and not (failed or extra), "failed_observables": failed,
            "extra_observables": extra, "observables": details}


def read_method(case, seed, method):
    from decoder.benchmark import prepare_case
    with closing(prepare_case(method, case.length, batch=case.batch, seed=seed)) as runner:
        runner.run()
        return {key: np.array(arr, copy=True) for key, arr in runner.output_numpy().items()}


def diagnose(case, seed, machine, output, *, original=None):
    """Write a replay report. Keep each runner's lifetime within a single readback."""
    report = {"case": case.name, "seed": seed, "machine": machine, "passed": False,
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "comparisons": {}, "method_errors": {}}
    position = case.length - 1
    cpu = read_method(case, seed, "cpu")
    if original is not None:
        context, readback = original
        report["original_failure"] = {**context, "vs_cpu": compare_outputs(cpu, readback, position)}
    compiled = None
    failed_methods = []
    for method in case.methods:  # The compiled baseline runs first.
        try:
            measured = read_method(case, seed, method)
            vs_cpu = compare_outputs(cpu, measured, position)
            if not vs_cpu["passed"]:
                failed_methods.append(method)
            comparisons = {"vs_cpu": vs_cpu}
            if method == "compiled":
                compiled = measured
            elif compiled is not None:
                comparisons["vs_compiled"] = compare_outputs(compiled, measured, position)
            report["comparisons"][method] = comparisons
            print(f"{case.name} seed {seed} {method}: "
                  + ("passed" if vs_cpu["passed"] else f"failed {vs_cpu['failed_observables']}"), flush=True)
        except (ValueError, RuntimeError, AssertionError, OSError, ImportError) as exc:
            report["method_errors"][method] = f"{type(exc).__name__}: {exc}"
    report["status"] = "complete"
    report["passed"] = not (failed_methods or report["method_errors"])
    output.parent.mkdir(parents=True, exist_ok=True)
    put_json(output, report)
    if failed_methods:
        # Preserve the complete observable comparison before the extra instrumented replay.
        method = failed_methods[0]
        try:
            from diagnostics.stack_stages import isolate
            report["stage_isolation"] = {"method": method, "findings": isolate(case, seed, method)}
        except (ValueError, RuntimeError, AssertionError, OSError, ImportError) as exc:
            report["stage_isolation"] = {"method": method, "error": f"{type(exc).__name__}: {exc}"}
        put_json(output, report)
    return report


def main():
    from .run import preflight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path("results/environment.json"))
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.length, args.batch) < 1 or args.seed < 0:
        parser.error("length and batch must be positive; seed must be nonnegative")
    _, machine = preflight(args.environment)
    report = diagnose(Case("stack", args.batch, args.length), args.seed, machine, args.out)
    print(args.out)
    return int(not report["passed"])


if __name__ == "__main__":
    sys.exit(main())
