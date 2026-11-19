"""Registered cases, timing sample schema, and numeric acceptance rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np


SCHEDULES = ("compiled", "staged", "hybrid", "single")
LENGTHS = (64, 256, 512, 1024, 2048, 4096)
MIN_REPEATS = 100
FIXTURES = 2
REPO = Path(__file__).resolve().parents[1]
SCHEMA = 1
FIELDS = ("session", "case", "fixture", "block", "order", "method",
          "wall_us", "native_gpu_us", "encoded_dispatches")


@dataclass(frozen=True)
class Case:
    scope: str
    batch: int
    length: int

    @property
    def name(self):
        return f"{self.scope}-b{self.batch}-l{self.length}"

    @property
    def methods(self):
        if self.scope == "attention" and self.batch == 1 and self.length in (64, 4096):
            return SCHEDULES + ("split",)
        return SCHEDULES


PILOT = [Case("attention", 1, n) for n in (64, 4096)]
ATTENTION = [Case("attention", 1, n) for n in LENGTHS] + [Case("attention", 8, 4096)]
STACK = [Case("stack", 1, n) for n in (64, 4096)]
SUITES = {"pilot": PILOT, "attention": ATTENTION, "stack": STACK, "full": ATTENTION + STACK}
TABLE = PILOT + [Case("attention", 8, 4096)] + STACK


def put_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_hash():
    """Identify all benchmark, shader, adapter, and oracle source, including uncommitted edits."""
    digest = hashlib.sha256()
    for folder in ("bench", "reference", "native", "mlx_backend", "decoder"):
        for p in sorted((REPO / folder).rglob("*")):
            if not p.is_file() or p.suffix not in (".py", ".mm", ".m", ".metal", ".h", ".cpp", ".c"):
                continue
            digest.update(str(p.relative_to(REPO)).encode() + b"\0" + p.read_bytes() + b"\0")
    return digest.hexdigest()


def finite_time(value, label, *, allow_zero=False):
    x = float(value)
    if not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        raise ValueError(f"Invalid {label}: {value!r}")
    return x


def compare_array(name, expected, measured, atol=2e-3, rtol=1e-2):
    """Vary equality by observable; compare floats in FP32, preserving integer route IDs."""
    a = np.asarray(expected)
    b = np.asarray(measured)
    if a.shape != b.shape:
        raise ValueError(f"{name} shape {a.shape} differs from {b.shape}")
    if a.dtype.kind in "uib" or (atol == 0 and rtol == 0):
        np.testing.assert_array_equal(a, b, err_msg=name)
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{name} contains nonfinite values")
    a, b = a.astype(np.float32), b.astype(np.float32)
    np.testing.assert_allclose(b, a, rtol=rtol, atol=atol, err_msg=name, equal_nan=False)
    return float(np.max(np.abs(a-b), initial=0.0))


def compare_attention(expected, measured, original, position):
    output, state = expected
    actual_output, actual_state = measured
    if state.keys() != actual_state.keys():
        raise ValueError(f"state keys differ: {state.keys()} vs {actual_state.keys()}")
    errors = {"output": compare_array("output", output, actual_output)}
    for key, arr in state.items():
        other = np.asarray(actual_state[key])
        if key in ("cache_k", "cache_v"):
            if arr.shape != other.shape:
                raise ValueError(f"{key} shape changed")
            slot = position % arr.shape[2]
            errors[key] = compare_array(key, arr[:, :, slot], other[:, :, slot], 2e-3, 2e-3)
            for region in (slice(None, slot), slice(slot + 1, None)):
                compare_array(key + " untouched", original[key][:, :, region],
                              other[:, :, region], 0, 0)
        else:
            errors[key] = compare_array(key, arr, other, 0, 0)
    return errors


def stack_regions(name, expected, actual, position):
    """Yield the exact slices and tolerances used to certify one stack output."""
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape:
        raise ValueError(f"{name} shape changed: {a.shape} vs {b.shape}")
    if name.endswith(("cache_k", "cache_v")):
        slot = position % a.shape[2]
        for label, region in (("before slot", slice(None, slot)), ("after slot", slice(slot+1, None))):
            yield label, a[:, :, region], b[:, :, region], 0.0, 0.0
        yield "written slot", a[:, :, slot], b[:, :, slot], 2e-3, 2e-3
    elif a.dtype.kind in "uib":
        yield "all", a, b, 0.0, 0.0
    else:
        yield "all", a, b, 2e-3, 1e-2


def equal_stack(expected, actual, position):
    if not isinstance(expected, dict) or expected.keys() != actual.keys():
        raise ValueError("Stack observable dictionaries differ")
    if not expected:
        raise ValueError("Empty stack observable")
    errors = {}
    for k, v in expected.items():
        for label, reference, measured, atol, rtol in stack_regions(k, v, actual[k], position):
            errors[k] = compare_array(f"{k} ({label})", reference, measured, atol, rtol)
    return errors
