"""Run the Metal schedules against the reference.attention buffer contract.

From the repository root: python -m native.runner --length 64 --samples 20
The CLI reports timings only for schedules whose outputs and cache state pass parity.
"""

from __future__ import annotations

import argparse
import ctypes as ct
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

SOURCE = Path(__file__).resolve().parent
METHODS = {"staged": 0, "single": 1, "split": 2}
TOKEN_KEYS = ("q", "k", "v", "r", "sink", "gate", "dk", "dv")
STATE_KEYS = ("cache_k", "cache_v", "hist_k", "hist_v")
WEIGHT_KEYS = ("wk", "wv", "rel")
BUFFER_NAMES = (
    None,
    *TOKEN_KEYS,
    *WEIGHT_KEYS,
    *STATE_KEYS,
    "prepared_q",
    "prepared_r",
    "ungated",
    "out",
    "partials",
    "q_scale",
    "r_scale",
)
SLOTS = {name: i for i, name in enumerate(BUFFER_NAMES) if name is not None}


class Config(ct.Structure):
    _fields_ = [
        (name, ct.c_uint32)
        for name in (
            "batch",
            "queries",
            "kv",
            "dim",
            "relative_dim",
            "extent",
            "capacity",
            "position",
            "window",
            "dynamic",
            "tile",
            "chunks",
        )
    ]


class Metrics(ct.Structure):
    _fields_ = [("wall_us", ct.c_double), ("gpu_us", ct.c_double), ("dispatches", ct.c_uint32)]


def compile_bridge(destination):
    if sys.platform != "darwin":
        raise RuntimeError("The native backend requires macOS and Metal")
    subprocess.run(
        [
            "xcrun",
            "clang++",
            "-O3",
            "-std=c++17",
            "-fobjc-arc",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-dynamiclib",
            str(SOURCE / "bridge.mm"),
            "-o",
            str(destination),
        ],
        check=True,
    )


@lru_cache(maxsize=1)
def _library():
    # dyld keeps a mapped image alive through the CDLL handle after its temporary
    # directory is removed. No compiler output goes in the source checkout.
    with tempfile.TemporaryDirectory(prefix="mac-megakernel-") as directory:
        target = Path(directory) / "decode.dylib"
        compile_bridge(target)
        library = ct.CDLL(str(target))
    library.mm_error.argtypes = []
    library.mm_error.restype = ct.c_char_p
    library.mm_open.argtypes = [
        ct.c_char_p,
        ct.POINTER(Config),
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_size_t),
    ]
    library.mm_open.restype = ct.c_void_p
    library.mm_run.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(Metrics)]
    library.mm_run.restype = ct.c_int
    library.mm_transfer.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_size_t, ct.c_bool]
    library.mm_transfer.restype = ct.c_int
    library.mm_bind.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_size_t]
    library.mm_bind.restype = ct.c_int
    library.mm_position.argtypes = [ct.c_void_p, ct.c_uint32]
    library.mm_position.restype = ct.c_int
    library.mm_close.argtypes = [ct.c_void_p]
    library.mm_close.restype = None
    return library


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float16:
        raise ValueError(f"{name}: expected FP16 {shape}, received {a.dtype} {a.shape}")
    return np.ascontiguousarray(a)


class NativeRunner:
    """Own device buffers and pipelines for one fixed geometry; operations are synchronous."""

    def __init__(self, token, weights, state, *, position, window=0, dynamic=True, tile=32):
        self.handle = None
        b, q, d = token["q"].shape
        h = token["k"].shape[1]
        e, r = weights["rel"].shape
        c = state["cache_k"].shape[2]
        if min(b, q, h, d, r, e, c, tile) <= 0 or q % h or d > 128 or r > 32 or q // h > 32:
            raise ValueError("Expected a valid GQA geometry, dim <= 128 and relative_dim <= 32")
        if not 0 <= position < 2**32 - 1 or window < 0:
            raise ValueError("Invalid position or window")
        length = min(position + 1, window) if window else position + 1
        if length > c:
            raise ValueError("Visible context exceeds the cache capacity")
        chunks = (length + tile - 1) // tile
        max_chunks = ((min(window, c) if window else c) + tile - 1) // tile
        self.config = Config(b, q, h, d, r, e, c, position, window, int(dynamic), tile, chunks)
        self.shapes = {
            "q": (b, q, d),
            "k": (b, h, d),
            "v": (b, h, d),
            "r": (b, q, r),
            "sink": (b, q),
            "gate": (b, q, d),
            "dk": (b, h, 4),
            "dv": (b, h, 4),
            "wk": (4, h, d),
            "wv": (4, h, d),
            "rel": (e, r),
            "cache_k": (b, h, c, d),
            "cache_v": (b, h, c, d),
            "hist_k": (b, h, 3, d),
            "hist_v": (b, h, 3, d),
            "prepared_q": (b, q, d),
            "prepared_r": (b, q, r),
            "ungated": (b, q, d),
            "out": (b, q, d),
            "partials": (b, q, max_chunks, d + 2),
            "q_scale": (q, d),
            "r_scale": (q, r),
        }
        # Device buffers are mutable; store independent snapshots for fixed-length trials.
        self.initial = {key: _array(state[key], self.shapes[key], key).copy() for key in STATE_KEYS}
        self.initial_position = position
        inputs = dict(weights)
        inputs.update(token)
        inputs.update(self.initial)
        arrays = {}
        data = (ct.c_void_p * len(BUFFER_NAMES))()
        sizes = (ct.c_size_t * len(BUFFER_NAMES))()
        for name, idx in SLOTS.items():
            shape = self.shapes[name]
            count = int(np.prod(shape, dtype=np.int64))
            if count >= 2**32:
                raise ValueError(f"{name} exceeds Metal uint indexing range")
            sizes[idx] = count * (4 if name == "partials" else 2)
            if name in inputs:
                a = _array(inputs[name], shape, name)
                arrays[name] = a
                data[idx] = a.ctypes.data
            elif name not in ("prepared_q", "prepared_r", "ungated", "out", "partials"):
                raise KeyError(name)
        self.lib = _library()
        self.handle = self.lib.mm_open(
            (SOURCE / "decode.metal").read_bytes(), ct.byref(self.config), data, sizes
        )
        self._check(self.handle)

    def _check(self, result):
        if not result:
            raise RuntimeError(self.lib.mm_error().decode("utf-8", errors="replace"))

    def _require_open(self):
        if not self.handle:
            raise RuntimeError("NativeRunner is closed")

    def _copy(self, key, array, write):
        self._require_open()
        self._check(
            self.lib.mm_transfer(self.handle, SLOTS[key], array.ctypes.data, array.nbytes, write)
        )

    def set_position(self, position):
        self._require_open()
        if not 0 <= position < 2**32 - 1:
            raise ValueError("Position is out of range")
        self._check(self.lib.mm_position(self.handle, position))
        self.config.position = position
        length = min(position + 1, self.config.window) if self.config.window else position + 1
        self.config.chunks = (length + self.config.tile - 1) // self.config.tile

    def set_token(self, token):
        """Copy only new projected token inputs into preallocated shared buffers."""
        for name in TOKEN_KEYS:
            self._copy(name, _array(token[name], self.shapes[name], name), True)

    def bind_external(self, name, buffer_address, byte_offset=0):
        """Bind a verified, live kDLMetal MTLBuffer, or None to revert to owned storage.

        The exporting array must be contiguous, with the expected FP16 shape and
        sufficient alignment. Synchronize its producing stream before run(). Retain
        the DLPack capsule until the consumer's command buffer has completed.
        """
        self._require_open()
        if name not in (*TOKEN_KEYS, "out") or byte_offset < 0:
            raise ValueError("Only projected token inputs and gated output can be rebound")
        self._check(self.lib.mm_bind(self.handle, SLOTS[name], buffer_address, byte_offset))

    def reset(self):
        for name, initial in self.initial.items():
            self._copy(name, initial, True)
        self.set_position(self.initial_position)

    def run(self, method):
        self._require_open()
        metrics = Metrics()
        self._check(self.lib.mm_run(self.handle, METHODS[method], ct.byref(metrics)))
        return {
            "wall_us": metrics.wall_us,
            "gpu_us": None if metrics.gpu_us < 0 else metrics.gpu_us,
            "dispatches": metrics.dispatches,
        }

    def _read(self, name):
        array = np.empty(self.shapes[name], dtype=np.float16)
        self._copy(name, array, False)
        return array

    def gated_numpy(self):
        return self._read("out")

    def outputs(self):
        return self.gated_numpy(), {name: self._read(name) for name in STATE_KEYS}

    def close(self):
        if self.handle:
            self.lib.mm_close(self.handle)
            self.handle = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--length", type=int, default=64, help="Visible sequence length including current token"
    )
    parser.add_argument(
        "--samples", type=int, default=0, help="Synchronized diagnostic samples after parity"
    )
    args = parser.parse_args()
    if args.build_only:
        _library()
        print("Objective-C++ Metal bridge compiled and loaded successfully.")
        return
    if args.samples < 0 or args.length < 1:
        parser.error("length must be positive and samples nonnegative")
    from reference.fixtures import make_case
    from reference.attention import step

    token, weights, state = make_case(length=args.length)
    before = {key: v.copy() for key, v in state.items()}
    expected = step(token, weights, state, position=args.length - 1)
    with NativeRunner(token, weights, before, position=args.length - 1) as runner:
        for method in METHODS:
            runner.reset()
            runner.run(method)
            out, actual = runner.outputs()
            np.testing.assert_allclose(out, expected, rtol=1e-2, atol=2e-3)
            for name in STATE_KEYS:
                if name.startswith("hist"):
                    np.testing.assert_array_equal(actual[name], state[name])
                else:
                    slot = (args.length - 1) % state[name].shape[2]
                    np.testing.assert_allclose(
                        actual[name][:, :, slot], state[name][:, :, slot], rtol=2e-3, atol=2e-3
                    )
                    np.testing.assert_array_equal(
                        actual[name][:, :, :slot], before[name][:, :, :slot]
                    )
                    np.testing.assert_array_equal(
                        actual[name][:, :, slot + 1 :], before[name][:, :, slot + 1 :]
                    )
        print(json.dumps({"validation": "passed", "methods": list(METHODS), "length": args.length}))
        for i in range(args.samples):
            for j in range(len(METHODS)):
                method = list(METHODS)[(j + i) % len(METHODS)]
                runner.reset()
                t0 = time.perf_counter_ns()
                metrics = runner.run(method)
                metrics["python_wall_us"] = (time.perf_counter_ns() - t0) / 1e3
                print(json.dumps(dict(method=method, trial=i, **metrics)))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
