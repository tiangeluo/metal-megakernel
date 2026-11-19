"""Reproducible FP16 projected inputs, synthetic prefix snapshots and weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .attention import empty_state, step


def _normal(rng, shape, sigma=1.0):
    return (rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma)).astype(np.float16)


def make_token(rng, batch, q_heads, kv_heads, dim, rel_dim=16):
    return {
        "q": _normal(rng, (batch, q_heads, dim)),
        "k": _normal(rng, (batch, kv_heads, dim)),
        "v": _normal(rng, (batch, kv_heads, dim)),
        "r": _normal(rng, (batch, q_heads, rel_dim)),
        "gate": _normal(rng, (batch, q_heads, dim)),
        "sink": _normal(rng, (batch, q_heads)),
        "dk": _normal(rng, (batch, kv_heads, 4), 0.35),
        "dv": _normal(rng, (batch, kv_heads, 4), 0.35),
    }


def make_weights(rng, q_heads, kv_heads, dim, rel_dim=16, extent=1024):
    return {
        "wk": _normal(rng, (4, kv_heads, dim), 0.4),
        "wv": _normal(rng, (4, kv_heads, dim), 0.4),
        "q_scale": (_normal(rng, (q_heads, dim), 0.25) + np.float16(1)),
        "r_scale": (_normal(rng, (q_heads, rel_dim), 0.25) + np.float16(1)),
        "rel": _normal(rng, (extent, rel_dim), rel_dim ** -0.5),
    }


def make_case(batch=1, q_heads=48, kv_heads=8, dim=128, length=64, seed=0,
              window=0, dynamic=True, rel_dim=16, extent=1024):
    """Length includes the current token. Build a prefilled operator snapshot.

    Historical K/V are seeded valid post-clip/normalization vectors. Tests of actual
    consecutive tokens use zero caches and real convolution histories instead.
    """
    if min(length, extent, rel_dim, q_heads, kv_heads) < 1 or q_heads % kv_heads:
        raise ValueError("positive dimensions and divisible grouped query heads required")
    if window < 0:
        raise ValueError("window must be nonnegative")
    rng = np.random.default_rng(seed)
    token = make_token(rng, batch, q_heads, kv_heads, dim, rel_dim)
    weights = make_weights(rng, q_heads, kv_heads, dim, rel_dim, extent)
    capacity = min(window, length) if window else length
    state = empty_state(batch, kv_heads, capacity, dim)
    history = min(3, length - 1)
    for key in ("hist_k", "hist_v"):
        state[key][:, :, :history] = _normal(rng, (batch, kv_heads, history, dim))
    first = max(0, length - 1 - capacity)
    slots = np.arange(first, length - 1) % capacity
    for key in ("cache_k", "cache_v"):
        values = _normal(rng, (batch, kv_heads, len(slots), dim)).astype(np.float32)
        mean_sq = np.mean(values * values, axis=-1, keepdims=True)
        if key == "cache_k":
            values *= np.minimum(1.0, 22.627416997969522 / (np.sqrt(mean_sq * dim) + 1e-6))
        else:
            values /= np.sqrt(mean_sq + 1e-6)
        state[key][:, :, slots] = values.astype(np.float16)
    return token, weights, state


def write_case(path, **params):
    """Write portable individual .npy arrays plus shapes and unambiguous cast data."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    token, weights, state = make_case(**params)
    expected = {k: v.copy() for k, v in state.items()}
    out = step(token, weights, expected, position=params["length"] - 1,
               window=params["window"], dynamic=params["dynamic"])
    bundle = {**token, **weights, **state, "expected_output": out,
              **{"expected_" + k: v for k, v in expected.items()}}
    for name, array in bundle.items():
        np.save(path / (name + ".npy"), array, allow_pickle=False)
    meta = {"parameters": params, "arrays": {k: list(v.shape) for k, v in bundle.items()}}
    (path / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    for name, default in (("length", 64), ("batch", 1), ("q_heads", 48), ("kv_heads", 8),
                          ("dim", 128), ("seed", 0), ("window", 0), ("rel_dim", 16),
                          ("extent", 1024)):
        parser.add_argument("--" + name.replace("_","-"), dest=name, type=int, default=default)
    parser.add_argument("--static", action="store_false", dest="dynamic")
    args = vars(parser.parse_args())
    write_case(args.pop("output"), **args)
