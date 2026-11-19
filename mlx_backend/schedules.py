"""MLX compiled and Metal-preparation/MLX-attention decode backends."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def require_gpu():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("MLX is missing. Install the pinned MLX release on Apple silicon.") from exc
    if not mx.metal.is_available():
        raise RuntimeError("No Metal device is available to MLX in this process.")
    mx.set_default_device(mx.gpu)
    return mx


def _clip(mx, x, limit):
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x * mx.minimum(1.0, limit / (norm + 1e-6))


def _conv(mx, raw, history, taps, dynamic_taps, dynamic):
    # [B,H,lag,D]. The original input is also the residual branch.
    samples = mx.concatenate([raw[:, :, None, :], history], axis=2)
    tap = taps.transpose(1, 0, 2)[None].astype(mx.float32)
    if dynamic:
        tap = tap + dynamic_taps[..., None].astype(mx.float32)
    convolved = raw.astype(mx.float32) + mx.sum(samples.astype(mx.float32) * tap, axis=2)
    return (convolved / (3.0 if dynamic else math.sqrt(5.0))).astype(mx.float16), samples[:, :, :3, :]


def _prepare_mlx(mx, token, weights, state, dynamic, tau):
    q = token["q"].astype(mx.float32) * weights["q_scale"].astype(mx.float32)
    r = token["r"].astype(mx.float32) * weights["r_scale"].astype(mx.float32)
    q = _clip(mx, q, math.sqrt(512.0)).astype(mx.float16)
    r = _clip(mx, r, 8.0).astype(mx.float16)
    q = (q.astype(mx.float32) * tau).astype(mx.float16)
    r = (r.astype(mx.float32) * tau).astype(mx.float16)
    k, hk = _conv(mx, token["k"], state["hist_k"], weights["wk"], token["dk"], dynamic)
    v, hv = _conv(mx, token["v"], state["hist_v"], weights["wv"], token["dv"], dynamic)
    k = _clip(mx, k.astype(mx.float32), math.sqrt(512.0)).astype(mx.float16)
    vf = v.astype(mx.float32)
    v = (vf * mx.rsqrt(mx.mean(vf * vf, axis=-1, keepdims=True) + 1e-6)).astype(mx.float16)
    return q, k, v, r, hk, hv


@lru_cache(maxsize=1)
def _metal_preparer():
    mx = require_gpu()
    return mx.fast.metal_kernel(
        name="inkling_decode_prepare",
        input_names=["q", "k", "v", "r", "dk", "dv", "qscale", "rscale",
                     "wk", "wv", "histk", "histv", "tau"],
        output_names=["qo", "ko", "vo", "ro", "hko", "hvo"],
        source=Path(__file__).with_name("prepare.metal").read_text(),
    )


def _prepare_metal(mx, token, weights, state, dynamic, tau):
    q, k, v, r = (token[name] for name in ("q", "k", "v", "r"))
    b, qh, dim = q.shape
    _, kh, _ = k.shape
    hk, hv = state["hist_k"], state["hist_v"]
    return _metal_preparer()(
        inputs=[q, k, v, r, token["dk"], token["dv"], weights["q_scale"],
                weights["r_scale"], weights["wk"], weights["wv"], hk, hv, tau],
        template=[("QH", qh), ("KH", kh), ("DIM", dim), ("RDIM", r.shape[-1]), ("DYN", dynamic)],
        grid=(32 * b * qh, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[q.shape, k.shape, v.shape, r.shape, hk.shape, hv.shape],
        output_dtypes=[mx.float16] * 6,
    )


def _attend(mx, token, weights, state, prepared, position, window):
    q, k, v, r, hist_k, hist_v = prepared
    b, qh, dim = q.shape
    kv_heads = k.shape[1]
    capacity = state["cache_k"].shape[2]
    visible = min(position + 1, capacity)
    slot = position % capacity
    # Keep physical order in a ring. SDPA is invariant to a shared permutation
    # of K, V and the bias, avoiding a sort/roll/copy after wrapping.
    offset = mx.array([slot], dtype=mx.int32)
    cache_k = mx.slice_update(state["cache_k"], k[:, :, None, :], offset, (2,))
    cache_v = mx.slice_update(state["cache_v"], v[:, :, None, :], offset, (2,))
    distance = (slot - mx.arange(visible, dtype=mx.int32) + capacity) % capacity
    extent = weights["rel"].shape[0]
    # Compute only learned distances which can be seen. Position-dependent R,
    # table product, mask construction and any physical-layout copies are timed.
    count = min(extent, visible)
    rel = mx.matmul(r.astype(mx.float32), weights["rel"][:count].T.astype(mx.float32))
    bias = mx.take(rel, mx.minimum(distance, count - 1), axis=-1).astype(mx.float16)
    bias = mx.where(distance < extent, bias, mx.zeros_like(bias))
    if window and visible > window:
        bias = mx.where(distance < window, bias, mx.array(-float("inf"), dtype=mx.float16))
    # Native sinks takes [heads]. Folding B into heads retains contiguous
    # GQA groups, one distinct sink per (batch, head), and no cross-batch keys.
    q = q.reshape(1, b * qh, 1, dim)
    keys = cache_k[:, :, :visible, :].reshape(1, b * kv_heads, visible, dim)
    values = cache_v[:, :, :visible, :].reshape(1, b * kv_heads, visible, dim)
    mask = bias.reshape(1, b * qh, 1, visible)
    sink = token["sink"].reshape(b * qh)
    output = mx.fast.scaled_dot_product_attention(q, keys, values, scale=dim ** -0.5, mask=mask, sinks=sink)
    output = output.reshape(b, qh, dim).astype(mx.float16)
    gated = (output.astype(mx.float32) * mx.sigmoid(token["gate"].astype(mx.float32))).astype(mx.float16)
    return gated, {"cache_k":cache_k, "cache_v":cache_v, "hist_k":hist_k, "hist_v":hist_v}


@lru_cache(maxsize=48)
def build_step(*, method, position, window=0, dynamic=True):
    """Return a reusable one-token callable over MLX dictionaries; state is functional."""
    if method not in ("compiled", "hybrid"):
        raise ValueError("method must be 'compiled' or 'hybrid'")
    if position < 0 or window < 0:
        raise ValueError("position and window must be nonnegative")
    mx = require_gpu()
    factor = 1.0 if not dynamic else 1.0 + 0.1 * math.log(max(1.0, (position + 1) / 8192.0))
    tau = mx.array([factor], dtype=mx.float32)
    def check(state):
        capacity = state["cache_k"].shape[2]
        if capacity < min(position + 1, window or position + 1):
            raise ValueError("cache cannot hold all visible tokens")
    if method == "compiled":
        @mx.compile
        def compiled(token, weights, state):
            prepared = _prepare_mlx(mx, token, weights, state, dynamic, tau)
            return _attend(mx, token, weights, state, prepared, position, window)
        def run(token, weights, state):
            check(state)
            return compiled(token, weights, state)
        return run
    # Keep the custom dispatch outside compile. The remaining SDPA/append/gate
    # region is still compiled, retaining MLX buffer donation and fast GQA.
    @mx.compile
    def tail(token, weights, state, prepared):
        return _attend(mx, token, weights, state, prepared, position, window)
    def run(token, weights, state):
        check(state)
        prepared = tuple(_prepare_metal(mx, token, weights, state, dynamic, tau))
        return tail(token, weights, state, prepared)
    return run


class _Case:
    """Prepared benchmark adapter. Never transfers to NumPy inside run()."""
    def __init__(self, token, weights, state, step):
        self.mx = require_gpu()
        self.token = {k:self.mx.array(v) for k, v in token.items()}
        self.weights = {k:self.mx.array(v) for k, v in weights.items()}
        self.host_state = {k:np.array(v, copy=True) for k, v in state.items()}
        self.step = step
        self.mx.eval(*self.token.values(), *self.weights.values())
        self.fresh()

    def fresh(self):
        self.output = None
        self.state = {k:self.mx.array(v) for k, v in self.host_state.items()}
        self.mx.eval(*self.state.values())

    def run(self):
        self.output, self.state = self.step(self.token, self.weights, self.state)
        self.mx.eval(self.output, *self.state.values())

    def readback(self):
        if self.output is None:
            raise RuntimeError("Call run() before reading back the output")
        return np.array(self.output), {k:np.array(v) for k, v in self.state.items()}


def prepare_case(token, weights, state, *, position, window=0, dynamic=True, method="compiled"):
    return _Case(token, weights, state, build_step(method=method, position=position, window=window, dynamic=dynamic))
