"""Cached, batch-vectorized NumPy reference for one projected decode step."""
from __future__ import annotations

import numpy as np

QK_LIMIT = 22.627416997969522
REL_LIMIT = 8.0
EPSILON = 1e-6


def clip(x, threshold):
    x = x.astype(np.float32)
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return (x * np.minimum(1.0, threshold / (norm + EPSILON))).astype(np.float16)


def shortconv(x, history, weights, taps=None):
    """Four taps in lag order. Raw history is read-only in this operation."""
    raw = x.astype(np.float32)
    total = raw.copy()
    for lag in range(4):
        gain = weights[lag].astype(np.float32)
        if taps is not None:
            gain = gain + taps[..., lag, None].astype(np.float32)
        sample = raw if lag == 0 else history[:, :, lag - 1].astype(np.float32)
        total += sample * gain
    divisor = np.float32(3.0 if taps is not None else np.sqrt(5.0))
    return (total / divisor).astype(np.float16)


def empty_state(batch, kv_heads, capacity, dim):
    if min(batch, kv_heads, capacity, dim) < 1:
        raise ValueError("state dimensions must be positive")
    return {
        key: np.zeros((batch, kv_heads, columns, dim), np.float16)
        for key, columns in (("cache_k", capacity), ("cache_v", capacity),
                             ("hist_k", 3), ("hist_v", 3))
    }


def step(token, weights, state, *, position, window=0, dynamic=True):
    """Return [batch,query_heads,head_dim] FP16 output and advance state in place."""
    batch, queries, dim = token["q"].shape
    heads = token["k"].shape[1]
    capacity = state["cache_k"].shape[2]
    if queries % heads or position < 0 or window < 0:
        raise ValueError("invalid head grouping, position or window")
    visible = min(position + 1, window) if window else position + 1
    if visible > capacity:
        raise ValueError("cache capacity is smaller than the active attention span")
    groups = queries // heads
    prepared = {}
    for key in ("k", "v"):
        taps = token["d" + key] if dynamic else None
        c = shortconv(token[key], state["hist_" + key], weights["w" + key], taps)
        if key == "k":
            prepared[key] = clip(c, QK_LIMIT)
        else:
            v = c.astype(np.float32)
            prepared[key] = (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True)
                                         + EPSILON)).astype(np.float16)
    q = clip(token["q"].astype(np.float32) * weights["q_scale"], QK_LIMIT)
    r = clip(token["r"].astype(np.float32) * weights["r_scale"], REL_LIMIT)
    if dynamic and position + 1 > 8192:
        tau = np.float32(1.0 + 0.1 * np.log((position + 1) / 8192.0))
        q = (q.astype(np.float32) * tau).astype(np.float16)
        r = (r.astype(np.float32) * tau).astype(np.float16)

    slot = position % capacity
    for key in ("k", "v"):
        state["cache_" + key][:, :, slot] = prepared[key]
        history = state["hist_" + key]
        history[:, :, 2] = history[:, :, 1]
        history[:, :, 1] = history[:, :, 0]
        history[:, :, 0] = token[key]
    # Chronological gathering keeps local wrapped rings in logical order.
    indices = np.arange(position + 1 - visible, position + 1) % capacity
    keys = state["cache_k"][:, :, indices].astype(np.float32)
    values = state["cache_v"][:, :, indices].astype(np.float32)
    grouped_q = q.astype(np.float32).reshape(batch, heads, groups, dim)
    logits = np.einsum("bhgd,bhld->bhgl", grouped_q, keys).reshape(batch, queries, visible)
    logits *= np.float32(dim ** -0.5)
    near = min(weights["rel"].shape[0], visible)
    bias = np.einsum("bqr,er->bqe", r.astype(np.float32), weights["rel"][:near].astype(np.float32))
    logits[:, :, -near:] += bias.astype(np.float16)[:, :, ::-1]
    sink = token["sink"].astype(np.float32)[..., None]
    peak = np.maximum(np.max(logits, axis=-1, keepdims=True), sink)
    exp_scores = np.exp(logits - peak)
    denom = np.sum(exp_scores, axis=-1, keepdims=True) + np.exp(sink - peak)
    probabilities = (exp_scores / denom).reshape(batch, heads, groups, visible)
    average = np.einsum("bhgl,bhld->bhgd", probabilities, values).reshape(batch, queries, dim).astype(np.float16)
    gate = token["gate"].astype(np.float32)
    # Equivalent sigmoid using logaddexp remains finite for the extreme fixtures.
    sigmoid = np.exp(-np.logaddexp(np.float32(0), -gate))
    return (average.astype(np.float32) * sigmoid).astype(np.float16)
