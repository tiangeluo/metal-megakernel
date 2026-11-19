"""Independent dense prefix oracle. FP64 arithmetic with explicit FP16 boundaries.

The API accepts a list of projected tokens and no decode-state objects. Delayed
raw projections come directly from the sequence; sliding attention is a mask
over absolute positions. Only requested query rows need be evaluated.
"""
from __future__ import annotations

import numpy as np


def prefill(tokens, weights, *, window=0, dynamic=True, positions=None):
    if not tokens or window < 0:
        raise ValueError("non-empty prefix and nonnegative window required")
    arrays = {k: np.stack([token[k] for token in tokens]).astype(np.float64)
              for k in tokens[0]}
    length, batch, queries, dim = arrays["q"].shape
    heads = arrays["k"].shape[2]
    if queries % heads:
        raise ValueError("query heads must be divisible by KV heads")
    if positions is None:
        positions = np.arange(length)
    else:
        positions = np.asarray(positions)
    if np.any(positions < 0) or np.any(positions >= length):
        raise IndexError("query index outside supplied prefix")

    def clip_head(x, limit):
        divisor = np.maximum(1, (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) / limit)
        return (x / divisor).astype(np.float16).astype(np.float64)

    prepared = {}
    for name in ("k", "v"):
        projections = arrays[name]
        # sliding_window_view exposes oldest→newest; canonical gains use lag 0=current.
        padded = np.pad(projections, ((3, 0), (0, 0), (0, 0), (0, 0)))
        samples = np.lib.stride_tricks.sliding_window_view(padded, 4, axis=0)[..., ::-1]
        gains = np.moveaxis(weights["w" + name].astype(np.float64), 0, -1)
        if dynamic:
            gains = gains + arrays["d" + name][..., None, :]
        filtered = (projections + np.sum(samples * gains, axis=-1)) / np.sqrt(9 if dynamic else 5)
        filtered = filtered.astype(np.float16).astype(np.float64)
        if name == "k":
            prepared[name] = clip_head(filtered, 22.627416997969522)
        else:
            prepared[name] = (filtered / np.sqrt((filtered ** 2).mean(axis=-1, keepdims=True)
                                                + 1e-6)).astype(np.float16).astype(np.float64)
    q = clip_head(arrays["q"] * weights["q_scale"], 22.627416997969522)
    r = clip_head(arrays["r"] * weights["r_scale"], 8.0)
    if dynamic:
        temp = 1 + 0.1 * np.log(np.maximum(1, (np.arange(length) + 1) / 8192))
        q = (q * temp[:, None, None, None]).astype(np.float16).astype(np.float64)
        r = (r * temp[:, None, None, None]).astype(np.float16).astype(np.float64)
    # Explicit extra, zero-valued sink in ordinary softmax; no separate denominator trick.
    result = np.empty((len(positions), batch, queries, dim), np.float16)
    table = weights["rel"].astype(np.float64)
    for row, t in enumerate(positions):
        start = max(0, int(t) + 1 - window) if window else 0
        distances = np.arange(int(t) - start, -1, -1)
        for h in range(queries):
            kv = h // (queries // heads)
            k = prepared["k"][start:t + 1, :, kv]
            content = np.einsum("bd,sbd->bs", q[t, :, h], k) / np.sqrt(dim)
            near = distances < len(table)
            relative = np.einsum("br,sr->bs", r[t, :, h], table[distances[near]])
            content[:, near] += relative.astype(np.float16)
            scores = np.column_stack((content, arrays["sink"][t, :, h]))
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            v = prepared["v"][start:t + 1, :, kv]
            average = np.einsum("bs,sbd->bd", probs[:, :-1], v).astype(np.float16)
            gate = arrays["gate"][t, :, h]
            result[row, :, h] = average.astype(np.float64) * np.exp(-np.logaddexp(0, -gate))
    return result, {k: v.astype(np.float16) for k, v in prepared.items()}
