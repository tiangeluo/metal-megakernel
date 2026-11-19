"""Shared model equations; NumPy verifies them, MLX keeps reductions and routing on device."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    width: int = 1152
    expert_width: int = 576
    query_heads: int = 12
    kv_heads: int = 2
    head_dim: int = 128
    rel_dim: int = 16
    window: int = 1024
    extent: int = 1024
    experts: int = 384
    top_k: int = 8

    def __post_init__(self):
        if (
            min(vars(self).values()) < 1
            or self.query_heads % self.kv_heads
            or self.top_k > self.experts
        ):
            raise ValueError(
                "Positive compatible dimensions and 1 <= top_k <= experts required"
            )

    @property
    def expert_bytes(self):
        return 2 * 3 * self.width * self.expert_width * self.experts * 2

    @classmethod
    def compact(cls):
        return cls(
            width=24,
            expert_width=12,
            query_heads=6,
            kv_heads=1,
            head_dim=8,
            rel_dim=4,
            window=4,
            extent=4,
        )


def normal(rng, shape, scale=1.0, dtype=np.float16):
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(dtype)


def expert_bank(rng, experts, in_features, out_features):
    # Bound peak scratch space to one expert. Timing never includes initialization.
    weights = np.empty((experts, in_features, out_features), dtype=np.float16)
    for expert in range(experts):
        weights[expert] = normal(rng, (in_features, out_features), in_features**-0.5)
    return weights


def make_layer(c, rng, *, global_layer):
    from reference.fixtures import make_weights

    q, h, d, w = c.query_heads, c.kv_heads, c.head_dim, c.width
    shapes = dict(
        q=(q, d), k=(h, d), v=(h, d), r=(q, c.rel_dim), sink=(q,), gate=(q, d)
    )
    if global_layer:
        shapes.update(dk=(h, 4), dv=(h, 4))
    reads = 3 if global_layer else 1

    def gain(shape):
        return 1.0 + normal(rng, shape, 0.06, dtype=np.float32)

    return dict(
        project={
            name: normal(rng, (w, math.prod(shape)), w**-0.5)
            for name, shape in shapes.items()
        },
        attention=make_weights(
            rng, q, h, d, c.rel_dim, c.extent if global_layer else c.window
        ),
        attention_read=normal(rng, (reads, w), 0.8, dtype=np.float32),
        attention_norm=gain((reads, w)),
        moe_read=normal(rng, (1, w), 0.8, dtype=np.float32),
        moe_norm=gain((1, w)),
        out=normal(rng, (q * d, w), (q * d) ** -0.5),
        post_attn=normal(rng, (4, w), 0.25),
        post_moe=normal(rng, (4, w), 0.25),
        router=normal(rng, (w, c.experts), w**-0.5, dtype=np.float32),
        bias=normal(rng, (c.experts,), 0.015, dtype=np.float32),
        gate_up=expert_bank(rng, c.experts, w, 2 * c.expert_width),
        down=expert_bank(rng, c.experts, c.expert_width, w),
    )


def make_state(c, rng, batch, capacity, position, *, global_layer):
    from reference.attention import empty_state

    capacity = capacity if global_layer else min(c.window, capacity)
    state = empty_state(batch, c.kv_heads, capacity, c.head_dim)
    # This is a seeded replay of an existing cache. The current slot and evicted
    # entries are overwritten by the attention function according to position.
    active = min(capacity, position)
    if active:
        shape = (batch, c.kv_heads, active, c.head_dim)
        keys = normal(rng, shape, dtype=np.float32)
        norm = np.sqrt(np.sum(keys * keys, axis=-1, keepdims=True))
        state["cache_k"][:, :, :active] = keys * np.minimum(
            1.0, 22.627416997969522 / (norm + 1e-6)
        )
        values = normal(rng, shape, dtype=np.float32)
        state["cache_v"][:, :, :active] = values / np.sqrt(
            np.mean(values**2, axis=-1, keepdims=True) + 1e-6
        )
    for name in ("hist_k", "hist_v"):
        available = min(position, 3)
        state[name][:, :, :available] = normal(
            rng, (batch, c.kv_heads, available, c.head_dim)
        )
    for name in ("post_attn", "post_moe"):
        hist = np.zeros((batch, 3, c.width), dtype=np.float16)
        hist[:, : min(3, position)] = normal(rng, (batch, min(3, position), c.width))
        state[name] = hist
    return state


def rms(xp, x, eps=1e-6):
    x = x.astype(xp.float32)
    return x / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + eps)


def depth_read(xp, completed, partial, queries, gamma):
    """completed=[B,2,W] (embedding and previous block). Output [B,Q,W]."""
    sources = xp.concatenate(
        (completed.astype(xp.float32), partial[:, None, :].astype(xp.float32)), axis=1
    )
    values = rms(xp, sources)
    scores = (
        np.einsum("qw,bsw->bqs", queries, values)
        if xp is np
        else queries @ xp.swapaxes(values, -1, -2)
    )
    scores = scores / math.sqrt(values.shape[-1])
    weights = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
    weights = weights / xp.sum(weights, axis=-1, keepdims=True)
    mixed = np.einsum("bqs,bsw->bqw", weights, values) if xp is np else weights @ values
    return (rms(xp, mixed) * gamma).astype(xp.float16)


def project(xp, x, w):
    if xp is np:
        # Reference dot product accumulates in FP32, with half inputs and output.
        return np.einsum("...i,ij->...j", x, w, dtype=np.float32).astype(np.float16)
    return x @ w


def sigmoid(xp, x):
    return 0.5 + 0.5 * xp.tanh(0.5 * x)


def route(xp, x, w, bias, top_k):
    logits = (
        np.einsum("bi,ie->be", x, w, dtype=np.float32)
        if xp is np
        else x.astype(xp.float32) @ w
    )
    scores = sigmoid(xp, logits)
    order = -(scores + bias)
    idx = (
        np.argsort(order, axis=-1, kind="stable")
        if xp is np
        else xp.argsort(order, axis=-1)
    )[:, :top_k]
    score = xp.take_along_axis(scores, idx, axis=-1)
    # Tie policy is ascending expert ID on equal corrected scores.
    return idx, 8.0 * score / (xp.sum(score, axis=-1, keepdims=True) + 1e-20)


def activation(xp, gate_up):
    gate, up = xp.split(gate_up.astype(xp.float32), 2, axis=-1)
    return (
        4.0 * xp.tanh(gate / 4.0) * sigmoid(xp, gate) * 25.0 * xp.tanh(up / 25.0)
    ).astype(xp.float16)


def moe(xp, x, layer, top_k):
    idx, weights = route(xp, x, layer["router"], layer["bias"], top_k)
    if xp is np:
        # Index one selected expert at a time; never copy the full bank per token.
        out = np.zeros(x.shape, dtype=np.float32)
        for b, row in enumerate(idx):
            for rank, e in enumerate(row):
                mid = activation(np, project(np, x[b], layer["gate_up"][e]))
                out[b] += (
                    project(np, mid, layer["down"][e]).astype(np.float32)
                    * weights[b, rank]
                )
    else:
        batch = x.shape[0]
        lhs = xp.broadcast_to(xp.arange(batch, dtype=xp.uint32)[:, None], idx.shape)
        mid = activation(xp, xp.gather_mm(x[:, None, :], layer["gate_up"], lhs, idx))
        order = xp.arange(batch * top_k, dtype=xp.uint32).reshape(idx.shape)
        ys = xp.gather_mm(mid, layer["down"], order, idx)[:, :, 0, :]
        out = xp.sum(ys.astype(xp.float32) * weights[:, :, None], axis=1)
    return out.astype(xp.float16), idx


def output_conv(xp, current, hist, taps):
    """Static residual Conv4. History index zero and tap one refer to t-1."""
    raw = xp.concatenate((current[:, None, :], hist), axis=1)
    conv = xp.sum(raw.astype(xp.float32) * taps.astype(xp.float32), axis=1)
    return ((current.astype(xp.float32) + conv) / math.sqrt(5.0)).astype(
        xp.float16
    ), raw[:, :3, :]


def prepare_token(xp, c, layer, completed, partial, global_layer):
    sources = depth_read(
        xp, completed, partial, layer["attention_read"], layer["attention_norm"]
    )
    qs = sources[:, 0, :]
    ks = sources[:, 1, :] if global_layer else qs
    vs = sources[:, 2, :] if global_layer else qs
    batch = qs.shape[0]
    token = {}
    for key, source, count in [
        ("q", qs, c.query_heads),
        ("k", ks, c.kv_heads),
        ("v", vs, c.kv_heads),
        ("r", qs, c.query_heads),
        ("gate", qs, c.query_heads),
    ]:
        token[key] = project(xp, source, layer["project"][key]).reshape(
            batch, count, -1
        )
    token["sink"] = project(xp, qs, layer["project"]["sink"])
    for key, source in [("dk", ks), ("dv", vs)]:
        token[key] = (
            project(xp, source, layer["project"][key]).reshape(batch, c.kv_heads, 4)
            if global_layer
            else xp.zeros((batch, c.kv_heads, 4), dtype=xp.float16)
        )
    return token


def finish_layer(xp, c, layer, completed, partial, heads, post_attn, post_moe):
    projected = project(xp, heads.reshape(partial.shape[0], -1), layer["out"])
    delta, h_attn = output_conv(xp, projected, post_attn, layer["post_attn"])
    residual = partial.astype(xp.float32) + delta.astype(xp.float32)
    expert_source = depth_read(
        xp, completed, residual, layer["moe_read"], layer["moe_norm"]
    )[:, 0, :]
    expert_out, ids = moe(xp, expert_source, layer, c.top_k)
    delta, h_moe = output_conv(xp, expert_out, post_moe, layer["post_moe"])
    return (residual + delta.astype(xp.float32)).astype(xp.float16), h_attn, h_moe, ids


def build_decoder(xp, c, attention_steps):
    """All schedules share the same compiled pre/post regions around each attention call."""
    prepare = []
    for global_layer in (False, True):

        def preparer(layer, completed, partial, is_global=global_layer):
            return prepare_token(xp, c, layer, completed, partial, is_global)

        prepare.append(preparer if xp is np else xp.compile(preparer))

    def finish(layer, completed, partial, heads, post_attn, post_moe):
        return finish_layer(
            xp, c, layer, completed, partial, heads, post_attn, post_moe
        )

    finish = finish if xp is np else xp.compile(finish)

    def decode(layers, completed, partial, states):
        outputs = {}
        for i, (layer, state, attention) in enumerate(
            zip(layers, states, attention_steps)
        ):
            token = prepare[i](layer, completed, partial)
            attn_state = {
                key: state[key] for key in ("cache_k", "cache_v", "hist_k", "hist_v")
            }
            heads, new_state = attention(token, layer["attention"], attn_state)
            partial, attn, experts, ids = finish(
                layer, completed, partial, heads, state["post_attn"], state["post_moe"]
            )
            expanded = dict(new_state, post_attn=attn, post_moe=experts)
            outputs.update(
                {f"layer{i + 10}_{key}": value for key, value in expanded.items()}
            )
            outputs[f"layer{i + 10}_route_ids"] = ids
        outputs["summary"] = partial
        return outputs

    return decode
