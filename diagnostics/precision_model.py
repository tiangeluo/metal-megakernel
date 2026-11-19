"""Alternate two-layer workload with FP32 expert activations and FP16 expert banks."""
from functools import lru_cache
import math

import numpy as np

from decoder import model
from decoder.benchmark import Case, numpy_attention

CONTROL = "fp32-moe-depth-and-experts-v1"
PRECISION_CONTRACT = (
    "FP32 MoE depth-read output, router input, gate/up and down projection outputs, "
    "expert activation and weighted sum. FP16 expert banks; one FP16 cast after mixing. "
    "Attention, projections outside the MoE, cache, convolution and residual storage "
    "retain the original cast points. The GPU uses a custom selected-expert matvec."
)


def expert_source(xp, completed, partial, layer):
    sources = xp.concatenate((completed.astype(xp.float32), partial[:, None].astype(xp.float32)), axis=1)
    values = model.rms(xp, sources)
    queries = layer["moe_read"]
    scores = (np.einsum("qw,bsw->bqs", queries, values) if xp is np
              else queries @ xp.swapaxes(values, -1, -2)) / math.sqrt(values.shape[-1])
    weights = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
    weights = weights / xp.sum(weights, axis=-1, keepdims=True)
    mixed = np.einsum("bqs,bsw->bqw", weights, values) if xp is np else weights @ values
    return (model.rms(xp, mixed) * layer["moe_norm"])[:, 0]


@lru_cache(maxsize=1)
def _projection_kernel():
    import mlx.core as mx
    # One 128-thread group computes 32 columns for one selected expert. Four groups
    # of 32 lanes divide the input dimension, with contiguous FP16 weight reads.
    return mx.fast.metal_kernel(
        name="fp32_moe_selected_projection", input_names=["x", "bank", "ids"],
        output_names=["result"], source="""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid % 32;
        uint part = tid / 32;
        uint col = threadgroup_position_in_grid.x * 32 + lane;
        uint route = threadgroup_position_in_grid.y;
        uint expert = ids[route];
        uint row = PER_ROUTE ? route : route / TOP_K;
        threadgroup float partials[128];
        float sum = 0.0f;
        if (col < OUT_DIM) {
            for (uint d = part; d < IN_DIM; d += 4) {
                size_t offset = (size_t(expert) * IN_DIM + d) * OUT_DIM + col;
                sum = fma(x[size_t(row) * IN_DIM + d], float(bank[offset]), sum);
            }
        }
        partials[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0 && col < OUT_DIM) {
            result[size_t(route) * OUT_DIM + col] =
                (partials[lane] + partials[lane + 32]) +
                (partials[lane + 64] + partials[lane + 96]);
        }
        """,
    )


def selected_project(xp, x, bank, ids):
    """x=[B,I] or [B,K,I], bank=[E,I,O], ids=[B,K]. Return FP32 [B,K,O]."""
    batch, top_k = ids.shape
    _, width, out = bank.shape
    per_route = x.ndim == 3
    if x.shape != ((batch, top_k, width) if per_route else (batch, width)):
        raise ValueError("Input shape does not match bank and route IDs")
    if x.dtype != xp.float32 or bank.dtype != xp.float16:
        raise ValueError("Selected projection requires FP32 inputs and FP16 weight storage")
    if xp is np:
        result = np.empty((batch, top_k, out), dtype=np.float32)
        for b, row in enumerate(ids):
            for rank, expert in enumerate(row):
                lhs = x[b, rank] if per_route else x[b]
                result[b, rank] = np.einsum("i,ij->j", lhs, bank[expert], dtype=np.float32)
        return result
    return _projection_kernel()(
        inputs=[x, bank, ids], template=[("TOP_K", top_k), ("IN_DIM", width),
                                        ("OUT_DIM", out), ("PER_ROUTE", per_route)],
        grid=(128 * ((out + 31) // 32), batch * top_k, 1), threadgroup=(128, 1, 1),
        output_shapes=[(batch, top_k, out)], output_dtypes=[xp.float32],
    )[0]


def moe(xp, x, layer, top_k):
    ids, weights = model.route(xp, x, layer["router"], layer["bias"], top_k)
    gate, up = xp.split(selected_project(xp, x, layer["gate_up"], ids), 2, axis=-1)
    mid = 4.0 * xp.tanh(gate / 4.0) * model.sigmoid(xp, gate) * 25.0 * xp.tanh(up / 25.0)
    output = selected_project(xp, mid, layer["down"], ids)
    return xp.sum(output * weights[:, :, None], axis=1).astype(xp.float16), ids


def finish(xp, config, layer, completed, partial, heads, hist_a, hist_e):
    projected = model.project(xp, heads.reshape(partial.shape[0], -1), layer["out"])
    delta, new_a = model.output_conv(xp, projected, hist_a, layer["post_attn"])
    residual = partial.astype(xp.float32) + delta.astype(xp.float32)
    expert_out, ids = moe(xp, expert_source(xp, completed, residual, layer), layer, config.top_k)
    delta, new_e = model.output_conv(xp, expert_out, hist_e, layer["post_moe"])
    return (residual + delta.astype(xp.float32)).astype(xp.float16), new_a, new_e, ids


def _decoder(case, config, position):
    xp = case.xp
    compile_fn = (lambda fn: fn) if xp is np else xp.compile
    steps = list(case.native)
    if not steps:
        for i, window in enumerate((config.window, 0)):
            if xp is np:
                steps.append(numpy_attention(position, window, bool(i)))
            else:
                from mlx_backend import build_step
                steps.append(build_step(method=case.method, position=position, window=window, dynamic=bool(i)))
    prepare = [compile_fn(lambda w, completed, partial, global_layer=flag:
                          model.prepare_token(xp, config, w, completed, partial, global_layer))
               for flag in (False, True)]
    compiled_finish = compile_fn(lambda w, completed, partial, heads, a, e:
                                 finish(xp, config, w, completed, partial, heads, a, e))

    def decode(layers, completed, partial, states):
        outputs = {}
        for i, (layer, state, step) in enumerate(zip(layers, states, steps)):
            token = prepare[i](layer, completed, partial)
            heads, updated = step(token, layer["attention"],
                                  {k: state[k] for k in ("cache_k", "cache_v", "hist_k", "hist_v")})
            partial, a, e, ids = compiled_finish(layer, completed, partial, heads, state["post_attn"], state["post_moe"])
            outputs.update({f"layer{i + 10}_{k}": v for k, v in
                            dict(updated, post_attn=a, post_moe=e, route_ids=ids).items()})
        outputs["summary"] = partial
        return outputs
    return decode


def prepare_case(method, length, batch=1, seed=0, *, config=None):
    config = config or model.Config()
    case = Case(method, config, length, batch, seed)
    try:
        case.decoder = _decoder(case, config, length - 1)
        case.case["precision_control"] = CONTROL
    except BaseException:
        case.close()
        raise
    return case
