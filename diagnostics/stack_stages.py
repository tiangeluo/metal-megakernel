"""Untimed, matched-input checks of the connected decoder's numerical boundaries."""

import numpy as np

from decoder import model
from decoder.benchmark import map_arrays, model_weights, numpy_attention, prepare_case
from reference.attention import step


def difference(expected, actual, *, atol=0.002, rtol=0.01):
    """Summarize elementwise differences after the operation's output cast."""
    ref, got = np.asarray(expected), np.asarray(actual)
    if ref.shape != got.shape:
        raise ValueError(f"Expected {ref.shape}, got {got.shape}")
    if ref.dtype.kind in "iub":
        atol = rtol = 0
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        return {"finite": False, "passed": False}
    ref, got = ref.astype(np.float64), got.astype(np.float64)
    err = np.abs(got - ref)
    bad = err > atol + rtol * np.abs(ref)
    return {"elements": ref.size, "different": int(np.count_nonzero(err)),
            "max_abs": float(err.max(initial=0.0)), "above_tolerance": int(bad.sum()),
            "passed": not bool(bad.any())}


def _host_expert_project(xs, ids, bank):
    # The CPU weight bank already exists. Only visit the selected experts.
    return np.stack([np.stack([model.project(np, x, bank[e])
                              for x, e in zip(row, experts)])
                     for row, experts in zip(xs, ids)])


def _expert_detail(xp, source, layer, host, top_k, actual_out):
    """Isolate each active expert projection, its FP16 cast, and the weighted sum."""
    ids, weights = model.route(xp, source, layer["router"], layer["bias"], top_k)
    hsource, hids = np.asarray(source), np.asarray(ids)
    cpu_ids, cpu_weights = model.route(np, hsource, host["router"], host["bias"], top_k)
    stats = {"route_ids": difference(cpu_ids, ids), "route_weights": difference(cpu_weights, weights)}
    xs = np.broadcast_to(hsource[:, None, :], (*hids.shape, hsource.shape[-1]))
    gate_ref = _host_expert_project(xs, hids, host["gate_up"])
    if xp is np:
        gate = gate_ref
    else:
        lhs = xp.broadcast_to(xp.arange(hsource.shape[0], dtype=xp.uint32)[:, None], ids.shape)
        gate = xp.gather_mm(source[:, None, :], layer["gate_up"], lhs, ids)[:, :, 0, :]
    stats["gate_up_fp16"] = difference(gate_ref, gate)
    mid = model.activation(xp, gate)
    stats["activation_fp16"] = difference(model.activation(np, np.asarray(gate)), mid)
    down_ref = _host_expert_project(np.asarray(mid), hids, host["down"])
    if xp is np:
        down = down_ref
    else:
        order = xp.arange(hids.size, dtype=xp.uint32).reshape(ids.shape)
        down = xp.gather_mm(mid[:, :, None, :], layer["down"], order, ids)[:, :, 0, :]
    stats["down_fp16"] = difference(down_ref, down)
    summed = xp.sum(down.astype(xp.float32) * weights[:, :, None], axis=1).astype(xp.float16)
    local_sum = np.sum(np.asarray(down).astype(np.float32) * np.asarray(weights)[:, :, None],
                       axis=1).astype(np.float16)
    stats["weighted_sum_fp16"] = difference(local_sum, summed)
    # Exposes any arithmetic change when observing components breaks graph fusion.
    stats["split_replay_vs_compiled_expert_output"] = difference(actual_out, summed)
    return stats


def _attention(method, runner, i, position, window):
    if runner.native:
        return runner.native[i]
    if method == "cpu":
        return numpy_attention(position, window, bool(i))
    from mlx_backend import build_step
    return build_step(method=method, position=position, window=window, dynamic=bool(i))


def _layer(runner, c, i, partial, position, original):
    xp, layer = runner.xp, runner.layers[i]
    host, completed = model_weights(c)[i], runner.completed
    state = runner.states[i]
    window = 0 if i else c.window
    compile_fn = (lambda fn: fn) if xp is np else xp.compile

    def prep(w, blocks, stream):
        return model.prepare_token(xp, c, w, blocks, stream, bool(i))

    def finish(w, blocks, stream, heads, attn_hist, moe_hist):
        return model.finish_layer(xp, c, w, blocks, stream, heads, attn_hist, moe_hist)

    token = compile_fn(prep)(layer, completed, partial)
    host_token = map_arrays(token, np.asarray)
    prep_ref = model.prepare_token(np, c, host, np.asarray(completed), np.asarray(partial), bool(i))
    stats = {"prepare": {k: difference(prep_ref[k], v) for k, v in host_token.items()}}
    src = model.depth_read(xp, completed, partial, layer["attention_read"], layer["attention_norm"])
    host_src = np.asarray(src)
    ref_src = model.depth_read(np, np.asarray(completed), np.asarray(partial),
                               host["attention_read"], host["attention_norm"])
    stats["attention_depth_read_fp16"] = difference(ref_src, src)
    projection_checks = {}
    for name, weight in host["project"].items():
        idx = {"k": 1, "dk": 1, "v": 2, "dv": 2}.get(name, 0) if i else 0
        projected = model.project(np, host_src[:, idx], weight).reshape(host_token[name].shape)
        projection_checks[name] = difference(projected, host_token[name])
    stats["input_projections_on_split_depth_read"] = projection_checks
    keys = ("cache_k", "cache_v", "hist_k", "hist_v")
    attn = _attention(runner.method, runner, i, position, window)
    heads, new = attn(token, layer["attention"], {k: state[k] for k in keys})
    if runner.native:
        new = attn.read_state()
    hstate = {k: np.copy(runner.initial[i][k]) for k in keys}
    oracle_heads = step(host_token, host["attention"], hstate, position=position, window=window, dynamic=bool(i))
    attention = {"gated_heads": difference(oracle_heads, heads)}
    for k in keys:
        got = np.asarray(new[k])
        if k.startswith("cache"):
            slot = position % got.shape[2]
            attention[k] = difference(hstate[k][:, :, slot], got[:, :, slot], atol=0.002, rtol=0.002)
            attention[k + "_untouched_equal"] = all(
                np.array_equal(hstate[k][:, :, region], got[:, :, region])
                for region in (slice(None, slot), slice(slot + 1, None)))
        else:
            attention[k] = difference(hstate[k], got, atol=0, rtol=0)
    stats["attention_on_observed_token"] = attention

    post = [state[k] for k in ("post_attn", "post_moe")]
    next_partial, attn_hist, moe_hist, route_ids = compile_fn(finish)(
        layer, completed, partial, heads, *post)
    ref_outputs = model.finish_layer(np, c, host, np.asarray(completed), np.asarray(partial),
                                     np.asarray(heads), *[np.asarray(v) for v in post])
    outputs = [next_partial, attn_hist, moe_hist, route_ids]
    stats["finish_on_observed_heads"] = {
        k: difference(ref, got) for k, ref, got in zip(
            ("partial", "post_attn", "post_moe", "route_ids"), ref_outputs, outputs)}
    prefix = f"layer{i + 10}_"
    checks = {k: difference(original[prefix+k], v, atol=0, rtol=0)
              for k, v in zip(("post_attn", "post_moe", "route_ids"), outputs[1:])}
    if i:
        checks["summary"] = difference(original["summary"], next_partial, atol=0, rtol=0)
    stats["replay_against_original_call"] = checks

    # post_attn/post_moe store the unfiltered projection/expert outputs in slot 0.
    projected, experts = attn_hist[:, 0, :], moe_hist[:, 0, :]
    projection_ref = model.project(np, np.asarray(heads).reshape(partial.shape[0], -1), host["out"])
    stats["output_projection_fp16"] = difference(projection_ref, projected)
    delta, _ = model.output_conv(xp, projected, post[0], layer["post_attn"])
    ref_delta, _ = model.output_conv(np, np.asarray(projected), np.asarray(post[0]), host["post_attn"])
    stats["attention_output_conv_fp16"] = difference(ref_delta, delta)
    residual = partial.astype(xp.float32) + delta.astype(xp.float32)
    source = model.depth_read(xp, completed, residual, layer["moe_read"], layer["moe_norm"])[:, 0, :]
    ref_source = model.depth_read(np, np.asarray(completed), np.asarray(residual),
                                  host["moe_read"], host["moe_norm"])[:, 0, :]
    stats["expert_depth_read_fp16"] = difference(ref_source, source)
    stats["expert_components"] = _expert_detail(xp, source, layer, host, c.top_k, experts)
    moe_delta, _ = model.output_conv(xp, experts, post[1], layer["post_moe"])
    ref_moe, _ = model.output_conv(np, np.asarray(experts), np.asarray(post[1]), host["post_moe"])
    stats["expert_output_conv_fp16"] = difference(ref_moe, moe_delta)
    stats["split_replay_vs_compiled_partial"] = difference(
        next_partial, (residual + moe_delta.astype(xp.float32)).astype(xp.float16))
    return next_partial, stats


def isolate(case, seed, method, *, config=None):
    """Compare GPU and CPU on the same stage inputs. Results are diagnostic only."""
    c = config or model.Config()
    runner = prepare_case(method, case.length, case.batch, seed, config=c)
    try:
        runner.run()
        # Keep only small outputs for replay/production equality. Native readback would
        # otherwise retain two extra copies of both KV caches. Expert weights stay put.
        original = {k: np.asarray(v).copy() for k, v in runner.result.items()
                    if k.endswith(("post_attn", "post_moe", "route_ids", "summary"))}
        runner.reset()
        partial = runner.partial
        layers = {}
        for i in range(2):
            partial, layers[str(i + 10)] = _layer(runner, c, i, partial, case.length - 1, original)
        return {"method": method, "seed": seed, "length": case.length, "batch": case.batch,
                "interpretation": "Untimed replay. Each local NumPy check receives the observed stage inputs. "
                                  "Split component graphs can round differently from the compiled finish. "
                                  "Replay comparisons expose that change. Local checks do not certify a timing.",
                "layers": layers}
    finally:
        runner.close()
