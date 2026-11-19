"""Swap captured layer-11 residuals and gated heads; run every combination through one CPU finish."""
import argparse
from contextlib import closing
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from bench.diagnose_stack import numbers
from bench.protocol import put_json, source_hash
from decoder import model
from decoder.benchmark import model_weights, prepare_case
from diagnostics.stack_stages import _attention, _host_expert_project


def copy(x):
    return np.array(x, copy=True)


def expert_readback(xp, config, layer, host, blocks, partial, hist_a, projected):
    """Expose the active-expert arithmetic as a separate graph; compare its result to the real finish."""
    delta, _ = model.output_conv(xp, projected, hist_a, layer["post_attn"])
    residual = partial.astype(xp.float32) + delta.astype(xp.float32)
    source = model.depth_read(xp, blocks, residual, layer["moe_read"], layer["moe_norm"])[:, 0]
    ids, weights = model.route(xp, source, layer["router"], layer["bias"], config.top_k)
    if xp is np:
        xs = np.broadcast_to(source[:, None, :], (*ids.shape, source.shape[-1]))
        gate = _host_expert_project(xs, ids, host["gate_up"])
        activation = model.activation(np, gate)
        down = _host_expert_project(activation, ids, host["down"])
    else:
        lhs = xp.broadcast_to(xp.arange(source.shape[0], dtype=xp.uint32)[:, None], ids.shape)
        gate = xp.gather_mm(source[:, None, :], layer["gate_up"], lhs, ids)[:, :, 0, :]
        activation = model.activation(xp, gate)
        order = xp.arange(source.shape[0] * config.top_k, dtype=xp.uint32).reshape(ids.shape)
        down = xp.gather_mm(activation[:, :, None, :], layer["down"], order, ids)[:, :, 0, :]
    mixed = xp.sum(down.astype(xp.float32) * weights[:, :, None], axis=1).astype(xp.float16)
    return {k: copy(v) for k, v in dict(expert_source=source, replay_routes=ids, route_weights=weights,
                                          gate_up=gate, activation=activation, down=down, replay_moe=mixed).items()}


def capture(method, config, length, seed):
    """Replay the regular call and verify every returned wrapper output before accepting its boundaries."""
    with closing(prepare_case(method, length, batch=1, seed=seed, config=config)) as runner:
        runner.run()
        original = {k: copy(v) for k, v in runner.result.items()
                    if k == "summary" or k.endswith(("post_attn", "post_moe", "route_ids"))}
        runner.reset()
        xp = runner.xp
        compile_fn = (lambda f: f) if xp is np else xp.compile

        def finish(layer, blocks, partial, heads, hist_a, hist_e):
            return model.finish_layer(xp, config, layer, blocks, partial, heads, hist_a, hist_e)

        finish = compile_fn(finish)
        partial = runner.partial
        common = {"completed": copy(runner.completed)}
        for i, window in enumerate((config.window, 0)):
            def prepare(layer, blocks, partial, global_layer=bool(i)):
                return model.prepare_token(xp, config, layer, blocks, partial, global_layer)

            layer, state = runner.layers[i], runner.states[i]
            token = compile_fn(prepare)(layer, runner.completed, partial)
            attn = _attention(method, runner, i, length - 1, window)
            attn_state = {k: state[k] for k in ("cache_k", "cache_v", "hist_k", "hist_v")}
            heads, _ = attn(token, layer["attention"], attn_state)
            if i == 1:
                layer11_input = partial
                inputs = {"residual": copy(partial), "heads": copy(heads)}
                common.update(hist_a=copy(state["post_attn"]), hist_e=copy(state["post_moe"]))
            partial, h_a, h_e, routes = finish(layer, runner.completed, partial, heads,
                                                state["post_attn"], state["post_moe"])
            for key, value in (("post_attn", h_a), ("post_moe", h_e), ("route_ids", routes)):
                key = f"layer{i + 10}_{key}"
                np.testing.assert_array_equal(original[key], np.asarray(value), err_msg=f"{method} {key} replay")
        np.testing.assert_array_equal(original["summary"], np.asarray(partial), err_msg=f"{method} summary replay")
        inputs.update(moe=original["layer11_post_moe"][:, 0, :], routes=original["layer11_route_ids"],
                      summary=original["summary"])
        inputs.update(expert_readback(xp, config, layer, model_weights(config)[1], runner.completed,
                                      layer11_input, state["post_attn"], h_a[:, 0]))
        return common, inputs


def compare(a, b):
    return numbers(a, b, 0.002, 0.01)


def expert_precision(layer, values, channels):
    """FP64 reference dot products share the replayed inputs, expert IDs and FP16 output casts."""
    ids = values["replay_routes"][0]
    src = values["expert_source"][0]
    up_ref, down_ref = [], []
    for i, expert in enumerate(ids):
        up_ref.append(np.einsum("i,ij->j", src, layer["gate_up"][expert], dtype=np.float64))
        down_weights = layer["down"][expert][:, channels]
        down_ref.append(np.einsum("i,ij->j", values["activation"][0, i], down_weights, dtype=np.float64))
    up_ref, down_ref = np.asarray(up_ref, dtype=np.float16), np.asarray(down_ref, dtype=np.float16)
    observed_down = values["down"][0][:, channels]
    weighted = observed_down.astype(np.float64) * values["route_weights"][0, :, None]
    return dict(replayed_routes_match=bool(np.array_equal(values["routes"], values["replay_routes"])),
                replayed_expert_output_exact=bool(np.array_equal(values["moe"], values["replay_moe"])),
                replayed_expert_output_vs_actual=compare(values["moe"], values["replay_moe"]),
                fp64_gate_up_vs_observed=compare(up_ref, values["gate_up"][0]),
                fp64_down_vs_observed_at_channels=compare(down_ref, observed_down),
                signed_expert_contributions=weighted.tolist(),
                sum_abs_contributions=np.abs(weighted).sum(axis=0).tolist(),
                signed_sum=weighted.sum(axis=0).tolist())


def analyze(config, common, captures):
    """All nine cells use identical host weights, histories, depth states and FP16 cast points."""
    observed, crossed = {}, {}
    channels = set()
    for left, right in combinations(captures, 2):
        a, b = captures[left]["moe"], captures[right]["moe"]
        observed[f"{left}_vs_{right}"] = compare(a, b)
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        channels.update(np.flatnonzero(np.abs(b32-a32) > 0.002 + 0.01*np.abs(a32)).tolist())
    channels = sorted(channels)
    authority = captures["cpu"]["moe"]
    raw = {}
    layer = model_weights(config)[1]
    for r, h in product(captures, repeat=2):
        name = f"residual_{r}__heads_{h}"
        summary, _, moe_hist, routes = model.finish_layer(np, config, layer, common["completed"],
                                                         captures[r]["residual"], captures[h]["heads"],
                                                         common["hist_a"], common["hist_e"])
        moe = moe_hist[:, 0, :]
        if r == h == "cpu":
            np.testing.assert_array_equal(moe, authority)
            np.testing.assert_array_equal(routes, captures["cpu"]["routes"])
            np.testing.assert_array_equal(summary, captures["cpu"]["summary"])
        raw[name] = moe
        crossed[name] = dict(vs_cpu=compare(authority, moe), routes=routes.tolist(),
                             channels=moe[0, channels].astype(float).tolist())
        if r == h:
            crossed[name]["observed_vs_cpu_finish_on_same_inputs"] = compare(moe, captures[r]["moe"])
    experts = {m: expert_precision(layer, v, channels) for m, v in captures.items()}
    return {"observed": observed, "channels": channels, "expert_precision": experts,
            "observed_channel_values": {m: v["moe"][0, channels].astype(float).tolist()
                                        for m, v in captures.items()},
            "crossed_cpu_finish": crossed}, raw


def run(length, seed, out, *, methods=("cpu", "compiled", "staged"), config=None):
    config = config or model.Config()
    if length < 1 or seed < 0 or not methods or methods[0] != "cpu" or len(set(methods)) != len(methods):
        raise ValueError("Use positive length, nonnegative seed, and unique methods starting with cpu.")
    common, captures = None, {}
    for method in methods:
        current, captures[method] = capture(method, config, length, seed)
        if common is None:
            common = current
        else:
            for k, v in common.items():
                np.testing.assert_array_equal(v, current[k], err_msg=f"shared {k}")
    result, crossed = analyze(config, common, captures)
    metadata = dict(seed=seed, length=length, batch=1, config=vars(config), methods=list(methods),
                    sources_sha256=source_hash(), device="CPU")
    metadata["diagnostic_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("stack_crossover.py", "stack_stages.py")
    }
    if any(method != "cpu" for method in methods):
        from mlx_backend import require_gpu
        metadata["device"] = require_gpu().device_info()["device_name"]
    arrays = dict(common, metadata=np.array(json.dumps(metadata)))
    arrays.update({f"{m}__{key}": v for m, values in captures.items() for key, v in values.items()})
    arrays.update(crossed)
    out = Path(out)
    if out.suffix != ".json":
        raise ValueError("Choose a .json output; the adjacent .npz holds captured arrays.")
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.with_suffix(".npz")
    with NamedTemporaryFile(dir=out.parent, delete=False) as f:
        np.savez_compressed(f, **arrays)
    Path(f.name).replace(archive)
    result.update(metadata, wrapper_replay_exact=True, capture=archive.name,
                  capture_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    put_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("cpu", "compiled", "staged", "single", "hybrid"),
                        default=("cpu", "compiled", "staged"))
    args = parser.parse_args()
    result = run(args.length, args.seed, args.out, methods=args.methods)
    print(f"Saved {args.out} and {result['capture']}. Compared {len(result['crossed_cpu_finish'])} input pairs.")
