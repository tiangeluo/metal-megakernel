"""Fixed-length seeded two-layer decode replay for the common benchmark driver."""

import argparse
from functools import lru_cache
import json

import numpy as np

from .model import Config, build_decoder, make_layer, make_state, normal


@lru_cache(maxsize=1)
def model_weights(config):
    rng = np.random.default_rng(17611)
    return tuple(make_layer(config, rng, global_layer=g) for g in (False, True))


def map_arrays(tree, func):
    if isinstance(tree, dict):
        return {k: map_arrays(v, func) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(map_arrays(v, func) for v in tree)
    return func(tree)


def leaves(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from leaves(v)
    elif isinstance(tree, tuple):
        for v in tree:
            yield from leaves(v)
    else:
        yield tree


@lru_cache(maxsize=1)
def device_weights(config):
    from mlx_backend import require_gpu

    mx = require_gpu()
    weights = map_arrays(model_weights(config), mx.array)
    mx.eval(*leaves(weights))
    return weights


def numpy_attention(position, window, dynamic):
    from reference.attention import step

    def attention(token, weights, state):
        return step(
            token, weights, state, position=position, window=window, dynamic=dynamic
        ), state

    return attention


class Case:
    def __init__(self, method, config, length, batch, seed):
        if min(length, batch) < 1:
            raise ValueError("length and batch must be positive")
        if method not in ("cpu", "compiled", "hybrid", "single", "staged"):
            raise ValueError(f"Unknown decoder method {method!r}")
        self.method = method
        self.case = {
            "method": method,
            "batch": batch,
            "length": length,
            "seed": seed,
            "geometry": vars(config),
        }
        if method == "cpu":
            self.xp = np
        else:
            from mlx_backend import require_gpu

            self.xp = require_gpu()
        self.result = None
        self.native = []
        weights = model_weights(config)
        rng = np.random.default_rng(seed)
        self.completed = normal(rng, (batch, 2, config.width))
        self.partial = normal(rng, (batch, config.width))
        position = length - 1
        self.initial = tuple(
            make_state(config, rng, batch, length, position, global_layer=g)
            for g in (False, True)
        )
        windows = (config.window, 0)
        if method == "cpu":
            self.layers = weights
            steps = [
                numpy_attention(position, window, i == 1)
                for i, window in enumerate(windows)
            ]
        else:
            self.layers = device_weights(config)
            if method in ("compiled", "hybrid"):
                from mlx_backend import build_step

                steps = [
                    build_step(
                        method=method, position=position, window=window, dynamic=i == 1
                    )
                    for i, window in enumerate(windows)
                ]
            else:
                from .metal_interop import MetalAttention

                for i, window in enumerate(windows):
                    self.native.append(
                        MetalAttention(
                            self.xp,
                            method,
                            config,
                            weights[i],
                            self.initial[i],
                            position,
                            window,
                            i == 1,
                        )
                    )
                steps = self.native
            self.completed, self.partial = map_arrays(
                (self.completed, self.partial), self.xp.array
            )
            self.xp.eval(self.completed, self.partial)
        self.decoder = build_decoder(self.xp, config, steps)
        self.reset()

    def reset(self):
        self.result = None
        if self.xp is np:
            self.states = map_arrays(self.initial, np.copy)
        elif self.native:
            # Only the output convolutions belong to MLX in this schedule. The
            # four attention buffers are already allocated and reset natively.
            self.states = tuple(
                {
                    k: (self.xp.array(v) if k.startswith("post_") else None)
                    for k, v in layer.items()
                }
                for layer in self.initial
            )
            self.xp.eval(*(v for v in leaves(self.states) if v is not None))
        else:
            # Fresh device cache snapshots allow buffer donation during append.
            self.states = map_arrays(self.initial, self.xp.array)
            self.xp.eval(*leaves(self.states))

        for method in self.native:
            method.runner.reset()

    def run(self):
        if self.states is None:
            raise RuntimeError("Reset the fixed-length replay before each run")
        self.result = self.decoder(
            self.layers, self.completed, self.partial, self.states
        )
        self.states = None
        if self.xp is not np:
            self.xp.eval(*self.result.values())
            self.xp.synchronize()

    def output_numpy(self):
        if self.result is None:
            raise RuntimeError("run() must finish before reading outputs")
        values = {key: np.asarray(value) for key, value in self.result.items()}
        for i, method in enumerate(self.native):
            for key, value in method.read_state().items():
                values[f"layer{i + 10}_{key}"] = value
        return values

    def close(self):
        for method in self.native:
            method.close()


def prepare_case(method, length, batch=1, seed=0, *, config=None):
    return Case(method, config or Config(), length, batch, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("cpu", "compiled", "staged", "single", "hybrid"),
        default="cpu",
    )
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    c = Config.compact() if args.compact else Config()
    case = prepare_case(args.method, args.length, seed=args.seed, config=c)
    try:
        case.run()
        out = case.output_numpy()
        if not all(np.isfinite(a).all() for a in out.values()):
            raise RuntimeError("Nonfinite stack result")
        print(
            json.dumps(
                dict(
                    method=args.method,
                    geometry=vars(c),
                    length=args.length,
                    expert_weight_bytes=c.expert_bytes,
                    output_arrays=len(out),
                    summary_shape=list(out["summary"].shape),
                    summary_l2=float(np.linalg.norm(out["summary"].astype(np.float32))),
                    routes=[out[f"layer{i}_route_ids"].tolist() for i in (10, 11)],
                ),
                indent=2,
            )
        )
    finally:
        case.close()


if __name__ == "__main__":
    main()
