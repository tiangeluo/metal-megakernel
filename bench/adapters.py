"""Prepare device-resident inputs, then expose separate reset, synchronous step and readback."""
from __future__ import annotations

import numpy as np

from .protocol import compare_attention, equal_stack


class Native:
    def __init__(self, runner, method):
        self.runner, self.method = runner, method

    def reset(self):
        self.runner.reset()

    def step(self):
        data = self.runner.run(self.method)
        return data.get("gpu_us"), data.get("dispatches")

    def read(self):
        return self.runner.outputs()



class MLX:
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.fresh()

    def step(self):
        self.runner.run()
        return None, None

    def read(self):
        return self.runner.readback()



class Stack:
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.reset()

    def step(self):
        self.runner.run()
        return None, None

    def read(self):
        return self.runner.output_numpy()

    def close(self):
        self.runner.close()


def prepare(case, seed):
    """Return runners, an independent readback validator and cleanup. No setup is timed."""
    if case.scope == "attention":
        return prepare_attention(case, seed)
    return prepare_stack(case, seed)


def prepare_attention(case, seed):
    from reference.fixtures import make_case
    from reference.attention import step
    from mlx_backend import prepare_case
    from native.runner import NativeRunner

    args = {"position": case.length-1, "window": 0, "dynamic": True}
    token, weights, initial = make_case(batch=case.batch, q_heads=48, kv_heads=8,
                                        dim=128, length=case.length, seed=seed, window=0,
                                        dynamic=True, rel_dim=16, extent=1024)
    expected_state = {k: np.array(v, copy=True) for k, v in initial.items()}
    output = step(token, weights, expected_state, **args)
    expected = (output, expected_state)
    native = NativeRunner(token, weights, initial, **args)
    try:
        runners = {}
        for method in case.methods:
            if method in ("staged", "single", "split"):
                runners[method] = Native(native, method)
            else:
                runners[method] = MLX(prepare_case(token, weights, initial, method=method, **args))
    except BaseException:
        native.close()
        raise

    def check(actual):
        return compare_attention(expected, actual, initial, args["position"])
    return runners, check, native.close


def prepare_stack(case, seed):
    from decoder.benchmark import prepare_case
    args = {"seed": seed, "batch": case.batch, "length": case.length}
    oracle = prepare_case(method="cpu", **args)
    oracle.reset()
    oracle.run()
    expected = oracle.output_numpy()
    oracle.close()
    del oracle
    runners = {}
    def close():
        for runner in runners.values():
            runner.close()
    try:
        for method in case.methods:
            runners[method] = Stack(prepare_case(method=method, **args))
    except BaseException:
        close()
        raise
    return runners, lambda actual: equal_stack(expected, actual, case.length - 1), close
