"""Exercise published head geometry against an independently computed prefix."""
import json

import numpy as np

from .attention import empty_state, step
from .fixtures import make_token, make_weights
from .prefill import prefill


def published_geometry(seed=20261118):
    rng = np.random.default_rng(seed)
    weights = make_weights(rng, 48, 8, 128)
    tokens = [make_token(rng, 1, 48, 8, 128) for _ in range(64)]
    state = empty_state(1, 8, 64, 128)
    outputs = np.stack([step(x, weights, state, position=i) for i, x in enumerate(tokens)])
    oracle, kv = prefill(tokens, weights)
    pairs = {
        "gated_output": (outputs, oracle, 1e-2),
        "cache_k": (state["cache_k"], kv["k"].transpose(1, 2, 0, 3), 2e-3),
        "cache_v": (state["cache_v"], kv["v"].transpose(1, 2, 0, 3), 2e-3),
    }
    report = {"seed": seed, "length": 64, "geometry": [48, 8, 128],
              "device": "CPU", "numpy_version": np.__version__, "comparisons": {}}
    for name, (actual, expected, rtol) in pairs.items():
        a = actual.astype(np.float64)
        b = expected.astype(np.float64)
        np.testing.assert_allclose(a, b, atol=0.002, rtol=rtol, equal_nan=False)
        delta = a - b
        report["comparisons"][name] = {
            "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12)),
        }
    return report


if __name__ == "__main__":
    print(json.dumps(published_geometry(), indent=2))
