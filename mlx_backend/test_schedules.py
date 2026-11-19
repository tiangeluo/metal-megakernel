"""Host algebra checks and real-device parity tests for both MLX schedules."""
import math
import unittest

import numpy as np

from reference.attention import step as oracle
from reference.fixtures import make_case
from mlx_backend.schedules import _attend, _prepare_mlx, prepare_case, require_gpu


class _HostFast:
    @staticmethod
    def scaled_dot_product_attention(q, k, v, *, scale, mask, sinks):
        """Independent dense SDPA with per-head sink, for checking argument layout."""
        repeat = q.shape[1] // k.shape[1]
        keys = np.repeat(k, repeat, axis=1).astype(np.float32)
        values = np.repeat(v, repeat, axis=1).astype(np.float32)
        scores = np.einsum("bhqd,bhld->bhql", q.astype(np.float32), keys) * np.float32(scale) + mask
        sink_column = np.broadcast_to(sinks[None, :, None, None], (*scores.shape[:-1], 1))
        scores = np.concatenate([scores, sink_column], axis=-1)
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return np.einsum("bhql,bhld->bhqd", p[..., :-1], values).astype(np.float16)


class _HostOps:
    """Only used in the CPU test; benchmark code always requires a Metal device."""
    fast = _HostFast()

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def matmul(a, b):
        return np.einsum("bqr,re->bqe", a, b)

    @staticmethod
    def rsqrt(x):
        return 1 / np.sqrt(x)

    @staticmethod
    def sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))

    @staticmethod
    def slice_update(x, value, offset, axes):
        if tuple(axes) != (2,):
            raise ValueError("unexpected cache axis")
        out = x.copy()
        start = int(offset[0])
        out[:, :, start:start + 1] = value
        return out


def assert_result(out, actual, expected, target):
    assert np.isfinite(out).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(out, expected, atol=0.002, rtol=0.003)
    for key in actual:
        if key.startswith("hist"):
            np.testing.assert_array_equal(actual[key], target[key])
        else:
            np.testing.assert_allclose(actual[key], target[key], atol=0.002, rtol=0.003)


class LayoutTests(unittest.TestCase):
    def test_host_attention_algebra_against_chronological_oracle(self):
        for batch, length, window, extent in [(1, 1, 0, 8), (3, 13, 0, 3),
                                               (2, 28, 7, 3), (2, 3, 5, 8),
                                               (1, 1025, 1024, 1024)]:
            with self.subTest(batch=batch, length=length, window=window):
                dynamic = window == 0
                token, weights, state = make_case(batch=batch, length=length, window=window,
                                                  q_heads=6, kv_heads=2, dim=32, extent=extent)
                if batch > 1:
                    token["sink"][:] += np.arange(batch, dtype=np.float16)[:, None] * 5
                target = {k:v.copy() for k, v in state.items()}
                expected = oracle(token, weights, target, position=length - 1, window=window,
                                  dynamic=dynamic)
                mx = _HostOps()
                tau = 1.0 if not dynamic else 1 + 0.1 * math.log(max(1, length / 8192))
                prepared = _prepare_mlx(mx, token, weights, state, dynamic, tau)
                out, actual = _attend(mx, token, weights, state, prepared, length - 1, window)
                assert_result(out, actual, expected, target)


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            require_gpu()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def test_compiled_and_hybrid_on_metal(self):
        for method in ("compiled", "hybrid"):
            for b, length, w in [(1, 1, 0), (2, 64, 0), (1, 1025, 1024)]:
                with self.subTest(method=method, b=b, length=length, window=w):
                    token, weights, state = make_case(batch=b, q_heads=12, kv_heads=2,
                                                      dim=128, length=length, window=w)
                    self._check(method, token, weights, state, length - 1, w)

    def _check(self, method, token, weights, state, pos, window):
        expected_state = {k:v.copy() for k, v in state.items()}
        expected = oracle(token, weights, expected_state, position=pos, window=window,
                          dynamic=window == 0)
        case = prepare_case(token, weights, state, position=pos, window=window,
                            dynamic=window == 0, method=method)
        for _ in range(2):
            case.fresh()
            case.run()
            out, actual = case.readback()
            assert_result(out, actual, expected, expected_state)


if __name__ == "__main__":
    unittest.main()
