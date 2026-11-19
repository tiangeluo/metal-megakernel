"""Exercise the actual Metal kernels, with explicit skips only for missing GPU access."""

import sys
import unittest

import numpy as np

from native.runner import METHODS, NativeRunner
from reference.attention import empty_state, step
from reference.fixtures import make_case, make_token, make_weights


@unittest.skipUnless(sys.platform == "darwin", "native Metal tests require macOS")
class MetalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        token, weights, state = make_case(q_heads=6, kv_heads=1, dim=7, length=2)
        try:
            NativeRunner(token, weights, state, position=1).close()
        except RuntimeError as exc:
            if "MTLCreateSystemDefaultDevice returned nil" in str(exc):
                raise unittest.SkipTest(str(exc)) from exc
            raise

    def verify(self, runner, output, expected, initial, position):
        actual, state = runner.outputs()
        np.testing.assert_allclose(actual, output, atol=2e-3, rtol=1e-2)
        for name in ("hist_k", "hist_v"):
            np.testing.assert_array_equal(state[name], expected[name])
        for name in ("cache_k", "cache_v"):
            slot = position % state[name].shape[2]
            np.testing.assert_allclose(
                state[name][:, :, slot], expected[name][:, :, slot], atol=2e-3, rtol=2e-3
            )
            np.testing.assert_array_equal(state[name][:, :, :slot], initial[name][:, :, :slot])
            np.testing.assert_array_equal(
                state[name][:, :, slot + 1 :], initial[name][:, :, slot + 1 :]
            )

    def test_grouped_decode_and_wrap(self):
        for length, window, dynamic in [(1, 0, True), (65, 0, True), (1025, 1024, False)]:
            token, weights, before = make_case(
                batch=2,
                q_heads=12,
                kv_heads=2,
                dim=128,
                length=length,
                window=window,
                dynamic=dynamic,
            )
            token["sink"][0] = np.float16(-80)
            token["sink"][1] = np.float16(80)
            token["q"] *= np.float16(6)
            token["r"] *= np.float16(6)
            expected = {k: v.copy() for k, v in before.items()}
            output = step(
                token, weights, expected, position=length - 1, window=window, dynamic=dynamic
            )
            with NativeRunner(
                token, weights, before, position=length - 1, window=window, dynamic=dynamic
            ) as runner:
                for method in METHODS:
                    with self.subTest(length=length, method=method):
                        runner.reset()
                        runner.run(method)
                        self.verify(runner, output, expected, before, length - 1)

    def test_consecutive_steps_cross_conv_and_window_boundaries(self):
        rng = np.random.default_rng(141)
        tokens = [make_token(rng, 1, 6, 1, 13, 5) for _ in range(13)]
        weights = make_weights(rng, 6, 1, 13, 5, 4)
        for dynamic, window in [(False, 5), (True, 0)]:
            initial = empty_state(1, 1, window or len(tokens), 13)
            with NativeRunner(
                tokens[0], weights, initial, position=0, window=window, dynamic=dynamic, tile=3
            ) as runner:
                for method in METHODS:
                    expected = {k: v.copy() for k, v in initial.items()}
                    runner.reset()
                    for pos, tok in enumerate(tokens):
                        with self.subTest(dynamic=dynamic, method=method, position=pos):
                            # Entries outside the new slot must equal prior GPU storage exactly.
                            _, before = runner.outputs() if pos else (None, initial)
                            out = step(
                                tok, weights, expected, position=pos, window=window, dynamic=dynamic
                            )
                            runner.set_token(tok)
                            runner.set_position(pos)
                            runner.run(method)
                            self.verify(runner, out, expected, before, pos)


if __name__ == "__main__":
    unittest.main()
