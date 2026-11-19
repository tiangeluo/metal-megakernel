import unittest

import numpy as np
from numpy.testing import assert_allclose as numpy_assert_allclose, assert_array_equal

from reference.attention import clip, empty_state, shortconv, step
from reference.fixtures import make_case, make_token, make_weights
from reference.prefill import prefill


def assert_allclose(*args, **kwargs):
    return numpy_assert_allclose(*args, equal_nan=False, **kwargs)


class AttentionContract(unittest.TestCase):
    def check_trajectory(self, *, length, window, dynamic, positions=None):
        rng = np.random.default_rng(72)
        b, q, h, d, r = 2, 4, 2, 5, 3
        w = make_weights(rng, q, h, d, rel_dim=r, extent=1024)
        tokens = [make_token(rng, b, q, h, d, r) for _ in range(length)]
        positions = list(range(length)) if positions is None else list(positions)
        dense, kv = prefill(tokens, w, window=window, dynamic=dynamic, positions=positions)
        capacity = min(length, window) if window else length
        state = empty_state(b, h, capacity, d)
        for t, token in enumerate(tokens):
            out = step(token, w, state, position=t, window=window, dynamic=dynamic)
            if t in positions:
                assert_allclose(out, dense[positions.index(t)], atol=2e-3, rtol=1e-2)
                for key in ("k", "v"):
                    indices = np.arange(max(0, t + 1 - capacity), t + 1)
                    actual = state["cache_"+key][:, :, indices % capacity]
                    assert_allclose(actual, kv[key][indices].transpose(1, 2, 0, 3),
                                    atol=2e-3, rtol=2e-3)
                    for lag in range(3):
                        expected = tokens[t-lag][key] if t >= lag else np.zeros_like(token[key])
                        assert_array_equal(state["hist_"+key][:, :, lag], expected)

    def test_global_consecutive_steps_against_dense(self):
        self.check_trajectory(length=11, window=0, dynamic=True)

    def test_local_repeated_ring_wrap_against_dense(self):
        self.check_trajectory(length=17, window=5, dynamic=False)

    def test_true_window_and_relative_extent_boundaries(self):
        for window, dynamic in ((1024, False), (0, True)):
            with self.subTest(window=window):
                self.check_trajectory(length=1026, window=window, dynamic=dynamic,
                                      positions=(0, 1, 2, 1022, 1023, 1024, 1025))

    def test_old_global_key_still_participates_beyond_bias_extent(self):
        # Three positions, extent two. The oldest (value 4) gets no relative term.
        for window, expected in ((0, 4/(2+1+1+1)), (2, 0)):
            token, weights, state = make_case(length=3, q_heads=1, kv_heads=1,
                dim=1, rel_dim=1, extent=2, window=window)
            for x in token.values():
                x.fill(0)
            for x in state.values():
                x.fill(0)
            token["r"].fill(1)
            token["gate"].fill(100)
            weights["r_scale"].fill(1)
            weights["rel"][:] = [[np.log(2)], [0]]
            if not window:
                state["cache_v"][0,0,0,0] = 4
            out = step(token, weights, state, position=2, window=window,
                       dynamic=not bool(window))
            assert_allclose(out.item(), expected, atol=0.001)

    def test_causal_tap_order_and_scale(self):
        x = np.full((1,1,1), 7, np.float16)
        hist = np.array([3, 2, 1], np.float16).reshape(1,1,3,1)
        w = np.array([1, 2, 3, 4], np.float16).reshape(4,1,1)
        static = shortconv(x, hist, w)
        dynamic = shortconv(x, hist, w, np.array([[[2, 0, 0, -2]]], np.float16))
        self.assertEqual(static.item(), np.float16((7 + 7 + 6 + 6 + 4)/np.sqrt(5)))
        self.assertEqual(dynamic.item(), np.float16((7 + 7 + 6 + 6 + 4 + 14 - 2)/3))

    def test_zero_sink_and_extreme_gates(self):
        token, w, state = make_case(q_heads=2, kv_heads=1, dim=2, length=1)
        for x in token.values():
            x.fill(0)
        w["wv"].fill(0)
        token["v"][:] = 3
        token["gate"][:] = [[[1000, -1000], [1000, 1000]]]
        token["sink"][:] = [[0, 1000]]
        out = step(token, w, state, position=0)
        assert_allclose(out, [[[0.5, 0], [0, 0]]], atol=0.001)
        self.assertTrue(np.isfinite(out).all())

    def test_query_scales_clip_before_dot(self):
        inp = np.array([[5, -1, 0], [60, 80, 0]], np.float32)
        res = clip(inp, 10)
        assert_array_equal(res, [[5, -1, 0], [6, 8, 0]])
        token, w, state = make_case(q_heads=1, kv_heads=1, dim=2, rel_dim=1,
                                    extent=1, length=1)
        for a in token.values():
            a.fill(0)
        for a in w.values():
            a.fill(0)
        token["gate"].fill(100)
        token["q"][:] = [1000, 20]
        token["k"][:] = [100, 0]
        token["v"].fill(3)
        w["q_scale"][:] = [0, 2]
        # Q direction preserved only if channel scales act before clipping.
        token["sink"].fill(0)
        out = step(token, w, state, position=0)
        assert_allclose(out, 0.5, atol=0.001)

    def test_local_capacity_error_leaves_state_intact(self):
        token, w, state = make_case(q_heads=1, kv_heads=1, dim=3, length=2)
        before = {k: a.copy() for k, a in state.items()}
        with self.assertRaises(ValueError):
            step(token, w, state, position=2, window=3, dynamic=False)
        for k in state:
            assert_array_equal(state[k], before[k])

    def test_fixture_reproducibility_and_untouched_slots(self):
        params = dict(batch=2, q_heads=6, kv_heads=2, dim=4, length=9, window=5)
        ta, wa, sa = make_case(**params)
        tb, wb, sb = make_case(**params)
        for left, right in ((ta,tb), (wa,wb), (sa,sb)):
            for k in left:
                assert_array_equal(left[k], right[k])
        step(ta, wa, sa, position=8, window=5, dynamic=False)
        frozen = np.arange(5) != 8 % 5
        for k in ("cache_k","cache_v"):
            assert_array_equal(sa[k][:,:,frozen], sb[k][:,:,frozen])

    def test_large_position_temperature_for_global_only(self):
        token, w, state = make_case(q_heads=1,kv_heads=1,dim=1,rel_dim=1,
                                    extent=1,length=16384,seed=1)
        # Only a single token contributes appreciable value, and the last token
        # has content logit zero. Its relative bias alone tests tau on R.
        for x in token.values():
            x.fill(0)
        for x in state.values():
            x.fill(0)
        token["r"].fill(1)
        token["gate"].fill(1000)
        token["v"].fill(3)
        token["sink"].fill(-1000)
        w["wv"].fill(0)
        w["r_scale"].fill(1)
        w["rel"].fill(8)
        actual = step(token,w,state,position=16383)
        tau = 1 + 0.1 * np.log(2)
        relative_logit = np.float16(np.float16(tau) * 8)
        expected = 1/(1 + 16383*np.exp(-float(relative_logit)))
        assert_allclose(actual,expected,atol=0.001)

    def test_published_geometry(self):
        t, w, state = make_case(length=64)
        before = {name: x.copy() for name, x in state.items()}
        y = step(t, w, state, position=63)
        self.assertEqual(y.shape, (1,48,128))
        self.assertTrue(np.isfinite(y).all())
        for name in ("cache_k","cache_v"):
            assert_array_equal(state[name][:,:,:63],before[name][:,:,:63])
        assert_array_equal(state["hist_v"][:,:,0],t["v"])


if __name__ == "__main__":
    unittest.main()
