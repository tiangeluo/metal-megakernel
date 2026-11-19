"""Independent scalar or time-major checks for the connected benchmark slice."""

from unittest import TestCase

import numpy as np

from decoder.benchmark import model_weights, numpy_attention, prepare_case
from decoder.model import (
    Config,
    activation,
    build_decoder,
    depth_read,
    make_state,
    moe,
    normal,
    output_conv,
    prepare_token,
    project,
    route,
)
from reference.prefill import prefill


class DecoderTests(TestCase):
    def test_depth_reads_are_normalized_and_ordered(self):
        rng = np.random.default_rng(45)
        b, w = 2, 8
        completed = normal(rng, (b, 2, w))
        partial = normal(rng, (b, w), 5)
        queries = normal(rng, (3, w), 2)
        gains = normal(rng, (3, w), 0.2) + 1
        actual = depth_read(np, completed, partial, queries, gains)
        expected = []
        for row in range(b):
            values = np.vstack([completed[row], partial[row]]).astype(np.float64)
            values /= np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + 1e-6)
            reads = []
            for i, query in enumerate(queries):
                logits = np.array(
                    [np.dot(query, value) / np.sqrt(w) for value in values]
                )
                probs = np.exp(logits - logits.max())
                mixed = (probs / probs.sum()) @ values
                reads.append(gains[i] * mixed / np.sqrt(np.mean(mixed**2) + 1e-6))
            expected.append(reads)
        np.testing.assert_allclose(actual, np.array(expected), rtol=0.001, atol=0.0005)
        self.assertGreater(float(np.max(np.abs(actual[:, 0] - actual[:, 1]))), 0.01)

    def test_router_bias_only_selects_and_ties_use_smaller_id(self):
        x = np.ones((1, 1), np.float16)
        router = np.array([[0.0, 0.0, 2.0, -2.0]], np.float32)
        ids, weights = route(np, x, router, np.array([0, 0, 0, 2], np.float32), 3)
        np.testing.assert_array_equal(ids, [[3, 2, 0]])
        base = 1 / (1 + np.exp(-np.array([-2, 2, 0], np.float64)))
        np.testing.assert_allclose(weights[0], 8 * base / base.sum(), rtol=1e-6)
        self.assertEqual(float(weights.sum()), 8.0)

    def test_situ_uses_raw_gate_in_sigmoid(self):
        gate = np.array([8.0, -8.0, 0.0], dtype=np.float16)
        up = np.array([45.0, 45.0, 45.0], dtype=np.float16)
        expected = (
            4
            * np.tanh(gate.astype(float) / 4)
            / (1 + np.exp(-gate.astype(float)))
            * 25
            * np.tanh(up.astype(float) / 25)
        )
        np.testing.assert_allclose(
            activation(np, np.r_[gate, up]), expected, rtol=0.0007, atol=0.00001
        )

    def test_output_convolution_matches_temporal_filter(self):
        rng = np.random.default_rng(3)
        xs = normal(rng, (8, 2, 9))
        taps = normal(rng, (4, 9))
        hist = np.zeros((2, 3, 9), np.float16)
        for t, x in enumerate(xs):
            actual, hist = output_conv(np, x, hist, taps)
            expected = x.astype(float) + sum(
                taps[lag].astype(float) * xs[t - lag] for lag in range(min(4, t + 1))
            )
            np.testing.assert_allclose(
                actual, expected / np.sqrt(5.0), rtol=0.001, atol=0.0005
            )
            np.testing.assert_array_equal(hist[:, 0], x)

    def test_moe_agrees_with_dense_expert_evaluation(self):
        c = Config.compact()
        rng = np.random.default_rng(5)
        layer = model_weights(c)[0]
        x = normal(rng, (2, c.width))
        actual, ids = moe(np, x, layer, c.top_k)
        for b in range(2):
            logits = np.einsum("w,we->e", x[b], layer["router"], dtype=np.float64)
            prob = 1 / (1 + np.exp(-logits))
            top = np.argsort(-(prob + layer["bias"]), kind="stable")[: c.top_k]
            np.testing.assert_array_equal(ids[b], top)
            # Dense reference evaluates every expert, then selects and combines eight.
            gu = np.einsum(
                "w,ewf->ef", x[b], layer["gate_up"], dtype=np.float64
            ).astype(np.float16)
            g, u = np.split(gu.astype(float), 2, axis=-1)
            hidden = (
                4 * np.tanh(g / 4) / (1 + np.exp(-g)) * 25 * np.tanh(u / 25)
            ).astype(np.float16)
            outs = np.einsum(
                "ef,efw->ew", hidden, layer["down"], dtype=np.float64
            ).astype(np.float16)
            expected = np.einsum(
                "e,ew->w", 8 * prob[top] / prob[top].sum(), outs[top].astype(float)
            )
            np.testing.assert_allclose(actual[b], expected, atol=0.0025, rtol=0.0025)

    def test_two_layer_sequence_against_dense_prefill(self):
        c = Config.compact()
        layers = model_weights(c)
        rng = np.random.default_rng(517)
        t, b = 9, 2
        completed = normal(rng, (t, b, 2, c.width))
        original = normal(rng, (t, b, c.width))
        # Full-sequence reference computes the first layer for every token, then
        # the second, with direct temporal convolution and independent dense attention.
        partial = original
        expected_cache = []
        expected_ids = []
        for i, layer in enumerate(layers):
            tokens = [
                prepare_token(np, c, layer, completed[j], partial[j], i == 1)
                for j in range(t)
            ]
            heads, kv = prefill(
                tokens, layer["attention"], window=0 if i else c.window, dynamic=i == 1
            )
            expected_cache.append(kv)
            proj = project(np, heads.reshape(t, b, -1), layer["out"])

            def conv_all(x, taps):
                arr = x.astype(np.float64)
                result = arr.copy()
                for lag in range(min(4, len(arr))):
                    result[lag:] += arr[: len(arr) - lag] * taps[lag].astype(np.float64)
                return (result / np.sqrt(5)).astype(np.float16)

            after = partial.astype(np.float32) + conv_all(proj, layer["post_attn"])
            expert_rows = []
            routes = []
            for j in range(t):
                src = depth_read(
                    np, completed[j], after[j], layer["moe_read"], layer["moe_norm"]
                )[:, 0]
                value, ids = moe(np, src, layer, c.top_k)
                routes.append(ids)
                expert_rows.append(value)
            partial = (
                after + conv_all(np.array(expert_rows), layer["post_moe"])
            ).astype(np.float16)
            expected_ids.append(np.stack(routes))
        states = [make_state(c, rng, b, t, 0, global_layer=i == 1) for i in range(2)]
        for j in range(t):
            steps = [numpy_attention(j, c.window, False), numpy_attention(j, 0, True)]
            out = build_decoder(np, c, steps)(layers, completed[j], original[j], states)
            np.testing.assert_allclose(
                out["summary"], partial[j], atol=0.012, rtol=0.003
            )
            for i in range(2):
                prefix = f"layer{i + 10}_"
                np.testing.assert_array_equal(
                    out[prefix + "route_ids"], expected_ids[i][j]
                )
                for key in states[i]:
                    states[i][key] = out[prefix + key]
                cap = states[i]["cache_k"].shape[2]
                for key in ("k", "v"):
                    for pos in range(max(0, j + 1 - cap), j + 1):
                        np.testing.assert_allclose(
                            states[i]["cache_" + key][:, :, pos % cap],
                            expected_cache[i][key][pos],
                            atol=0.003,
                            rtol=0.003,
                        )

    def test_case_reset_and_observables(self):
        c = Config.compact()
        self.assertEqual(Config().expert_bytes, 3057647616)
        case = prepare_case("cpu", 64, batch=2, seed=9, config=c)
        case.run()
        before = {k: v.copy() for k, v in case.output_numpy().items()}
        self.assertEqual(len(before), 15)
        case.reset()
        case.run()
        after = case.output_numpy()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
        other = prepare_case("cpu", 64, batch=2, seed=12, config=c)
        other.run()
        self.assertFalse(
            np.array_equal(
                before["layer11_route_ids"], other.output_numpy()["layer11_route_ids"]
            )
        )
