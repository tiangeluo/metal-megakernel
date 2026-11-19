from dataclasses import replace

import numpy as np
import pytest

from bench.protocol import equal_stack
from decoder import model
from diagnostics import precision_model as control


def test_moe_matches_independent_expert_equations():
    rng = np.random.default_rng(327)
    config = replace(model.Config.compact(), experts=11, top_k=3)
    layer = model.make_layer(config, rng, global_layer=True)
    # Exercise nontrivial batch and routing, including indices shared across batches.
    x = rng.standard_normal((3, config.width)).astype(np.float32)
    values, ids = control.moe(np, x, layer, config.top_k)
    scores = 1 / (1 + np.exp(-x.astype(np.float64) @ layer["router"].astype(np.float64)))
    order = np.argsort(-(scores + layer["bias"]), axis=1, kind="stable")[:, :config.top_k]
    np.testing.assert_array_equal(ids, order)
    expected = np.zeros(x.shape)
    for b, row in enumerate(order):
        weights = 8 * scores[b, row] / scores[b, row].sum()
        for weight, expert in zip(weights, row):
            gate, up = np.split(x[b].astype(np.float64) @ layer["gate_up"][expert].astype(np.float64), 2)
            active = 4 * np.tanh(gate/4) / (1 + np.exp(-gate)) * 25 * np.tanh(up/25)
            expected[b] += weight * (active @ layer["down"][expert].astype(np.float64))
    np.testing.assert_allclose(values, expected.astype(np.float16), atol=5e-4, rtol=5e-4)
    assert values.dtype == np.float16


def test_depth_read_retains_unrounded_source_and_original_cast():
    rng = np.random.default_rng(17)
    c = replace(model.Config.compact(), experts=12)
    layer = model.make_layer(c, rng, global_layer=True)
    complete = model.normal(rng, (2, 2, c.width))
    residual = model.normal(rng, (2, c.width), dtype=np.float32)
    source = control.expert_source(np, complete, residual, layer)
    original = model.depth_read(np, complete, residual, layer['moe_read'], layer['moe_norm'])[:, 0]
    assert source.dtype == np.float32
    np.testing.assert_array_equal(source.astype(np.float16), original)
    assert np.any(source != original.astype(np.float32))


def test_connected_control_reset_and_outputs():
    c = model.Config.compact()
    case = control.prepare_case('cpu', 9, batch=2, seed=113, config=c)
    try:
        case.run()
        original = {k: v.copy() for k, v in case.output_numpy().items()}
        assert len(original) == 15
        for layer in case.layers:
            assert layer['gate_up'].dtype == layer['down'].dtype == np.float16
        case.reset()
        case.run()
        equal_stack(original, case.output_numpy(), 8)
    finally:
        case.close()


def test_gpu_matvec_irregular_and_batch():
    from mlx_backend import require_gpu
    from scripts.environment import probe
    if not probe().get("passed"):
        pytest.skip("Metal is unavailable to this process")
    mx = require_gpu()
    rng = np.random.default_rng(7)
    ids = np.array([[2, 5, 0], [0, 3, 1]], np.uint32)
    bank = model.normal(rng, (6, 37, 45))
    for shape in [(2, 37), (2, 3, 37)]:
        x = model.normal(rng, shape, dtype=np.float32)
        actual = np.asarray(control.selected_project(mx, mx.array(x), mx.array(bank), mx.array(ids)))
        expected = control.selected_project(np, x, bank, ids)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
