import hashlib
import json

import numpy as np

from decoder import model
from decoder.benchmark import model_weights
from diagnostics.stack_crossover import analyze, capture, run


def test_cpu_artifacts_round_trip(tmp_path):
    out = tmp_path / "replay.json"
    report = run(6, 20261119, out, methods=("cpu",), config=model.Config.compact())
    assert report == json.loads(out.read_text())
    archive = out.with_suffix(".npz")
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report["capture_sha256"]
    with np.load(archive, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["cpu__moe"], data["residual_cpu__heads_cpu"])
        np.testing.assert_array_equal(data["cpu__routes"], data["cpu__replay_routes"])
        assert data["cpu__residual"].dtype == np.float16
        assert json.loads(data["metadata"].item())["length"] == 6


def test_residual_and_heads_are_crossed_independently():
    config = model.Config.compact()
    common, first = capture("cpu", config, 6, 20261118)
    _, second = capture("cpu", config, 6, 20261119)
    rows, actual = analyze(config, common, {"cpu": first, "second": second})
    assert len(rows["crossed_cpu_finish"]) == 4
    channels = rows["channels"]
    assert channels
    for method, observed in (("cpu", first), ("second", second)):
        stats = rows["expert_precision"][method]
        assert stats["fp64_down_vs_observed_at_channels"]["size"] == len(channels) * config.top_k
        assert stats["replayed_routes_match"]
        expected = observed["down"][0][:, channels].astype(float) * observed["route_weights"][0, :, None]
        np.testing.assert_array_equal(stats["signed_expert_contributions"], expected)
    for r_name, h_name, r, h in (("cpu", "second", first, second), ("second", "cpu", second, first)):
        _, _, expected, _ = model.finish_layer(np, config, model_weights(config)[1], common["completed"],
                                               r["residual"], h["heads"], common["hist_a"], common["hist_e"])
        np.testing.assert_array_equal(actual[f"residual_{r_name}__heads_{h_name}"], expected[:, 0])
    assert not np.array_equal(actual["residual_cpu__heads_second"], actual["residual_second__heads_cpu"])
