import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bench.diagnose_stack import compare_outputs, diagnose
from bench.protocol import Case, equal_stack
from bench.run import ParityError, check_step, measure


def fixture():
    return {
        "layer10_cache_k": np.zeros((1, 1, 3, 2), dtype=np.float16),
        "layer11_route_ids": np.array([[5, 6]], dtype=np.uint32),
        "layer11_post_moe": np.zeros((1, 3, 2), dtype=np.float16),
    }


def test_inspects_all_states_and_uses_the_certification_tolerances():
    expected, actual = fixture(), fixture()
    actual["layer10_cache_k"][0, 0, 0, 0] = 0.00001  # An untouched cache entry.
    actual["layer10_cache_k"][0, 0, 1, 0] = 0.001    # The written slot.
    actual["layer11_route_ids"][0] = [6, 5]
    actual["layer11_post_moe"][0, 0, 1] = -0.0037
    result = compare_outputs(expected, actual, 1)
    assert not result["passed"]
    assert result["failed_observables"] == list(expected)
    for key in expected:
        with pytest.raises(AssertionError):
            equal_stack({key: expected[key]}, {key: actual[key]}, 1)
    details = result["observables"]
    assert details["layer10_cache_k"]["regions"]["before slot"]["violations"] == 1
    assert details["layer10_cache_k"]["regions"]["written slot"]["violations"] == 0
    assert details["layer11_route_ids"]["actual"] == [[6, 5]]
    worst = details["layer11_post_moe"]["regions"]["all"]["worst_violation"]
    assert worst["index"] == [0, 0, 1]


def test_records_cross_schedule_comparisons_and_raw_failed_readback(tmp_path):
    case = Case("stack", 1, 64)
    original = fixture()
    original["layer11_post_moe"][0, 1, 1] = 1
    path = tmp_path / "diagnosis.json"
    with patch("bench.diagnose_stack.read_method", side_effect=lambda *args: fixture()):
        report = diagnose(case, 111, {}, path, original=({"method": "staged"}, original))
    assert report == json.loads(path.read_text())
    assert report["passed"]
    assert report["original_failure"]["vs_cpu"]["failed_observables"] == ["layer11_post_moe"]
    assert report["comparisons"]["single"]["vs_compiled"]["passed"]
    assert set(report["comparisons"]) == set(case.methods)


def test_shared_device_drift_is_visible_and_still_fails_cpu_gate(tmp_path):
    def read(_case, _seed, method):
        data = fixture()
        if method != "cpu":
            data["layer11_post_moe"][0, 0, 0] = -0.0037
        return data
    with patch("bench.diagnose_stack.read_method", side_effect=read), \
            patch("diagnostics.stack_stages.isolate", return_value={}) as isolate:
        case = Case("stack", 1, 64)
        report = diagnose(case, 222, {}, tmp_path / "report.json")
    isolate.assert_called_once_with(case, 222, "compiled")
    assert not report["passed"]
    assert report["comparisons"]["staged"]["vs_compiled"]["passed"]
    assert report["comparisons"]["staged"]["vs_cpu"]["failed_observables"] == ["layer11_post_moe"]


def test_benchmark_error_retains_context_and_remains_a_failure_after_replay(tmp_path):
    case = Case("stack", 1, 64)
    runner = Mock()
    actual = fixture()
    runner.read.return_value = actual
    def fail(_):
        raise AssertionError("bad output")
    with pytest.raises(ParityError) as caught:
        check_step(case, 222, 1, "staged", "after timing", runner, fail)
    failure = caught.value
    assert failure.context == dict(case=case.name, seed=222, fixture=1,
                                   method="staged", phase="after timing")
    actual["layer11_route_ids"][0] = 0
    assert failure.readback["layer11_route_ids"].tolist() == [[5, 6]]
    args = SimpleNamespace(session="test", seed=221, trials=1, warmups=1)
    with patch("bench.run.run_case", side_effect=failure), patch("bench.diagnose_stack.diagnose") as replay:
        with pytest.raises(AssertionError, match="Diagnostic replay"):
            measure(case, args, {}, tmp_path)
    assert replay.call_args.args[1] == 222
    assert replay.call_args.args[3] == Path(tmp_path, "stack-b1-l64-fixture1.diagnostic.json")
