"""Control reports require full independent sessions and all registered observables."""
import json
from itertools import count

import numpy as np
import pytest

from bench.protocol import put_json
from diagnostics import precision_study as study
from diagnostics.precision_model import CONTROL, PRECISION_CONTRACT


class Stub:
    def __init__(self, method, seed, fail=False):
        self.method = method
        self.seed = seed
        self.fail = fail

    def reset(self):
        pass

    def run(self):
        pass

    def output_numpy(self):
        outputs = {name: np.zeros((1, 1, 2), dtype=np.float16) for name in study.FIELDS}
        for name in ("layer10_route_ids", "layer11_route_ids"):
            outputs[name] = np.array([[self.seed % 384]], dtype=np.int32)
        if self.fail and self.method in ("staged", "single"):
            outputs["layer11_post_moe"] += 1
        return outputs

    def close(self):
        pass


def build_spec(name):
    return dict(schema=study.SCHEMA, control=CONTROL, contract=PRECISION_CONTRACT,
                session=name, machine={"sources_sha256": "test-history"},
                control_sha256="test-control", trials=study.MIN_REPEATS,
                fixtures=study.FIXTURES, seed=study.SEED, warmups=study.WARMUPS)


def test_failed_control_keeps_all_preflight_failures_but_no_timings(tmp_path):
    def prepare(method, **args):
        return Stub(method, args["seed"], fail=True)
    case = study.STACK[0]
    assert not study.measure(tmp_path, case, build_spec("a"), prepare)
    rows = json.loads((tmp_path/(case.name+".failures.json")).read_text())["failures"]
    assert {(r["seed"], r["method"]) for r in rows} == {
        (study.SEED+i, method) for i in range(2) for method in ("staged", "single")}
    assert all(r["comparison"]["failed_observables"] == ["layer11_post_moe"] for r in rows)
    assert not list(tmp_path.glob("*.csv"))


def test_report_requires_matching_sessions_and_complete_checked_cases(tmp_path, monkeypatch):
    # The synthetic runner does no work and can finish within a single clock tick.
    monkeypatch.setattr(study.time, "perf_counter_ns", count(0, 1000).__next__)
    def prepare(method, **args):
        return Stub(method, args["seed"])
    for name in study.SESSIONS:
        spec = build_spec(name)
        folder = tmp_path/name
        folder.mkdir()
        put_json(folder/"manifest.json", {"status":"complete", "spec":spec,
                                        "environment":{"gpu_probe":{"passed":True}}})
        for case in study.STACK:
            assert study.measure(folder, case, spec, prepare)

    manifest = tmp_path/"b/manifest.json"
    bad = json.loads(manifest.read_text())
    bad["spec"]["control_sha256"] = "stale"
    put_json(manifest, bad)
    with pytest.raises(ValueError, match="differ"):
        study.report(tmp_path)
    bad["spec"] = build_spec("b")
    put_json(manifest, bad)
    study.report(tmp_path)
    assert "FP32" in (tmp_path/"report.md").read_text()

    proof_path = tmp_path/"a"/(study.STACK[0].name+".proof.json")
    proof = json.loads(proof_path.read_text())
    del proof["post_oracle_max_abs"]["0"]["compiled"]["summary"]
    put_json(proof_path, proof)
    with pytest.raises(ValueError, match="observables"):
        study.report(tmp_path)
    proof_path.unlink()
    with pytest.raises(FileNotFoundError):
        study.report(tmp_path)
