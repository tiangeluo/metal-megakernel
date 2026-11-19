"""A hardware replication must acquire all cases into a new and exclusive output tree."""
from pathlib import Path
import subprocess

import pytest

from scripts import replicate
from scripts.study import study_lock


def test_fresh_remeasurements_bind_every_report_to_this_run(tmp_path):
    calls = []

    def record(argv, *, check, pass_fds):
        assert check and pass_fds
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    root = replicate.collect("repeat", results=tmp_path, invoke=record)
    assert root == tmp_path / "replications/repeat"
    acquisition = [c for c in calls if "bench.run" in c]
    assert [c[c.index("--session")+1] for c in acquisition] == list(replicate.ATTENTION_SESSIONS)
    assert all(c[c.index("--out")+1] == str(root / "attention") for c in acquisition)
    assert all(c[c.index("--environment")+1].startswith(str(root / "attention") + "/") for c in acquisition)
    attention_report, = [c for c in calls if "bench.report" in c]
    assert attention_report[attention_report.index("--input")+1] == str(root / "attention")
    assert attention_report[attention_report.index("--out")+1] == str(root / "attention-report")
    precision = [c for c in calls if "diagnostics.precision_study" in c]
    assert all(c[c.index("--out")+1] == str(root / "precision-control") for c in precision)
    assert [c[-1] for c in precision] == ["a", "b", "--report-only"]
    assert calls[-1][-3:] == ["analysis.decoder_performance", "--replication", str(root)]
    before = list(calls)
    with pytest.raises(ValueError, match="already exists"):
        replicate.collect("repeat", results=tmp_path, invoke=record)
    assert calls == before


def test_concurrent_study_refused_before_output_creation(tmp_path):
    with study_lock(tmp_path), pytest.raises(RuntimeError, match="already running"):
        replicate.collect(results=tmp_path)
    assert not (tmp_path / "replications").exists()


def test_traces_use_the_replications_own_environment_and_exports(tmp_path, monkeypatch):
    from scripts import capture_traces

    calls = []

    def record(argv, **kwargs):
        calls.append(argv)
        if "scripts/environment.py" in argv:
            environment = Path(argv[argv.index("--out") + 1])
            environment.parent.mkdir(parents=True, exist_ok=True)
            environment.write_text("{}")
        return subprocess.CompletedProcess(argv, 0)

    def capture(folder, environment):
        assert folder.parent / "environment.json" == environment
        assert environment.is_file() and folder.is_dir()
        return True

    monkeypatch.setattr(capture_traces, "capture", capture)
    root = replicate.collect(traces=True, results=tmp_path, invoke=record)
    assert calls[-1][1:] == [
        "-m", "analysis.inspect_trace", str(root / "traces/capture"), "--save-exports",
        str(root / "traces/xml"), "--out", str(root / "traces/observations.json"),
    ]
