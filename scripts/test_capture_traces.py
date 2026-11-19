import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from scripts import capture_traces as traces


def test_dry_run_targets_four_bounded_child_processes(tmp_path, capsys):
    output = tmp_path / "traces"
    env = tmp_path / "host-probe.json"
    with patch("sys.argv", ["capture", "--out", str(output), "--environment", str(env), "--dry-run"]):
        assert traces.main() == 0
    commands = json.loads(capsys.readouterr().out)
    assert [cmd[cmd.index("--method") + 1] for cmd in commands] == list(traces.METHODS)
    for cmd in commands:
        assert cmd[cmd.index("--launch") + 1:cmd.index("--launch") + 3] == ["--", traces.sys.executable]
        assert "--all-processes" not in cmd
        assert cmd[cmd.index("--time-limit") + 1] == "30s"
        assert cmd[cmd.index("--environment") + 1] == str(env)
        assert Path(cmd[cmd.index("--output") + 1]).parent == output
    assert not output.exists()


def test_trace_bundle_needs_completed_target_and_inspection(tmp_path):
    def recorder(args, **kwargs):
        method = args[args.index("--method") + 1]
        Path(args[args.index("--output") + 1]).mkdir()
        if method != "hybrid":
            Path(args[args.index("--target-stdout") + 1]).write_text(
                f"Captured candidate interval: {traces.CASE}/{method}\n")
        return subprocess.CompletedProcess(args, 0)

    with patch.object(traces, "catalog_has", return_value=True), patch.object(subprocess, "run", recorder):
        assert not traces.capture(tmp_path, tmp_path / "environment.json")
    report = json.loads((tmp_path / "capture.json").read_text())
    assert report["status"] == "incomplete"
    assert report["captures"]["hybrid"]["status"] == "incomplete"
    assert report["captures"]["single"]["status"] == "awaiting_inspection"
