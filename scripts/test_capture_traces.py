from importlib import metadata, util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from scripts import capture_traces as traces


def test_dry_run_targets_four_bounded_child_processes(tmp_path, capsys):
    output = tmp_path / "traces"
    env = tmp_path / "host-probe.json"
    with patch("sys.argv", ["capture", "--out", str(output), "--environment", str(env), "--dry-run"]):
        assert traces.main() == 0
    commands = json.loads(capsys.readouterr().out)
    assert [cmd[cmd.index("--method") + 1] for cmd in commands] == list(traces.METHODS)
    for cmd in commands:
        assert cmd[cmd.index("--launch") + 1:cmd.index("--launch") + 5] == [
            "--", str(Path(traces.sys.executable).resolve()), "-I", "-S"]
        assert "--all-processes" not in cmd
        assert cmd[cmd.index("--time-limit") + 1] == "30s"
        assert cmd[cmd.index("--environment") + 1] == str(env)
        assert Path(cmd[cmd.index("--output") + 1]).parent == output
    assert not output.exists()


@pytest.mark.skipif(util.find_spec("mlx") is None, reason="MLX wheel required to test launcher environment")
def test_real_trace_target_with_empty_environment_and_foreign_cwd(tmp_path):
    cmd = traces.record_command("single", tmp_path, tmp_path / "environment.json")
    cmd = cmd[cmd.index("--launch") + 2:cmd.index("--case")] + ["--help"]
    sub = subprocess.run(cmd, cwd=tmp_path, env={}, capture_output=True, text=True, check=True)
    prefix = "Trace target packages: "
    packages = json.loads(next(line[len(prefix):] for line in sub.stdout.splitlines() if line.startswith(prefix)))
    assert packages == {key: metadata.version(key) for key in ("numpy", "mlx", "mlx-metal")}
    assert "--case" in sub.stdout


def test_trace_bundle_needs_completed_target_and_inspection(tmp_path):
    output = tmp_path / "capture"
    output.mkdir()
    env = tmp_path / "environment.json"
    env.write_text("{}")

    def recorder(args, **kwargs):
        if "export" in args:
            Path(args[args.index("--output") + 1]).write_text(
                '<trace-toc><run><data><table schema="os-signpost-interval"/></data></run></trace-toc>')
            return subprocess.CompletedProcess(args, 0)
        method = args[args.index("--method") + 1]
        Path(args[args.index("--output") + 1]).mkdir()
        if method != "hybrid":
            Path(args[args.index("--target-stdout") + 1]).write_text(
                f"Captured candidate interval: {traces.CASE}/{method}\n")
        else:
            Path(args[args.index("--target-stdout") + 1]).write_text(
                "Traceback (most recent call last):\nModuleNotFoundError: No module named 'mlx'\n")
        return subprocess.CompletedProcess(args, 54 if method == "hybrid" else 0)

    with patch.object(traces, "catalog_has", return_value=True), patch.object(subprocess, "run", recorder):
        assert not traces.capture(output, env)
        with pytest.raises(FileExistsError):
            traces.capture(output, env)
    report = json.loads((output / "capture.json").read_text())
    assert report["status"] == "incomplete"
    assert report["captures"]["hybrid"]["status"] == "incomplete"
    assert report["captures"]["single"]["status"] == "awaiting_inspection"
    assert report["captures"]["single"]["table_schemas"] == ["os-signpost-interval"]
    assert report["captures"]["hybrid"]["xctrace_exit"] == 54
    assert "No module named 'mlx'" in report["captures"]["hybrid"]["target_error"]
