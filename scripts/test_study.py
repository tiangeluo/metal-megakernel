"""A failed stack replay must preserve the independent attention report."""
from pathlib import Path
import subprocess

import pytest

from scripts.study import collect


@pytest.mark.parametrize("stack_passes", [True, False])
def test_study_reports_independent_scopes(tmp_path, stack_passes):
    commands = []

    def execute(argv, *, check):
        commands.append(argv)
        failed = "bench.diagnose_stack" in argv and not stack_passes
        return subprocess.CompletedProcess(argv, int(failed))

    if stack_passes:
        collect("repeat", tmp_path, execute)
    else:
        with pytest.raises(RuntimeError, match="Attention report ready"):
            collect("repeat", tmp_path, execute)

    diagnostics = [args for args in commands if "bench.diagnose_stack" in args]
    assert len({Path(args[-1]).name for args in diagnostics}) == 4
    suites = [(a[a.index("--suite") + 1], a[a.index("--session") + 1])
              for a in commands if "bench.run" in a]
    expected = [("pilot", "repeat-pilot"), ("attention", "repeat-a"), ("attention", "repeat-b")]
    if stack_passes:
        expected += [("stack", "repeat-a"), ("stack", "repeat-b")]
    assert suites == expected
    reports = [a[a.index("--scope") + 1] for a in commands if "bench.report" in a]
    assert reports == (["attention", "full"] if stack_passes else ["attention"])
