from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from verifix.cli import app


runner = CliRunner()


BUGGY_SOURCE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


BUGGY_TESTS = """from buggy import find_max


def test_fail_one():
    assert find_max([3, 1, 2]) == 3


def test_fail_two():
    assert find_max([5, 4, 3]) == 5


def test_pass_one():
    assert find_max([7]) == 7


def test_pass_two():
    assert find_max([2, 2, 2]) == 2
"""


def _write_repairable_project(root: Path) -> tuple[Path, Path]:
    project = root / "repairable"
    project.mkdir(parents=True)
    buggy_file = project / "buggy.py"
    buggy_file.write_text(BUGGY_SOURCE, encoding="utf-8")
    (project / "test_buggy.py").write_text(BUGGY_TESTS, encoding="utf-8")
    return project, buggy_file


def _write_unfixable_project(root: Path) -> tuple[Path, Path]:
    project = root / "unfixable"
    project.mkdir(parents=True)
    buggy_file = project / "buggy.py"
    buggy_file.write_text("def impossible(x):\n    return 0\n", encoding="utf-8")
    (project / "test_buggy.py").write_text(
        """from buggy import impossible


def test_fail_only():
    assert impossible(1) == 42
""",
        encoding="utf-8",
    )
    return project, buggy_file


def _write_config(path: Path, *, iterations: int, verbose: bool = False) -> None:
    payload = {
        "mcts_iterations": iterations,
        "mcts_max_depth": 1,
        "mcts_time_budget_seconds": 20.0,
        "max_validations": 100,
        "max_candidates_per_node": 4,
        "fl_top_n_lines": 1,
        "fl_algorithm": "ochiai",
        "test_timeout_seconds": 5.0,
        "working_dir": str(path.parent / "work"),
        "python_executable": "c:/Users/sunil/OneDrive/Desktop/VeriFIX/.venv/Scripts/python.exe",
        "verbose": verbose,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_cli_config_show_prints_yaml_output() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "mcts_iterations" in result.output
    assert "fl_algorithm" in result.output


def test_cli_config_init_creates_config_yaml(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        created = Path("config.yaml")
        assert created.exists()


def test_cli_repair_returns_zero_on_repaired_bug(tmp_path: Path) -> None:
    project, buggy_file = _write_repairable_project(tmp_path)
    config_path = tmp_path / "config_success.yaml"
    _write_config(config_path, iterations=50)

    result = runner.invoke(
        app,
        [
            "repair",
            "--file",
            str(buggy_file),
            "--project-root",
            str(project),
            "--failing-tests",
            "test_buggy.py::test_fail_one,test_buggy.py::test_fail_two",
            "--passing-tests",
            "test_buggy.py::test_pass_one,test_buggy.py::test_pass_two",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "PATCH #1" in result.output


def test_cli_repair_returns_one_on_unfixable_bug(tmp_path: Path) -> None:
    project, buggy_file = _write_unfixable_project(tmp_path)
    config_path = tmp_path / "config_fail.yaml"
    _write_config(config_path, iterations=10)

    result = runner.invoke(
        app,
        [
            "repair",
            "--file",
            str(buggy_file),
            "--project-root",
            str(project),
            "--failing-tests",
            "test_buggy.py::test_fail_only",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1


def test_cli_repair_verbose_includes_progress_output(tmp_path: Path) -> None:
    project, buggy_file = _write_repairable_project(tmp_path)
    config_path = tmp_path / "config_verbose.yaml"
    _write_config(config_path, iterations=50, verbose=True)

    result = runner.invoke(
        app,
        [
            "repair",
            "--file",
            str(buggy_file),
            "--project-root",
            str(project),
            "--failing-tests",
            "test_buggy.py::test_fail_one,test_buggy.py::test_fail_two",
            "--passing-tests",
            "test_buggy.py::test_pass_one,test_buggy.py::test_pass_two",
            "--config",
            str(config_path),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Building AST" in result.output


def test_cli_repair_v3_help_is_available() -> None:
    result = runner.invoke(app, ["repair-v3", "--help"])
    assert result.exit_code == 0
    assert "rollout-mode" in result.output
