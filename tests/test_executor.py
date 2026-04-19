from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport
from verifix.validator.executor import ConcreteValidator, ValidationBudgetExceeded, validate_patch


BUGGY_SOURCE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""

CORRECT_SOURCE = BUGGY_SOURCE.replace("<", ">")

BASE_TESTS = """from buggy import find_max


def test_fail_one():
    assert find_max([3, 1, 2]) == 3


def test_fail_two():
    assert find_max([5, 4, 3]) == 5


def test_pass_one():
    assert find_max([7]) == 7


def test_pass_two():
    assert find_max([2, 2, 2]) == 2
"""


@pytest.fixture
def fixture_project(tmp_path: Path) -> Path:
    root = tmp_path / "fixture_project"
    root.mkdir(parents=True)
    (root / "buggy.py").write_text(BUGGY_SOURCE, encoding="utf-8")
    (root / "test_buggy.py").write_text(BASE_TESTS, encoding="utf-8")
    return root


def _config(working_dir: Path, timeout: float = 5.0, max_validations: int = 100) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=2,
        working_dir=str(working_dir),
        python_executable=sys.executable,
        test_timeout_seconds=timeout,
        max_validations=max_validations,
    )


def _bug_report(project: Path, source: str = BUGGY_SOURCE, passing_tests: list[str] | None = None) -> BugReport:
    return BugReport(
        bug_id="Chart-1",
        language="python",
        buggy_source=source,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail_one", "test_buggy.py::test_fail_two"],
        passing_tests=passing_tests
        if passing_tests is not None
        else ["test_buggy.py::test_pass_one", "test_buggy.py::test_pass_two"],
        project_root=str(project),
        metadata={"benchmark": "toy"},
    )


def test_validate_patch_original_source_not_plausible(fixture_project: Path, tmp_path: Path) -> None:
    result = validate_patch(
        patched_source=BUGGY_SOURCE,
        bug_report=_bug_report(fixture_project),
        config=_config(tmp_path / "work"),
        state_id="s1",
    )

    assert result.is_plausible is False
    assert result.all_failing_tests_pass is False


def test_validate_patch_correct_source_is_plausible(fixture_project: Path, tmp_path: Path) -> None:
    result = validate_patch(
        patched_source=CORRECT_SOURCE,
        bug_report=_bug_report(fixture_project),
        config=_config(tmp_path / "work"),
        state_id="s2",
    )

    assert result.compiled is True
    assert result.is_plausible is True
    assert result.all_failing_tests_pass is True
    assert result.no_regression is True


def test_validate_patch_syntax_error_returns_compiled_false(fixture_project: Path, tmp_path: Path) -> None:
    result = validate_patch(
        patched_source="def broken(:\n    pass\n",
        bug_report=_bug_report(fixture_project),
        config=_config(tmp_path / "work"),
        state_id="s3",
    )

    assert result.compiled is False
    assert isinstance(result.compile_error, str)
    assert result.tests_passed == []


def test_validate_patch_identifies_passed_and_failed_tests(fixture_project: Path, tmp_path: Path) -> None:
    report = _bug_report(fixture_project)
    result = validate_patch(
        patched_source=BUGGY_SOURCE,
        bug_report=report,
        config=_config(tmp_path / "work"),
        state_id="s4",
    )

    assert "test_buggy.py::test_pass_one" in result.tests_passed
    assert "test_buggy.py::test_pass_two" in result.tests_passed
    assert "test_buggy.py::test_fail_one" in result.tests_failed
    assert "test_buggy.py::test_fail_two" in result.tests_failed


def test_workspace_cleanup_after_validation(fixture_project: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    validate_patch(
        patched_source=CORRECT_SOURCE,
        bug_report=_bug_report(fixture_project),
        config=_config(work_dir),
        state_id="cleanup",
    )

    leftovers = [p for p in work_dir.glob("verifix_*") if p.exists()]
    assert leftovers == []


def test_concrete_validator_budget_exceeded_raises(fixture_project: Path, tmp_path: Path) -> None:
    validator = ConcreteValidator(config=_config(tmp_path / "work", max_validations=1))
    report = _bug_report(fixture_project)

    validator.validate(CORRECT_SOURCE, report)
    with pytest.raises(ValidationBudgetExceeded):
        validator.validate(CORRECT_SOURCE, report)


def test_validate_patch_timeout_returns_quickly(tmp_path: Path) -> None:
    project = tmp_path / "timeout_project"
    project.mkdir(parents=True)
    (project / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (project / "test_buggy.py").write_text(
        """import time
from buggy import f


def test_slow_fail():
    time.sleep(60)
    assert f() == 1
""",
        encoding="utf-8",
    )

    report = BugReport(
        bug_id="T-1",
        language="python",
        buggy_source="def f():\n    return 1\n",
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_slow_fail"],
        passing_tests=[],
        project_root=str(project),
        metadata={},
    )

    cfg = _config(tmp_path / "work", timeout=1.0)

    start = time.monotonic()
    result = validate_patch(
        patched_source="def f():\n    return 1\n",
        bug_report=report,
        config=cfg,
        state_id="timeout",
    )
    elapsed = time.monotonic() - start

    assert elapsed < 3.0
    assert result.is_plausible is False
    assert result.all_failing_tests_pass is False


@pytest.mark.windows_timing
def test_parallel_regression_runs_faster_than_sequential(tmp_path: Path) -> None:
    single_test_time = 1.2

    project = tmp_path / "parallel_project"
    project.mkdir(parents=True)
    (project / "buggy.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (project / "test_buggy.py").write_text(
        f"""import time
from buggy import f


def test_fail_to_fix():
    assert f(2) == 2


def test_pass_one():
    time.sleep({single_test_time})
    assert f(1) == 1


def test_pass_two():
    time.sleep({single_test_time})
    assert f(2) == 2


def test_pass_three():
    time.sleep({single_test_time})
    assert f(3) == 3


def test_pass_four():
    time.sleep({single_test_time})
    assert f(4) == 4
""",
        encoding="utf-8",
    )

    report = BugReport(
        bug_id="P-1",
        language="python",
        buggy_source="def f(x):\n    return x\n",
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail_to_fix"],
        passing_tests=[
            "test_buggy.py::test_pass_one",
            "test_buggy.py::test_pass_two",
            "test_buggy.py::test_pass_three",
            "test_buggy.py::test_pass_four",
        ],
        project_root=str(project),
        metadata={},
    )

    cfg = _config(tmp_path / "work", timeout=10.0)

    start = time.monotonic()
    validate_patch(
        patched_source="def f(x):\n    return x\n",
        bug_report=report,
        config=cfg,
        state_id="parallel",
    )
    elapsed = time.monotonic() - start

    threshold_multiplier = 2.0
    if sys.platform.startswith("win"):
        threshold_multiplier = 2.8

    assert elapsed < (threshold_multiplier * single_test_time)
