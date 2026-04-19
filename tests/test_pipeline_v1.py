from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.pipeline import repair_agent as repair_agent_module
from verifix.pipeline.repair_agent import RepairAgent


BUGGY_SOURCE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


TEST_SOURCE = """from buggy import find_max


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
    project = tmp_path / "fixture_project"
    project.mkdir(parents=True)
    (project / "buggy.py").write_text(BUGGY_SOURCE, encoding="utf-8")
    (project / "test_buggy.py").write_text(TEST_SOURCE, encoding="utf-8")
    return project


@pytest.fixture
def fast_config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=80,
        mcts_max_depth=1,
        mcts_time_budget_seconds=25.0,
        max_validations=200,
        max_candidates_per_node=4,
        fl_top_n_lines=1,
        fl_algorithm="ochiai",
        python_executable=sys.executable,
        test_timeout_seconds=5.0,
        working_dir=str(tmp_path / "work"),
    )


@pytest.fixture
def base_bug_report(fixture_project: Path) -> BugReport:
    return BugReport(
        bug_id="Chart-1",
        language="python",
        buggy_source=BUGGY_SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail_one", "test_buggy.py::test_fail_two"],
        passing_tests=["test_buggy.py::test_pass_one", "test_buggy.py::test_pass_two"],
        project_root=str(fixture_project),
        metadata={"benchmark": "toy"},
    )


@pytest.fixture
def pipeline_result(base_bug_report: BugReport, fast_config: VerifixConfig) -> RepairResult:
    random.seed(1234)
    agent = RepairAgent(config=fast_config)
    return agent.repair(base_bug_report)


def test_repair_returns_repair_result(pipeline_result: RepairResult) -> None:
    assert isinstance(pipeline_result, RepairResult)


def test_repair_success_true_for_find_max_bug(pipeline_result: RepairResult) -> None:
    assert pipeline_result.success is True


def test_top_ranked_patch_is_plausible(pipeline_result: RepairResult) -> None:
    assert pipeline_result.ranked_patches
    assert pipeline_result.ranked_patches[0].validation.is_plausible is True


def test_top_patch_diff_contains_added_removed_lines(pipeline_result: RepairResult) -> None:
    diff = pipeline_result.ranked_patches[0].diff
    assert "\n-" in diff
    assert "\n+" in diff


def test_wall_time_within_expected_bound(
    pipeline_result: RepairResult,
    fast_config: VerifixConfig,
) -> None:
    assert pipeline_result.wall_time_seconds > 0
    assert pipeline_result.wall_time_seconds < fast_config.mcts_time_budget_seconds + 5


def test_no_failing_tests_raises_assertion(base_bug_report: BugReport, fast_config: VerifixConfig) -> None:
    agent = RepairAgent(config=fast_config)
    empty_fail = BugReport(
        bug_id=base_bug_report.bug_id,
        language=base_bug_report.language,
        buggy_source=base_bug_report.buggy_source,
        file_path=base_bug_report.file_path,
        failing_tests=[],
        passing_tests=base_bug_report.passing_tests,
        project_root=base_bug_report.project_root,
        metadata=base_bug_report.metadata,
    )

    with pytest.raises(AssertionError):
        agent.repair(empty_fail)


def test_repair_from_file_reads_disk_and_matches_direct_call(
    fixture_project: Path,
    fast_config: VerifixConfig,
) -> None:
    random.seed(777)
    agent = RepairAgent(config=fast_config)

    direct_report = BugReport(
        bug_id="buggy",
        language="python",
        buggy_source=BUGGY_SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail_one", "test_buggy.py::test_fail_two"],
        passing_tests=[],
        project_root=str(fixture_project),
        metadata={"source": "repair_from_file"},
    )

    direct_result = agent.repair(direct_report)

    random.seed(777)
    from_file_result = agent.repair_from_file(
        file_path=str(fixture_project / "buggy.py"),
        test_ids=["test_buggy.py::test_fail_one", "test_buggy.py::test_fail_two"],
        project_root=str(fixture_project),
    )

    assert from_file_result.success == direct_result.success
    assert len(from_file_result.ranked_patches) == len(direct_result.ranked_patches)


def test_error_field_none_on_success_and_set_on_exception(
    pipeline_result: RepairResult,
    base_bug_report: BugReport,
    fast_config: VerifixConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pipeline_result.error is None

    def _boom(*args: object, **kwargs: object):
        raise RuntimeError("forced localization failure")

    monkeypatch.setattr(repair_agent_module, "localize_faults", _boom)
    agent = RepairAgent(config=fast_config)
    failed = agent.repair(base_bug_report)

    assert failed.success is False
    assert isinstance(failed.error, str)
    assert "forced localization failure" in failed.error
