from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verifix.parser.fault_localizer import (
    TestCoverageRecord,
    collect_coverage,
    localize_faults,
    ochiai_score,
)


BUGGY_MODULE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:  # BUG: should be >
            max_val = arr[i]
    return max_val
"""


TEST_MODULE = """from buggy import find_max


def test_fail_case_one():
    assert find_max([3, 1, 2]) == 3


def test_fail_case_two():
    assert find_max([5, 4, 3]) == 5


def test_pass_case_one():
    assert find_max([7]) == 7


def test_pass_case_two():
    assert find_max([2, 2, 2]) == 2
"""


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    project = tmp_path / "temp_project"
    project.mkdir(parents=True)
    (project / "buggy.py").write_text(BUGGY_MODULE, encoding="utf-8")
    (project / "test_buggy.py").write_text(TEST_MODULE, encoding="utf-8")
    return project


def _test_ids() -> tuple[list[str], list[str]]:
    failing = [
        "test_buggy.py::test_fail_case_one",
        "test_buggy.py::test_fail_case_two",
    ]
    passing = [
        "test_buggy.py::test_pass_case_one",
        "test_buggy.py::test_pass_case_two",
    ]
    return failing, passing


def test_collect_coverage_returns_record_per_test(temp_project: Path) -> None:
    failing, passing = _test_ids()
    all_tests = [*failing, *passing]

    records = collect_coverage(
        project_root=str(temp_project),
        source_file="buggy.py",
        test_ids=all_tests,
        python_executable=sys.executable,
    )

    assert len(records) == len(all_tests)
    assert {rec.test_id for rec in records} == set(all_tests)


def test_collect_coverage_marks_pass_fail_correctly(temp_project: Path) -> None:
    failing, passing = _test_ids()
    all_tests = [*failing, *passing]

    records = collect_coverage(
        project_root=str(temp_project),
        source_file="buggy.py",
        test_ids=all_tests,
        python_executable=sys.executable,
    )
    by_id = {rec.test_id: rec for rec in records}

    assert all(by_id[test_id].passed is False for test_id in failing)
    assert all(by_id[test_id].passed is True for test_id in passing)


def test_ochiai_score_bug_line_higher_than_surrounding_lines(temp_project: Path) -> None:
    failing, passing = _test_ids()
    records = collect_coverage(
        project_root=str(temp_project),
        source_file="buggy.py",
        test_ids=[*failing, *passing],
        python_executable=sys.executable,
    )

    scores = ochiai_score(records)
    by_line = {item.line: item.score for item in scores}

    # The mutation-prone assignment line should be more suspicious than surrounding control flow.
    assert by_line[5] > by_line[4]
    assert by_line[5] > by_line[6]


def test_ochiai_score_line_covered_only_by_passing_is_zero() -> None:
    records = [
        TestCoverageRecord(test_id="f1", passed=False, covered_lines=frozenset({1, 2})),
        TestCoverageRecord(test_id="p1", passed=True, covered_lines=frozenset({2, 10})),
    ]

    scores = ochiai_score(records)
    score_by_line = {item.line: item.score for item in scores}

    assert score_by_line[10] == 0.0


def test_localize_faults_returns_at_most_top_n_results(temp_project: Path) -> None:
    failing, passing = _test_ids()

    result = localize_faults(
        project_root=str(temp_project),
        source_file="buggy.py",
        failing_tests=failing,
        passing_tests=passing,
        algorithm="ochiai",
        top_n=3,
        python_executable=sys.executable,
    )

    assert len(result) <= 3


def test_localize_faults_fallback_returns_first_n_lines(temp_project: Path) -> None:
    fallback = localize_faults(
        project_root=str(temp_project),
        source_file="buggy.py",
        failing_tests=[],
        passing_tests=[],
        algorithm="ochiai",
        top_n=4,
        existing_coverage=[
            TestCoverageRecord(test_id="f1", passed=False, covered_lines=frozenset()),
            TestCoverageRecord(test_id="p1", passed=True, covered_lines=frozenset()),
        ],
    )

    assert [item.line for item in fallback] == [1, 2, 3, 4]
    assert all(item.score == 0.0 for item in fallback)


def test_localize_faults_uses_cache_on_second_call(temp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failing, passing = _test_ids()
    cache_dir = temp_project / "cache"

    first = localize_faults(
        project_root=str(temp_project),
        source_file="buggy.py",
        failing_tests=failing,
        passing_tests=passing,
        algorithm="ochiai",
        top_n=5,
        python_executable=sys.executable,
        cache_dir=str(cache_dir),
    )

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("collect_coverage should not be called when cache exists")

    monkeypatch.setattr("verifix.parser.fault_localizer.collect_coverage", _fail_if_called)

    second = localize_faults(
        project_root=str(temp_project),
        source_file="buggy.py",
        failing_tests=failing,
        passing_tests=passing,
        algorithm="ochiai",
        top_n=5,
        python_executable=sys.executable,
        cache_dir=str(cache_dir),
    )

    assert second == first
