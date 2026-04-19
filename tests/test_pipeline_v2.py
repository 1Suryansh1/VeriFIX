from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport
from verifix.pipeline.repair_agent_v2 import RepairAgentV2, RepairResultV2


BUGGY_SOURCE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


FIXED_SOURCE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
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
    project = tmp_path / "fixture_project_v2"
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
        working_dir=str(tmp_path / "work_v2"),
    )


@pytest.fixture
def base_bug_report(fixture_project: Path) -> BugReport:
    return BugReport(
        bug_id="Chart-1-v2",
        language="python",
        buggy_source=BUGGY_SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail_one", "test_buggy.py::test_fail_two"],
        passing_tests=["test_buggy.py::test_pass_one", "test_buggy.py::test_pass_two"],
        project_root=str(fixture_project),
        metadata={
            "benchmark": "toy-v2",
            "test_cases": [
                {"input": [[3, 1, 2]], "expected": 3},
                {"input": [[5, 4, 3]], "expected": 5},
                {"input": [[2, 2, 2]], "expected": 2},
            ],
        },
    )


@pytest.fixture
def v2_result(base_bug_report: BugReport, fast_config: VerifixConfig) -> RepairResultV2:
    random.seed(2026)
    agent = RepairAgentV2(config=fast_config)
    return agent.repair(base_bug_report)


def test_repair_returns_repair_result_v2(v2_result: RepairResultV2) -> None:
    assert isinstance(v2_result, RepairResultV2)


def test_end_to_end_best_evidence_is_high_trust(v2_result: RepairResultV2) -> None:
    assert v2_result.best_evidence is not None
    assert v2_result.best_evidence.trust_level == "HIGH"


def test_success_true_when_high_or_medium_exists(v2_result: RepairResultV2) -> None:
    assert any(item.trust_level in {"HIGH", "MEDIUM"} for item in v2_result.evidence_list)
    assert v2_result.success is True


def test_result_contains_v1_result(v2_result: RepairResultV2) -> None:
    assert v2_result.v1_result is not None
    assert v2_result.v1_result.bug_id == "Chart-1-v2"


def test_to_dict_is_json_serializable(v2_result: RepairResultV2) -> None:
    payload = v2_result.to_dict()
    json.dumps(payload)


def test_to_markdown_contains_trust_level_and_diff_block(v2_result: RepairResultV2) -> None:
    report_md = v2_result.to_markdown()
    assert "Trust Level" in report_md
    assert "```diff" in report_md
