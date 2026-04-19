from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from verifix.cli import app
from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v2 import RepairAgentV2
from verifix.verifier.evidence_report import compute_trust_score
from verifix.verifier.fuzzer import FuzzResult
from verifix.verifier.smt_layer import SMTResult


pytestmark = pytest.mark.integration


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


def test_fail_three():
    assert find_max([1, 3, 2]) == 3


def test_pass_one():
    assert find_max([7]) == 7


def test_pass_two():
    assert find_max([2, 2, 2]) == 2
"""


def _config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=80,
        mcts_max_depth=1,
        mcts_time_budget_seconds=30.0,
        max_validations=200,
        max_candidates_per_node=4,
        fl_top_n_lines=1,
        fl_algorithm="ochiai",
        python_executable=sys.executable,
        test_timeout_seconds=5.0,
        working_dir=str(tmp_path / "work"),
    )


def _build_project(root: Path, name: str) -> tuple[Path, BugReport]:
    project = root / name
    project.mkdir(parents=True, exist_ok=True)

    (project / "buggy.py").write_text(BUGGY_SOURCE, encoding="utf-8")
    (project / "test_buggy.py").write_text(TEST_SOURCE, encoding="utf-8")

    report = BugReport(
        bug_id=name,
        language="python",
        buggy_source=BUGGY_SOURCE,
        file_path="buggy.py",
        failing_tests=[
            "test_buggy.py::test_fail_one",
            "test_buggy.py::test_fail_two",
            "test_buggy.py::test_fail_three",
        ],
        passing_tests=["test_buggy.py::test_pass_one", "test_buggy.py::test_pass_two"],
        project_root=str(project),
        metadata={
            "benchmark": "integration",
            "reference_source": FIXED_SOURCE,
            "test_cases": [
                {"input": [[3, 1, 2]], "expected": 3},
                {"input": [[5, 4, 3]], "expected": 5},
                {"input": [[1, 3, 2]], "expected": 3},
                {"input": [[2, 2, 2]], "expected": 2},
            ],
        },
    )
    return project, report


def _v1_patch_trust_proxy(patch) -> float:
    neutral_fuzz = FuzzResult(
        survived=True,
        total_inputs_tested=0,
        failing_inputs=[],
        coverage_achieved=0.0,
        fuzz_time_seconds=0.0,
        strategy_used="tests_only",
    )
    neutral_smt = SMTResult(
        smt_applicable=False,
        smt_passed=False,
        counterexample=None,
        property_checked="none",
        solver_time_ms=0.0,
        verdict="NOT_APPLICABLE",
    )
    score, _level = compute_trust_score(patch.validation, neutral_fuzz, neutral_smt, patch.edit_sequence)
    return score


def _reload_v2_result(json_path: Path) -> SimpleNamespace:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    best_payload = payload.get("best_evidence")
    if best_payload is None:
        return SimpleNamespace(best_evidence=None)
    return SimpleNamespace(
        best_evidence=SimpleNamespace(
            trust_score=float(best_payload["trust_score"]),
        )
    )


def test_v1_full_pipeline(tmp_path: Path) -> None:
    _project, bug_report = _build_project(tmp_path, "v1_full")
    config = _config(tmp_path)

    random.seed(42)
    result = RepairAgent(config=config).repair(bug_report)

    assert result.success is True
    assert result.ranked_patches
    assert result.ranked_patches[0].validation.is_plausible is True
    assert ">" in result.ranked_patches[0].diff
    assert result.total_validations_run > 0
    assert result.wall_time_seconds < 180


def test_v2_full_pipeline(tmp_path: Path) -> None:
    _project, bug_report = _build_project(tmp_path, "v2_full")
    config = _config(tmp_path)

    random.seed(43)
    result = RepairAgentV2(config=config).repair(bug_report)

    assert result.best_evidence is not None
    assert result.best_evidence.trust_level in {"HIGH", "MEDIUM"}
    assert result.best_evidence.fuzz_result.survived is True
    assert result.best_evidence.smt_result.verdict in {"VERIFIED", "UNKNOWN", "NOT_APPLICABLE"}
    assert result.best_evidence.trust_level == "HIGH" or result.best_evidence.trust_score > 50.0


@pytest.mark.timeout(240)
def test_v1_vs_v2_overfit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    v1_agent = RepairAgent(config=config)
    v2_agent = RepairAgentV2(config=config)

    reports: list[BugReport] = []
    for idx in range(3):
        _project, report = _build_project(tmp_path, f"overfit_case_{idx}")
        reports.append(report)

    v1_scores: list[float] = []
    v2_scores: list[float] = []

    for report in reports:
        random.seed(100 + len(report.bug_id))
        v1_result = v1_agent.repair(report)

        random.seed(200 + len(report.bug_id))
        v2_result = v2_agent.repair(report)

        assert v1_result.ranked_patches
        assert v2_result.evidence_list

        v1_scores.extend([_v1_patch_trust_proxy(patch) for patch in v1_result.ranked_patches])
        v2_scores.extend([evidence.trust_score for evidence in v2_result.evidence_list])

    assert v1_scores and v2_scores
    assert (sum(v2_scores) / len(v2_scores)) > (sum(v1_scores) / len(v1_scores))


def test_cli_repair_command(tmp_path: Path) -> None:
    project, _bug_report = _build_project(tmp_path, "cli_full")
    config = _config(tmp_path)

    config_path = tmp_path / "integration_config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=True), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair",
            "--file",
            str(project / "buggy.py"),
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


def test_result_serialization_roundtrip(tmp_path: Path) -> None:
    _project, bug_report = _build_project(tmp_path, "serialize_full")
    config = _config(tmp_path)

    random.seed(123)
    original = RepairAgentV2(config=config).repair(bug_report)
    assert original.best_evidence is not None

    payload_path = tmp_path / "v2_result.json"
    payload_path.write_text(json.dumps(original.to_dict(), indent=2), encoding="utf-8")

    reloaded = _reload_v2_result(payload_path)
    assert reloaded.best_evidence is not None
    assert reloaded.best_evidence.trust_score == original.best_evidence.trust_score
