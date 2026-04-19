from __future__ import annotations

from pathlib import Path

from verifix.benchmarks import ablation_runner as ar
from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, EditOperator, RankedPatch, RepairResult, ValidationResult
from verifix.parser.fault_localizer import SuspiciousnessScore


def _config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        mcts_time_budget_seconds=2.0,
        max_validations=10,
        max_candidates_per_node=2,
        fl_top_n_lines=1,
        test_timeout_seconds=2.0,
        working_dir=str(tmp_path / "work"),
    )


def _bug_report(tmp_path: Path) -> BugReport:
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "buggy.py").write_text("def f(x):\n    return x < 0\n", encoding="utf-8")

    return BugReport(
        bug_id="Abl-1",
        language="python",
        buggy_source="def f(x):\n    return x < 0\n",
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=["test_buggy.py::test_pass"],
        project_root=str(project),
        metadata={},
    )


def _validation(plausible: bool = True) -> ValidationResult:
    return ValidationResult(
        state_id="s1",
        compiled=True,
        tests_passed=["t1"] if plausible else [],
        tests_failed=[] if plausible else ["t1"],
        all_failing_tests_pass=plausible,
        no_regression=plausible,
        is_plausible=plausible,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=1.0,
    )


def test_standard_ablations_count() -> None:
    assert len(ar.STANDARD_ABLATIONS) == 5
    names = {item.name for item in ar.STANDARD_ABLATIONS}
    assert "v2-full" in names


def test_run_ablation_returns_required_keys(tmp_path: Path, monkeypatch) -> None:
    def _fake_repair(_self, report: BugReport) -> RepairResult:
        patch = RankedPatch(
            rank=1,
            edit_sequence=[],
            patched_source=report.buggy_source.replace("<", ">"),
            validation=_validation(True),
            score=0.9,
            diff="--- a/buggy.py\n+++ b/buggy.py\n",
        )
        return RepairResult(
            bug_id=report.bug_id,
            success=True,
            ranked_patches=[patch],
            total_states_explored=3,
            total_validations_run=2,
            wall_time_seconds=0.2,
            search_tree_depth=1,
            error=None,
        )

    monkeypatch.setattr(ar.RepairAgent, "repair", _fake_repair)

    summary = ar.run_ablation(
        ablation=ar.AblationConfig(name="baseline", description="desc"),
        bug_reports=[_bug_report(tmp_path)],
        base_config=_config(tmp_path),
        output_dir=str(tmp_path / "out"),
    )

    keys = {
        "ablation_name",
        "bugs_total",
        "bugs_repaired",
        "repair_rate",
        "avg_plausible_patches",
        "avg_validations_used",
        "avg_wall_time",
        "high_trust_patches",
        "fuzz_rejection_rate",
        "per_bug",
    }
    assert keys.issubset(summary.keys())


def test_greedy_search_agent_repair_returns_repair_result(tmp_path: Path, monkeypatch) -> None:
    report = _bug_report(tmp_path)

    edit = Edit(
        operator=EditOperator.REPLACE_OPERATOR,
        node_id="n1",
        node_type="Compare",
        line_number=2,
        original_text="<",
        replacement_text=">",
        metadata={},
    )

    monkeypatch.setattr(
        ar,
        "localize_faults",
        lambda **kwargs: [
            SuspiciousnessScore(
                line=2,
                score=1.0,
                algorithm="ochiai",
                covered_by_failing=1,
                covered_by_passing=0,
            )
        ],
    )
    monkeypatch.setattr(ar, "get_candidate_edits", lambda *args, **kwargs: [edit])
    monkeypatch.setattr(
        ar,
        "apply_edit_sequence",
        lambda source, edits: (source.replace("<", ">"), [True]),
    )
    monkeypatch.setattr(ar, "validate_patch", lambda **kwargs: _validation(True))

    agent = ar.GreedySearchAgent(_config(tmp_path))
    result = agent.repair(report)

    assert isinstance(result, RepairResult)
    assert result.success is True
    assert len(result.ranked_patches) == 1


def test_print_ablation_table_no_error(capsys) -> None:
    ar.print_ablation_table(
        {
            "baseline-v1-tests-only": {
                "repair_rate": 0.425,
                "avg_validations": 87.3,
                "avg_time": 45.2,
                "fuzz_rejection_rate": None,
            },
            "v2-full": {
                "repair_rate": 0.468,
                "avg_validations": 93.5,
                "avg_time": 89.1,
                "fuzz_rejection_rate": 0.182,
            },
        }
    )
    output = capsys.readouterr().out
    assert "Ablation" in output
    assert "Repair Rate" in output
