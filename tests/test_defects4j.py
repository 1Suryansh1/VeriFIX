from __future__ import annotations

from pathlib import Path

from verifix.benchmarks.defects4j import Defects4JBenchmark, Defects4JLoader, create_d4py_dataset_from_examples
from verifix.core.config import VerifixConfig
from verifix.core.models import Edit, EditOperator, RankedPatch, RepairResult, ValidationResult


def _config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        mcts_time_budget_seconds=1.0,
        max_validations=5,
        max_candidates_per_node=2,
        fl_top_n_lines=1,
        test_timeout_seconds=1.0,
        working_dir=str(tmp_path / "work"),
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


def test_create_d4py_dataset_from_examples_creates_structure(tmp_path: Path) -> None:
    create_d4py_dataset_from_examples(str(tmp_path))

    for bug_id in ["Example-1", "Example-2", "Example-3"]:
        bug_dir = tmp_path / bug_id
        assert bug_dir.exists()
        assert (bug_dir / "buggy.py").exists()
        assert (bug_dir / "fixed.py").exists()
        assert (bug_dir / "failing_tests.txt").exists()
        assert (bug_dir / "passing_tests.txt").exists()
        assert (bug_dir / "metadata.json").exists()


def test_loader_load_bug_from_toy_dataset_returns_bug_report(tmp_path: Path) -> None:
    create_d4py_dataset_from_examples(str(tmp_path))
    loader = Defects4JLoader(str(tmp_path), format="d4j_python")

    report = loader.load_bug("Example-1")

    assert report.bug_id == "Example-1"
    assert report.language == "python"
    assert report.file_path == "buggy.py"
    assert len(report.failing_tests) > 0


def test_benchmark_run_single_bug_returns_summary_with_keys(tmp_path: Path) -> None:
    create_d4py_dataset_from_examples(str(tmp_path))
    benchmark = Defects4JBenchmark(str(tmp_path), config=_config(tmp_path))

    def _fake_repair(report) -> RepairResult:
        fixed_path = Path(str(report.metadata.get("fixed_source_path", "")))
        patched_source = fixed_path.read_text(encoding="utf-8") if fixed_path.exists() else report.buggy_source
        patch = RankedPatch(
            rank=1,
            edit_sequence=[
                Edit(
                    operator=EditOperator.REPLACE_OPERATOR,
                    node_id="n1",
                    node_type="Compare",
                    line_number=1,
                    original_text="<",
                    replacement_text=">",
                    metadata={},
                )
            ],
            patched_source=patched_source,
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

    benchmark.agent.repair = _fake_repair
    summary = benchmark.run(bug_ids=["Example-1"], output_dir=str(tmp_path / "results"))

    expected_keys = {
        "total_bugs",
        "plausible_patches",
        "correct_patches",
        "plausible_rate",
        "correct_rate",
        "avg_time_seconds",
        "avg_validations",
        "terminated_by_budget",
        "per_bug",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["total_bugs"] == 1


def test_compute_correct_rate_true_when_patched_equals_fixed(tmp_path: Path) -> None:
    create_d4py_dataset_from_examples(str(tmp_path))
    benchmark = Defects4JBenchmark(str(tmp_path), config=_config(tmp_path))

    fixed_source = (tmp_path / "Example-1" / "fixed.py").read_text(encoding="utf-8")
    result = RepairResult(
        bug_id="Example-1",
        success=True,
        ranked_patches=[
            RankedPatch(
                rank=1,
                edit_sequence=[],
                patched_source=fixed_source,
                validation=_validation(True),
                score=1.0,
                diff="",
            )
        ],
        total_states_explored=1,
        total_validations_run=1,
        wall_time_seconds=0.1,
        search_tree_depth=1,
        error=None,
    )

    assert benchmark.compute_correct_rate(result, str(tmp_path / "Example-1" / "fixed.py")) is True


def test_compute_correct_rate_false_when_patched_differs(tmp_path: Path) -> None:
    create_d4py_dataset_from_examples(str(tmp_path))
    benchmark = Defects4JBenchmark(str(tmp_path), config=_config(tmp_path))

    result = RepairResult(
        bug_id="Example-1",
        success=True,
        ranked_patches=[
            RankedPatch(
                rank=1,
                edit_sequence=[],
                patched_source="def unrelated():\n    return 1\n",
                validation=_validation(True),
                score=1.0,
                diff="",
            )
        ],
        total_states_explored=1,
        total_validations_run=1,
        wall_time_seconds=0.1,
        search_tree_depth=1,
        error=None,
    )

    assert benchmark.compute_correct_rate(result, str(tmp_path / "Example-1" / "fixed.py")) is False
