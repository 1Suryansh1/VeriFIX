from __future__ import annotations

from pathlib import Path

from verifix.analysis.overfit_detector import (
    OverfitAnalysis,
    compare_overfit_rates,
    compute_overfit_rate,
    generate_overfit_report,
)
from verifix.core.config import VerifixConfig
from verifix.core.models import Edit, EditOperator, RankedPatch, ValidationResult


def _config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        fl_top_n_lines=1,
        working_dir=str(tmp_path / "work"),
    )


def _validation() -> ValidationResult:
    return ValidationResult(
        state_id="s1",
        compiled=True,
        tests_passed=["t_fail_1", "t_pass_1"],
        tests_failed=[],
        all_failing_tests_pass=True,
        no_regression=True,
        is_plausible=True,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=1.0,
    )


def _patch(rank: int, patched_source: str) -> RankedPatch:
    return RankedPatch(
        rank=rank,
        edit_sequence=[
            Edit(
                operator=EditOperator.REPLACE_OPERATOR,
                node_id=f"n{rank}",
                node_type="Compare",
                line_number=2,
                original_text="<",
                replacement_text=">",
                metadata={},
            )
        ],
        patched_source=patched_source,
        validation=_validation(),
        score=0.9,
        diff="--- a/buggy.py\n+++ b/buggy.py\n",
    )


def test_compute_overfit_rate_reference_match_top_patch_correct_rate_one(tmp_path: Path) -> None:
    reference = "def f(x):\n    return x > 0\n"
    patches = [_patch(1, reference)]

    analysis = compute_overfit_rate(
        plausible_patches=patches,
        reference_source=reference,
        holdout_tests=None,
        project_root=str(tmp_path),
        config=_config(tmp_path),
    )

    assert analysis.correct_rate == 1.0
    assert analysis.overfit_rate == 0.0
    assert analysis.total_correct == 1


def test_compute_overfit_rate_reference_mismatch_detects_overfit(tmp_path: Path) -> None:
    reference = "def f(x):\n    return x > 0\n"
    patches = [_patch(1, "def f(x):\n    return x < 0\n")]

    analysis = compute_overfit_rate(
        plausible_patches=patches,
        reference_source=reference,
        holdout_tests=None,
        project_root=str(tmp_path),
        config=_config(tmp_path),
    )

    assert analysis.total_overfitted == 1
    assert analysis.overfit_rate == 1.0
    assert analysis.per_patch_verdict[0]["verdict"] == "OVERFITTED"


def test_compare_overfit_rates_reports_v2_better() -> None:
    v1 = OverfitAnalysis(
        total_plausible=10,
        total_correct=4,
        total_overfitted=0,
        overfit_rate=0.0,
        correct_rate=0.0,
        detection_method="reference",
        per_patch_verdict=[],
    )
    v2 = OverfitAnalysis(
        total_plausible=10,
        total_correct=8,
        total_overfitted=0,
        overfit_rate=0.0,
        correct_rate=0.0,
        detection_method="combined",
        per_patch_verdict=[],
    )

    comparison = compare_overfit_rates(v1, v2)
    assert comparison["verdict"] == "V2_BETTER"
    assert comparison["reduction"] > 0


def test_generate_overfit_report_contains_markdown_table() -> None:
    v1 = OverfitAnalysis(
        total_plausible=5,
        total_correct=2,
        total_overfitted=0,
        overfit_rate=0.0,
        correct_rate=0.0,
        detection_method="reference",
        per_patch_verdict=[],
    )
    v2 = OverfitAnalysis(
        total_plausible=5,
        total_correct=4,
        total_overfitted=0,
        overfit_rate=0.0,
        correct_rate=0.0,
        detection_method="combined",
        per_patch_verdict=[],
    )

    report = generate_overfit_report(v1, v2, "ToyBenchmark")
    assert "| System | Plausible | Correct | Overfitted | Overfit Rate |" in report
    assert "## Overfit Analysis: ToyBenchmark" in report


def test_overfit_analysis_rate_formula() -> None:
    analysis = OverfitAnalysis(
        total_plausible=4,
        total_correct=1,
        total_overfitted=0,
        overfit_rate=0.0,
        correct_rate=0.0,
        detection_method="reference",
        per_patch_verdict=[],
    )

    assert analysis.total_overfitted == 3
    assert analysis.overfit_rate == 3 / 4
