from __future__ import annotations

from verifix.core.models import (
    BugReport,
    Edit,
    EditOperator,
    RankedPatch,
    ValidationResult,
)
from verifix.verifier.evidence_report import (
    build_evidence_report,
    compute_trust_score,
    detect_risk_flags,
    evidence_to_json,
    evidence_to_markdown,
    generate_fix_explanation,
)
from verifix.verifier.fuzzer import FuzzResult
from verifix.verifier.smt_layer import SMTResult


def _edit(
    operator: EditOperator,
    line: int,
    original: str,
    replacement: str,
) -> Edit:
    return Edit(
        operator=operator,
        node_id=f"n{line}",
        node_type="Expr",
        line_number=line,
        original_text=original,
        replacement_text=replacement,
        metadata={},
    )


def _validation(
    *,
    all_failing: bool,
    no_regression: bool,
    passed: list[str] | None = None,
    failed: list[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        state_id="s1",
        compiled=True,
        tests_passed=passed or [],
        tests_failed=failed or [],
        all_failing_tests_pass=all_failing,
        no_regression=no_regression,
        is_plausible=all_failing and no_regression,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=12.0,
    )


def _fuzz(*, survived: bool, total: int, failing_inputs: list[str] | None = None) -> FuzzResult:
    return FuzzResult(
        survived=survived,
        total_inputs_tested=total,
        failing_inputs=failing_inputs or [],
        coverage_achieved=0.4,
        fuzz_time_seconds=0.2,
        strategy_used="random",
    )


def _smt(verdict: str) -> SMTResult:
    return SMTResult(
        smt_applicable=True,
        smt_passed=verdict == "VERIFIED",
        counterexample=None,
        property_checked="x>=0",
        solver_time_ms=8.0,
        verdict=verdict,
    )


def _bug_report(metadata: dict | None = None, passing_tests: list[str] | None = None) -> BugReport:
    return BugReport(
        bug_id="B-1",
        language="python",
        buggy_source="def f(x):\n    return x < 0\n",
        file_path="prog.py",
        failing_tests=["tests::test_bug"],
        passing_tests=passing_tests if passing_tests is not None else ["tests::test_ok"],
        project_root=".",
        metadata=metadata or {},
    )


def _ranked_patch(edits: list[Edit], validation: ValidationResult | None = None) -> RankedPatch:
    val = validation or _validation(all_failing=True, no_regression=True)
    return RankedPatch(
        rank=1,
        patched_source="def f(x):\n    return x > 0\n",
        validation=val,
        score=0.9,
        edit_sequence=edits,
        diff="- return x < 0\n+ return x > 0\n",
    )


def test_compute_trust_score_high_for_strong_evidence() -> None:
    edits = [_edit(EditOperator.REPLACE_OPERATOR, 2, "<", ">")]
    score, level = compute_trust_score(
        _validation(all_failing=True, no_regression=True),
        _fuzz(survived=True, total=120),
        _smt("VERIFIED"),
        edits,
    )
    assert 75 <= score <= 100
    assert level == "HIGH"


def test_compute_trust_score_low_when_checks_fail() -> None:
    edits = [
        _edit(EditOperator.REPLACE_LITERAL, 1, "0", "1"),
        _edit(EditOperator.REPLACE_LITERAL, 2, "1", "2"),
        _edit(EditOperator.REPLACE_LITERAL, 3, "2", "3"),
        _edit(EditOperator.REPLACE_LITERAL, 4, "3", "4"),
    ]
    score, level = compute_trust_score(
        _validation(all_failing=False, no_regression=False),
        _fuzz(survived=False, total=3),
        _smt("COUNTEREXAMPLE_FOUND"),
        edits,
    )
    assert 0 <= score <= 49
    assert level in {"UNVERIFIED", "LOW"}


def test_generate_fix_explanation_operator_template() -> None:
    explanation = generate_fix_explanation(
        [_edit(EditOperator.REPLACE_OPERATOR, 8, "<", ">")]
    )
    assert "Fixed off-direction comparison" in explanation
    assert "line 8" in explanation


def test_generate_fix_explanation_multiple_edits_chains_text() -> None:
    explanation = generate_fix_explanation(
        [
            _edit(EditOperator.REPLACE_LITERAL, 2, "0", "1"),
            _edit(EditOperator.DELETE_STMT, 5, "print('x')", ""),
        ],
        context="inside boundary branch",
    )
    assert "Additionally" in explanation
    assert "Context: inside boundary branch" in explanation


def test_detect_risk_flags_collects_multiple_conditions() -> None:
    validation = _validation(
        all_failing=True,
        no_regression=False,
        passed=["tests::test_bug"],
        failed=["tests::timeout_case"],
    )
    fuzz_result = _fuzz(survived=False, total=5, failing_inputs=["42"])
    bug_report = _bug_report(
        metadata={"smt_verdict": "COUNTEREXAMPLE_FOUND", "edit_count": 5},
        passing_tests=[],
    )
    flags = detect_risk_flags(validation, fuzz_result, bug_report)

    assert "no_regression_tests" in flags
    assert "shallow_fuzz" in flags
    assert "fuzz_failed" in flags
    assert "smt_counterexample" in flags
    assert "high_edit_count" in flags
    assert "test_timeout_suspected" in flags


def test_build_evidence_report_populates_fields() -> None:
    edits = [_edit(EditOperator.NEGATE_CONDITION, 3, "if cond:", "if not cond:")]
    validation = _validation(all_failing=True, no_regression=True, passed=["tests::test_bug"])
    evidence = build_evidence_report(
        ranked_patch=_ranked_patch(edits, validation),
        validation=validation,
        fuzz_result=_fuzz(survived=True, total=60),
        smt_result=_smt("UNKNOWN"),
        bug_report=_bug_report(),
    )

    assert evidence.patch.rank == 1
    assert evidence.trust_level in {"HIGH", "MEDIUM", "LOW", "UNVERIFIED"}
    assert evidence.fix_explanation
    assert isinstance(evidence.risk_flags, list)


def test_evidence_to_markdown_contains_sections() -> None:
    edits = [_edit(EditOperator.REPLACE_OPERATOR, 2, "<", ">")]
    validation = _validation(
        all_failing=True,
        no_regression=True,
        passed=["tests::test_bug"],
        failed=[],
    )
    evidence = build_evidence_report(
        ranked_patch=_ranked_patch(edits, validation),
        validation=validation,
        fuzz_result=_fuzz(survived=True, total=12),
        smt_result=_smt("VERIFIED"),
        bug_report=_bug_report(),
    )

    md = evidence_to_markdown(evidence)
    assert "# Patch Evidence Report" in md
    assert "## What Changed" in md
    assert "## Test Evidence" in md
    assert "## Fuzz Evidence" in md
    assert "## SMT Evidence" in md
    assert "## Risk Flags" in md


def test_evidence_to_json_is_serializable_and_complete() -> None:
    edits = [_edit(EditOperator.REPLACE_LITERAL, 10, "0", "-1")]
    validation = _validation(all_failing=True, no_regression=False)
    evidence = build_evidence_report(
        ranked_patch=_ranked_patch(edits, validation),
        validation=validation,
        fuzz_result=_fuzz(survived=False, total=7, failing_inputs=["-99"]),
        smt_result=_smt("COUNTEREXAMPLE_FOUND"),
        bug_report=_bug_report(metadata={"source": "unit"}),
    )
    payload = evidence_to_json(evidence)

    assert "patch" in payload
    assert "validation" in payload
    assert "fuzz_result" in payload
    assert "smt_result" in payload
    assert payload["trust_level"] in {"HIGH", "MEDIUM", "LOW", "UNVERIFIED"}
    assert isinstance(payload["risk_flags"], list)
