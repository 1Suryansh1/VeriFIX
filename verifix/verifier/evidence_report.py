from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verifix.core.models import BugReport, Edit, EditOperator, RankedPatch, ValidationResult
from verifix.verifier.fuzzer import FuzzResult
from verifix.verifier.smt_layer import SMTResult


@dataclass(frozen=False)
class PatchEvidence:
    patch: RankedPatch
    validation: ValidationResult
    fuzz_result: FuzzResult
    smt_result: SMTResult
    trust_score: float
    trust_level: str
    fix_explanation: str
    risk_flags: list[str]


def compute_trust_score(
    validation: ValidationResult,
    fuzz_result: FuzzResult,
    smt_result: SMTResult,
    edit_sequence: list[Edit],
) -> tuple[float, str]:
    base = 0.0

    if validation.all_failing_tests_pass:
        base += 25
    if validation.no_regression:
        base += 15

    if fuzz_result.survived:
        base += 20
        base += min(15, (fuzz_result.total_inputs_tested / 50.0) * 15)
    else:
        base -= 20

    if smt_result.verdict == "VERIFIED":
        base += 25
    elif smt_result.verdict == "UNKNOWN":
        base += 10
    elif smt_result.verdict == "COUNTEREXAMPLE_FOUND":
        base -= 15
    elif smt_result.verdict == "NOT_APPLICABLE":
        base += 5

    base -= max(0, len(edit_sequence) - 1) * 5

    score = max(0.0, min(100.0, base))
    if score >= 75:
        level = "HIGH"
    elif score >= 50:
        level = "MEDIUM"
    elif score >= 25:
        level = "LOW"
    else:
        level = "UNVERIFIED"

    return score, level


def generate_fix_explanation(edit_sequence: list[Edit], context: str = "") -> str:
    if not edit_sequence:
        return "No edit details available for this patch."

    fragments: list[str] = []
    for edit in edit_sequence:
        op_value = edit.operator.value if hasattr(edit.operator, "value") else str(edit.operator)
        orig = edit.original_text
        new = edit.replacement_text if edit.replacement_text is not None else "<deleted>"

        if op_value == EditOperator.REPLACE_OPERATOR.value and "<" in orig and ">" in new:
            fragments.append(
                f"Fixed off-direction comparison on line {edit.line_number}: changed `{orig}` to `{new}`, "
                "reversing the comparison direction."
            )
        elif op_value == EditOperator.NEGATE_CONDITION.value:
            fragments.append(
                f"Fixed inverted guard condition on line {edit.line_number}: added `not` to reverse the predicate."
            )
        elif op_value == EditOperator.REPLACE_LITERAL.value:
            fragments.append(
                f"Fixed incorrect literal value on line {edit.line_number}: changed `{orig}` to `{new}`."
            )
        elif op_value == EditOperator.DELETE_STMT.value:
            fragments.append(f"Removed redundant/erroneous statement on line {edit.line_number}.")
        else:
            fragments.append(
                f"Applied {op_value} edit on line {edit.line_number}: `{orig}` -> `{new}`."
            )

    explanation = " Additionally, ".join(fragments)
    if context:
        return f"{explanation} Context: {context}"
    return explanation


def detect_risk_flags(
    validation: ValidationResult,
    fuzz_result: FuzzResult,
    bug_report: BugReport,
) -> list[str]:
    flags: list[str] = []

    if not bug_report.passing_tests:
        flags.append("no_regression_tests")

    if fuzz_result.total_inputs_tested < 10:
        flags.append("shallow_fuzz")

    if not fuzz_result.survived:
        flags.append("fuzz_failed")

    smt_verdict = str(bug_report.metadata.get("smt_verdict", ""))
    if smt_verdict == "COUNTEREXAMPLE_FOUND":
        flags.append("smt_counterexample")

    edit_count = bug_report.metadata.get("edit_count")
    if isinstance(edit_count, int) and edit_count > 3:
        flags.append("high_edit_count")

    if any("timeout" in test_id.lower() for test_id in validation.tests_failed):
        flags.append("test_timeout_suspected")

    return flags


def build_evidence_report(
    ranked_patch: RankedPatch,
    validation: ValidationResult,
    fuzz_result: FuzzResult,
    smt_result: SMTResult,
    bug_report: BugReport,
) -> PatchEvidence:
    score, level = compute_trust_score(validation, fuzz_result, smt_result, ranked_patch.edit_sequence)
    explanation = generate_fix_explanation(ranked_patch.edit_sequence)

    metadata = dict(bug_report.metadata)
    metadata["smt_verdict"] = smt_result.verdict
    metadata["edit_count"] = len(ranked_patch.edit_sequence)

    bug_report_for_risk = BugReport(
        bug_id=bug_report.bug_id,
        language=bug_report.language,
        buggy_source=bug_report.buggy_source,
        file_path=bug_report.file_path,
        failing_tests=list(bug_report.failing_tests),
        passing_tests=list(bug_report.passing_tests),
        project_root=bug_report.project_root,
        metadata=metadata,
    )
    risks = detect_risk_flags(validation, fuzz_result, bug_report_for_risk)

    return PatchEvidence(
        patch=ranked_patch,
        validation=validation,
        fuzz_result=fuzz_result,
        smt_result=smt_result,
        trust_score=score,
        trust_level=level,
        fix_explanation=explanation,
        risk_flags=risks,
    )


def evidence_to_markdown(evidence: PatchEvidence) -> str:
    trust_emoji = {
        "HIGH": "🟢",
        "MEDIUM": "🟡",
        "LOW": "🟠",
        "UNVERIFIED": "🔴",
    }.get(evidence.trust_level, "⚪")

    lines = [
        "# Patch Evidence Report",
        f"**Trust Level**: {trust_emoji} {evidence.trust_level} (score: {evidence.trust_score:.0f}/100)",
        "",
        "## What Changed",
        evidence.fix_explanation,
        "",
        "```diff",
        evidence.patch.diff.rstrip("\n"),
        "```",
        "",
        "## Test Evidence",
        "| Test | Status |",
        "|------|--------|",
    ]

    for test_id in sorted(evidence.validation.tests_passed):
        lines.append(f"| {test_id} | ✅ PASS |")
    for test_id in sorted(evidence.validation.tests_failed):
        lines.append(f"| {test_id} | ❌ FAIL |")

    failing_total = len(evidence.validation.tests_passed) + len(evidence.validation.tests_failed)
    failing_status = "✅" if evidence.validation.all_failing_tests_pass else "❌"
    regression_status = "✅" if evidence.validation.no_regression else "❌"

    lines.extend(
        [
            "",
            f"**Originally Failing Tests**: {len(evidence.validation.tests_passed)}/{max(1, failing_total)} now passing {failing_status}",
            f"**Regression Check**: {'No regressions' if evidence.validation.no_regression else 'Regression detected'} {regression_status}",
            "",
            "## Fuzz Evidence",
            f"- Strategy: {evidence.fuzz_result.strategy_used.upper()}",
            f"- Inputs tested: {evidence.fuzz_result.total_inputs_tested}",
            f"- Result: {'✅ All inputs survived' if evidence.fuzz_result.survived else '❌ Failing fuzz inputs found'}",
            "",
            "## SMT Evidence",
            f"- Verdict: {'✅' if evidence.smt_result.verdict == 'VERIFIED' else '⚠️'} {evidence.smt_result.verdict}",
            f"- Property: {evidence.smt_result.property_checked}",
            f"- Solver time: {evidence.smt_result.solver_time_ms:.0f}ms",
            "",
            "## Risk Flags",
        ]
    )

    if evidence.risk_flags:
        for flag in evidence.risk_flags:
            lines.append(f"⚠️ {flag}")
    else:
        lines.append("None")

    return "\n".join(lines)


def evidence_to_json(evidence: PatchEvidence) -> dict:
    return {
        "patch": evidence.patch.to_dict(),
        "validation": {
            "state_id": evidence.validation.state_id,
            "compiled": evidence.validation.compiled,
            "tests_passed": list(evidence.validation.tests_passed),
            "tests_failed": list(evidence.validation.tests_failed),
            "all_failing_tests_pass": evidence.validation.all_failing_tests_pass,
            "no_regression": evidence.validation.no_regression,
            "is_plausible": evidence.validation.is_plausible,
            "compile_error": evidence.validation.compile_error,
            "runtime_error": evidence.validation.runtime_error,
            "execution_time_ms": evidence.validation.execution_time_ms,
        },
        "fuzz_result": {
            "survived": evidence.fuzz_result.survived,
            "total_inputs_tested": evidence.fuzz_result.total_inputs_tested,
            "failing_inputs": list(evidence.fuzz_result.failing_inputs),
            "coverage_achieved": evidence.fuzz_result.coverage_achieved,
            "fuzz_time_seconds": evidence.fuzz_result.fuzz_time_seconds,
            "strategy_used": evidence.fuzz_result.strategy_used,
        },
        "smt_result": {
            "smt_applicable": evidence.smt_result.smt_applicable,
            "smt_passed": evidence.smt_result.smt_passed,
            "counterexample": evidence.smt_result.counterexample,
            "property_checked": evidence.smt_result.property_checked,
            "solver_time_ms": evidence.smt_result.solver_time_ms,
            "verdict": evidence.smt_result.verdict,
        },
        "trust_score": evidence.trust_score,
        "trust_level": evidence.trust_level,
        "fix_explanation": evidence.fix_explanation,
        "risk_flags": list(evidence.risk_flags),
    }


__all__ = [
    "PatchEvidence",
    "compute_trust_score",
    "generate_fix_explanation",
    "detect_risk_flags",
    "build_evidence_report",
    "evidence_to_markdown",
    "evidence_to_json",
]
