from __future__ import annotations

import time

import z3

from verifix.core.models import Edit, EditOperator, ValidationResult
from verifix.verifier import smt_layer
from verifix.verifier.smt_layer import (
    SMTResult,
    build_z3_formula,
    check_patch_semantics,
    extract_changed_expression,
    smt_screen_patches,
)


def _edit(operator: EditOperator, original: str, replacement: str | None) -> Edit:
    return Edit(
        operator=operator,
        node_id="n1",
        node_type="Expr",
        line_number=5,
        original_text=original,
        replacement_text=replacement,
        metadata={},
    )


def _validation() -> ValidationResult:
    return ValidationResult(
        state_id="s1",
        compiled=True,
        tests_passed=["t1"],
        tests_failed=[],
        all_failing_tests_pass=True,
        no_regression=True,
        is_plausible=True,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=1.0,
    )


def test_extract_changed_expression_replace_operator_comparison_type() -> None:
    edit = _edit(EditOperator.REPLACE_OPERATOR, "<", ">")
    changed = extract_changed_expression("x < y", "x > y", edit)

    assert changed is not None
    assert changed["type"] == "comparison_op"


def test_build_z3_formula_comparison_returns_bool_ref() -> None:
    changed = {
        "type": "comparison_op",
        "original_expr": "x < y",
        "patched_expr": "x > y",
    }
    formulas = build_z3_formula(changed, {"x": "Int", "y": "Int"})

    assert formulas is not None
    original, patched = formulas
    assert z3.is_bool(original)
    assert z3.is_bool(patched)


def test_check_patch_semantics_equivalence_detects_not_equivalent() -> None:
    changed = {
        "type": "comparison_op",
        "original_expr": "x < y",
        "patched_expr": "x > y",
        "timeout_ms": 500,
    }
    result = check_patch_semantics(changed, {"x": "Int", "y": "Int"}, property_type="equivalence")

    assert result.smt_applicable is True
    assert result.smt_passed is False
    assert result.verdict == "COUNTEREXAMPLE_FOUND"
    assert isinstance(result.counterexample, dict)


def test_check_patch_semantics_literal_change_detects_difference() -> None:
    changed = {
        "type": "literal_change",
        "old_value": 0,
        "new_value": 1,
        "variable": "x",
        "timeout_ms": 500,
    }
    result = check_patch_semantics(changed, {"x": "Int"}, property_type="equivalence")

    assert result.smt_applicable is True
    assert result.smt_passed is False
    assert result.verdict == "COUNTEREXAMPLE_FOUND"
    assert isinstance(result.counterexample, dict)


def test_check_patch_semantics_equivalence_detects_equivalent() -> None:
    changed = {
        "type": "comparison_op",
        "original_expr": "x < y",
        "patched_expr": "x < y",
        "timeout_ms": 500,
    }
    result = check_patch_semantics(changed, {"x": "Int", "y": "Int"}, property_type="equivalence")

    assert result.smt_applicable is True
    assert result.smt_passed is True
    assert result.verdict == "VERIFIED"
    assert result.counterexample is None


def test_smt_screen_patches_attaches_smt_results() -> None:
    patches = [
        ([_edit(EditOperator.REPLACE_OPERATOR, "<", ">")], "x > y", _validation()),
        ([_edit(EditOperator.REPLACE_LITERAL, "0", "1")], "x == 1", _validation()),
        ([_edit(EditOperator.DELETE_STMT, "return x", None)], "return x", _validation()),
    ]

    screened = smt_screen_patches(patches, original_source="x < y", top_k=3, timeout_ms=1000)

    assert len(screened) == 3
    assert all(len(item) == 4 for item in screened)
    assert all(isinstance(item[3], SMTResult) for item in screened)


def test_smt_screen_patches_timeout_returns_unknown(monkeypatch) -> None:
    original_check = smt_layer.check_patch_semantics

    def _slow_check(changed_expr, context_variables, property_type="equivalence"):
        time.sleep(0.2)
        return original_check(changed_expr, context_variables, property_type)

    monkeypatch.setattr(smt_layer, "check_patch_semantics", _slow_check)

    patches = [
        ([_edit(EditOperator.REPLACE_OPERATOR, "<", ">")], "x > y", _validation()),
    ]

    start = time.monotonic()
    screened = smt_screen_patches(patches, original_source="x < y", top_k=1, timeout_ms=50)
    elapsed = time.monotonic() - start

    assert screened[0][3].verdict == "UNKNOWN"
    assert elapsed < 1.0
