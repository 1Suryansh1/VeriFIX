from __future__ import annotations

import ast
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

import z3

from verifix.core.models import Edit, EditOperator, ValidationResult


@dataclass(frozen=False)
class SMTResult:
    smt_applicable: bool
    smt_passed: bool
    counterexample: dict | None
    property_checked: str
    solver_time_ms: float
    verdict: str


def extract_changed_expression(
    original_source: str,
    patched_source: str,
    edit: Edit,
) -> dict | None:
    del original_source
    del patched_source

    original = edit.original_text.strip()
    patched = (edit.replacement_text or "").strip()

    if edit.operator == EditOperator.REPLACE_OPERATOR:
        if original in _COMPARISON_OPERATORS and patched in _COMPARISON_OPERATORS:
            return {
                "type": "comparison_op",
                "original": original,
                "patched": patched,
                "operands": ["x", "y"],
                "operand_types": ["int", "int"],
            }

        if original in _ARITHMETIC_OPERATORS and patched in _ARITHMETIC_OPERATORS:
            return {
                "type": "arithmetic_op",
                "original": original,
                "patched": patched,
                "operands": ["x", "y"],
                "operand_types": ["int", "int"],
            }

        return {
            "type": "comparison_op",
            "original_expr": original,
            "patched_expr": patched,
        }

    if edit.operator == EditOperator.REPLACE_LITERAL:
        try:
            old_value = ast.literal_eval(original)
            new_value = ast.literal_eval(patched)
        except Exception:
            try:
                old_value = int(original)
                new_value = int(patched)
            except Exception:
                return None

        if not isinstance(old_value, int) or not isinstance(new_value, int):
            return None

        return {
            "type": "literal_change",
            "old_value": old_value,
            "new_value": new_value,
            "variable": "x",
        }

    if edit.operator == EditOperator.NEGATE_CONDITION:
        return {
            "type": "negate_condition",
            "original_condition": original,
            "patched_condition": patched,
        }

    return None


def build_z3_formula(
    changed_expr: dict,
    context_variables: dict[str, str],
) -> tuple[Any, Any] | None:
    variables = _build_context_variables(context_variables)

    expr_type = changed_expr.get("type")
    if expr_type == "comparison_op":
        original_expr = changed_expr.get("original_expr")
        patched_expr = changed_expr.get("patched_expr")

        if isinstance(original_expr, str) and isinstance(patched_expr, str):
            original_formula = _ast_expr_to_z3(ast.parse(original_expr, mode="eval").body, variables)
            patched_formula = _ast_expr_to_z3(ast.parse(patched_expr, mode="eval").body, variables)
            if original_formula is None or patched_formula is None:
                return None
            return original_formula, patched_formula

        operands = changed_expr.get("operands", ["x", "y"])
        if not isinstance(operands, list) or len(operands) < 2:
            return None

        left = variables.get(str(operands[0]))
        right = variables.get(str(operands[1]))
        if left is None or right is None:
            return None

        original_op = changed_expr.get("original")
        patched_op = changed_expr.get("patched")

        original_formula = _comparison_formula(original_op, left, right)
        patched_formula = _comparison_formula(patched_op, left, right)
        if original_formula is None or patched_formula is None:
            return None
        return original_formula, patched_formula

    if expr_type == "arithmetic_op":
        operands = changed_expr.get("operands", ["x", "y"])
        if not isinstance(operands, list) or len(operands) < 2:
            return None
        left = variables.get(str(operands[0]))
        right = variables.get(str(operands[1]))
        if left is None or right is None:
            return None

        original_formula = _arithmetic_formula(changed_expr.get("original"), left, right)
        patched_formula = _arithmetic_formula(changed_expr.get("patched"), left, right)
        if original_formula is None or patched_formula is None:
            return None
        return original_formula, patched_formula

    if expr_type == "negate_condition":
        original_condition = changed_expr.get("original_condition")
        patched_condition = changed_expr.get("patched_condition")
        if not isinstance(original_condition, str) or not isinstance(patched_condition, str):
            return None

        original_formula = _ast_expr_to_z3(ast.parse(original_condition, mode="eval").body, variables)
        patched_formula = _ast_expr_to_z3(ast.parse(patched_condition, mode="eval").body, variables)
        if original_formula is None or patched_formula is None:
            return None
        return original_formula, patched_formula

    if expr_type == "literal_change":
        variable_name = str(changed_expr.get("variable", "x"))
        variable = variables.get(variable_name)
        if variable is None:
            return None

        old_value = changed_expr.get("old_value")
        new_value = changed_expr.get("new_value")
        if not isinstance(old_value, int) or not isinstance(new_value, int):
            return None

        return variable == z3.IntVal(old_value), variable == z3.IntVal(new_value)

    return None


def check_patch_semantics(
    changed_expr: dict,
    context_variables: dict[str, str],
    property_type: str = "monotone",
) -> SMTResult:
    start = time.monotonic()
    formulas = build_z3_formula(changed_expr, context_variables)
    if formulas is None:
        return SMTResult(
            smt_applicable=False,
            smt_passed=False,
            counterexample=None,
            property_checked=property_type,
            solver_time_ms=(time.monotonic() - start) * 1000.0,
            verdict="NOT_APPLICABLE",
        )

    original_formula, patched_formula = formulas

    solver = z3.Solver()
    timeout_ms = int(changed_expr.get("timeout_ms", 5000))
    solver.set(timeout=max(1, timeout_ms))

    property_name = property_type.lower()

    if property_name == "equivalence":
        solver.add(original_formula != patched_formula)
        check = solver.check()
        elapsed = (time.monotonic() - start) * 1000.0

        if check == z3.sat:
            model = solver.model()
            return SMTResult(
                smt_applicable=True,
                smt_passed=False,
                counterexample=_model_to_dict(model),
                property_checked=property_name,
                solver_time_ms=elapsed,
                verdict="COUNTEREXAMPLE_FOUND",
            )
        if check == z3.unsat:
            return SMTResult(
                smt_applicable=True,
                smt_passed=True,
                counterexample=None,
                property_checked=property_name,
                solver_time_ms=elapsed,
                verdict="VERIFIED",
            )
        return SMTResult(
            smt_applicable=True,
            smt_passed=False,
            counterexample=None,
            property_checked=property_name,
            solver_time_ms=elapsed,
            verdict="UNKNOWN",
        )

    if property_name == "soundness":
        if changed_expr.get("type") != "comparison_op":
            return SMTResult(
                smt_applicable=False,
                smt_passed=False,
                counterexample=None,
                property_checked=property_name,
                solver_time_ms=(time.monotonic() - start) * 1000.0,
                verdict="NOT_APPLICABLE",
            )

        ctx = _build_context_variables(context_variables)
        x = ctx.get("x")
        y = ctx.get("y")
        if x is None or y is None:
            return SMTResult(
                smt_applicable=False,
                smt_passed=False,
                counterexample=None,
                property_checked=property_name,
                solver_time_ms=(time.monotonic() - start) * 1000.0,
                verdict="NOT_APPLICABLE",
            )

        target = z3.ForAll([x, y], (x > y) == (z3.Not(x < y) | (x == y)))
        solver.add(z3.Not(target))
        check = solver.check()
        elapsed = (time.monotonic() - start) * 1000.0

        if check == z3.unsat:
            return SMTResult(True, True, None, property_name, elapsed, "VERIFIED")
        if check == z3.sat:
            return SMTResult(True, False, _model_to_dict(solver.model()), property_name, elapsed, "COUNTEREXAMPLE_FOUND")
        return SMTResult(True, False, None, property_name, elapsed, "UNKNOWN")

    if property_name == "monotone":
        elapsed = (time.monotonic() - start) * 1000.0
        return SMTResult(
            smt_applicable=True,
            smt_passed=False,
            counterexample=None,
            property_checked=property_name,
            solver_time_ms=elapsed,
            verdict="UNKNOWN",
        )

    elapsed = (time.monotonic() - start) * 1000.0
    return SMTResult(
        smt_applicable=False,
        smt_passed=False,
        counterexample=None,
        property_checked=property_name,
        solver_time_ms=elapsed,
        verdict="NOT_APPLICABLE",
    )


def smt_screen_patches(
    patches: list[tuple[list[Edit], str, ValidationResult]],
    original_source: str,
    top_k: int = 5,
    timeout_ms: float = 5000.0,
) -> list[tuple[list[Edit], str, ValidationResult, SMTResult]]:
    del original_source

    if top_k <= 0:
        return []

    candidates = patches[:top_k]
    screened: list[tuple[list[Edit], str, ValidationResult, SMTResult]] = []

    for edits, patched_source, validation in candidates:
        if not edits:
            screened.append((edits, patched_source, validation, _not_applicable_result("equivalence")))
            continue

        changed_expr = extract_changed_expression("", patched_source, edits[-1])
        if changed_expr is None:
            screened.append((edits, patched_source, validation, _not_applicable_result("equivalence")))
            continue

        context = _infer_context_from_expr(changed_expr)
        changed_expr = dict(changed_expr)
        changed_expr["timeout_ms"] = int(timeout_ms)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(check_patch_semantics, changed_expr, context, "equivalence")
            try:
                smt_result = future.result(timeout=max(0.001, timeout_ms / 1000.0))
            except FuturesTimeoutError:
                smt_result = SMTResult(
                    smt_applicable=True,
                    smt_passed=False,
                    counterexample=None,
                    property_checked="equivalence",
                    solver_time_ms=float(timeout_ms),
                    verdict="UNKNOWN",
                )

        screened.append((edits, patched_source, validation, smt_result))

    verdict_order = {"VERIFIED": 0, "UNKNOWN": 1, "COUNTEREXAMPLE_FOUND": 2, "NOT_APPLICABLE": 3}
    screened.sort(key=lambda item: verdict_order.get(item[3].verdict, 9))
    return screened


def _build_context_variables(context_variables: dict[str, str]) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    for name, sort in context_variables.items():
        lowered = sort.lower()
        if lowered == "bool":
            variables[name] = z3.Bool(name)
        elif lowered in {"real", "float"}:
            variables[name] = z3.Real(name)
        else:
            variables[name] = z3.Int(name)

    if not variables:
        variables["x"] = z3.Int("x")
        variables["y"] = z3.Int("y")

    return variables


def _comparison_formula(op: str, left: Any, right: Any) -> Any | None:
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    return None


def _arithmetic_formula(op: str, left: Any, right: Any) -> Any | None:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    return None


def _ast_expr_to_z3(node: ast.AST, variables: dict[str, Any]) -> Any | None:
    if isinstance(node, ast.Name):
        if node.id not in variables:
            variables[node.id] = z3.Int(node.id)
        return variables[node.id]

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return z3.BoolVal(node.value)
        if isinstance(node.value, int):
            return z3.IntVal(node.value)
        return None

    if isinstance(node, ast.UnaryOp):
        operand = _ast_expr_to_z3(node.operand, variables)
        if operand is None:
            return None
        if isinstance(node.op, ast.Not):
            return z3.Not(operand)
        if isinstance(node.op, ast.USub):
            return -operand
        return None

    if isinstance(node, ast.BinOp):
        left = _ast_expr_to_z3(node.left, variables)
        right = _ast_expr_to_z3(node.right, variables)
        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return None

    if isinstance(node, ast.BoolOp):
        values = [_ast_expr_to_z3(value, variables) for value in node.values]
        if any(value is None for value in values):
            return None
        if isinstance(node.op, ast.And):
            return z3.And(values)
        if isinstance(node.op, ast.Or):
            return z3.Or(values)
        return None

    if isinstance(node, ast.Compare):
        left = _ast_expr_to_z3(node.left, variables)
        if left is None:
            return None

        comparisons: list[Any] = []
        current_left = left
        for op, comparator in zip(node.ops, node.comparators):
            right = _ast_expr_to_z3(comparator, variables)
            if right is None:
                return None

            if isinstance(op, ast.Lt):
                comparisons.append(current_left < right)
            elif isinstance(op, ast.Gt):
                comparisons.append(current_left > right)
            elif isinstance(op, ast.LtE):
                comparisons.append(current_left <= right)
            elif isinstance(op, ast.GtE):
                comparisons.append(current_left >= right)
            elif isinstance(op, ast.Eq):
                comparisons.append(current_left == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(current_left != right)
            else:
                return None

            current_left = right

        return z3.And(comparisons) if len(comparisons) > 1 else comparisons[0]

    return None


def _model_to_dict(model: Any) -> dict:
    output: dict[str, Any] = {}
    for decl in model.decls():
        output[decl.name()] = str(model[decl])
    return output


def _infer_context_from_expr(changed_expr: dict) -> dict[str, str]:
    variables: dict[str, str] = {}

    def collect_from_text(text: str) -> None:
        try:
            expr = ast.parse(text, mode="eval")
        except Exception:
            return
        for node in ast.walk(expr):
            if isinstance(node, ast.Name):
                variables[node.id] = "Int"

    for key in ["original_expr", "patched_expr", "original_condition", "patched_condition"]:
        value = changed_expr.get(key)
        if isinstance(value, str):
            collect_from_text(value)

    operands = changed_expr.get("operands")
    if isinstance(operands, list):
        for operand in operands:
            if isinstance(operand, str) and operand:
                variables[operand] = "Int"

    if "variable" in changed_expr and isinstance(changed_expr["variable"], str):
        variables[changed_expr["variable"]] = "Int"

    if not variables:
        variables = {"x": "Int", "y": "Int"}

    return variables


def _not_applicable_result(property_checked: str) -> SMTResult:
    return SMTResult(
        smt_applicable=False,
        smt_passed=False,
        counterexample=None,
        property_checked=property_checked,
        solver_time_ms=0.0,
        verdict="NOT_APPLICABLE",
    )


_COMPARISON_OPERATORS = {"<", ">", "<=", ">=", "==", "!="}
_ARITHMETIC_OPERATORS = {"+", "-", "*", "/"}


__all__ = [
    "extract_changed_expression",
    "build_z3_formula",
    "check_patch_semantics",
    "SMTResult",
    "smt_screen_patches",
]
