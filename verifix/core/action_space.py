from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verifix.core.models import EditOperator

# Research-facing action map used by synthetic generation, training, and inference.
ACTION_MAP: dict[str, int] = {
    "replace_comparison_operator": 0,
    "negate_condition": 1,
    "replace_literal": 2,
    "delete_stmt": 3,
    "swap_operands": 4,
    "replace_arithmetic_operator": 5,
    "unwrap_if_body": 6,
    "replace_variable": 7,
    "insert_stmt_before": 8,
    "wrap_condition": 9,
    "rewrite_subscript_index": 10,
    "rewrite_call_argument": 11,
    "insert_stmt_after": 12,
    "wrap_expression": 13,
    "replace_return_expr": 14,
}

NUM_ACTIONS = 15
ACTION_ID_TO_NAME: dict[int, str] = {idx: name for name, idx in ACTION_MAP.items()}

_ACTION_NAME_TO_OPERATOR: dict[str, EditOperator] = {
    "replace_comparison_operator": EditOperator.REPLACE_OPERATOR,
    "negate_condition": EditOperator.NEGATE_CONDITION,
    "replace_literal": EditOperator.REPLACE_LITERAL,
    "delete_stmt": EditOperator.DELETE_STMT,
    "swap_operands": EditOperator.SWAP_OPERANDS,
    "replace_arithmetic_operator": EditOperator.REPLACE_OPERATOR,
    "unwrap_if_body": EditOperator.UNWRAP_BLOCK,
    "replace_variable": EditOperator.REPLACE_EXPR,
    "insert_stmt_before": EditOperator.INSERT_STMT_BEFORE,
    "wrap_condition": EditOperator.WRAP_CONDITION,
    "rewrite_subscript_index": EditOperator.REPLACE_EXPR,
    "rewrite_call_argument": EditOperator.REPLACE_EXPR,
    "insert_stmt_after": EditOperator.INSERT_STMT_AFTER,
    "wrap_expression": EditOperator.REPLACE_EXPR,
    "replace_return_expr": EditOperator.REPLACE_EXPR,
}


def _ensure_contract() -> None:
    if len(ACTION_MAP) != NUM_ACTIONS:
        raise ValueError(
            f"ACTION_MAP must contain exactly {NUM_ACTIONS} entries, found {len(ACTION_MAP)}"
        )

    expected_ids = set(range(NUM_ACTIONS))
    mapped_ids = set(ACTION_MAP.values())
    if mapped_ids != expected_ids:
        raise ValueError(
            f"ACTION_MAP ids must be contiguous [0, {NUM_ACTIONS - 1}], got {sorted(mapped_ids)}"
        )

    if set(_ACTION_NAME_TO_OPERATOR.keys()) != set(ACTION_MAP.keys()):
        raise ValueError("Action/operator contract mismatch: names differ between ACTION_MAP and adapter")


def action_name_to_id(action_name: str) -> int:
    try:
        return ACTION_MAP[action_name]
    except KeyError as exc:
        raise ValueError(f"Unknown action name: {action_name}") from exc


def action_id_to_name(action_id: int) -> str:
    try:
        return ACTION_ID_TO_NAME[action_id]
    except KeyError as exc:
        raise ValueError(f"Unknown action id: {action_id}") from exc


def action_name_to_operator(action_name: str) -> EditOperator:
    try:
        return _ACTION_NAME_TO_OPERATOR[action_name]
    except KeyError as exc:
        raise ValueError(f"Unknown action name: {action_name}") from exc


def operator_to_action_id(
    operator: EditOperator | str,
    metadata: Mapping[str, Any] | None = None,
) -> int:
    """
    Resolve an EditOperator value into one of the research action ids.

    Notes:
    - replace_operator is ambiguous between comparison/arithmetic replacement.
      metadata["target"] is used when available ("comparison" or "arithmetic").
      If absent, we default to comparison for deterministic behavior.
    """
    operator_value = operator.value if isinstance(operator, EditOperator) else str(operator)
    target = str((metadata or {}).get("target", "")).strip().lower()

    if operator_value == EditOperator.REPLACE_OPERATOR.value:
        if target in {"arithmetic", "bitwise"}:
            return ACTION_MAP["replace_arithmetic_operator"]
        return ACTION_MAP["replace_comparison_operator"]

    if operator_value == EditOperator.NEGATE_CONDITION.value:
        return ACTION_MAP["negate_condition"]
    if operator_value == EditOperator.REPLACE_LITERAL.value:
        return ACTION_MAP["replace_literal"]
    if operator_value == EditOperator.DELETE_STMT.value:
        return ACTION_MAP["delete_stmt"]
    if operator_value == EditOperator.SWAP_OPERANDS.value:
        return ACTION_MAP["swap_operands"]
    if operator_value == EditOperator.UNWRAP_BLOCK.value:
        return ACTION_MAP["unwrap_if_body"]
    if operator_value == EditOperator.REPLACE_EXPR.value:
        if target in {"subscript_index", "subscript_base"}:
            return ACTION_MAP["rewrite_subscript_index"]
        if target in {"call_argument", "call_keyword_argument", "call_callee", "call_unwrap"}:
            return ACTION_MAP["rewrite_call_argument"]
        if target == "expression_wrap":
            return ACTION_MAP["wrap_expression"]
        if target == "return_expr":
            return ACTION_MAP["replace_return_expr"]
        return ACTION_MAP["replace_variable"]
    if operator_value == EditOperator.INSERT_STMT_BEFORE.value:
        return ACTION_MAP["insert_stmt_before"]
    if operator_value == EditOperator.INSERT_STMT_AFTER.value:
        return ACTION_MAP["insert_stmt_after"]
    if operator_value == EditOperator.WRAP_CONDITION.value:
        return ACTION_MAP["wrap_condition"]

    raise ValueError(f"Unsupported operator for action mapping: {operator_value}")


def assert_action_contract() -> None:
    _ensure_contract()


_ensure_contract()

__all__ = [
    "ACTION_MAP",
    "NUM_ACTIONS",
    "ACTION_ID_TO_NAME",
    "action_name_to_id",
    "action_id_to_name",
    "action_name_to_operator",
    "operator_to_action_id",
    "assert_action_contract",
]
