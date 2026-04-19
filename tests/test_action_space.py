from __future__ import annotations

import pytest

from verifix.core.action_space import (
    ACTION_MAP,
    NUM_ACTIONS,
    action_id_to_name,
    action_name_to_id,
    operator_to_action_id,
)
from verifix.core.models import EditOperator


def test_action_map_is_contiguous_and_expected_size() -> None:
    assert NUM_ACTIONS == 15
    assert len(ACTION_MAP) == 15
    assert set(ACTION_MAP.values()) == set(range(NUM_ACTIONS))


def test_action_name_round_trip() -> None:
    for name, idx in ACTION_MAP.items():
        assert action_name_to_id(name) == idx
        assert action_id_to_name(idx) == name


def test_replace_operator_mapping_uses_target_metadata() -> None:
    comparison_id = operator_to_action_id(
        EditOperator.REPLACE_OPERATOR,
        metadata={"target": "comparison"},
    )
    arithmetic_id = operator_to_action_id(
        EditOperator.REPLACE_OPERATOR,
        metadata={"target": "arithmetic"},
    )

    assert comparison_id == ACTION_MAP["replace_comparison_operator"]
    assert arithmetic_id == ACTION_MAP["replace_arithmetic_operator"]
    assert operator_to_action_id(
        EditOperator.REPLACE_OPERATOR,
        metadata={"target": "bitwise"},
    ) == ACTION_MAP["replace_arithmetic_operator"]


def test_alias_operator_values_map_to_research_ids() -> None:
    assert operator_to_action_id(EditOperator.REPLACE_EXPR) == ACTION_MAP["replace_variable"]
    assert operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "subscript_index"},
    ) == ACTION_MAP["rewrite_subscript_index"]
    assert operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "subscript_base"},
    ) == ACTION_MAP["rewrite_subscript_index"]
    assert operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "call_keyword_argument"},
    ) == ACTION_MAP["rewrite_call_argument"]
    assert operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "call_unwrap"},
    ) == ACTION_MAP["rewrite_call_argument"]
    assert operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "expression_wrap"},
    ) == ACTION_MAP["wrap_expression"]
    assert operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "return_expr"},
    ) == ACTION_MAP["replace_return_expr"]
    assert operator_to_action_id(EditOperator.UNWRAP_BLOCK) == ACTION_MAP["unwrap_if_body"]
    assert operator_to_action_id(EditOperator.INSERT_STMT_AFTER) == ACTION_MAP["insert_stmt_after"]


def test_unknown_action_name_raises() -> None:
    with pytest.raises(ValueError):
        action_name_to_id("does_not_exist")
