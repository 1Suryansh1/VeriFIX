from __future__ import annotations

from verifix.core.models import Edit, EditOperator
from verifix.edit_dsl.applicator import apply_edit, apply_edit_sequence, generate_diff, validate_syntax


ORIGINAL = """
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


def _edit(
    operator: EditOperator,
    line_number: int,
    original_text: str,
    replacement_text: str | None,
) -> Edit:
    return Edit(
        operator=operator,
        node_id="node-1",
        node_type="Any",
        line_number=line_number,
        original_text=original_text,
        replacement_text=replacement_text,
        metadata={},
    )


def test_apply_edit_replace_operator_changes_lt_to_gt() -> None:
    edit = _edit(EditOperator.REPLACE_OPERATOR, 5, "<", ">")

    modified, success = apply_edit(ORIGINAL, edit)

    assert success is True
    assert "if arr[i] > max_val:" in modified


def test_apply_edit_delete_stmt_removes_target_line() -> None:
    edit = _edit(EditOperator.DELETE_STMT, 4, "for i in range(1, len(arr)):", None)

    modified, success = apply_edit(ORIGINAL, edit)

    assert success is True
    assert "for i in range(1, len(arr)):" not in modified


def test_apply_edit_returns_false_when_original_not_found_near_line() -> None:
    edit = _edit(EditOperator.REPLACE_OPERATOR, 5, "this_text_does_not_exist", ">")

    modified, success = apply_edit(ORIGINAL, edit)

    assert success is False
    assert modified == ORIGINAL


def test_apply_edit_sequence_applies_two_edits_in_order() -> None:
    edits = [
        _edit(EditOperator.REPLACE_OPERATOR, 5, "<", ">"),
        _edit(EditOperator.REPLACE_LITERAL, 3, "0", "1"),
    ]

    modified, flags = apply_edit_sequence(ORIGINAL, edits)

    assert flags == [True, True]
    assert "if arr[i] > max_val:" in modified
    assert "arr[1]" in modified


def test_validate_syntax_valid_python() -> None:
    ok, err = validate_syntax(ORIGINAL, language="python")
    assert ok is True
    assert err is None


def test_validate_syntax_invalid_python() -> None:
    ok, err = validate_syntax("def f(: pass", language="python")
    assert ok is False
    assert isinstance(err, str)
    assert err


def test_generate_diff_contains_headers() -> None:
    modified = ORIGINAL.replace("<", ">")
    diff_text = generate_diff(ORIGINAL, modified, file_path="buggy.py")
    assert "---" in diff_text
    assert "+++" in diff_text


def test_delete_stmt_that_empties_block_inserts_pass() -> None:
    src = """
def f(x):
    if x > 0:
        x = x + 1
    return x
"""
    edit = _edit(EditOperator.DELETE_STMT, 4, "x = x + 1", None)

    modified, success = apply_edit(src, edit)
    syntax_ok, _ = validate_syntax(modified)

    assert success is True
    assert "pass" in modified
    assert syntax_ok is True


def test_apply_edit_sequence_continues_after_failure() -> None:
    edits = [
        _edit(EditOperator.REPLACE_OPERATOR, 5, "<", ">"),
        _edit(EditOperator.REPLACE_OPERATOR, 5, "missing_op", "=="),
    ]

    modified, flags = apply_edit_sequence(ORIGINAL, edits)

    assert flags == [True, False]
    assert "if arr[i] > max_val:" in modified


def test_apply_edit_wrap_condition_rewrites_inline_condition() -> None:
    src = """
def detect_cycle(node):
    hare = node
    if hare.successor is None:
        return False
"""
    edit = _edit(
        EditOperator.WRAP_CONDITION,
        4,
        "hare.successor is None",
        "hare is None or (hare.successor is None)",
    )

    modified, success = apply_edit(src, edit)

    assert success is True
    assert "if hare is None or (hare.successor is None):" in modified


def test_apply_edit_replace_expr_can_replace_multiline_statement_block() -> None:
    src = """
def f(arr):
    for x in arr:
        yield x
"""
    edit = _edit(
        EditOperator.REPLACE_EXPR,
        3,
        "for x in arr:\n        yield x",
        "for x in arr[1:]:\n        yield x",
    )

    modified, success = apply_edit(src, edit)

    assert success is True
    assert "for x in arr[1:]:" in modified


def test_insert_after_compound_statement_inserts_after_block() -> None:
    src = """
def f(tokens):
    for token in tokens:
        pass
    return tokens
"""
    edit = _edit(EditOperator.INSERT_STMT_AFTER, 3, "for token in tokens:", "seen.append(token)")

    modified, success = apply_edit(src, edit)
    syntax_ok, _ = validate_syntax(modified)

    assert success is True
    assert syntax_ok is True
    assert "for token in tokens:\n        pass\n    seen.append(token)" in modified
