from __future__ import annotations

import ast
import difflib
import textwrap
from enum import Enum

from verifix.core.models import Edit, EditOperator


def apply_edit(source: str, edit: Edit) -> tuple[str, bool]:
    operator_value = edit.operator.value if isinstance(edit.operator, Enum) else str(edit.operator)

    replace_like_ops = {
        EditOperator.REPLACE_EXPR,
        EditOperator.REPLACE_LITERAL,
        EditOperator.NEGATE_CONDITION,
        EditOperator.WRAP_CONDITION,
        EditOperator.REPLACE_OPERATOR,
        EditOperator.SWAP_OPERANDS,
    }

    if edit.operator in replace_like_ops or operator_value == "replace_variable":
        return _apply_replace_near_line(source, edit)

    if edit.operator == EditOperator.DELETE_STMT:
        return _apply_delete_statement(source, edit)

    if edit.operator in {EditOperator.INSERT_STMT_BEFORE, EditOperator.INSERT_STMT_AFTER}:
        return _apply_insert_statement(source, edit)

    if edit.operator == EditOperator.UNWRAP_BLOCK:
        return _apply_unwrap_block(source, edit)

    return source, False


def apply_edit_sequence(source: str, edits: list[Edit]) -> tuple[str, list[bool]]:
    current = source
    success_flags: list[bool] = []

    for edit in edits:
        current, success = apply_edit(current, edit)
        success_flags.append(success)

    return current, success_flags


def validate_syntax(source: str, language: str = "python") -> tuple[bool, str | None]:
    normalized = language.lower()

    if normalized == "java":
        return True, None

    if normalized == "python":
        try:
            ast.parse(source)
            return True, None
        except SyntaxError as exc:
            return False, str(exc)

    return False, f"Unsupported language: {language}"


def generate_diff(original: str, modified: str, file_path: str = "buggy.py") -> str:
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=3,
    )
    return "".join(diff_lines)


def _apply_replace_near_line(source: str, edit: Edit) -> tuple[str, bool]:
    if not edit.original_text or edit.replacement_text is None:
        return source, False

    # Some operators replace full multi-line statements (e.g., for-loop rewrites).
    # Try an exact block replacement first before line-local replacement.
    if "\n" in edit.original_text:
        replaced = source.replace(edit.original_text, edit.replacement_text, 1)
        if replaced != source:
            return replaced, True

    lines = source.splitlines(keepends=True)
    target_index = max(0, edit.line_number - 1)
    start_index = max(0, target_index - 3)
    end_index = min(len(lines) - 1, target_index + 3)

    search_order = sorted(range(start_index, end_index + 1), key=lambda idx: abs(idx - target_index))

    for idx in search_order:
        if edit.original_text not in lines[idx]:
            continue
        lines[idx] = lines[idx].replace(edit.original_text, edit.replacement_text, 1)
        return "".join(lines), True

    return source, False


def _apply_delete_statement(source: str, edit: Edit) -> tuple[str, bool]:
    lines = source.splitlines(keepends=True)
    idx = edit.line_number - 1

    if idx < 0 or idx >= len(lines):
        return source, False

    deleted_line = lines[idx]
    deleted_indent = _leading_whitespace(deleted_line)
    deleted_stripped = deleted_line.strip()
    removed_header = deleted_stripped.endswith(":")

    del lines[idx]

    if removed_header:
        lines = _dedent_following_block(lines, idx, deleted_indent)

    lines = _insert_pass_for_empty_blocks(lines)
    return "".join(lines), True


def _apply_insert_statement(source: str, edit: Edit) -> tuple[str, bool]:
    if edit.replacement_text is None:
        return source, False

    lines = source.splitlines(keepends=True)
    idx = edit.line_number - 1

    if idx < 0 or idx >= len(lines):
        return source, False

    insert_idx = idx
    indent = _leading_whitespace(lines[idx])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is not None:
        target_stmt: ast.stmt | None = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt):
                continue
            if int(getattr(node, "lineno", -1)) != edit.line_number:
                continue
            target_stmt = node
            break

        if target_stmt is not None:
            indent = _leading_whitespace(lines[int(target_stmt.lineno) - 1])
            if edit.operator == EditOperator.INSERT_STMT_BEFORE:
                insert_idx = int(target_stmt.lineno) - 1
            else:
                end_lineno = int(getattr(target_stmt, "end_lineno", target_stmt.lineno) or target_stmt.lineno)
                insert_idx = end_lineno
        elif edit.operator == EditOperator.INSERT_STMT_AFTER:
            insert_idx = idx + 1
    elif edit.operator == EditOperator.INSERT_STMT_AFTER:
        insert_idx = idx + 1

    insert_block = _indent_block(edit.replacement_text, indent)
    lines[insert_idx:insert_idx] = insert_block

    return "".join(lines), True


def _apply_unwrap_block(source: str, edit: Edit) -> tuple[str, bool]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, False

    target_node: ast.If | ast.For | ast.While | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While)) and int(getattr(node, "lineno", -1)) == edit.line_number:
            target_node = node
            break

    if target_node is None or not target_node.body:
        return source, False

    if target_node.end_lineno is None:
        return source, False

    body_start = int(target_node.body[0].lineno)
    body_end = int(target_node.body[-1].end_lineno or target_node.body[-1].lineno)
    source_lines = source.splitlines(keepends=True)

    if body_start < 1 or body_end > len(source_lines):
        return source, False

    body_text = "".join(source_lines[body_start - 1 : body_end])
    dedented = textwrap.dedent(body_text)
    replacement_lines = dedented.splitlines(keepends=True)

    block_start = int(target_node.lineno)
    block_end = int(target_node.end_lineno)

    source_lines[block_start - 1 : block_end] = replacement_lines
    source_lines = _insert_pass_for_empty_blocks(source_lines)
    return "".join(source_lines), True


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _indent_block(text: str, indent: str) -> list[str]:
    stripped = text.strip("\n")
    if not stripped:
        return []

    lines: list[str] = []
    for line in stripped.splitlines():
        if line.strip():
            lines.append(f"{indent}{line}\n")
        else:
            lines.append("\n")
    return lines


def _dedent_following_block(lines: list[str], start_idx: int, removed_indent: str) -> list[str]:
    removed_indent_len = len(removed_indent)
    indent_unit = "    "

    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue

        current_indent = _leading_whitespace(line)
        if len(current_indent) <= removed_indent_len:
            break

        if current_indent.startswith(removed_indent + indent_unit):
            lines[idx] = line[len(indent_unit) :]
        elif current_indent.startswith(removed_indent + "\t"):
            lines[idx] = line[1:]
        else:
            lines[idx] = line[len(removed_indent) :]

    return lines


def _insert_pass_for_empty_blocks(lines: list[str]) -> list[str]:
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if not stripped or not stripped.endswith(":"):
            idx += 1
            continue

        header_indent = _leading_whitespace(line)
        next_idx = idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1

        needs_pass = next_idx >= len(lines) or len(_leading_whitespace(lines[next_idx])) <= len(header_indent)
        if needs_pass:
            pass_line = f"{header_indent}    pass\n"
            lines.insert(idx + 1, pass_line)
            idx += 2
            continue

        idx += 1

    return lines


__all__ = [
    "apply_edit",
    "apply_edit_sequence",
    "validate_syntax",
    "generate_diff",
]
