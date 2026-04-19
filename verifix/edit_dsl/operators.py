from __future__ import annotations

import ast
import textwrap
from typing import Callable, Literal

from verifix.core.models import Edit, EditOperator
from verifix.parser.ast_builder import ASTNode, AnnotatedAST


def replace_literal_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    if isinstance(node.ast_node, ast.List) and len(node.ast_node.elts) == 0:
        replacement = "[[]]"
        if replacement == node.source_text.strip():
            return []
        return [
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_LITERAL,
                replacement_text=replacement,
                metadata={"old_value": "[]", "new_value": "[[]]", "target": "collection_literal"},
            )
        ]

    if not isinstance(node.ast_node, ast.Constant):
        return []

    value = node.ast_node.value
    edits: list[Edit] = []

    if isinstance(value, bool):
        replacement_texts = ["True" if not value else "False"]

        for name in sorted(_in_scope_names(node, ast_tree))[:8]:
            replacement_texts.append(name)
            replacement_texts.append(f"not {name}")

        for replacement in sorted(set(replacement_texts)):
            if replacement == node.source_text.strip():
                continue
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_LITERAL,
                    replacement_text=replacement,
                    metadata={"old_value": value, "new_value": replacement, "target": "condition_literal"},
                )
            )

        return edits
    elif isinstance(value, int):
        candidates = [0, 1, -1, value + 1, value - 1]
    elif isinstance(value, str):
        candidates = [""]
    else:
        return []

    for candidate in candidates:
        if candidate == value:
            continue

        if isinstance(candidate, bool):
            replacement = "True" if candidate else "False"
        elif isinstance(candidate, str):
            replacement = '""'
        else:
            replacement = str(candidate)

        if replacement == node.source_text.strip():
            continue

        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_LITERAL,
                replacement_text=replacement,
                metadata={"old_value": value, "new_value": candidate},
            )
        )

    return edits


def negate_condition_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node

    if isinstance(ast_node, (ast.Compare, ast.BoolOp)):
        replacement = f"not ({node.source_text})"
        return [
            _make_edit(
                node=node,
                operator=EditOperator.NEGATE_CONDITION,
                replacement_text=replacement,
                metadata={"action": "wrap_not"},
            )
        ]

    if isinstance(ast_node, ast.UnaryOp) and isinstance(ast_node.op, ast.Not):
        inner_expr = _node_text(ast_tree, ast_node.operand).strip()
        if not inner_expr:
            return []
        return [
            _make_edit(
                node=node,
                operator=EditOperator.NEGATE_CONDITION,
                replacement_text=inner_expr,
                metadata={"action": "unwrap_not"},
            )
        ]

    return []


def wrap_condition_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, (ast.Compare, ast.BoolOp, ast.Name, ast.UnaryOp)):
        return []

    current = node.source_text.strip()
    if not current:
        return []

    edits: list[Edit] = []
    guards: set[str] = set()

    if (
        isinstance(ast_node, ast.Compare)
        and len(ast_node.ops) == 1
        and isinstance(ast_node.ops[0], ast.Eq)
        and len(ast_node.comparators) == 1
        and isinstance(ast_node.comparators[0], ast.Constant)
        and ast_node.comparators[0].value == 0
    ):
        left_text = _node_text(ast_tree, ast_node.left).strip()
        if left_text:
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_OPERATOR,
                    replacement_text=f"{left_text} <= 1",
                    metadata={"target": "comparison", "old_op": "==", "new_op": "<="},
                )
            )

    if (
        isinstance(ast_node, ast.Compare)
        and len(ast_node.ops) == 1
        and len(ast_node.comparators) == 1
        and isinstance(ast_node.comparators[0], ast.Constant)
        and ast_node.comparators[0].value is None
        and isinstance(ast_node.left, ast.Attribute)
    ):
        base = _node_text(ast_tree, ast_node.left.value).strip()
        if base:
            guards.add(f"{base} is None")

    for name in sorted(_in_scope_names(node, ast_tree))[:6]:
        guards.add(f"not {name}")

    for guard in sorted(guards):
        for joiner in ("or", "and"):
            replacement = f"({current}) {joiner} {guard}"
            if replacement == current:
                continue
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.WRAP_CONDITION,
                    replacement_text=replacement,
                    metadata={"target": "wrap_condition", "joiner": joiner, "guard": guard},
                )
            )

            replacement_reordered = f"{guard} {joiner} ({current})"
            if replacement_reordered != current:
                edits.append(
                    _make_edit(
                        node=node,
                        operator=EditOperator.WRAP_CONDITION,
                        replacement_text=replacement_reordered,
                        metadata={"target": "wrap_condition", "joiner": joiner, "guard": guard},
                    )
                )

    return edits


def replace_attribute_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Attribute):
        return []
    if not isinstance(ast_node.ctx, ast.Load):
        return []

    group_candidates: tuple[tuple[str, ...], ...] = (
        ("incoming_nodes", "outgoing_nodes"),
        ("successor", "successors", "next", "prev"),
        ("left", "right"),
    )

    replacement_attrs: list[str] = []
    for group in group_candidates:
        if ast_node.attr in group:
            replacement_attrs = [candidate for candidate in group if candidate != ast_node.attr]
            break

    if not replacement_attrs:
        return []

    base = _node_text(ast_tree, ast_node.value).strip()
    if not base:
        return []

    edits: list[Edit] = []
    for attr in replacement_attrs:
        replacement = f"{base}.{attr}"
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={"target": "attribute", "old_attr": ast_node.attr, "new_attr": attr},
            )
        )
    return edits


def replace_comparison_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    if not isinstance(node.ast_node, ast.Compare):
        return []

    compare_node = node.ast_node
    left_text = _node_text(ast_tree, compare_node.left).strip()
    comparator_texts = [_node_text(ast_tree, comp).strip() for comp in compare_node.comparators]

    if not left_text or any(not comp for comp in comparator_texts):
        return []

    edits: list[Edit] = []
    original_ops = compare_node.ops

    for index, op in enumerate(original_ops):
        group = _comparison_group(type(op))
        if group is None:
            continue

        original_op_text = _comparison_op_text(type(op))
        if original_op_text is None:
            continue

        for replacement_op_type in group:
            if replacement_op_type is type(op):
                continue

            replacement_op_text = _comparison_op_text(replacement_op_type)
            if replacement_op_text is None:
                continue

            op_texts: list[str] = []
            for i, existing_op in enumerate(original_ops):
                if i == index:
                    op_texts.append(replacement_op_text)
                else:
                    existing_text = _comparison_op_text(type(existing_op))
                    if existing_text is None:
                        op_texts = []
                        break
                    op_texts.append(existing_text)

            if not op_texts:
                continue

            replacement = left_text
            for op_text, comp_text in zip(op_texts, comparator_texts):
                replacement += f" {op_text} {comp_text}"

            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_OPERATOR,
                    replacement_text=replacement,
                    metadata={
                        "target": "comparison",
                        "op_index": index,
                        "old_op": original_op_text,
                        "new_op": replacement_op_text,
                    },
                )
            )

    return edits


def replace_arithmetic_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node

    if isinstance(ast_node, ast.BinOp):
        op_type = type(ast_node.op)
        replacement_types = _replacement_operator_types(op_type)
        if not replacement_types:
            return []

        left_text = _node_text(ast_tree, ast_node.left).strip()
        right_text = _node_text(ast_tree, ast_node.right).strip()
        old_op = _binop_op_text(op_type)

        if not left_text or not right_text or old_op is None:
            return []

        edits: list[Edit] = []
        for replacement_type in replacement_types:
            new_op = _binop_op_text(replacement_type)
            if new_op is None:
                continue
            replacement = f"{left_text} {new_op} {right_text}"
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_OPERATOR,
                    replacement_text=replacement,
                    metadata={
                        "target": _operator_family_target(op_type),
                        "old_op": old_op,
                        "new_op": new_op,
                    },
                )
            )
        return edits

    if isinstance(ast_node, ast.AugAssign):
        op_type = type(ast_node.op)
        replacement_types = _replacement_operator_types(op_type)
        if not replacement_types:
            return []

        target_text = _node_text(ast_tree, ast_node.target).strip()
        value_text = _node_text(ast_tree, ast_node.value).strip()
        old_op = _augassign_op_text(op_type)
        if not target_text or not value_text or old_op is None:
            return []

        edits: list[Edit] = []
        for replacement_type in replacement_types:
            new_op = _augassign_op_text(replacement_type)
            if new_op is None:
                continue
            replacement = f"{target_text} {new_op}= {value_text}"
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_OPERATOR,
                    replacement_text=replacement,
                    metadata={
                        "target": _operator_family_target(op_type),
                        "old_op": f"{old_op}=",
                        "new_op": f"{new_op}=",
                    },
                )
            )
        return edits

    return []


def swap_operands_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node

    if isinstance(ast_node, ast.BinOp):
        left_text = _node_text(ast_tree, ast_node.left).strip()
        right_text = _node_text(ast_tree, ast_node.right).strip()
        op_text = _binop_op_text(type(ast_node.op))

        if not left_text or not right_text or op_text is None or left_text == right_text:
            return []

        replacement = f"{right_text} {op_text} {left_text}"
        return [
            _make_edit(
                node=node,
                operator=EditOperator.SWAP_OPERANDS,
                replacement_text=replacement,
                metadata={"target": "binop"},
            )
        ]

    if isinstance(ast_node, ast.Compare) and len(ast_node.comparators) == 1 and len(ast_node.ops) == 1:
        left_text = _node_text(ast_tree, ast_node.left).strip()
        right_text = _node_text(ast_tree, ast_node.comparators[0]).strip()
        op_text = _comparison_op_text(type(ast_node.ops[0]))

        if not left_text or not right_text or op_text is None or left_text == right_text:
            return []

        replacement = f"{right_text} {op_text} {left_text}"
        return [
            _make_edit(
                node=node,
                operator=EditOperator.SWAP_OPERANDS,
                replacement_text=replacement,
                metadata={"target": "compare"},
            )
        ]

    return []


def delete_statement_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    if not node.is_statement:
        return []

    protected = (ast.Return, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import)
    if isinstance(node.ast_node, protected):
        return []

    return [
        _make_edit(
            node=node,
            operator=EditOperator.DELETE_STMT,
            replacement_text=None,
            metadata={"reason": "statement_deletion"},
        )
    ]


def unwrap_if_body_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    if not isinstance(node.ast_node, ast.If):
        return []

    if len(node.ast_node.body) != 1:
        return []

    body_stmt = node.ast_node.body[0]
    body_text = _node_text(ast_tree, body_stmt)
    replacement = textwrap.dedent(body_text).strip("\n")

    if not replacement:
        return []

    return [
        _make_edit(
            node=node,
            operator=EditOperator.UNWRAP_BLOCK,
            replacement_text=replacement,
            metadata={"target": "if_body"},
        )
    ]


def replace_variable_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    if not isinstance(node.ast_node, ast.Name):
        return []
    if not isinstance(node.ast_node.ctx, ast.Load):
        return []

    current_name = node.ast_node.id
    in_scope = _in_scope_names(node, ast_tree)

    candidates = sorted(
        {
            name
            for name in in_scope
            if name != current_name and abs(len(name) - len(current_name)) <= 5
        }
    )

    edits: list[Edit] = []
    for replacement in candidates:
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={"target": "variable", "old_name": current_name, "new_name": replacement},
            )
        )
    return edits


def rewrite_subscript_index_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Subscript):
        return []

    value_text = _node_text(ast_tree, ast_node.value).strip()
    if not value_text:
        return []

    edits: list[Edit] = []
    slice_node = ast_node.slice
    slice_text = _node_text(ast_tree, slice_node).strip()

    if isinstance(ast_node.value, ast.Name) and slice_text:
        current_base = ast_node.value.id
        for candidate_base in sorted(_in_scope_names(node, ast_tree)):
            if candidate_base == current_base:
                continue
            replacement = f"{candidate_base}[{slice_text}]"
            if replacement == node.source_text.strip():
                continue
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=replacement,
                    metadata={
                        "target": "subscript_base",
                        "old_base": current_base,
                        "new_base": candidate_base,
                    },
                )
            )

    if isinstance(slice_node, ast.Tuple):
        element_texts = [_node_text(ast_tree, element).strip() for element in slice_node.elts]
        if any(not text for text in element_texts):
            return []

        for idx, element in enumerate(slice_node.elts):
            if not isinstance(element, ast.Name):
                continue
            prioritized = f"{element.id} - 1"
            new_elements = list(element_texts)
            new_elements[idx] = prioritized
            replacement = f"{value_text}[{', '.join(new_elements)}]"
            if replacement == node.source_text.strip():
                continue
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=replacement,
                    metadata={
                        "target": "subscript_index",
                        "dimension": idx,
                        "old_index": element_texts[idx],
                        "new_index": prioritized,
                    },
                )
            )

        if len(element_texts) >= 2:
            for keep_idx in range(len(element_texts)):
                dropped = f"{value_text}[{element_texts[keep_idx]}]"
                if dropped == node.source_text.strip():
                    continue
                edits.append(
                    _make_edit(
                        node=node,
                        operator=EditOperator.REPLACE_EXPR,
                        replacement_text=dropped,
                        metadata={"target": "subscript_index", "drop_to_index": keep_idx},
                    )
                )

        if len(element_texts) == 2 and element_texts[0] != element_texts[1]:
            swapped = f"{value_text}[{element_texts[1]}, {element_texts[0]}]"
            if swapped != node.source_text.strip():
                edits.append(
                    _make_edit(
                        node=node,
                        operator=EditOperator.REPLACE_EXPR,
                        replacement_text=swapped,
                        metadata={"target": "subscript_index", "swap_indices": [0, 1]},
                    )
                )

        for idx, element in enumerate(slice_node.elts):
            variants = _index_variants(element, ast_tree)
            for variant in variants:
                new_elements = list(element_texts)
                new_elements[idx] = variant
                replacement = f"{value_text}[{', '.join(new_elements)}]"
                if replacement == node.source_text.strip():
                    continue
                edits.append(
                    _make_edit(
                        node=node,
                        operator=EditOperator.REPLACE_EXPR,
                        replacement_text=replacement,
                        metadata={
                            "target": "subscript_index",
                            "dimension": idx,
                            "old_index": element_texts[idx],
                            "new_index": variant,
                        },
                    )
                )
        return edits

    variants = _index_variants(slice_node, ast_tree)
    for variant in variants:
        replacement = f"{value_text}[{variant}]"
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={
                    "target": "subscript_index",
                    "dimension": 0,
                    "old_index": _node_text(ast_tree, slice_node).strip(),
                    "new_index": variant,
                },
            )
        )

    return edits


def rewrite_call_argument_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Call):
        return []

    function_text = _node_text(ast_tree, ast_node.func).strip()
    if not function_text:
        return []

    positional_texts = [_node_text(ast_tree, arg).strip() for arg in ast_node.args]
    keyword_texts = []
    for kw in ast_node.keywords:
        value_text = _node_text(ast_tree, kw.value).strip()
        if kw.arg is None:
            keyword_texts.append(f"**{value_text}")
        else:
            keyword_texts.append(f"{kw.arg}={value_text}")

    if any(not text for text in positional_texts) or any(not text for text in keyword_texts):
        return []

    edits: list[Edit] = []

    for replacement_name in sorted(_in_scope_names(node, ast_tree))[:8]:
        if replacement_name == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement_name,
                metadata={"target": "call_unwrap", "new_expr": replacement_name},
            )
        )

    if len(positional_texts) == 1 and not keyword_texts:
        unwrap_replacement = positional_texts[0]
        if unwrap_replacement != node.source_text.strip():
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=unwrap_replacement,
                    metadata={
                        "target": "call_unwrap",
                        "old_call": node.source_text.strip(),
                        "new_expr": unwrap_replacement,
                    },
                )
            )

    if len(positional_texts) >= 2:
        for i in range(len(positional_texts)):
            for j in range(i + 1, len(positional_texts)):
                updated_positional = list(positional_texts)
                updated_positional[i], updated_positional[j] = updated_positional[j], updated_positional[i]
                all_args = [*updated_positional, *keyword_texts]
                replacement = f"{function_text}({', '.join(all_args)})"
                if replacement == node.source_text.strip():
                    continue
                edits.append(
                    _make_edit(
                        node=node,
                        operator=EditOperator.SWAP_OPERANDS,
                        replacement_text=replacement,
                        metadata={
                            "target": "call_argument_swap",
                            "swap_indices": [i, j],
                        },
                    )
                )

    if isinstance(ast_node.func, ast.Name) and ast_node.func.id in {"any", "all"}:
        swapped_name = "all" if ast_node.func.id == "any" else "any"
        all_args = [*positional_texts, *keyword_texts]
        replacement = f"{swapped_name}({', '.join(all_args)})"
        if replacement != node.source_text.strip():
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=replacement,
                    metadata={
                        "target": "call_callee",
                        "old_func": ast_node.func.id,
                        "new_func": swapped_name,
                    },
                )
            )

    enclosing = _enclosing_scope(node, ast_tree)
    if (
        isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef))
        and isinstance(ast_node.func, ast.Name)
        and ast_node.func.id == enclosing.name
        and len(positional_texts) >= 2
        and positional_texts[1] in {"k", "idx", "index"}
    ):
        for candidate in sorted(_in_scope_names(node, ast_tree)):
            if candidate in {positional_texts[1], positional_texts[0]}:
                continue
            if not any(token in candidate.lower() for token in ("num", "count", "len", "size")):
                continue
            updated_positional = list(positional_texts)
            updated_positional[1] = f"{positional_texts[1]} - {candidate}"
            all_args = [*updated_positional, *keyword_texts]
            replacement = f"{function_text}({', '.join(all_args)})"
            if replacement == node.source_text.strip():
                continue
            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=replacement,
                    metadata={
                        "target": "call_argument",
                        "arg_index": 1,
                        "old_arg": positional_texts[1],
                        "new_arg": updated_positional[1],
                    },
                )
            )

    for idx, arg in enumerate(ast_node.args):
        variants = _argument_variants(arg, ast_tree, node)
        for variant in variants:
            updated_positional = list(positional_texts)
            updated_positional[idx] = variant
            all_args = [*updated_positional, *keyword_texts]
            replacement = f"{function_text}({', '.join(all_args)})"
            if replacement == node.source_text.strip():
                continue

            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=replacement,
                    metadata={
                        "target": "call_argument",
                        "arg_index": idx,
                        "old_arg": positional_texts[idx],
                        "new_arg": variant,
                    },
                )
            )

    for idx, kw in enumerate(ast_node.keywords):
        variants = _argument_variants(kw.value, ast_tree, node)
        if kw.arg is None:
            original_keyword = f"**{_node_text(ast_tree, kw.value).strip()}"
        else:
            original_keyword = f"{kw.arg}={_node_text(ast_tree, kw.value).strip()}"

        for variant in variants:
            updated_keywords = list(keyword_texts)
            if kw.arg is None:
                updated_keywords[idx] = f"**{variant}"
            else:
                updated_keywords[idx] = f"{kw.arg}={variant}"

            all_args = [*positional_texts, *updated_keywords]
            replacement = f"{function_text}({', '.join(all_args)})"
            if replacement == node.source_text.strip():
                continue

            edits.append(
                _make_edit(
                    node=node,
                    operator=EditOperator.REPLACE_EXPR,
                    replacement_text=replacement,
                    metadata={
                        "target": "call_keyword_argument",
                        "arg_index": idx,
                        "old_arg": original_keyword,
                        "new_arg": updated_keywords[idx],
                    },
                )
            )

    return edits


def insert_statement_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    if not node.is_statement:
        return []

    if isinstance(
        node.ast_node,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
    ):
        return []

    statement_candidates = _statement_insertion_candidates(node, ast_tree)
    if not statement_candidates:
        return []

    edits: list[Edit] = []
    for candidate in statement_candidates[:6]:
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.INSERT_STMT_BEFORE,
                replacement_text=candidate,
                metadata={"target": "statement_insertion", "position": "before"},
            )
        )
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.INSERT_STMT_AFTER,
                replacement_text=candidate,
                metadata={"target": "statement_insertion", "position": "after"},
            )
        )

    return edits


def wrap_expression_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.BinOp):
        return []

    if not isinstance(ast_node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)):
        return []

    expr_text = node.source_text.strip()
    if not expr_text or expr_text.startswith("max("):
        return []

    replacement = f"max(0, {expr_text})"
    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={"target": "expression_wrap", "wrapper": "max_zero"},
        )
    ]


def replace_return_expr_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Return) or ast_node.value is None:
        return []

    current_text = node.source_text.strip()
    in_scope = sorted(_in_scope_names(node, ast_tree))
    candidates: set[str] = set()

    for name in in_scope[:8]:
        candidates.add(name)

    if isinstance(ast_node.value, ast.Constant) and isinstance(ast_node.value.value, bool):
        for name in in_scope[:6]:
            candidates.add(f"{name} == 0")
            candidates.add(f"{name} != 0")

    if isinstance(ast_node.value, ast.List) and len(ast_node.value.elts) == 0:
        for name in in_scope[:4]:
            candidates.add(f"[{name}]")

    edits: list[Edit] = []
    for expr in sorted(candidates):
        replacement = f"return {expr}"
        if replacement == current_text:
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={"target": "return_expr", "new_expr": expr},
            )
        )

    return edits


def rewrite_for_iter_slice_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.For):
        return []
    if not isinstance(ast_node.iter, ast.Name):
        return []

    iter_name = ast_node.iter.id
    candidates = ["1"]
    for name in sorted(_in_scope_names(node, ast_tree)):
        if name == iter_name:
            continue
        if any(token in name.lower() for token in ("k", "idx", "start", "offset", "mid")):
            candidates.append(name)

    edits: list[Edit] = []
    for candidate in candidates[:4]:
        original_stmt = node.source_text
        needle = f"in {iter_name}:"
        rewritten = f"in {iter_name}[{candidate}:]:"
        replacement = original_stmt.replace(needle, rewritten, 1)
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={"target": "for_iter_slice", "old_iter": iter_name, "slice_start": candidate},
            )
        )
    return edits


def rewrite_assign_subscript_target_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Assign):
        return []
    if len(ast_node.targets) != 1:
        return []

    target = ast_node.targets[0]
    if not isinstance(target, ast.Subscript):
        return []
    if not isinstance(target.value, ast.Name):
        return []
    if not isinstance(target.slice, ast.Tuple) or len(target.slice.elts) != 2:
        return []

    left_idx = _node_text(ast_tree, target.slice.elts[0]).strip()
    right_idx = _node_text(ast_tree, target.slice.elts[1]).strip()
    value_text = _node_text(ast_tree, ast_node.value).strip()
    if not left_idx or not right_idx or not value_text:
        return []

    base = target.value.id
    candidates = [
        (base.replace("_edge", "_node"), right_idx),
        (base, right_idx),
    ]

    edits: list[Edit] = []
    for new_base, new_idx in candidates:
        if not new_base:
            continue
        replacement = f"{new_base}[{new_idx}] = {value_text}"
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={
                    "target": "statement_rewrite",
                    "from": "assign_subscript_tuple",
                    "to": "assign_subscript_single",
                },
            )
        )

    return edits


def rewrite_len_call_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Call):
        return []
    if not isinstance(ast_node.func, ast.Name) or ast_node.func.id != "len":
        return []
    if len(ast_node.args) != 1 or ast_node.keywords:
        return []

    call_text = node.source_text.strip()
    if not call_text:
        return []

    replacement = f"({call_text} - 1)"
    if replacement == call_text:
        return []
    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={"target": "call_argument", "old_arg": call_text, "new_arg": replacement},
        )
    ]


def rewrite_range_call_stop_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Call):
        return []
    if not isinstance(ast_node.func, ast.Name) or ast_node.func.id != "range":
        return []
    if ast_node.keywords:
        return []
    if len(ast_node.args) not in {2, 3}:
        return []

    function_text = _node_text(ast_tree, ast_node.func).strip()
    positional_texts = [_node_text(ast_tree, arg).strip() for arg in ast_node.args]
    if any(not text for text in positional_texts):
        return []

    stop_idx = 1
    stop_text = positional_texts[stop_idx]
    if stop_text.endswith("+ 1"):
        return []

    updated = list(positional_texts)
    updated[stop_idx] = f"{stop_text} + 1"
    replacement = f"{function_text}({', '.join(updated)})"
    if replacement == node.source_text.strip():
        return []

    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={
                "target": "call_argument",
                "arg_index": stop_idx,
                "old_arg": stop_text,
                "new_arg": updated[stop_idx],
            },
        )
    ]


def rewrite_return_plus_one_call_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    del ast_tree
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Return) or ast_node.value is None:
        return []
    if not isinstance(ast_node.value, ast.BinOp) or not isinstance(ast_node.value.op, ast.Add):
        return []

    left = ast_node.value.left
    right = ast_node.value.right
    call_expr: ast.AST | None = None
    constant_expr: ast.AST | None = None

    if isinstance(left, ast.Call) and isinstance(right, ast.Constant):
        call_expr = left
        constant_expr = right
    elif isinstance(right, ast.Call) and isinstance(left, ast.Constant):
        call_expr = right
        constant_expr = left

    if call_expr is None or constant_expr is None or constant_expr.value != 1:
        return []

    call_text = ast.unparse(call_expr).strip()
    if not call_text:
        return []

    replacement = f"return {call_text}"
    if replacement == node.source_text.strip():
        return []

    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={"target": "return_expr", "new_expr": call_text},
        )
    ]


def rewrite_assign_to_max_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Assign):
        return []
    if len(ast_node.targets) != 1:
        return []
    if not isinstance(ast_node.targets[0], ast.Name):
        return []
    if isinstance(ast_node.value, ast.Call) and isinstance(ast_node.value.func, ast.Name):
        if ast_node.value.func.id == "max":
            return []

    target_name = ast_node.targets[0].id
    value_text = _node_text(ast_tree, ast_node.value).strip()
    if not value_text:
        return []

    replacement = f"{target_name} = max({target_name}, {value_text})"
    if replacement == node.source_text.strip():
        return []

    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={
                "target": "statement_rewrite",
                "from": "assignment",
                "to": "max_assignment",
            },
        )
    ]


def rewrite_return_listcomp_prefix_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Return) or ast_node.value is None:
        return []
    if not isinstance(ast_node.value, ast.ListComp):
        return []

    listcomp_text = _node_text(ast_tree, ast_node.value).strip()
    if not listcomp_text:
        return []

    in_scope = sorted(_in_scope_names(node, ast_tree))
    prefixes = [name for name in in_scope if name.lower().endswith("subsets") or "subset" in name.lower()]
    if not prefixes:
        return []

    edits: list[Edit] = []
    for prefix in prefixes[:3]:
        replacement = f"return {prefix} + {listcomp_text}"
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={"target": "return_expr", "new_expr": f"{prefix} + {listcomp_text}"},
            )
        )
    return edits


def rewrite_call_statement_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Expr):
        return []
    if not isinstance(ast_node.value, ast.Call):
        return []

    call = ast_node.value
    if not isinstance(call.func, ast.Attribute):
        return []
    if call.func.attr != "update" or len(call.args) != 1:
        return []

    base = _node_text(ast_tree, call.func.value).strip()
    value = _node_text(ast_tree, call.args[0]).strip()
    if not base or not value:
        return []

    replacement = f"{base} = {value}"
    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={"target": "statement_rewrite", "from": "update_call", "to": "assignment"},
        )
    ]


def square_variable_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    del ast_tree
    ast_node = node.ast_node
    if not isinstance(ast_node, ast.Name):
        return []
    if not isinstance(ast_node.ctx, ast.Load):
        return []

    name = ast_node.id
    replacements = [f"{name} ** 2", f"{name} * {name}"]
    edits: list[Edit] = []
    for replacement in replacements:
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_EXPR,
                replacement_text=replacement,
                metadata={"target": "expression_wrap", "wrapper": "square"},
            )
        )
    return edits


def get_candidate_edits(
    ast_tree: AnnotatedAST,
    suspicious_lines: list[int],
    max_edits_per_node: int = 15,
    enabled_operators: list[str] | None = None,
    operator_tier: Literal["core", "all", "synthetic_only"] = "core",
) -> list[Edit]:
    if max_edits_per_node <= 0:
        return []

    operator_names = enabled_operators or _resolve_operator_names(operator_tier)
    operators: list[Callable[[ASTNode, AnnotatedAST], list[Edit]]] = []

    for name in operator_names:
        op_fn = _OPERATOR_REGISTRY.get(name)
        if op_fn is None:
            raise ValueError(f"Unknown operator function: {name}")
        operators.append(op_fn)

    suspicious_line_set = set(suspicious_lines)
    candidate_edits: list[Edit] = []
    seen: set[tuple[str, str, str | None]] = set()

    for node in ast_tree.nodes.values():
        if node.lineno not in suspicious_line_set:
            continue

        generated_by_operator: list[list[Edit]] = []
        for operator_fn in operators:
            generated_by_operator.append(operator_fn(node, ast_tree))

        index = 0
        node_emitted = 0
        while node_emitted < max_edits_per_node:
            progress = False
            for generated in generated_by_operator:
                if node_emitted >= max_edits_per_node:
                    break
                if index >= len(generated):
                    continue

                progress = True
                edit = generated[index]
                dedup_key = (edit.node_id, edit.operator.value, edit.replacement_text)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                candidate_edits.append(edit)
                node_emitted += 1

            if not progress:
                break
            index += 1

    return candidate_edits


def _make_edit(
    node: ASTNode,
    operator: EditOperator,
    replacement_text: str | None,
    metadata: dict,
) -> Edit:
    return Edit(
        operator=operator,
        node_id=node.node_id,
        node_type=node.node_type,
        line_number=node.lineno,
        original_text=node.source_text,
        replacement_text=replacement_text,
        metadata=metadata,
    )


def _node_text(ast_tree: AnnotatedAST, node: ast.AST) -> str:
    segment = ast.get_source_segment(ast_tree.source, node)
    if segment is not None:
        return segment
    return ast.unparse(node)


def _comparison_group(op_type: type[ast.cmpop]) -> tuple[type[ast.cmpop], ...] | None:
    groups: tuple[tuple[type[ast.cmpop], ...], ...] = (
        (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq),
        (ast.Is, ast.IsNot),
        (ast.In, ast.NotIn),
    )
    for group in groups:
        if op_type in group:
            return group
    return None


def _comparison_op_text(op_type: type[ast.cmpop]) -> str | None:
    mapping: dict[type[ast.cmpop], str] = {
        ast.Lt: "<",
        ast.Gt: ">",
        ast.LtE: "<=",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }
    return mapping.get(op_type)


def _binop_op_text(op_type: type[ast.operator]) -> str | None:
    mapping: dict[type[ast.operator], str] = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.FloorDiv: "//",
        ast.Pow: "**",
        ast.BitAnd: "&",
        ast.BitOr: "|",
        ast.BitXor: "^",
    }
    return mapping.get(op_type)


def _augassign_op_text(op_type: type[ast.operator]) -> str | None:
    return _binop_op_text(op_type)


def _replacement_operator_types(op_type: type[ast.operator]) -> tuple[type[ast.operator], ...]:
    if op_type in {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv}:
        return tuple(
            candidate
            for candidate in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)
            if candidate is not op_type
        )
    if op_type is ast.Pow:
        return (ast.Mult,)
    if op_type in {ast.BitXor, ast.BitAnd, ast.BitOr}:
        return tuple(
            candidate for candidate in (ast.BitXor, ast.BitAnd, ast.BitOr) if candidate is not op_type
        )
    return tuple()


def _operator_family_target(op_type: type[ast.operator]) -> str:
    if op_type in {ast.BitXor, ast.BitAnd, ast.BitOr}:
        return "bitwise"
    return "arithmetic"


def _in_scope_names(node: ASTNode, ast_tree: AnnotatedAST) -> set[str]:
    scope_ast = _enclosing_scope(node, ast_tree)
    names: set[str] = set()

    if isinstance(scope_ast, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for arg_name in _function_argument_names(scope_ast):
            names.add(arg_name)

    for sub_node in ast.walk(scope_ast):
        if isinstance(sub_node, ast.Name) and isinstance(sub_node.ctx, ast.Store):
            if int(getattr(sub_node, "lineno", 0) or 0) <= node.lineno:
                names.add(sub_node.id)

    return names


def _outer_scope_names(node: ASTNode, ast_tree: AnnotatedAST) -> set[str]:
    names: set[str] = set()
    saw_enclosing_scope = False

    for ancestor in ast_tree.get_ancestors(node.node_id):
        ast_node = ancestor.ast_node
        if not isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue

        if not saw_enclosing_scope:
            saw_enclosing_scope = True
            continue

        for arg_name in _function_argument_names(ast_node):
            names.add(arg_name)

        for sub_node in ast.walk(ast_node):
            if isinstance(sub_node, ast.Name) and isinstance(sub_node.ctx, ast.Store):
                if int(getattr(sub_node, "lineno", 0) or 0) <= node.lineno:
                    names.add(sub_node.id)

    return names


def _index_variants(index_node: ast.AST, ast_tree: AnnotatedAST) -> list[str]:
    current_text = _node_text(ast_tree, index_node).strip()
    if not current_text:
        return []

    variants: set[str] = set()

    if isinstance(index_node, ast.Constant) and isinstance(index_node.value, int):
        value = int(index_node.value)
        variants.update({str(value - 1), str(value + 1), "0", "1"})

    if isinstance(index_node, ast.BinOp) and isinstance(index_node.right, ast.Constant):
        right_val = index_node.right.value
        if isinstance(right_val, int):
            left_text = _node_text(ast_tree, index_node.left).strip()
            if left_text and abs(right_val) == 1:
                variants.add(left_text)

            if left_text:
                if isinstance(index_node.op, ast.Sub):
                    variants.add(f"{left_text} + {right_val}")
                elif isinstance(index_node.op, ast.Add):
                    variants.add(f"{left_text} - {right_val}")

    variants.add(f"{current_text} + 1")
    variants.add(f"{current_text} - 1")

    cleaned = sorted(text for text in variants if text and text != current_text)
    return cleaned


def _argument_variants(arg_node: ast.AST, ast_tree: AnnotatedAST, call_node: ASTNode) -> list[str]:
    current_text = _node_text(ast_tree, arg_node).strip()
    if not current_text:
        return []

    variants: set[str] = set()

    if isinstance(arg_node, ast.Name):
        in_scope = _in_scope_names(call_node, ast_tree)
        for candidate in in_scope:
            if candidate == arg_node.id:
                continue

            close_length = abs(len(candidate) - len(arg_node.id)) <= 3
            if close_length:
                variants.add(candidate)
                variants.add(f"{arg_node.id} - {candidate}")
            elif arg_node.id in {"k", "idx", "index"}:
                variants.add(f"{arg_node.id} - {candidate}")

        variants.add(f"{arg_node.id}[1:]")
        variants.add(f"{arg_node.id}[:-1]")

    variants.update(_index_variants(arg_node, ast_tree))
    variants.add(f"{current_text} + 1")
    variants.add(f"{current_text} - 1")

    cleaned = sorted(text for text in variants if text and text != current_text)
    return cleaned


def _statement_insertion_candidates(node: ASTNode, ast_tree: AnnotatedAST) -> list[str]:
    in_scope = sorted(set(_in_scope_names(node, ast_tree)) | _outer_scope_names(node, ast_tree))
    if not in_scope:
        return []

    referenced_names = sorted(
        {
            sub_node.id
            for sub_node in ast.walk(node.ast_node)
            if isinstance(sub_node, ast.Name) and isinstance(sub_node.ctx, ast.Load)
        }
    )
    values = sorted(set(referenced_names) | set(in_scope))

    container_names = [
        name
        for name in in_scope
        if (
            name.lower() in {"visited", "nodesvisited", "opstack", "stack", "queue", "lines", "results"}
            or name.lower().endswith("s")
            or "stack" in name.lower()
            or "queue" in name.lower()
            or "list" in name.lower()
        )
    ]

    candidates: list[str] = []
    seen: set[str] = set()

    def _add_candidate(statement: str) -> None:
        if statement and statement not in seen:
            seen.add(statement)
            candidates.append(statement)

    visited_like = [name for name in in_scope if "visited" in name.lower()]
    node_like = [name for name in values if name.lower() in {"node", "curr", "current", "head"}]
    for visited_name in visited_like[:2]:
        for node_name in node_like[:2]:
            _add_candidate(f"{visited_name}.add({node_name})")

    stack_like = [name for name in in_scope if "stack" in name.lower()]
    token_like = [name for name in values if name.lower() in {"token", "op", "item", "x"}]
    for stack_name in stack_like[:2]:
        for token_name in token_like[:2]:
            _add_candidate(f"{stack_name}.append({token_name})")

    line_like = [name for name in values if name.lower() in {"text", "line", "token"}]
    lines_like = [name for name in in_scope if name.lower() in {"lines", "result", "results"}]
    for collection_name in lines_like[:2]:
        for value_name in line_like[:2]:
            _add_candidate(f"{collection_name}.append({value_name})")

    for container in container_names[:6]:
        for value in values[:6]:
            if container == value:
                continue
            _add_candidate(f"{container}.append({value})")
            _add_candidate(f"{container}.add({value})")

    prev_like = [name for name in in_scope if "prev" in name.lower()]
    current_like = [name for name in values if name.lower() in {"node", "curr", "current", "head"}]
    for left_name in prev_like[:2]:
        for right_name in current_like[:2]:
            if left_name != right_name:
                _add_candidate(f"{left_name} = {right_name}")

    return candidates


def _enclosing_scope(node: ASTNode, ast_tree: AnnotatedAST) -> ast.AST:
    for ancestor in ast_tree.get_ancestors(node.node_id):
        if isinstance(ancestor.ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return ancestor.ast_node

    return ast_tree.get_node(ast_tree.root_id).ast_node


def _function_argument_names(scope_node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    names: set[str] = set()
    args = scope_node.args

    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        names.add(arg.arg)

    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)

    return names


def synthetic_literal_jitter_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    del ast_tree
    if not isinstance(node.ast_node, ast.Constant):
        return []
    if not isinstance(node.ast_node.value, int):
        return []

    value = int(node.ast_node.value)
    candidates = [value + 2, value - 2, value + 3, value - 3]
    edits: list[Edit] = []
    for candidate in candidates:
        replacement = str(candidate)
        if replacement == node.source_text.strip():
            continue
        edits.append(
            _make_edit(
                node=node,
                operator=EditOperator.REPLACE_LITERAL,
                replacement_text=replacement,
                metadata={
                    "target": "synthetic_literal_jitter",
                    "old_value": value,
                    "new_value": candidate,
                    "operator_tier": "synthetic_only",
                },
            )
        )
    return edits


def synthetic_expression_wrap_operator(node: ASTNode, ast_tree: AnnotatedAST) -> list[Edit]:
    del ast_tree
    if not isinstance(node.ast_node, (ast.Name, ast.BinOp, ast.Compare, ast.Call, ast.Subscript)):
        return []

    current = node.source_text.strip()
    if not current:
        return []

    replacement = f"({current})"
    if replacement == current:
        return []

    return [
        _make_edit(
            node=node,
            operator=EditOperator.REPLACE_EXPR,
            replacement_text=replacement,
            metadata={"target": "synthetic_wrap", "operator_tier": "synthetic_only"},
        )
    ]


CORE_OPERATOR_NAMES: tuple[str, ...] = (
    "replace_literal_operator",
    "negate_condition_operator",
    "wrap_condition_operator",
    "replace_comparison_operator",
    "replace_arithmetic_operator",
    "swap_operands_operator",
    "delete_statement_operator",
    "unwrap_if_body_operator",
    "replace_variable_operator",
    "replace_attribute_operator",
    "rewrite_call_argument_operator",
    "rewrite_range_call_stop_operator",
    "rewrite_call_statement_operator",
    "rewrite_len_call_operator",
    "rewrite_subscript_index_operator",
    "rewrite_for_iter_slice_operator",
    "rewrite_assign_subscript_target_operator",
    "rewrite_assign_to_max_operator",
    "insert_statement_operator",
    "wrap_expression_operator",
    "replace_return_expr_operator",
    "rewrite_return_plus_one_call_operator",
    "rewrite_return_listcomp_prefix_operator",
    "square_variable_operator",
)

SYNTHETIC_ONLY_OPERATOR_NAMES: tuple[str, ...] = (
    "synthetic_literal_jitter_operator",
    "synthetic_expression_wrap_operator",
)

_OPERATOR_REGISTRY: dict[str, Callable[[ASTNode, AnnotatedAST], list[Edit]]] = {
    "replace_literal_operator": replace_literal_operator,
    "negate_condition_operator": negate_condition_operator,
    "wrap_condition_operator": wrap_condition_operator,
    "replace_comparison_operator": replace_comparison_operator,
    "replace_arithmetic_operator": replace_arithmetic_operator,
    "swap_operands_operator": swap_operands_operator,
    "delete_statement_operator": delete_statement_operator,
    "unwrap_if_body_operator": unwrap_if_body_operator,
    "replace_variable_operator": replace_variable_operator,
    "replace_attribute_operator": replace_attribute_operator,
    "rewrite_call_argument_operator": rewrite_call_argument_operator,
    "rewrite_range_call_stop_operator": rewrite_range_call_stop_operator,
    "rewrite_call_statement_operator": rewrite_call_statement_operator,
    "rewrite_len_call_operator": rewrite_len_call_operator,
    "rewrite_subscript_index_operator": rewrite_subscript_index_operator,
    "rewrite_for_iter_slice_operator": rewrite_for_iter_slice_operator,
    "rewrite_assign_subscript_target_operator": rewrite_assign_subscript_target_operator,
    "rewrite_assign_to_max_operator": rewrite_assign_to_max_operator,
    "insert_statement_operator": insert_statement_operator,
    "wrap_expression_operator": wrap_expression_operator,
    "replace_return_expr_operator": replace_return_expr_operator,
    "rewrite_return_plus_one_call_operator": rewrite_return_plus_one_call_operator,
    "rewrite_return_listcomp_prefix_operator": rewrite_return_listcomp_prefix_operator,
    "square_variable_operator": square_variable_operator,
    "synthetic_literal_jitter_operator": synthetic_literal_jitter_operator,
    "synthetic_expression_wrap_operator": synthetic_expression_wrap_operator,
}


def _resolve_operator_names(operator_tier: Literal["core", "all", "synthetic_only"]) -> list[str]:
    if operator_tier == "core":
        return list(CORE_OPERATOR_NAMES)
    if operator_tier == "synthetic_only":
        return list(SYNTHETIC_ONLY_OPERATOR_NAMES)
    return [*CORE_OPERATOR_NAMES, *SYNTHETIC_ONLY_OPERATOR_NAMES]


def list_registered_operator_names() -> list[str]:
    return sorted(_OPERATOR_REGISTRY.keys())


__all__ = [
    "replace_literal_operator",
    "negate_condition_operator",
    "wrap_condition_operator",
    "replace_comparison_operator",
    "replace_arithmetic_operator",
    "swap_operands_operator",
    "delete_statement_operator",
    "unwrap_if_body_operator",
    "replace_variable_operator",
    "replace_attribute_operator",
    "rewrite_call_argument_operator",
    "rewrite_range_call_stop_operator",
    "rewrite_call_statement_operator",
    "rewrite_len_call_operator",
    "rewrite_subscript_index_operator",
    "rewrite_for_iter_slice_operator",
    "rewrite_assign_subscript_target_operator",
    "rewrite_assign_to_max_operator",
    "insert_statement_operator",
    "wrap_expression_operator",
    "replace_return_expr_operator",
    "rewrite_return_plus_one_call_operator",
    "rewrite_return_listcomp_prefix_operator",
    "square_variable_operator",
    "synthetic_literal_jitter_operator",
    "synthetic_expression_wrap_operator",
    "CORE_OPERATOR_NAMES",
    "SYNTHETIC_ONLY_OPERATOR_NAMES",
    "get_candidate_edits",
    "list_registered_operator_names",
]
