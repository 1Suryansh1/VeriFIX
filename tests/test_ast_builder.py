from __future__ import annotations

import pytest

from verifix.parser.ast_builder import ParseError, build_ast, source_from_ast


BUGGY_SOURCE = """
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:   # BUG: should be >
            max_val = arr[i]
    return max_val
"""


def test_build_ast_returns_annotated_ast_with_expected_node_count() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    # Count expected for this fixture with positional Python nodes + Module root.
    assert len(annotated.nodes) == 29


def test_every_node_has_non_empty_node_id() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    assert all(node.node_id for node in annotated.nodes.values())


def test_node_ids_are_unique_within_tree() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    node_ids = [node.node_id for node in annotated.nodes.values()]
    assert len(node_ids) == len(set(node_ids))


def test_node_id_stability_same_source_same_ids() -> None:
    first = build_ast(BUGGY_SOURCE, "src/find_max.py")
    second = build_ast(BUGGY_SOURCE, "src/find_max.py")
    assert set(first.nodes.keys()) == set(second.nodes.keys())


def test_get_nodes_at_line_returns_if_node() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    nodes_line_5 = annotated.get_nodes_at_line(5)
    assert any(node.node_type == "If" for node in nodes_line_5)


def test_find_by_type_compare_returns_at_least_one() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    matches = annotated.find_by_type("Compare")
    assert len(matches) >= 1


def test_source_from_ast_no_modifications_returns_original_source() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    rebuilt = source_from_ast(annotated, {})
    assert rebuilt == BUGGY_SOURCE


def test_source_from_ast_replace_compare_text_produces_expected_source() -> None:
    annotated = build_ast(BUGGY_SOURCE, "src/find_max.py")
    compare_nodes = annotated.find_by_type("Compare")
    assert compare_nodes
    compare_node = compare_nodes[0]

    modified = source_from_ast(annotated, {compare_node.node_id: "arr[i] > max_val"})

    assert "if arr[i] > max_val:" in modified
    assert "if arr[i] < max_val:" not in modified


def test_build_ast_invalid_python_raises_parse_error() -> None:
    invalid_source = "def broken(:\n    pass\n"
    with pytest.raises(ParseError):
        build_ast(invalid_source, "src/broken.py")
