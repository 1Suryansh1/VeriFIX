from __future__ import annotations

from verifix.models.pyg_converter import ASTtoPyG, annotated_ast_from_dict
from verifix.parser.ast_builder import build_ast


SOURCE = """
def f(x):
    if x < 10:
        return x + 1
    return x
"""


def test_convert_returns_feature_and_edge_tensors() -> None:
    annotated = build_ast(SOURCE, "sample.py")
    converter = ASTtoPyG()

    data, node_id_to_idx, labels = converter.convert(annotated)

    assert data.x.shape[1] == 5
    assert data.edge_index.shape[0] == 2
    assert len(node_id_to_idx) == data.x.shape[0]
    assert labels is None


def test_convert_with_fault_label_marks_single_node() -> None:
    annotated = build_ast(SOURCE, "sample.py")
    first_node = next(iter(annotated.nodes.values()))
    converter = ASTtoPyG()

    data, node_id_to_idx, labels = converter.convert(annotated, bug_node_id=first_node.node_id)

    assert labels is not None
    assert labels.shape[0] == data.x.shape[0]
    assert float(labels.sum().item()) == 1.0
    assert float(labels[node_id_to_idx[first_node.node_id]].item()) == 1.0


def test_annotated_ast_from_dict_roundtrip() -> None:
    annotated = build_ast(SOURCE, "sample.py")
    payload = annotated.to_dict()

    restored = annotated_ast_from_dict(payload)
    assert restored.file_path == annotated.file_path
    assert restored.root_id == annotated.root_id
    assert set(restored.nodes.keys()) == set(annotated.nodes.keys())
