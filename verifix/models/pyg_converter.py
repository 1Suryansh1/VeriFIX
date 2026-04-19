from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from verifix.parser.ast_builder import ASTNode, AnnotatedAST

# Complete Python AST node type vocabulary.
PYTHON_AST_VOCAB = {
    "UNKNOWN": 0,
    "Module": 1,
    "FunctionDef": 2,
    "AsyncFunctionDef": 3,
    "ClassDef": 4,
    "Return": 5,
    "Delete": 6,
    "Assign": 7,
    "AugAssign": 8,
    "AnnAssign": 9,
    "For": 10,
    "AsyncFor": 11,
    "While": 12,
    "If": 13,
    "With": 14,
    "AsyncWith": 15,
    "Raise": 16,
    "Try": 17,
    "Assert": 18,
    "Import": 19,
    "ImportFrom": 20,
    "Global": 21,
    "Nonlocal": 22,
    "Expr": 23,
    "Pass": 24,
    "Break": 25,
    "Continue": 26,
    "BoolOp": 27,
    "NamedExpr": 28,
    "BinOp": 29,
    "UnaryOp": 30,
    "Lambda": 31,
    "IfExp": 32,
    "Dict": 33,
    "Set": 34,
    "ListComp": 35,
    "SetComp": 36,
    "DictComp": 37,
    "GeneratorExp": 38,
    "Await": 39,
    "Yield": 40,
    "YieldFrom": 41,
    "Compare": 42,
    "Call": 43,
    "FormattedValue": 44,
    "JoinedStr": 45,
    "Constant": 46,
    "Attribute": 47,
    "Subscript": 48,
    "Starred": 49,
    "Name": 50,
    "List": 51,
    "Tuple": 52,
    "Slice": 53,
    "Load": 54,
    "Store": 55,
    "Del": 56,
    "Add": 57,
    "Sub": 58,
    "Mult": 59,
    "MatMult": 60,
    "Div": 61,
    "Mod": 62,
    "Pow": 63,
    "LShift": 64,
    "RShift": 65,
    "BitOr": 66,
    "BitXor": 67,
    "BitAnd": 68,
    "FloorDiv": 69,
    "Invert": 70,
    "Not": 71,
    "UAdd": 72,
    "USub": 73,
    "Eq": 74,
    "NotEq": 75,
    "Lt": 76,
    "LtE": 77,
    "Gt": 78,
    "GtE": 79,
    "Is": 80,
    "IsNot": 81,
    "In": 82,
    "NotIn": 83,
    "arguments": 84,
    "arg": 85,
}
VOCAB_SIZE = 86
FEATURE_DIM = 5


def annotated_ast_from_dict(payload: dict[str, Any]) -> AnnotatedAST:
    raw_nodes = payload.get("nodes", {})
    nodes: dict[str, ASTNode] = {}

    for node_id, raw_node in raw_nodes.items():
        nodes[node_id] = ASTNode(
            node_id=str(raw_node.get("node_id", node_id)),
            node_type=str(raw_node.get("node_type", "UNKNOWN")),
            lineno=int(raw_node.get("lineno", 0) or 0),
            col_offset=int(raw_node.get("col_offset", 0) or 0),
            end_lineno=(
                int(raw_node["end_lineno"]) if raw_node.get("end_lineno") is not None else None
            ),
            end_col_offset=(
                int(raw_node["end_col_offset"])
                if raw_node.get("end_col_offset") is not None
                else None
            ),
            source_text=str(raw_node.get("source_text", "")),
            parent_id=(
                str(raw_node["parent_id"]) if raw_node.get("parent_id") is not None else None
            ),
            children_ids=[str(item) for item in raw_node.get("children_ids", [])],
            is_statement=bool(raw_node.get("is_statement", False)),
            is_expression=bool(raw_node.get("is_expression", False)),
            ast_node=None,
        )

    line_to_nodes_raw = payload.get("line_to_nodes", {})
    line_to_nodes: dict[int, list[str]] = {}
    for key, value in line_to_nodes_raw.items():
        line_to_nodes[int(key)] = [str(item) for item in value]

    return AnnotatedAST(
        source=str(payload.get("source", "")),
        file_path=str(payload.get("file_path", "unknown.py")),
        language=str(payload.get("language", "python")),
        nodes=nodes,
        root_id=str(payload.get("root_id", "")),
        line_to_nodes=line_to_nodes,
    )


class ASTtoPyG:
    def convert(
        self,
        annotated_ast: AnnotatedAST,
        ochiai_scores: dict[int, float] | None = None,
        bug_node_id: str | None = None,
    ) -> tuple[Data, dict[str, int], Tensor | None]:
        """
        Returns:
          data: PyG Data object
          node_id_to_idx: mapping node_id -> integer index
          fault_labels: [num_nodes, 1] tensor when bug_node_id is provided
        """
        nodes = sorted(
            annotated_ast.nodes.values(),
            key=lambda item: (item.lineno, item.col_offset, item.node_id),
        )
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(nodes)}

        depth_memo: dict[str, int] = {}

        def get_depth(node_id: str) -> int:
            if node_id in depth_memo:
                return depth_memo[node_id]
            node = annotated_ast.nodes[node_id]
            if node.parent_id is None or node.parent_id not in annotated_ast.nodes:
                depth_memo[node_id] = 0
            else:
                depth_memo[node_id] = 1 + get_depth(node.parent_id)
            return depth_memo[node_id]

        max_depth = max((get_depth(node.node_id) for node in nodes), default=1)
        if max_depth <= 0:
            max_depth = 1

        features: list[list[float]] = []
        for node in nodes:
            type_id = PYTHON_AST_VOCAB.get(node.node_type, PYTHON_AST_VOCAB["UNKNOWN"])
            depth_norm = get_depth(node.node_id) / max_depth
            ochiai = 0.0
            if ochiai_scores is not None:
                ochiai = float(ochiai_scores.get(node.lineno, 0.0))

            features.append(
                [
                    type_id / VOCAB_SIZE,
                    float(node.is_statement),
                    float(node.is_expression),
                    depth_norm,
                    ochiai,
                ]
            )

        x_tensor = torch.tensor(features, dtype=torch.float)

        edges: list[list[int]] = []
        for node in nodes:
            child_idx = node_id_to_idx[node.node_id]
            if node.parent_id is None:
                continue
            parent_idx = node_id_to_idx.get(node.parent_id)
            if parent_idx is None:
                continue
            edges.append([parent_idx, child_idx])
            edges.append([child_idx, parent_idx])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        fault_labels: Tensor | None = None
        if bug_node_id is not None:
            labels = torch.zeros((len(nodes), 1), dtype=torch.float)
            if bug_node_id in node_id_to_idx:
                labels[node_id_to_idx[bug_node_id], 0] = 1.0
            fault_labels = labels

        data = Data(x=x_tensor, edge_index=edge_index)
        return data, node_id_to_idx, fault_labels


__all__ = [
    "PYTHON_AST_VOCAB",
    "VOCAB_SIZE",
    "FEATURE_DIM",
    "annotated_ast_from_dict",
    "ASTtoPyG",
]
