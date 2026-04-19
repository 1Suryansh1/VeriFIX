from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any

import asttokens


class ParseError(ValueError):
    def __init__(self, source: str, file_path: str, original_exception: Exception) -> None:
        self.source = source
        self.file_path = file_path
        self.original_exception = original_exception
        message = f"Failed to parse source for {file_path}: {original_exception}"
        super().__init__(message)


@dataclass(frozen=False)
class ASTNode:
    node_id: str
    node_type: str
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None
    source_text: str
    parent_id: str | None
    children_ids: list[str]
    is_statement: bool
    is_expression: bool
    ast_node: Any = field(repr=False)


@dataclass(frozen=False)
class AnnotatedAST:
    source: str
    file_path: str
    language: str
    nodes: dict[str, ASTNode]
    root_id: str
    line_to_nodes: dict[int, list[str]]

    def get_node(self, node_id: str) -> ASTNode:
        return self.nodes[node_id]

    def get_nodes_at_line(self, line: int) -> list[ASTNode]:
        return [self.nodes[node_id] for node_id in self.line_to_nodes.get(line, [])]

    def get_statements(self) -> list[ASTNode]:
        return [node for node in self.nodes.values() if node.is_statement]

    def get_expressions(self) -> list[ASTNode]:
        return [node for node in self.nodes.values() if node.is_expression]

    def get_children(self, node_id: str) -> list[ASTNode]:
        node = self.get_node(node_id)
        return [self.nodes[child_id] for child_id in node.children_ids]

    def get_parent(self, node_id: str) -> ASTNode | None:
        parent_id = self.get_node(node_id).parent_id
        return self.nodes.get(parent_id) if parent_id else None

    def get_ancestors(self, node_id: str) -> list[ASTNode]:
        ancestors: list[ASTNode] = []
        current = self.get_node(node_id)
        while current is not None:
            ancestors.append(current)
            if current.parent_id is None:
                break
            current = self.nodes[current.parent_id]
        return ancestors

    def find_by_type(self, node_type: str) -> list[ASTNode]:
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def to_dict(self) -> dict[str, Any]:
        serialized_nodes: dict[str, dict[str, Any]] = {}
        for node_id, node in self.nodes.items():
            serialized_nodes[node_id] = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "lineno": node.lineno,
                "col_offset": node.col_offset,
                "end_lineno": node.end_lineno,
                "end_col_offset": node.end_col_offset,
                "source_text": node.source_text,
                "parent_id": node.parent_id,
                "children_ids": list(node.children_ids),
                "is_statement": node.is_statement,
                "is_expression": node.is_expression,
            }

        return {
            "source": self.source,
            "file_path": self.file_path,
            "language": self.language,
            "nodes": serialized_nodes,
            "root_id": self.root_id,
            "line_to_nodes": {line: list(node_ids) for line, node_ids in self.line_to_nodes.items()},
        }


def build_ast(source: str, file_path: str, language: str = "python") -> AnnotatedAST:
    normalized_language = language.lower()

    if normalized_language == "java":
        raise NotImplementedError("Java AST support coming in V1.1 — use tree-sitter")
    if normalized_language != "python":
        raise ValueError(f"Unsupported language: {language}")

    try:
        tokens = asttokens.ASTTokens(source, parse=True)
        tree = tokens.tree
    except Exception as exc:
        raise ParseError(source, file_path, exc) from exc

    nodes: dict[str, ASTNode] = {}
    line_to_nodes: dict[int, list[str]] = {}

    def should_include(node: ast.AST) -> bool:
        return isinstance(node, ast.Module) or (
            hasattr(node, "lineno") and hasattr(node, "col_offset")
        )

    def make_node_id(node: ast.AST) -> str:
        lineno = int(getattr(node, "lineno", 0) or 0)
        col_offset = int(getattr(node, "col_offset", 0) or 0)
        end_lineno = int(getattr(node, "end_lineno", 0) or 0)
        end_col_offset = int(getattr(node, "end_col_offset", 0) or 0)
        type_name = type(node).__name__
        material = f"{file_path}:{lineno}:{col_offset}:{end_lineno}:{end_col_offset}:{type_name}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

    def make_unique_node_id(node: ast.AST) -> str:
        base_material = make_node_id(node)
        if base_material not in nodes:
            return base_material

        # Some real-world files contain multiple nodes with identical positional/type tuples.
        # Keep IDs deterministic by deriving a stable suffix from collision order.
        collision_index = 1
        while True:
            candidate_material = f"{base_material}:{collision_index}"
            candidate = hashlib.sha256(candidate_material.encode("utf-8")).hexdigest()[:12]
            if candidate not in nodes:
                return candidate
            collision_index += 1

    def node_source_text(node: ast.AST) -> str:
        if isinstance(node, ast.Module):
            return source
        try:
            return tokens.get_text(node)
        except Exception:
            text = ast.get_source_segment(source, node)
            return text if text is not None else ""

    def visit(node: ast.AST, parent_id: str | None) -> None:
        current_parent_id = parent_id

        if should_include(node):
            node_id = make_unique_node_id(node)

            lineno = int(getattr(node, "lineno", 0) or 0)
            col_offset = int(getattr(node, "col_offset", 0) or 0)
            end_lineno = getattr(node, "end_lineno", None)
            end_col_offset = getattr(node, "end_col_offset", None)

            annotated = ASTNode(
                node_id=node_id,
                node_type=type(node).__name__,
                lineno=lineno,
                col_offset=col_offset,
                end_lineno=int(end_lineno) if end_lineno is not None else None,
                end_col_offset=int(end_col_offset) if end_col_offset is not None else None,
                source_text=node_source_text(node),
                parent_id=parent_id,
                children_ids=[],
                is_statement=isinstance(node, ast.stmt),
                is_expression=isinstance(node, ast.expr),
                ast_node=node,
            )

            nodes[node_id] = annotated
            if lineno > 0:
                line_to_nodes.setdefault(lineno, []).append(node_id)

            if parent_id is not None:
                nodes[parent_id].children_ids.append(node_id)

            current_parent_id = node_id

        for child in ast.iter_child_nodes(node):
            visit(child, current_parent_id)

    visit(tree, None)

    root_nodes = [node for node in nodes.values() if node.parent_id is None]
    if len(root_nodes) != 1:
        raise ValueError("Expected exactly one root node in annotated AST")

    return AnnotatedAST(
        source=source,
        file_path=file_path,
        language=normalized_language,
        nodes=nodes,
        root_id=root_nodes[0].node_id,
        line_to_nodes=line_to_nodes,
    )


def source_from_ast(annotated_ast: AnnotatedAST, modified_nodes: dict[str, str]) -> str:
    if not modified_nodes:
        return annotated_ast.source

    lines = annotated_ast.source.splitlines(keepends=True)

    def line_col_to_offset(lineno: int, col_offset: int) -> int:
        if lineno <= 0:
            return 0
        prefix = "".join(lines[: lineno - 1])
        return len(prefix) + col_offset

    replacements: list[tuple[int, int, str]] = []
    for node_id, new_text in modified_nodes.items():
        node = annotated_ast.get_node(node_id)
        if node.end_lineno is None or node.end_col_offset is None:
            raise ValueError(f"Node {node_id} has no end position and cannot be replaced")

        start = line_col_to_offset(node.lineno, node.col_offset)
        end = line_col_to_offset(node.end_lineno, node.end_col_offset)
        replacements.append((start, end, new_text))

    updated_source = annotated_ast.source
    for start, end, replacement_text in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated_source = updated_source[:start] + replacement_text + updated_source[end:]

    return updated_source


__all__ = [
    "ParseError",
    "ASTNode",
    "AnnotatedAST",
    "build_ast",
    "source_from_ast",
]
