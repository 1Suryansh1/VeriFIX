from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=False)
class BugReport:
    bug_id: str
    language: str
    buggy_source: str
    file_path: str
    failing_tests: list[str]
    passing_tests: list[str]
    project_root: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BugReport:
        return cls(
            bug_id=str(d["bug_id"]),
            language=str(d["language"]),
            buggy_source=str(d["buggy_source"]),
            file_path=str(d["file_path"]),
            failing_tests=list(d.get("failing_tests", [])),
            passing_tests=list(d.get("passing_tests", [])),
            project_root=str(d["project_root"]),
            metadata=dict(d.get("metadata", {})),
        )


class EditOperator(str, Enum):
    REPLACE_EXPR = "replace_expr"
    DELETE_STMT = "delete_stmt"
    INSERT_STMT_BEFORE = "insert_stmt_before"
    INSERT_STMT_AFTER = "insert_stmt_after"
    SWAP_OPERANDS = "swap_operands"
    REPLACE_LITERAL = "replace_literal"
    NEGATE_CONDITION = "negate_condition"
    REPLACE_OPERATOR = "replace_operator"
    UNWRAP_BLOCK = "unwrap_block"
    WRAP_CONDITION = "wrap_condition"


@dataclass(frozen=True)
class Edit:
    operator: EditOperator
    node_id: str
    node_type: str
    line_number: int
    original_text: str
    replacement_text: str | None
    metadata: dict[str, Any]


@dataclass(frozen=False)
class RepairState:
    bug_report: BugReport
    edit_sequence: list[Edit]
    current_source: str
    depth: int
    state_id: str = field(default="", init=False)
    is_terminal: bool = False

    def __post_init__(self) -> None:
        self.depth = len(self.edit_sequence)
        self.state_id = hashlib.sha256(self.current_source.encode("utf-8")).hexdigest()

    def apply_edit(self, edit: Edit) -> RepairState:
        updated_source = self._apply_text_edit(edit)
        return RepairState(
            bug_report=self.bug_report,
            edit_sequence=[*self.edit_sequence, edit],
            current_source=updated_source,
            depth=self.depth + 1,
            is_terminal=False,
        )

    def _apply_text_edit(self, edit: Edit) -> str:
        if not self.current_source:
            return self.current_source

        lines = self.current_source.splitlines(keepends=True)
        idx = edit.line_number - 1

        if 0 <= idx < len(lines):
            original_line = lines[idx]

            if edit.operator == EditOperator.DELETE_STMT or edit.replacement_text is None:
                if edit.original_text and edit.original_text in original_line:
                    lines[idx] = original_line.replace(edit.original_text, "", 1)
                else:
                    lines[idx] = ""
            else:
                replacement = edit.replacement_text
                if edit.original_text and edit.original_text in original_line:
                    lines[idx] = original_line.replace(edit.original_text, replacement, 1)
                else:
                    if original_line.endswith("\n") and not replacement.endswith("\n"):
                        replacement = replacement + "\n"
                    lines[idx] = replacement

            candidate = "".join(lines)
            if candidate != self.current_source:
                return candidate

        if edit.operator == EditOperator.DELETE_STMT or edit.replacement_text is None:
            if edit.original_text:
                return self.current_source.replace(edit.original_text, "", 1)
            return self.current_source

        if edit.original_text:
            return self.current_source.replace(edit.original_text, edit.replacement_text, 1)
        return self.current_source


@dataclass(frozen=False)
class ValidationResult:
    state_id: str
    compiled: bool
    tests_passed: list[str]
    tests_failed: list[str]
    all_failing_tests_pass: bool
    no_regression: bool
    is_plausible: bool
    compile_error: str | None
    runtime_error: str | None
    execution_time_ms: float

    def __post_init__(self) -> None:
        self.is_plausible = self.all_failing_tests_pass and self.no_regression


@dataclass(frozen=False)
class RankedPatch:
    rank: int
    edit_sequence: list[Edit]
    patched_source: str
    validation: ValidationResult
    score: float
    diff: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "edit_sequence": [_edit_to_dict(e) for e in self.edit_sequence],
            "patched_source": self.patched_source,
            "validation": _validation_to_dict(self.validation),
            "score": self.score,
            "diff": self.diff,
        }


@dataclass(frozen=False)
class RepairResult:
    bug_id: str
    success: bool
    ranked_patches: list[RankedPatch]
    total_states_explored: int
    total_validations_run: int
    wall_time_seconds: float
    search_tree_depth: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "success": self.success,
            "ranked_patches": [patch.to_dict() for patch in self.ranked_patches],
            "total_states_explored": self.total_states_explored,
            "total_validations_run": self.total_validations_run,
            "wall_time_seconds": self.wall_time_seconds,
            "search_tree_depth": self.search_tree_depth,
            "error": self.error,
        }


def _edit_to_dict(edit: Edit) -> dict[str, Any]:
    return {
        "operator": edit.operator.value,
        "node_id": edit.node_id,
        "node_type": edit.node_type,
        "line_number": edit.line_number,
        "original_text": edit.original_text,
        "replacement_text": edit.replacement_text,
        "metadata": dict(edit.metadata),
    }


def _validation_to_dict(validation: ValidationResult) -> dict[str, Any]:
    return {
        "state_id": validation.state_id,
        "compiled": validation.compiled,
        "tests_passed": list(validation.tests_passed),
        "tests_failed": list(validation.tests_failed),
        "all_failing_tests_pass": validation.all_failing_tests_pass,
        "no_regression": validation.no_regression,
        "is_plausible": validation.is_plausible,
        "compile_error": validation.compile_error,
        "runtime_error": validation.runtime_error,
        "execution_time_ms": validation.execution_time_ms,
    }


__all__ = [
    "BugReport",
    "EditOperator",
    "Edit",
    "RepairState",
    "ValidationResult",
    "RankedPatch",
    "RepairResult",
]
