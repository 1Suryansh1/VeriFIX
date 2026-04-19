from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, RepairState, ValidationResult
from verifix.edit_dsl.applicator import apply_edit_sequence, validate_syntax
from verifix.edit_dsl.operators import get_candidate_edits
from verifix.parser.ast_builder import build_ast


@dataclass(frozen=False)
class MCTSNode:
    state: RepairState
    parent: MCTSNode | None
    children: list[MCTSNode] = field(default_factory=list)
    unexplored_edits: list[Edit] = field(default_factory=list)
    visit_count: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    validation_result: ValidationResult | None = None

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")

        if self.parent is None:
            return self.total_reward

        parent_visits = max(self.parent.visit_count, 1)
        exploitation = self.total_reward / self.visit_count
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.unexplored_edits) == 0 and len(self.children) > 0

    def best_child(self, exploration_constant: float = 1.414) -> MCTSNode:
        if not self.children:
            raise ValueError("No children available for best_child")
        return max(self.children, key=lambda child: child.ucb1_score(exploration_constant))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def initialize_root(
    bug_report: BugReport,
    config: VerifixConfig,
    suspicious_lines: list[int],
) -> MCTSNode:
    annotated = build_ast(
        source=bug_report.buggy_source,
        file_path=bug_report.file_path,
        language=bug_report.language,
    )

    candidates = get_candidate_edits(
        annotated,
        suspicious_lines=suspicious_lines,
        max_edits_per_node=config.max_candidates_per_node,
    )
    candidates = _prioritize_candidates(candidates, suspicious_lines)

    root_metadata = dict(bug_report.metadata)
    root_metadata["_suspicious_lines"] = list(suspicious_lines)

    root_bug_report = BugReport(
        bug_id=bug_report.bug_id,
        language=bug_report.language,
        buggy_source=bug_report.buggy_source,
        file_path=bug_report.file_path,
        failing_tests=list(bug_report.failing_tests),
        passing_tests=list(bug_report.passing_tests),
        project_root=bug_report.project_root,
        metadata=root_metadata,
    )

    root_state = RepairState(
        bug_report=root_bug_report,
        edit_sequence=[],
        current_source=root_bug_report.buggy_source,
        depth=0,
        is_terminal=False,
    )

    return MCTSNode(
        state=root_state,
        parent=None,
        children=[],
        unexplored_edits=candidates,
        visit_count=0,
        total_reward=0.0,
        is_terminal=False,
        validation_result=None,
    )


def expand_node(
    node: MCTSNode,
    config: VerifixConfig,
) -> MCTSNode | None:
    if not node.unexplored_edits:
        return None

    # Loop to retry failed edits instead of giving up on the first failure.
    edit = None
    updated_source = node.state.current_source
    while node.unexplored_edits:
        candidate_edit = node.unexplored_edits.pop(0)
        candidate_source, edit_success = apply_edit_sequence(
            node.state.current_source, [candidate_edit]
        )
        if not edit_success or not edit_success[0]:
            continue

        syntax_ok, _ = validate_syntax(
            candidate_source, language=node.state.bug_report.language
        )
        if not syntax_ok:
            continue

        edit = candidate_edit
        updated_source = candidate_source
        break

    if edit is None:
        return None

    child_state = RepairState(
        bug_report=node.state.bug_report,
        edit_sequence=[*node.state.edit_sequence, edit],
        current_source=updated_source,
        depth=node.state.depth + 1,
        is_terminal=False,
    )

    child_terminal = child_state.depth >= config.mcts_max_depth
    child_state.is_terminal = child_terminal

    suspicious_lines = list(node.state.bug_report.metadata.get("_suspicious_lines", []))
    child_unexplored: list[Edit] = []

    if not child_terminal:
        annotated = build_ast(
            source=child_state.current_source,
            file_path=child_state.bug_report.file_path,
            language=child_state.bug_report.language,
        )
        child_unexplored = get_candidate_edits(
            annotated,
            suspicious_lines=suspicious_lines,
            max_edits_per_node=config.max_candidates_per_node,
        )
        child_unexplored = _prioritize_candidates(child_unexplored, suspicious_lines)

    child = MCTSNode(
        state=child_state,
        parent=node,
        children=[],
        unexplored_edits=child_unexplored,
        visit_count=0,
        total_reward=0.0,
        is_terminal=child_terminal,
        validation_result=None,
    )
    node.children.append(child)
    return child


def rollout(
    node: MCTSNode,
    config: VerifixConfig,
    max_steps: int = 3,
) -> float:
    if node.is_terminal or node.state.is_terminal:
        return 0.5

    temp_source = node.state.current_source
    temp_depth = node.state.depth
    candidate_edits = list(node.unexplored_edits)

    if not candidate_edits:
        return 0.5

    rng = _stable_rng(
        config,
        "rollout",
        node.state.state_id,
        str(node.visit_count),
        str(max_steps),
    )

    for _ in range(max_steps):
        if not candidate_edits:
            break

        idx = rng.randrange(len(candidate_edits))
        edit = candidate_edits.pop(idx)

        temp_source, success_flags = apply_edit_sequence(temp_source, [edit])
        if not success_flags or not success_flags[0]:
            continue

        syntax_ok, _ = validate_syntax(temp_source, language=node.state.bug_report.language)
        if not syntax_ok:
            return 0.0

        temp_depth += 1
        if temp_depth >= config.mcts_max_depth:
            return 0.5

    return 0.5


def backpropagate(node: MCTSNode, reward: float) -> None:
    current: MCTSNode | None = node
    while current is not None:
        current.visit_count += 1
        current.total_reward += reward
        current = current.parent


def _stable_rng(config: VerifixConfig, *parts: str) -> random.Random:
    material = "|".join(parts)
    seed_material = f"{config.random_seed}|{material}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _prioritize_candidates(candidates: list[Edit], suspicious_lines: list[int]) -> list[Edit]:
    line_rank = {line: idx for idx, line in enumerate(suspicious_lines)}
    default_line_rank = len(suspicious_lines) + 1000
    operator_rank = {
        "replace_operator": 0,
        "replace_expr": 1,
        "swap_operands": 2,
        "replace_literal": 3,
        "negate_condition": 4,
        "insert_stmt_before": 5,
        "insert_stmt_after": 5,
        "delete_stmt": 6,
        "unwrap_block": 7,
        "wrap_condition": 8,
    }

    def _key(edit: Edit) -> tuple[int, int, int, int, str]:
        line_priority = line_rank.get(edit.line_number, default_line_rank)
        op_priority = operator_rank.get(edit.operator.value, 99)
        replacement = edit.replacement_text or ""
        text_delta = abs(len(replacement) - len(edit.original_text or ""))
        target_score = 0 if edit.metadata.get("target") in {"comparison", "subscript_index", "call_argument"} else 1
        return (line_priority, op_priority, target_score, text_delta, edit.node_id)

    return sorted(candidates, key=_key)


__all__ = [
    "MCTSNode",
    "initialize_root",
    "expand_node",
    "rollout",
    "backpropagate",
]
