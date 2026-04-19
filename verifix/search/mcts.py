from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, ValidationResult
from verifix.search.state import MCTSNode, backpropagate, expand_node, initialize_root, rollout


class ValidatorProtocol(Protocol):
    def validate(self, source: str, bug_report: BugReport) -> ValidationResult: ...


@dataclass(frozen=False)
class MCTSSearchResult:
    plausible_patches: list[tuple[list[Edit], str, ValidationResult]]
    total_iterations: int
    total_validations: int
    wall_time_seconds: float
    tree_depth_reached: int
    terminated_by: str


def select(root: MCTSNode, exploration_constant: float) -> MCTSNode:
    node = root

    while True:
        if node.is_leaf() or node.unexplored_edits:
            return node
        node = node.best_child(exploration_constant)


def mcts_search(
    bug_report: BugReport,
    suspicious_lines: list[int],
    validator: ValidatorProtocol,
    config: VerifixConfig,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> MCTSSearchResult:
    root = initialize_root(bug_report, config, suspicious_lines)
    plausible_patches: list[tuple[list[Edit], str, ValidationResult]] = []
    validations_used = 0
    iterations_run = 0
    tree_depth_reached = root.state.depth
    terminated_by = "budget_exhausted"

    start_time = time.monotonic()

    for iteration in range(config.mcts_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed > config.mcts_time_budget_seconds:
            terminated_by = "time_limit"
            break

        if validations_used >= config.max_validations:
            terminated_by = "validation_cap"
            break

        if not _has_expandable_node(root):
            terminated_by = "all_explored"
            break

        node = select(root, config.mcts_exploration_constant)

        if not node.is_fully_expanded() and not node.is_terminal:
            child = expand_node(node, config)
            if child is None:
                backpropagate(node, 0.0)
                iterations_run += 1
                if progress_callback is not None:
                    progress_callback(iteration, validations_used, len(plausible_patches))
                continue
            node = child
            tree_depth_reached = max(tree_depth_reached, node.state.depth)

        should_validate = node.is_terminal or (
            config.validate_all_nodes and node.state.depth > 0
        )

        if should_validate and validations_used < config.max_validations:
            if node.validation_result is None:
                result = validator.validate(node.state.current_source, bug_report)
                validations_used += 1
                node.validation_result = result
            else:
                result = node.validation_result

            if result.is_plausible:
                plausible_patches.append((node.state.edit_sequence, node.state.current_source, result))

            reward = (
                1.0
                if result.is_plausible
                else (0.5 if result.all_failing_tests_pass else (0.2 if result.compiled else 0.0))
            )
        else:
            reward = rollout(node, config)

        backpropagate(node, reward)
        iterations_run += 1

        if progress_callback is not None:
            progress_callback(iteration, validations_used, len(plausible_patches))

    wall_time_seconds = time.monotonic() - start_time

    return MCTSSearchResult(
        plausible_patches=plausible_patches,
        total_iterations=iterations_run,
        total_validations=validations_used,
        wall_time_seconds=wall_time_seconds,
        tree_depth_reached=tree_depth_reached,
        terminated_by=terminated_by,
    )


def get_best_patches(
    result: MCTSSearchResult,
    max_patches: int = 10,
) -> list[tuple[list[Edit], str, ValidationResult]]:
    if max_patches <= 0:
        return []

    best_by_source: dict[str, tuple[float, tuple[list[Edit], str, ValidationResult]]] = {}

    for patch in result.plausible_patches:
        edits, source, validation = patch
        score = (
            (1.0 if validation.all_failing_tests_pass else 0.0)
            + (0.5 if validation.no_regression else 0.0)
            - (0.1 * len(edits))
        )

        existing = best_by_source.get(source)
        if existing is None or score > existing[0]:
            best_by_source[source] = (score, patch)

    ranked = sorted(
        best_by_source.values(),
        key=lambda item: (
            -item[0],
            len(item[1][0]),
            item[1][1],
        ),
    )

    return [entry[1] for entry in ranked[:max_patches]]


def _has_expandable_node(node: MCTSNode) -> bool:
    if node.unexplored_edits:
        return True
    return any(_has_expandable_node(child) for child in node.children)


__all__ = [
    "ValidatorProtocol",
    "MCTSSearchResult",
    "select",
    "mcts_search",
    "get_best_patches",
]
