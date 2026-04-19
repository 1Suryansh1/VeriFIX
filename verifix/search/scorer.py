from __future__ import annotations

import difflib
import warnings

from verifix.core.models import BugReport, Edit, RepairState


def edit_distance_score(state: RepairState, bug_report: BugReport) -> float:
    similarity = difflib.SequenceMatcher(
        None,
        bug_report.buggy_source,
        state.current_source,
    ).ratio()
    score = 1.0 - similarity

    if score > 0.3:
        score = 0.3

    return max(0.0, min(1.0, score))


def operator_priority_score(edits: list[Edit]) -> float:
    if not edits:
        return 0.0

    priority_map = {
        "replace_operator": 0.9,
        "negate_condition": 0.85,
        "replace_literal": 0.7,
        "replace_variable": 0.65,
        "replace_expr": 0.65,
        "delete_stmt": 0.5,
        "swap_operands": 0.5,
        "insert_stmt_before": 0.4,
        "insert_stmt_after": 0.4,
        "unwrap_block": 0.3,
    }

    total = 0.0
    for edit in edits:
        operator_value = edit.operator.value if hasattr(edit.operator, "value") else str(edit.operator)
        total += priority_map.get(operator_value, 0.0)

    return total / len(edits)


def suspiciousness_alignment_score(
    state: RepairState,
    suspicious_scores: dict[int, float],
) -> float:
    if not state.edit_sequence:
        return 0.0

    best = 0.0
    for edit in state.edit_sequence:
        best = max(best, suspicious_scores.get(edit.line_number, 0.0))
    return best


def composite_score(
    state: RepairState,
    bug_report: BugReport,
    suspicious_scores: dict[int, float],
    weights: dict[str, float] | None = None,
) -> float:
    default_weights = {
        "edit_distance": 0.2,
        "operator_priority": 0.4,
        "suspiciousness": 0.4,
    }
    resolved_weights = dict(default_weights)
    if weights is not None:
        resolved_weights.update(weights)

    total_weight = sum(resolved_weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        warnings.warn(
            f"Composite score weights sum to {total_weight:.6f}, expected 1.0; normalizing.",
            RuntimeWarning,
            stacklevel=2,
        )

    if total_weight <= 0.0:
        return 0.0

    edit_distance = edit_distance_score(state, bug_report)
    operator_priority = operator_priority_score(state.edit_sequence)
    suspiciousness = suspiciousness_alignment_score(state, suspicious_scores)

    weighted_sum = (
        resolved_weights.get("edit_distance", 0.0) * edit_distance
        + resolved_weights.get("operator_priority", 0.0) * operator_priority
        + resolved_weights.get("suspiciousness", 0.0) * suspiciousness
    )

    normalized = weighted_sum / total_weight
    return max(0.0, min(1.0, normalized))


def score_patch_candidate(
    edits: list[Edit],
    original_source: str,
    patched_source: str,
    suspicious_lines: list[int],
) -> float:
    del suspicious_lines

    score = 1.0

    if len(edits) > 1:
        score -= 0.05 * (len(edits) - 1)

    changed_lines = _count_changed_lines(original_source, patched_source)
    if changed_lines > 5:
        score -= 0.1

    return max(0.0, min(1.0, score))


def _count_changed_lines(original_source: str, patched_source: str) -> int:
    original_lines = original_source.splitlines()
    patched_lines = patched_source.splitlines()

    matcher = difflib.SequenceMatcher(None, original_lines, patched_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)

    return changed


__all__ = [
    "edit_distance_score",
    "operator_priority_score",
    "suspiciousness_alignment_score",
    "composite_score",
    "score_patch_candidate",
]
