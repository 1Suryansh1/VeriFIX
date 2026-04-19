from __future__ import annotations

import pytest

from verifix.core.models import BugReport, Edit, EditOperator, RepairState
from verifix.search.scorer import (
    composite_score,
    edit_distance_score,
    operator_priority_score,
    score_patch_candidate,
)


ORIGINAL = """
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


def _bug_report() -> BugReport:
    return BugReport(
        bug_id="Chart-1",
        language="python",
        buggy_source=ORIGINAL,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=["test_buggy.py::test_pass"],
        project_root="C:/tmp/project",
        metadata={"benchmark": "toy"},
    )


def _edit(line_number: int = 5) -> Edit:
    return Edit(
        operator=EditOperator.REPLACE_OPERATOR,
        node_id="n1",
        node_type="Compare",
        line_number=line_number,
        original_text="<",
        replacement_text=">",
        metadata={},
    )


def _state(source: str, edits: list[Edit]) -> RepairState:
    return RepairState(
        bug_report=_bug_report(),
        edit_sequence=edits,
        current_source=source,
        depth=len(edits),
    )


def test_edit_distance_score_zero_for_unchanged_source() -> None:
    state = _state(ORIGINAL, [])
    score = edit_distance_score(state, _bug_report())
    assert score == 0.0


def test_edit_distance_score_positive_for_changed_source() -> None:
    patched = ORIGINAL.replace("<", ">")
    state = _state(patched, [_edit()])
    score = edit_distance_score(state, _bug_report())
    assert score > 0.0


def test_operator_priority_score_replace_operator_is_point_nine() -> None:
    score = operator_priority_score([_edit()])
    assert score == pytest.approx(0.9, rel=1e-6)


def test_composite_score_with_all_components_is_bounded() -> None:
    patched = ORIGINAL.replace("<", ">")
    state = _state(patched, [_edit()])
    score = composite_score(
        state=state,
        bug_report=_bug_report(),
        suspicious_scores={5: 0.8},
    )
    assert 0.0 <= score <= 1.0


def test_score_patch_candidate_penalizes_longer_edit_sequences() -> None:
    patched = ORIGINAL.replace("<", ">")
    one_edit_score = score_patch_candidate([_edit()], ORIGINAL, patched, suspicious_lines=[5])
    three_edit_score = score_patch_candidate(
        [_edit(), _edit(), _edit()],
        ORIGINAL,
        patched,
        suspicious_lines=[5],
    )

    assert three_edit_score < one_edit_score
