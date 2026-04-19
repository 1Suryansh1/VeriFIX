from __future__ import annotations

import time

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, EditOperator, RepairState, ValidationResult
from verifix.search.mcts import MCTSSearchResult, get_best_patches, mcts_search, select
from verifix.search.state import MCTSNode


BUGGY_SOURCE = """
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


class MockValidator:
    def validate(self, source: str, bug_report: BugReport) -> ValidationResult:
        plausible = ">" in source
        return ValidationResult(
            state_id="s",
            compiled=True,
            tests_passed=["failing"] if plausible else [],
            tests_failed=[] if plausible else ["failing"],
            all_failing_tests_pass=plausible,
            no_regression=plausible,
            is_plausible=plausible,
            compile_error=None,
            runtime_error=None,
            execution_time_ms=1.0,
        )


class SlowValidator(MockValidator):
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    def validate(self, source: str, bug_report: BugReport) -> ValidationResult:
        time.sleep(self.delay_seconds)
        return super().validate(source, bug_report)


def _bug_report() -> BugReport:
    return BugReport(
        bug_id="Chart-1",
        language="python",
        buggy_source=BUGGY_SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=["test_buggy.py::test_pass"],
        project_root="C:/tmp/project",
        metadata={"benchmark": "toy"},
    )


def _config(**overrides: object) -> VerifixConfig:
    base = {
        "mcts_iterations": 50,
        "mcts_max_depth": 1,
        "mcts_time_budget_seconds": 5.0,
        "max_validations": 100,
        "max_candidates_per_node": 10,
    }
    base.update(overrides)
    return VerifixConfig(**base)


def _state() -> RepairState:
    return RepairState(
        bug_report=_bug_report(),
        edit_sequence=[],
        current_source=BUGGY_SOURCE,
        depth=0,
    )


def _edit(line_number: int = 5) -> Edit:
    return Edit(
        operator=EditOperator.REPLACE_OPERATOR,
        node_id="n",
        node_type="Compare",
        line_number=line_number,
        original_text="<",
        replacement_text=">",
        metadata={},
    )


def _validation(
    plausible: bool,
    all_failing: bool,
    no_regression: bool,
) -> ValidationResult:
    return ValidationResult(
        state_id="s",
        compiled=True,
        tests_passed=[],
        tests_failed=[],
        all_failing_tests_pass=all_failing,
        no_regression=no_regression,
        is_plausible=plausible,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=1.0,
    )


def test_mcts_search_completes_without_error() -> None:
    result = mcts_search(
        bug_report=_bug_report(),
        suspicious_lines=[5],
        validator=MockValidator(),
        config=_config(),
    )
    assert isinstance(result, MCTSSearchResult)


def test_mcts_search_returns_result_with_expected_fields() -> None:
    result = mcts_search(
        bug_report=_bug_report(),
        suspicious_lines=[5],
        validator=MockValidator(),
        config=_config(),
    )

    assert isinstance(result.plausible_patches, list)
    assert isinstance(result.total_iterations, int)
    assert isinstance(result.total_validations, int)
    assert isinstance(result.wall_time_seconds, float)
    assert isinstance(result.tree_depth_reached, int)
    assert result.terminated_by in {
        "budget_exhausted",
        "validation_cap",
        "time_limit",
        "all_explored",
    }


def test_mcts_search_finds_plausible_patch_with_mock_validator() -> None:
    result = mcts_search(
        bug_report=_bug_report(),
        suspicious_lines=[5],
        validator=MockValidator(),
        config=_config(mcts_iterations=80),
    )

    assert len(result.plausible_patches) >= 1


def test_select_prefers_unvisited_branch_over_fully_expanded_non_leaf() -> None:
    root = MCTSNode(state=_state(), parent=None, visit_count=10, unexplored_edits=[])

    fully_expanded_non_leaf = MCTSNode(
        state=_state(),
        parent=root,
        visit_count=5,
        total_reward=5.0,
        unexplored_edits=[],
    )
    fully_expanded_non_leaf.children = [
        MCTSNode(state=_state(), parent=fully_expanded_non_leaf, visit_count=2, total_reward=1.0)
    ]

    unvisited_leaf = MCTSNode(state=_state(), parent=root, visit_count=0, unexplored_edits=[])

    root.children = [fully_expanded_non_leaf, unvisited_leaf]

    selected = select(root, exploration_constant=1.414)

    assert selected is unvisited_leaf
    assert not (selected.is_fully_expanded() and not selected.is_leaf())


def test_mcts_search_respects_time_budget() -> None:
    result = mcts_search(
        bug_report=_bug_report(),
        suspicious_lines=[5],
        validator=SlowValidator(delay_seconds=0.03),
        config=_config(mcts_iterations=200, mcts_time_budget_seconds=0.1, max_validations=1000),
    )

    assert result.terminated_by == "time_limit"


def test_mcts_search_respects_max_validations_cap() -> None:
    result = mcts_search(
        bug_report=_bug_report(),
        suspicious_lines=[5],
        validator=MockValidator(),
        config=_config(mcts_iterations=200, max_validations=2, mcts_time_budget_seconds=10.0),
    )

    assert result.terminated_by == "validation_cap"
    assert result.total_validations == 2


def test_get_best_patches_deduplicates_identical_sources() -> None:
    short_edits = [_edit()]
    long_edits = [_edit(), _edit()]

    result = MCTSSearchResult(
        plausible_patches=[
            (long_edits, "same_source", _validation(True, True, True)),
            (short_edits, "same_source", _validation(True, True, True)),
        ],
        total_iterations=2,
        total_validations=2,
        wall_time_seconds=0.01,
        tree_depth_reached=1,
        terminated_by="all_explored",
    )

    best = get_best_patches(result, max_patches=10)

    assert len(best) == 1
    assert len(best[0][0]) == 1


def test_get_best_patches_prefers_shorter_when_scores_equal() -> None:
    short_patch = ([_edit()], "source_short", _validation(True, True, False))
    long_patch = (
        [_edit(), _edit(), _edit(), _edit(), _edit(), _edit()],
        "source_long",
        _validation(True, True, True),
    )
    # Scores are equal at 0.9; shorter sequence should rank first.

    result = MCTSSearchResult(
        plausible_patches=[long_patch, short_patch],
        total_iterations=2,
        total_validations=2,
        wall_time_seconds=0.01,
        tree_depth_reached=1,
        terminated_by="all_explored",
    )

    ranked = get_best_patches(result, max_patches=10)

    assert ranked[0][1] == "source_short"
    assert len(ranked[0][0]) == 1
