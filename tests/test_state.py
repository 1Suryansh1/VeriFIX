from __future__ import annotations

import math

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, EditOperator, RepairState
from verifix.search.state import MCTSNode, backpropagate, expand_node, initialize_root


BUGGY_SOURCE = """
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
        buggy_source=BUGGY_SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail_case_one"],
        passing_tests=["test_buggy.py::test_pass_case_one"],
        project_root="C:/tmp/project",
        metadata={"benchmark": "toy"},
    )


def _small_config() -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=3,
        max_candidates_per_node=10,
    )


def test_initialize_root_has_non_empty_unexplored_edits() -> None:
    root = initialize_root(_bug_report(), _small_config(), suspicious_lines=[5])
    assert root.parent is None
    assert len(root.unexplored_edits) > 0


def test_ucb1_score_infinity_when_unvisited() -> None:
    state = RepairState(
        bug_report=_bug_report(),
        edit_sequence=[],
        current_source=BUGGY_SOURCE,
        depth=0,
    )
    node = MCTSNode(state=state, parent=None)
    assert math.isinf(node.ucb1_score())


def test_ucb1_score_known_values() -> None:
    state = RepairState(
        bug_report=_bug_report(),
        edit_sequence=[],
        current_source=BUGGY_SOURCE,
        depth=0,
    )
    parent = MCTSNode(state=state, parent=None, visit_count=10, total_reward=0.0)
    child = MCTSNode(state=state, parent=parent, visit_count=3, total_reward=2.0)

    actual = child.ucb1_score(exploration_constant=1.414)
    expected = (2.0 / 3.0) + 1.414 * math.sqrt(math.log(10) / 3.0)

    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_expand_node_returns_child_with_incremented_depth() -> None:
    config = _small_config()
    root = initialize_root(_bug_report(), config, suspicious_lines=[5])

    valid_edit = Edit(
        operator=EditOperator.REPLACE_OPERATOR,
        node_id="cmp-node",
        node_type="Compare",
        line_number=5,
        original_text="<",
        replacement_text=">",
        metadata={},
    )
    root.unexplored_edits = [valid_edit]

    child = expand_node(root, config)

    assert child is not None
    assert child.state.depth == root.state.depth + 1
    assert child.parent is root


def test_expand_node_returns_none_for_syntax_broken_edit() -> None:
    config = _small_config()
    root = initialize_root(_bug_report(), config, suspicious_lines=[5])

    invalid_edit = Edit(
        operator=EditOperator.REPLACE_OPERATOR,
        node_id="cmp-node",
        node_type="Compare",
        line_number=5,
        original_text="<",
        replacement_text="<<<",
        metadata={},
    )
    root.unexplored_edits = [invalid_edit]

    child = expand_node(root, config)

    assert child is None


def test_is_fully_expanded_false_when_unexplored_exists() -> None:
    state = RepairState(
        bug_report=_bug_report(),
        edit_sequence=[],
        current_source=BUGGY_SOURCE,
        depth=0,
    )
    node = MCTSNode(
        state=state,
        parent=None,
        unexplored_edits=[
            Edit(
                operator=EditOperator.REPLACE_LITERAL,
                node_id="n1",
                node_type="Constant",
                line_number=3,
                original_text="0",
                replacement_text="1",
                metadata={},
            )
        ],
    )
    assert node.is_fully_expanded() is False


def test_backpropagate_updates_ancestors() -> None:
    base_state = RepairState(
        bug_report=_bug_report(),
        edit_sequence=[],
        current_source=BUGGY_SOURCE,
        depth=0,
    )
    root = MCTSNode(state=base_state, parent=None)
    child = MCTSNode(state=base_state, parent=root)
    grandchild = MCTSNode(state=base_state, parent=child)

    backpropagate(grandchild, reward=0.75)

    assert grandchild.visit_count == 1
    assert child.visit_count == 1
    assert root.visit_count == 1
    assert grandchild.total_reward == 0.75
    assert child.total_reward == 0.75
    assert root.total_reward == 0.75


def test_best_child_returns_highest_ucb1() -> None:
    state = RepairState(
        bug_report=_bug_report(),
        edit_sequence=[],
        current_source=BUGGY_SOURCE,
        depth=0,
    )
    parent = MCTSNode(state=state, parent=None, visit_count=20)
    child_a = MCTSNode(state=state, parent=parent, visit_count=1, total_reward=1.0)
    child_b = MCTSNode(state=state, parent=parent, visit_count=5, total_reward=6.0)
    parent.children = [child_a, child_b]

    best = parent.best_child(exploration_constant=1.414)

    assert best is child_a
