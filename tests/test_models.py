import json
from dataclasses import FrozenInstanceError

import pytest

from verifix.core.models import (
    BugReport,
    Edit,
    EditOperator,
    RankedPatch,
    RepairResult,
    RepairState,
    ValidationResult,
)


def _sample_bug_report() -> BugReport:
    return BugReport(
        bug_id="Chart-1",
        language="python",
        buggy_source="x = 1\n",
        file_path="src/module.py",
        failing_tests=["TestFoo::test_bar"],
        passing_tests=["TestFoo::test_baz"],
        project_root="C:/tmp/project",
        metadata={"benchmark": "Defects4J", "version": "1.2"},
    )


def _sample_edit() -> Edit:
    return Edit(
        operator=EditOperator.REPLACE_LITERAL,
        node_id="node-1",
        node_type="Constant",
        line_number=1,
        original_text="1",
        replacement_text="2",
        metadata={"note": "replace one literal"},
    )


def test_bug_report_construction_and_from_dict_round_trip() -> None:
    payload = {
        "bug_id": "Chart-1",
        "language": "python",
        "buggy_source": "x = 1\n",
        "file_path": "src/module.py",
        "failing_tests": ["TestFoo::test_bar"],
        "passing_tests": ["TestFoo::test_baz"],
        "project_root": "C:/tmp/project",
        "metadata": {"benchmark": "Defects4J", "version": "1.2"},
    }

    report = BugReport.from_dict(payload)

    assert report.bug_id == payload["bug_id"]
    assert report.language == payload["language"]
    assert report.buggy_source == payload["buggy_source"]
    assert report.file_path == payload["file_path"]
    assert report.failing_tests == payload["failing_tests"]
    assert report.passing_tests == payload["passing_tests"]
    assert report.project_root == payload["project_root"]
    assert report.metadata == payload["metadata"]


def test_edit_is_immutable() -> None:
    edit = _sample_edit()

    with pytest.raises(FrozenInstanceError):
        edit.line_number = 2


def test_repair_state_id_is_deterministic() -> None:
    report = _sample_bug_report()
    src = "a = 1\nb = 2\n"
    state_a = RepairState(
        bug_report=report,
        edit_sequence=[],
        current_source=src,
        depth=0,
        is_terminal=False,
    )
    state_b = RepairState(
        bug_report=report,
        edit_sequence=[],
        current_source=src,
        depth=99,
        is_terminal=True,
    )

    assert state_a.state_id == state_b.state_id


def test_repair_state_apply_edit_returns_new_state_without_mutation() -> None:
    report = _sample_bug_report()
    initial_source = "a = 1\nb = 2\n"
    state = RepairState(
        bug_report=report,
        edit_sequence=[],
        current_source=initial_source,
        depth=0,
        is_terminal=False,
    )
    edit = Edit(
        operator=EditOperator.REPLACE_LITERAL,
        node_id="node-2",
        node_type="Constant",
        line_number=1,
        original_text="1",
        replacement_text="3",
        metadata={},
    )

    next_state = state.apply_edit(edit)

    assert next_state is not state
    assert state.edit_sequence == []
    assert state.current_source == initial_source
    assert state.depth == 0
    assert next_state.edit_sequence == [edit]
    assert next_state.current_source == "a = 3\nb = 2\n"
    assert next_state.depth == 1


def test_validation_result_is_plausible_only_when_both_conditions_true() -> None:
    both_true = ValidationResult(
        state_id="s1",
        compiled=True,
        tests_passed=["t1"],
        tests_failed=[],
        all_failing_tests_pass=True,
        no_regression=True,
        is_plausible=False,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=10.0,
    )
    one_false_a = ValidationResult(
        state_id="s2",
        compiled=True,
        tests_passed=[],
        tests_failed=["t1"],
        all_failing_tests_pass=False,
        no_regression=True,
        is_plausible=True,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=12.0,
    )
    one_false_b = ValidationResult(
        state_id="s3",
        compiled=True,
        tests_passed=[],
        tests_failed=["t1"],
        all_failing_tests_pass=True,
        no_regression=False,
        is_plausible=True,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=11.0,
    )

    assert both_true.is_plausible is True
    assert one_false_a.is_plausible is False
    assert one_false_b.is_plausible is False


def test_repair_result_to_dict_is_json_serializable() -> None:
    validation = ValidationResult(
        state_id="abc123",
        compiled=True,
        tests_passed=["TestFoo::test_bar"],
        tests_failed=[],
        all_failing_tests_pass=True,
        no_regression=True,
        is_plausible=True,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=25.5,
    )
    patch = RankedPatch(
        rank=1,
        edit_sequence=[_sample_edit()],
        patched_source="x = 2\n",
        validation=validation,
        score=0.95,
        diff="--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
    )
    result = RepairResult(
        bug_id="Chart-1",
        success=True,
        ranked_patches=[patch],
        total_states_explored=10,
        total_validations_run=5,
        wall_time_seconds=1.23,
        search_tree_depth=2,
        error=None,
    )

    serialized = result.to_dict()
    dumped = json.dumps(serialized)

    assert isinstance(dumped, str)
    assert serialized["bug_id"] == "Chart-1"
    assert serialized["ranked_patches"][0]["edit_sequence"][0]["operator"] == "replace_literal"
