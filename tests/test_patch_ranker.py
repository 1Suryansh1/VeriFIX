from __future__ import annotations

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, EditOperator, ValidationResult
from verifix.search.mcts import MCTSSearchResult
from verifix.validator.patch_ranker import rank_patches, summarize_results


ORIGINAL_SOURCE = """
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:
            max_val = arr[i]
    return max_val
"""


def _config() -> VerifixConfig:
    return VerifixConfig(mcts_iterations=10, mcts_max_depth=2)


def _bug_report() -> BugReport:
    return BugReport(
        bug_id="Chart-1",
        language="python",
        buggy_source=ORIGINAL_SOURCE,
        file_path="buggy.py",
        failing_tests=["f1"],
        passing_tests=["p1"],
        project_root="C:/tmp/project",
        metadata={},
    )


def _edit(node_id: str, line: int = 5) -> Edit:
    return Edit(
        operator=EditOperator.REPLACE_OPERATOR,
        node_id=node_id,
        node_type="Compare",
        line_number=line,
        original_text="<",
        replacement_text=">",
        metadata={},
    )


def _validation(plausible: bool = True) -> ValidationResult:
    if plausible:
        return ValidationResult(
            state_id="s",
            compiled=True,
            tests_passed=["f1", "p1"],
            tests_failed=[],
            all_failing_tests_pass=True,
            no_regression=True,
            is_plausible=True,
            compile_error=None,
            runtime_error=None,
            execution_time_ms=1.0,
        )

    return ValidationResult(
        state_id="s",
        compiled=True,
        tests_passed=[],
        tests_failed=["f1"],
        all_failing_tests_pass=False,
        no_regression=True,
        is_plausible=False,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=1.0,
    )


def _search_result(patches: list[tuple[list[Edit], str, ValidationResult]]) -> MCTSSearchResult:
    return MCTSSearchResult(
        plausible_patches=patches,
        total_iterations=1,
        total_validations=1,
        wall_time_seconds=0.1,
        tree_depth_reached=1,
        terminated_by="all_explored",
    )


def test_rank_patches_filters_non_plausible() -> None:
    patches = [
        ([_edit("a")], ORIGINAL_SOURCE.replace("<", ">"), _validation(True)),
        ([_edit("b")], ORIGINAL_SOURCE.replace("<", "=="), _validation(False)),
    ]

    ranked = rank_patches(_search_result(patches), ORIGINAL_SOURCE, _bug_report(), _config())
    assert len(ranked) == 1


def test_rank_patches_deduplicates_identical_sources() -> None:
    same_source = ORIGINAL_SOURCE.replace("<", ">")
    patches = [
        ([_edit("a")], same_source, _validation(True)),
        ([_edit("b"), _edit("c")], same_source, _validation(True)),
    ]

    ranked = rank_patches(_search_result(patches), ORIGINAL_SOURCE, _bug_report(), _config())
    assert len(ranked) == 1


def test_rank_patches_prefers_shorter_sequence_among_duplicates() -> None:
    same_source = ORIGINAL_SOURCE.replace("<", ">")
    short = [_edit("a")]
    long = [_edit("b"), _edit("c")]

    patches = [(long, same_source, _validation(True)), (short, same_source, _validation(True))]
    ranked = rank_patches(_search_result(patches), ORIGINAL_SOURCE, _bug_report(), _config())

    assert len(ranked[0].edit_sequence) == 1


def test_rank_patches_returns_at_most_max_patches() -> None:
    patches = [
        ([_edit("a")], ORIGINAL_SOURCE.replace("<", ">"), _validation(True)),
        ([_edit("b")], ORIGINAL_SOURCE.replace("<", ">="), _validation(True)),
        ([_edit("c")], ORIGINAL_SOURCE.replace("<", "<="), _validation(True)),
    ]

    ranked = rank_patches(_search_result(patches), ORIGINAL_SOURCE, _bug_report(), _config(), max_patches=2)
    assert len(ranked) == 2


def test_rank_patches_scores_descending_order() -> None:
    better = ([_edit("a")], ORIGINAL_SOURCE.replace("<", ">"), _validation(True))
    worse = (
        [_edit("b"), _edit("c"), _edit("d")],
        ORIGINAL_SOURCE.replace("<", ">") + "\n# extra\n# x\n# y\n# z\n# w\n# q\n",
        _validation(True),
    )

    ranked = rank_patches(
        _search_result([worse, better]),
        ORIGINAL_SOURCE,
        _bug_report(),
        _config(),
    )

    scores = [patch.score for patch in ranked]
    assert scores == sorted(scores, reverse=True)


def test_summarize_results_contains_patch_header_and_diff() -> None:
    patches = [
        ([_edit("a")], ORIGINAL_SOURCE.replace("<", ">"), _validation(True)),
    ]
    ranked = rank_patches(_search_result(patches), ORIGINAL_SOURCE, _bug_report(), _config())

    summary = summarize_results(ranked, bug_id="Chart-1")
    assert "PATCH #1" in summary
    assert "--- a/buggy.py" in summary


def test_ranked_patch_rank_is_sequential() -> None:
    patches = [
        ([_edit("a")], ORIGINAL_SOURCE.replace("<", ">"), _validation(True)),
        ([_edit("b")], ORIGINAL_SOURCE.replace("<", ">="), _validation(True)),
        ([_edit("c")], ORIGINAL_SOURCE.replace("<", "<="), _validation(True)),
    ]
    ranked = rank_patches(_search_result(patches), ORIGINAL_SOURCE, _bug_report(), _config())

    assert [patch.rank for patch in ranked] == list(range(1, len(ranked) + 1))
