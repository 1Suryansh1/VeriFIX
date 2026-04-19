from __future__ import annotations

import time
from pathlib import Path

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport
from verifix.verifier.fuzzer import FuzzStrategy, FuzzTarget, fuzz_patch, generate_fuzz_inputs, infer_signature


CORRECT_SOURCE = """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
"""


OVERFITTED_SOURCE = """def find_max(arr):
    return 100
"""


ORIGINAL_TEST_CASES = [
    {"input": [[3, 1, 2]], "expected": 3},
    {"input": [[5, 4, 3]], "expected": 5},
    {"input": [[2, 2, 2]], "expected": 2},
]


def _config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        mcts_time_budget_seconds=2.0,
        max_validations=10,
        fl_top_n_lines=1,
        test_timeout_seconds=5.0,
        working_dir=str(tmp_path / "fuzz_work"),
    )


def _bug_report(reference_source: str) -> BugReport:
    return BugReport(
        bug_id="Fuzz-1",
        language="python",
        buggy_source="def placeholder(x):\n    return x\n",
        file_path="buggy.py",
        failing_tests=["t1"],
        passing_tests=["t2"],
        project_root=".",
        metadata={"reference_source": reference_source},
    )


def test_fuzz_patch_correct_source_survives(tmp_path: Path) -> None:
    result = fuzz_patch(
        patched_source=CORRECT_SOURCE,
        function_name="find_max",
        original_test_cases=ORIGINAL_TEST_CASES,
        bug_report=_bug_report(CORRECT_SOURCE),
        config=_config(tmp_path),
        strategy=FuzzStrategy.BOUNDARY,
    )

    assert result.survived is True
    assert result.total_inputs_tested > 0


def test_fuzz_patch_overfitted_source_fails_with_failing_inputs(tmp_path: Path) -> None:
    result = fuzz_patch(
        patched_source=OVERFITTED_SOURCE,
        function_name="find_max",
        original_test_cases=ORIGINAL_TEST_CASES,
        bug_report=_bug_report(CORRECT_SOURCE),
        config=_config(tmp_path),
        strategy=FuzzStrategy.BOUNDARY,
    )

    assert result.survived is False
    assert len(result.failing_inputs) > 0


def test_generate_fuzz_inputs_boundary_contains_empty_and_singleton_list() -> None:
    target = FuzzTarget(
        function_name="find_max",
        source=CORRECT_SOURCE,
        signature={"arr": "list"},
        test_cases=ORIGINAL_TEST_CASES,
    )

    inputs = generate_fuzz_inputs(target, strategy=FuzzStrategy.BOUNDARY, n_inputs=50)

    assert any(inp and inp[0] == [] for inp in inputs)
    assert any(inp and inp[0] == [0] for inp in inputs)


def test_generate_fuzz_inputs_mutation_expands_test_inputs() -> None:
    target = FuzzTarget(
        function_name="find_max",
        source=CORRECT_SOURCE,
        signature={"arr": "list"},
        test_cases=ORIGINAL_TEST_CASES,
    )

    mutated = generate_fuzz_inputs(target, strategy=FuzzStrategy.MUTATION, n_inputs=50)

    assert len(mutated) > len(ORIGINAL_TEST_CASES)


def test_infer_signature_infers_list_for_arr_from_test_cases() -> None:
    signature = infer_signature(CORRECT_SOURCE, "find_max", test_cases=ORIGINAL_TEST_CASES)
    assert signature["arr"] == "list"


def test_fuzz_patch_completes_within_ten_seconds_for_fifty_inputs(tmp_path: Path) -> None:
    start = time.monotonic()
    result = fuzz_patch(
        patched_source=CORRECT_SOURCE,
        function_name="find_max",
        original_test_cases=ORIGINAL_TEST_CASES,
        bug_report=_bug_report(CORRECT_SOURCE),
        config=_config(tmp_path),
        strategy=FuzzStrategy.RANDOM,
    )
    elapsed = time.monotonic() - start

    assert result.total_inputs_tested == 50
    assert elapsed < 10.0
