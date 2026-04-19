from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from verifix.benchmarks.quixbugs import QuixBugsBenchmark, QuixBugsLoader
from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, RepairResult


QUIXBUGS_ROOT = Path(os.environ.get("QUIXBUGS_ROOT", "quixbugs")).resolve()
QUIXBUGS_AVAILABLE = QUIXBUGS_ROOT.exists()

pytestmark = pytest.mark.skipif(
    not QUIXBUGS_AVAILABLE,
    reason="QuixBugs repository not found. Set QUIXBUGS_ROOT or clone to ./quixbugs",
)


def _small_config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        mcts_time_budget_seconds=1.0,
        max_validations=1,
        max_candidates_per_node=2,
        fl_top_n_lines=1,
        test_timeout_seconds=2.0,
        python_executable=sys.executable,
        working_dir=str(tmp_path / "work"),
    )


def test_load_program_returns_valid_bug_report() -> None:
    loader = QuixBugsLoader(str(QUIXBUGS_ROOT))
    report = loader.load_program("find_in_sorted")

    assert isinstance(report, BugReport)
    assert report.bug_id == "QuixBugs-find_in_sorted"
    assert report.language == "python"
    assert report.file_path == "find_in_sorted.py"
    assert report.project_root


def test_generated_test_file_is_valid_python(tmp_path: Path) -> None:
    loader = QuixBugsLoader(str(QUIXBUGS_ROOT))
    testcase_path = QUIXBUGS_ROOT / "python_testcases" / "find_in_sorted.json"
    if not testcase_path.exists():
        testcase_path = QUIXBUGS_ROOT / "json_testcases" / "find_in_sorted.json"
    test_file = loader.generate_test_file(
        program_name="find_in_sorted",
        testcases_json=testcase_path.read_text(encoding="utf-8"),
        output_dir=str(tmp_path),
    )

    content = Path(test_file).read_text(encoding="utf-8")
    ast.parse(content)
    assert Path(test_file).exists()


def test_correct_programs_use_single_canonical_definition() -> None:
    correct_dir = QUIXBUGS_ROOT / "correct_python_programs"
    target_programs = [
        "breadth_first_search",
        "is_valid_parenthesization",
        "possible_change",
        "max_sublist_sum",
        "mergesort",
        "sieve",
        "lis",
        "reverse_linked_list",
        "rpn_eval",
        "detect_cycle",
        "find_first_in_sorted",
        "next_permutation",
        "quicksort",
        "sqrt",
        "to_base",
        "powerset",
        "get_factors",
    ]

    for program in target_programs:
        path = correct_dir / f"{program}.py"
        source = path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        definitions = [
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name == program
        ]
        assert len(definitions) == 1, f"Expected single definition for {program}"


def test_run_single_returns_result_dict_with_success_key(tmp_path: Path) -> None:
    benchmark = QuixBugsBenchmark(str(QUIXBUGS_ROOT), config=_small_config(tmp_path))

    def _fake_repair(_: BugReport) -> RepairResult:
        return RepairResult(
            bug_id="QuixBugs-find_in_sorted",
            success=False,
            ranked_patches=[],
            total_states_explored=1,
            total_validations_run=0,
            wall_time_seconds=0.01,
            search_tree_depth=0,
            error=None,
        )

    benchmark.agent.repair = _fake_repair
    result = benchmark.run_single("find_in_sorted")

    assert isinstance(result, dict)
    assert "success" in result
    assert "time" in result


def test_run_all_summary_structure(tmp_path: Path) -> None:
    benchmark = QuixBugsBenchmark(str(QUIXBUGS_ROOT), config=_small_config(tmp_path))

    reports = [
        BugReport(
            bug_id=f"QuixBugs-prog{i}",
            language="python",
            buggy_source="def f():\n    return 1\n",
            file_path=f"prog{i}.py",
            failing_tests=["t1"],
            passing_tests=["t2"],
            project_root=str(tmp_path),
            metadata={"program_name": f"prog{i}"},
        )
        for i in range(5)
    ]

    benchmark.loader.load_all = lambda language="python": reports

    def _fake_run_single(program_name: str) -> dict:
        return {
            "program": program_name,
            "bug_id": f"QuixBugs-{program_name}",
            "success": True,
            "time": 0.5,
            "top_patch_score": 0.9,
            "validations": 2,
            "result": {},
            "error": None,
        }

    benchmark.run_single = _fake_run_single

    summary = benchmark.run_all(max_programs=3, parallel=False, output_dir=str(tmp_path / "out"))

    assert isinstance(summary, dict)
    assert {
        "total",
        "attempted_total",
        "dataset_total",
        "repaired",
        "repair_rate",
        "dataset_repair_rate",
        "avg_time_seconds",
        "avg_validations",
        "per_program",
    }.issubset(summary.keys())
    assert summary["total"] == 3
    assert summary["attempted_total"] == 3
    assert summary["dataset_total"] == 5
    assert len(summary["per_program"]) == 3
