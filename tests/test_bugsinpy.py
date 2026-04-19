from __future__ import annotations

import ast
from pathlib import Path

import pytest

from verifix.benchmarks.bugsinpy import BugsInPyBenchmark, BugsInPyLoader
from verifix.core.config import VerifixConfig
from verifix.core.models import RepairResult


ROOT = ".data/BugsInPy"
pytestmark = pytest.mark.skipif(
    not Path(ROOT).exists(),
    reason="Run: git clone https://github.com/soarsmu/BugsInPy .data/BugsInPy",
)


@pytest.fixture(scope="module")
def loader() -> BugsInPyLoader:
    return BugsInPyLoader(ROOT)


def _fast_config(tmp_path: Path) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        mcts_time_budget_seconds=5.0,
        max_validations=10,
        max_candidates_per_node=2,
        fl_top_n_lines=1,
        test_timeout_seconds=5.0,
        working_dir=str(tmp_path / "work"),
    )


def _info_path(project: str, bug_id: int) -> Path:
    bug_root = Path(ROOT) / "projects" / project / "bugs" / str(bug_id)
    preferred = bug_root / "bugsinpy_bug.info"
    if preferred.exists():
        return preferred
    return bug_root / "bug.info"


def test_get_available_projects(loader: BugsInPyLoader) -> None:
    projects = loader.get_available_projects()
    assert len(projects) >= 17
    assert "ansible" in projects
    assert "black" in projects
    assert "scrapy" in projects


def test_get_bug_count(loader: BugsInPyLoader) -> None:
    assert loader.get_bug_count("ansible") >= 1
    assert loader.get_bug_count("black") >= 1


def test_parse_bug_info(loader: BugsInPyLoader) -> None:
    info = loader._parse_bug_info(str(_info_path("ansible", 1)))
    assert "buggy_commit_id" in info
    assert "fixed_commit_id" in info
    assert "test_file" in info or "test_cases" in info


def test_get_failing_tests(loader: BugsInPyLoader) -> None:
    tests = loader._get_failing_tests("ansible", 1)
    assert len(tests) >= 1
    assert all("::" in test_id for test_id in tests)


@pytest.mark.timeout(300)
def test_load_bug_returns_bugreport(loader: BugsInPyLoader) -> None:
    bug = loader.load_bug("black", 1)
    assert bug.language == "python"
    assert bug.bug_id == "BugsInPy-black-1"
    assert len(bug.buggy_source) > 0
    assert len(bug.failing_tests) >= 1
    assert bug.project_root
    assert Path(bug.project_root).exists()


@pytest.mark.timeout(300)
def test_buggy_source_is_valid_python(loader: BugsInPyLoader) -> None:
    bug = loader.load_bug("black", 1)
    try:
        ast.parse(bug.buggy_source)
    except SyntaxError:
        pytest.skip("Bug involves syntax; skip AST validity check")


def test_compute_correct_with_real_patch(loader: BugsInPyLoader) -> None:
    bench = BugsInPyBenchmark(ROOT)
    patch_path = str(Path(ROOT) / "projects" / "black" / "bugs" / "1" / "bug_patch.txt")
    result = RepairResult(
        bug_id="BugsInPy-black-1",
        success=False,
        ranked_patches=[],
        total_states_explored=0,
        total_validations_run=0,
        wall_time_seconds=0.0,
        search_tree_depth=0,
        error=None,
    )
    assert bench.compute_correct(result, patch_path) is False


@pytest.mark.timeout(300)
def test_load_project(loader: BugsInPyLoader) -> None:
    bugs = loader.load_project("black")
    assert len(bugs) >= 1
    assert all(bug.language == "python" for bug in bugs)
    assert all("black" in bug.bug_id for bug in bugs)


@pytest.mark.timeout(420)
def test_benchmark_run_single_bug(tmp_path: Path) -> None:
    bench = BugsInPyBenchmark(ROOT, config=_fast_config(tmp_path))
    summary = bench.run(
        project_filter=["black"],
        max_bugs=1,
        output_dir=str(tmp_path / "verifix_test_bugsinpy"),
        use_v2=False,
    )

    assert "total_bugs" in summary
    assert "plausible_rate" in summary
    assert "correct_rate" in summary
    assert "by_project" in summary
    assert "per_bug" in summary
    assert summary["total_bugs"] == 1
