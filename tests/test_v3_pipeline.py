from __future__ import annotations

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.pipeline.repair_agent_v3 import RepairAgentV3


def _bug_report() -> BugReport:
    return BugReport(
        bug_id="v3-fallback",
        language="python",
        buggy_source="def f(x):\n    return x\n",
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=[],
        project_root=".",
        metadata={},
    )


def test_repair_v3_falls_back_without_checkpoint(monkeypatch) -> None:
    config = VerifixConfig(mcts_iterations=10, mcts_max_depth=1, fl_top_n_lines=1)
    agent = RepairAgentV3(config=config, checkpoint_path=None, device="cpu")

    expected = RepairResult(
        bug_id="v3-fallback",
        success=False,
        ranked_patches=[],
        total_states_explored=0,
        total_validations_run=0,
        wall_time_seconds=0.0,
        search_tree_depth=0,
        error=None,
    )

    monkeypatch.setattr(agent._baseline, "repair", lambda _report: expected)
    result = agent.repair(_bug_report())

    assert result.mode == "fallback-concrete"
    assert result.repair_result.bug_id == "v3-fallback"
