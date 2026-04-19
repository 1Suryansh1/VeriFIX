from __future__ import annotations

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, ValidationResult
from verifix.models.latent_jepa import JEPATransitionPredictor, MultiTaskRepairGAT
from verifix.models.pyg_converter import ASTtoPyG
from verifix.search.mcts_latent import latent_guided_search


SOURCE = """
def solve(arr):
    total = 0
    for i in range(0, len(arr)):
        if arr[i] < total:
            total = total + arr[i]
    return total
"""


class _MockValidator:
    def validate(self, source: str, bug_report: BugReport) -> ValidationResult:
        del bug_report
        plausible = ">" in source
        return ValidationResult(
            state_id="mock",
            compiled=True,
            tests_passed=["t"] if plausible else [],
            tests_failed=[] if plausible else ["t"],
            all_failing_tests_pass=plausible,
            no_regression=plausible,
            is_plausible=plausible,
            compile_error=None,
            runtime_error=None,
            execution_time_ms=1.0,
        )


def test_latent_search_reports_depth_floor() -> None:
    report = BugReport(
        bug_id="latent-depth",
        language="python",
        buggy_source=SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=[],
        project_root=".",
        metadata={},
    )

    config = VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        max_validations=5,
        max_candidates_per_node=4,
        max_patch_candidates=20,
        fl_top_n_lines=3,
        v3_min_rollout_depth=3,
        v3_branch_per_state=3,
        v3_critic_threshold=0.2,
    )

    model = MultiTaskRepairGAT()
    predictor = JEPATransitionPredictor()
    converter = ASTtoPyG()

    result, diag = latent_guided_search(
        bug_report=report,
        validator=_MockValidator(),
        config=config,
        model=model,
        predictor=predictor,
        converter=converter,
        rollout_mode="hybrid",
        critic_threshold=0.2,
    )

    assert diag.depth_floor == 3
    assert diag.depth_reached >= 1
    assert result.total_iterations >= 1


def test_latent_mode_uses_predictor(monkeypatch) -> None:
    report = BugReport(
        bug_id="latent-uses-predictor",
        language="python",
        buggy_source=SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=[],
        project_root=".",
        metadata={},
    )

    config = VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        max_validations=5,
        max_candidates_per_node=4,
        max_patch_candidates=20,
        fl_top_n_lines=3,
        v3_min_rollout_depth=2,
        v3_branch_per_state=3,
        v3_critic_threshold=0.2,
    )

    model = MultiTaskRepairGAT()
    predictor = JEPATransitionPredictor()
    converter = ASTtoPyG()

    calls = {"count": 0}
    original_forward = predictor.forward

    def counted_forward(*args, **kwargs):
        calls["count"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(predictor, "forward", counted_forward)

    latent_guided_search(
        bug_report=report,
        validator=_MockValidator(),
        config=config,
        model=model,
        predictor=predictor,
        converter=converter,
        rollout_mode="latent",
        critic_threshold=0.2,
    )

    assert calls["count"] > 0


def test_hybrid_mode_does_not_use_predictor(monkeypatch) -> None:
    report = BugReport(
        bug_id="hybrid-skips-predictor",
        language="python",
        buggy_source=SOURCE,
        file_path="buggy.py",
        failing_tests=["test_buggy.py::test_fail"],
        passing_tests=[],
        project_root=".",
        metadata={},
    )

    config = VerifixConfig(
        mcts_iterations=10,
        mcts_max_depth=1,
        max_validations=5,
        max_candidates_per_node=4,
        max_patch_candidates=20,
        fl_top_n_lines=3,
        v3_min_rollout_depth=2,
        v3_branch_per_state=3,
        v3_critic_threshold=0.2,
    )

    model = MultiTaskRepairGAT()
    predictor = JEPATransitionPredictor()
    converter = ASTtoPyG()

    calls = {"count": 0}
    original_forward = predictor.forward

    def counted_forward(*args, **kwargs):
        calls["count"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(predictor, "forward", counted_forward)

    latent_guided_search(
        bug_report=report,
        validator=_MockValidator(),
        config=config,
        model=model,
        predictor=predictor,
        converter=converter,
        rollout_mode="hybrid",
        critic_threshold=0.2,
    )

    assert calls["count"] == 0
