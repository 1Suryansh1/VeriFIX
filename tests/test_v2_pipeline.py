from __future__ import annotations

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, EditOperator, ValidationResult
from verifix.verifier.fuzzer import FuzzResult
from verifix.verifier.smt_layer import SMTResult
from verifix.verifier.v2_pipeline import VerificationFunnel


def _config() -> VerifixConfig:
    return VerifixConfig(mcts_iterations=10, fl_top_n_lines=1)


def _validation() -> ValidationResult:
    return ValidationResult(
        state_id="s",
        compiled=True,
        tests_passed=["tests::failing_case"],
        tests_failed=[],
        all_failing_tests_pass=True,
        no_regression=True,
        is_plausible=True,
        compile_error=None,
        runtime_error=None,
        execution_time_ms=3.0,
    )


def _edits(count: int, prefix: str = "e") -> list[Edit]:
    edits: list[Edit] = []
    for idx in range(count):
        edits.append(
            Edit(
                operator=EditOperator.REPLACE_OPERATOR,
                node_id=f"{prefix}-{idx}",
                node_type="Expr",
                line_number=idx + 1,
                original_text="<",
                replacement_text=">",
                metadata={},
            )
        )
    return edits


def _bug_report() -> BugReport:
    return BugReport(
        bug_id="V2-1",
        language="python",
        buggy_source="def target(x, y):\n    return x < y\n",
        file_path="buggy.py",
        failing_tests=["tests::failing_case"],
        passing_tests=["tests::regression_case"],
        project_root=".",
        metadata={"test_cases": [{"input": [1, 2], "expected": True}]},
    )


def _plausible_patches() -> list[tuple[list[Edit], str, ValidationResult]]:
    validation = _validation()
    return [
        (_edits(1, "good1"), "def target(x, y):\n    return x > y\n", validation),
        (_edits(1, "overfit1"), "def target(x, y):\n    return 0  # OVERFIT_A\n", validation),
        (_edits(1, "good2"), "def target(x, y):\n    return y < x\n", validation),
        (_edits(1, "overfit2"), "def target(x, y):\n    return 1  # OVERFIT_B\n", validation),
        (_edits(7, "uncertain"), "def target(x, y):\n    return x != y  # UNCERTAIN\n", validation),
    ]


def _mock_fuzz_patch(*args, **kwargs) -> FuzzResult:
    patched_source = kwargs.get("patched_source", "")
    if "OVERFIT_A" in patched_source or "OVERFIT_B" in patched_source:
        return FuzzResult(
            survived=False,
            total_inputs_tested=50,
            failing_inputs=[{"input": [1, 2], "error": "mismatch"}],
            coverage_achieved=0.6,
            fuzz_time_seconds=0.01,
            strategy_used="boundary",
        )
    return FuzzResult(
        survived=True,
        total_inputs_tested=50,
        failing_inputs=[],
        coverage_achieved=0.8,
        fuzz_time_seconds=0.01,
        strategy_used="boundary",
    )


def _mock_smt_screen_patches(patches, original_source, top_k=5):
    del original_source
    out = []
    for edits, source, validation in patches[:top_k]:
        if "UNCERTAIN" in source:
            smt = SMTResult(
                smt_applicable=True,
                smt_passed=False,
                counterexample=None,
                property_checked="equivalence",
                solver_time_ms=5.0,
                verdict="UNKNOWN",
            )
        else:
            smt = SMTResult(
                smt_applicable=True,
                smt_passed=True,
                counterexample={},
                property_checked="equivalence",
                solver_time_ms=5.0,
                verdict="VERIFIED",
            )
        out.append((edits, source, validation, smt))
    return out


def test_run_returns_only_fuzz_survivors(monkeypatch) -> None:
    import verifix.verifier.v2_pipeline as v2_pipeline_module

    monkeypatch.setattr(v2_pipeline_module, "fuzz_patch", _mock_fuzz_patch)
    monkeypatch.setattr(v2_pipeline_module, "smt_screen_patches", _mock_smt_screen_patches)

    funnel = VerificationFunnel(_config())
    evidence = funnel.run(_plausible_patches(), _bug_report(), _bug_report().buggy_source)

    assert len(evidence) == 3
    assert all("OVERFIT" not in item.patch.patched_source for item in evidence)


def test_output_sorted_by_trust_score_descending(monkeypatch) -> None:
    import verifix.verifier.v2_pipeline as v2_pipeline_module

    monkeypatch.setattr(v2_pipeline_module, "fuzz_patch", _mock_fuzz_patch)
    monkeypatch.setattr(v2_pipeline_module, "smt_screen_patches", _mock_smt_screen_patches)

    funnel = VerificationFunnel(_config())
    evidence = funnel.run(_plausible_patches(), _bug_report(), _bug_report().buggy_source)

    scores = [item.trust_score for item in evidence]
    assert scores == sorted(scores, reverse=True)


def test_high_trust_patches_come_first(monkeypatch) -> None:
    import verifix.verifier.v2_pipeline as v2_pipeline_module

    monkeypatch.setattr(v2_pipeline_module, "fuzz_patch", _mock_fuzz_patch)
    monkeypatch.setattr(v2_pipeline_module, "smt_screen_patches", _mock_smt_screen_patches)

    funnel = VerificationFunnel(_config())
    evidence = funnel.run(_plausible_patches(), _bug_report(), _bug_report().buggy_source)

    levels = [item.trust_level for item in evidence]
    assert levels[:2] == ["HIGH", "HIGH"]
    assert levels[2] in {"MEDIUM", "LOW", "UNVERIFIED"}


def test_get_fuzz_rejection_rate_is_point_four_for_two_of_five(monkeypatch) -> None:
    import verifix.verifier.v2_pipeline as v2_pipeline_module

    monkeypatch.setattr(v2_pipeline_module, "fuzz_patch", _mock_fuzz_patch)
    monkeypatch.setattr(v2_pipeline_module, "smt_screen_patches", _mock_smt_screen_patches)

    funnel = VerificationFunnel(_config())
    funnel.run(_plausible_patches(), _bug_report(), _bug_report().buggy_source)

    rate = funnel.get_fuzz_rejection_rate(funnel.last_run_results)
    assert rate == 0.4


def test_get_summary_stats_returns_expected_trust_counts(monkeypatch) -> None:
    import verifix.verifier.v2_pipeline as v2_pipeline_module

    monkeypatch.setattr(v2_pipeline_module, "fuzz_patch", _mock_fuzz_patch)
    monkeypatch.setattr(v2_pipeline_module, "smt_screen_patches", _mock_smt_screen_patches)

    funnel = VerificationFunnel(_config())
    evidence = funnel.run(_plausible_patches(), _bug_report(), _bug_report().buggy_source)
    stats = funnel.get_summary_stats(evidence)

    assert stats["total_patches"] == 5
    assert stats["high_trust"] == 2
    assert stats["medium_trust"] == 1
    assert stats["low_trust"] == 0
    assert stats["fuzz_rejection_rate"] == 0.4
    assert abs(stats["smt_verification_rate"] - (2.0 / 3.0)) < 1e-9


def test_empty_input_is_handled_gracefully() -> None:
    funnel = VerificationFunnel(_config())
    evidence = funnel.run([], _bug_report(), _bug_report().buggy_source)

    assert evidence == []
    assert funnel.get_fuzz_rejection_rate(funnel.last_run_results) == 0.0
