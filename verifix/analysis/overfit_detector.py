from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, RankedPatch
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v2 import RepairAgentV2
from verifix.validator.executor import ExecutionSandbox


@dataclass(frozen=False)
class OverfitAnalysis:
    total_plausible: int
    total_correct: int
    total_overfitted: int
    overfit_rate: float
    correct_rate: float
    detection_method: str
    per_patch_verdict: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.total_overfitted = max(0, self.total_plausible - self.total_correct)
        self.correct_rate = (self.total_correct / self.total_plausible) if self.total_plausible else 0.0
        self.overfit_rate = (self.total_overfitted / self.total_plausible) if self.total_plausible else 0.0


def compute_overfit_rate(
    plausible_patches: list[RankedPatch],
    reference_source: str | None,
    holdout_tests: list[str] | None,
    project_root: str,
    config: VerifixConfig,
) -> OverfitAnalysis:
    ref_norm = _normalize_source(reference_source) if reference_source is not None else None
    holdout = [test_id for test_id in (holdout_tests or []) if test_id]

    detection_signals = {
        "reference": ref_norm is not None,
        "holdout": bool(holdout),
        "fuzz": any(_extract_fuzz_survival(patch) is not None for patch in plausible_patches),
    }
    enabled = [name for name, enabled in detection_signals.items() if enabled]
    if len(enabled) >= 2:
        detection_method = "combined"
    elif enabled:
        detection_method = enabled[0]
    else:
        detection_method = "fuzz"

    per_patch: list[dict[str, Any]] = []
    correct_count = 0

    for patch in plausible_patches:
        reference_match: bool | None = None
        holdout_passed: bool | None = None
        fuzz_survived: bool | None = _extract_fuzz_survival(patch)

        if ref_norm is not None:
            reference_match = _normalize_source(patch.patched_source) == ref_norm

        if holdout:
            target_file = _infer_target_file(patch)
            holdout_passed = _run_holdout_tests(
                patched_source=patch.patched_source,
                project_root=project_root,
                target_file=target_file,
                holdout_tests=holdout,
                config=config,
            )

        evidence_results = [
            value
            for value in [reference_match, holdout_passed, fuzz_survived]
            if value is not None
        ]
        is_correct = all(evidence_results) if evidence_results else True

        if is_correct:
            correct_count += 1

        per_patch.append(
            {
                "rank": patch.rank,
                "plausible": bool(patch.validation.is_plausible),
                "correct": is_correct,
                "verdict": "CORRECT" if is_correct else "OVERFITTED",
                "reference_match": reference_match,
                "holdout_passed": holdout_passed,
                "fuzz_survived": fuzz_survived,
            }
        )

    analysis = OverfitAnalysis(
        total_plausible=len(plausible_patches),
        total_correct=correct_count,
        total_overfitted=0,
        overfit_rate=0.0,
        correct_rate=0.0,
        detection_method=detection_method,
        per_patch_verdict=per_patch,
    )
    return analysis


def compare_overfit_rates(
    v1_analysis: OverfitAnalysis,
    v2_analysis: OverfitAnalysis,
) -> dict[str, Any]:
    reduction = v1_analysis.overfit_rate - v2_analysis.overfit_rate
    if v1_analysis.overfit_rate > 0.0:
        reduction_percent = (reduction / v1_analysis.overfit_rate) * 100.0
    else:
        reduction_percent = 0.0

    if abs(reduction) < 1e-12:
        verdict = "NO_DIFFERENCE"
    elif reduction > 0:
        verdict = "V2_BETTER"
    else:
        verdict = "V1_BETTER"

    return {
        "v1_overfit_rate": v1_analysis.overfit_rate,
        "v2_overfit_rate": v2_analysis.overfit_rate,
        "reduction": reduction,
        "reduction_percent": reduction_percent,
        "verdict": verdict,
    }


def generate_overfit_report(
    v1_analysis: OverfitAnalysis,
    v2_analysis: OverfitAnalysis,
    benchmark_name: str,
) -> str:
    comparison = compare_overfit_rates(v1_analysis, v2_analysis)
    line = (
        f"V2 reduces the overfit rate by {comparison['reduction_percent']:.1f}% "
        f"compared to V1 (from {v1_analysis.overfit_rate * 100:.1f}% "
        f"to {v2_analysis.overfit_rate * 100:.1f}%)."
    )

    return "\n".join(
        [
            f"## Overfit Analysis: {benchmark_name}",
            "",
            "| System | Plausible | Correct | Overfitted | Overfit Rate |",
            "|--------|-----------|---------|------------|--------------|",
            (
                f"| V1 (Tests Only) | {v1_analysis.total_plausible} | {v1_analysis.total_correct} | "
                f"{v1_analysis.total_overfitted} | {v1_analysis.overfit_rate * 100:.1f}% |"
            ),
            (
                f"| V2 (Tests+Fuzz+SMT) | {v2_analysis.total_plausible} | {v2_analysis.total_correct} | "
                f"{v2_analysis.total_overfitted} | {v2_analysis.overfit_rate * 100:.1f}% |"
            ),
            "",
            line,
            f"Detection method: {v2_analysis.detection_method}.",
        ]
    )


def run_benchmark_overfit_study(
    benchmark_loader,
    bug_ids: list[str],
    config: VerifixConfig,
    output_dir: str,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_agent = RepairAgent(config)
    v2_agent = RepairAgentV2(config)

    studies: dict[str, Any] = {}
    v1_rates: list[float] = []
    v2_rates: list[float] = []

    resolved_bug_ids = list(bug_ids)
    if not resolved_bug_ids and hasattr(benchmark_loader, "load_all"):
        resolved_bug_ids = [item.bug_id for item in benchmark_loader.load_all()]

    for bug_id in resolved_bug_ids:
        report = benchmark_loader.load_bug(bug_id)
        reference_source = _resolve_reference_source(report)
        holdout_tests = _resolve_holdout_tests(report)

        v1_result = v1_agent.repair(report)
        v2_result = v2_agent.repair(report)

        v1_analysis = compute_overfit_rate(
            plausible_patches=v1_result.ranked_patches,
            reference_source=reference_source,
            holdout_tests=holdout_tests,
            project_root=report.project_root,
            config=config,
        )

        v2_patches = [evidence.patch for evidence in v2_result.evidence_list]
        v2_analysis = compute_overfit_rate(
            plausible_patches=v2_patches,
            reference_source=reference_source,
            holdout_tests=holdout_tests,
            project_root=report.project_root,
            config=config,
        )

        comparison = compare_overfit_rates(v1_analysis, v2_analysis)
        studies[bug_id] = {
            "v1": _analysis_to_dict(v1_analysis),
            "v2": _analysis_to_dict(v2_analysis),
            "comparison": comparison,
        }

        v1_rates.append(v1_analysis.overfit_rate)
        v2_rates.append(v2_analysis.overfit_rate)

        (out_dir / f"{bug_id}.json").write_text(json.dumps(studies[bug_id], indent=2), encoding="utf-8")

    summary = {
        "total_bugs": len(studies),
        "avg_v1_overfit_rate": (sum(v1_rates) / len(v1_rates)) if v1_rates else 0.0,
        "avg_v2_overfit_rate": (sum(v2_rates) / len(v2_rates)) if v2_rates else 0.0,
        "per_bug": studies,
    }
    summary["overall_comparison"] = compare_overfit_rates(
        OverfitAnalysis(
            total_plausible=1,
            total_correct=0,
            total_overfitted=0,
            overfit_rate=summary["avg_v1_overfit_rate"],
            correct_rate=0.0,
            detection_method="combined",
            per_patch_verdict=[],
        ),
        OverfitAnalysis(
            total_plausible=1,
            total_correct=0,
            total_overfitted=0,
            overfit_rate=summary["avg_v2_overfit_rate"],
            correct_rate=0.0,
            detection_method="combined",
            per_patch_verdict=[],
        ),
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _normalize_source(source: str | None) -> str:
    if source is None:
        return ""
    lines = [line.rstrip() for line in source.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _infer_target_file(patch: RankedPatch) -> str:
    for line in patch.diff.splitlines():
        if line.startswith("--- a/"):
            candidate = line[len("--- a/") :].strip()
            if candidate and candidate != "/dev/null":
                return candidate
        if line.startswith("+++ b/"):
            candidate = line[len("+++ b/") :].strip()
            if candidate and candidate != "/dev/null":
                return candidate
    return "buggy.py"


def _run_holdout_tests(
    patched_source: str,
    project_root: str,
    target_file: str,
    holdout_tests: list[str],
    config: VerifixConfig,
) -> bool:
    sandbox = ExecutionSandbox(
        project_root=project_root,
        working_dir=config.working_dir,
        python_executable=config.python_executable,
    )
    placeholder_report = BugReport(
        bug_id="holdout",
        language="python",
        buggy_source="",
        file_path=target_file,
        failing_tests=[],
        passing_tests=[],
        project_root=project_root,
        metadata={},
    )

    workspace_path: str | None = None
    try:
        workspace_path = sandbox.setup_workspace(placeholder_report)
        sandbox.write_patched_file(workspace_path, target_file, patched_source)
        results = sandbox.run_tests(workspace_path, holdout_tests, timeout_seconds=config.test_timeout_seconds)
        return all(passed for passed, _output in results.values())
    except Exception:
        return False
    finally:
        if workspace_path is not None:
            sandbox.cleanup_workspace(workspace_path)


def _extract_fuzz_survival(patch: RankedPatch) -> bool | None:
    text = " ".join(
        [
            patch.validation.runtime_error or "",
            patch.validation.compile_error or "",
            patch.diff,
        ]
    ).lower()

    if "fuzz_failed" in text or "fuzz_reject" in text:
        return False
    if "fuzz_passed" in text or "fuzz_survived" in text:
        return True
    return None


def _resolve_reference_source(report: BugReport) -> str | None:
    inline_reference = report.metadata.get("reference_source")
    if isinstance(inline_reference, str) and inline_reference.strip():
        return inline_reference

    fixed_path = report.metadata.get("fixed_source_path")
    if isinstance(fixed_path, str) and fixed_path.strip() and Path(fixed_path).exists():
        return Path(fixed_path).read_text(encoding="utf-8")
    return None


def _resolve_holdout_tests(report: BugReport) -> list[str] | None:
    raw = report.metadata.get("holdout_tests")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return None


def _analysis_to_dict(analysis: OverfitAnalysis) -> dict[str, Any]:
    return {
        "total_plausible": analysis.total_plausible,
        "total_correct": analysis.total_correct,
        "total_overfitted": analysis.total_overfitted,
        "overfit_rate": analysis.overfit_rate,
        "correct_rate": analysis.correct_rate,
        "detection_method": analysis.detection_method,
        "per_patch_verdict": list(analysis.per_patch_verdict),
    }


__all__ = [
    "OverfitAnalysis",
    "compute_overfit_rate",
    "compare_overfit_rates",
    "generate_overfit_report",
    "run_benchmark_overfit_study",
]
