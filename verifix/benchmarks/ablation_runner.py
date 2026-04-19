from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig
from verifix.core.models import BugReport, RankedPatch, RepairResult, ValidationResult
from verifix.edit_dsl.applicator import apply_edit_sequence, generate_diff, validate_syntax
from verifix.edit_dsl.operators import get_candidate_edits
from verifix.parser.ast_builder import build_ast
from verifix.parser.fault_localizer import localize_faults
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v2 import RepairAgentV2, RepairResultV2
from verifix.search.scorer import operator_priority_score
from verifix.validator.executor import validate_patch
from verifix.verifier.fuzzer import FuzzResult
from verifix.verifier.smt_layer import SMTResult
from verifix.verifier import v2_pipeline as v2_pipeline_module


logger = logging.getLogger("verifix.ablation")


@dataclass(frozen=False)
class AblationConfig:
    name: str
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    use_v2_pipeline: bool = False
    disable_fuzzing: bool = False
    disable_smt: bool = False
    search_strategy: str = "mcts"


STANDARD_ABLATIONS: list[AblationConfig] = [
    AblationConfig(
        name="baseline-v1-tests-only",
        description="V1: MCTS + constrained edits + test-only validation (no fuzz, no SMT)",
        use_v2_pipeline=False,
    ),
    AblationConfig(
        name="v2-tests-fuzz",
        description="V2: MCTS + constrained edits + tests + fuzzing (no SMT)",
        use_v2_pipeline=True,
        disable_smt=True,
    ),
    AblationConfig(
        name="v2-full",
        description="V2: MCTS + constrained edits + tests + fuzz + SMT",
        use_v2_pipeline=True,
    ),
    AblationConfig(
        name="greedy-search",
        description="Greedy search (no MCTS, no UCB1) + test-only validation",
        search_strategy="greedy",
        use_v2_pipeline=False,
    ),
    AblationConfig(
        name="beam-search-k3",
        description="Beam search k=3 (no MCTS) + test-only validation",
        search_strategy="beam",
        use_v2_pipeline=False,
        config_overrides={"beam_fallback_k": 3},
    ),
]


class GreedySearchAgent:
    def __init__(self, config: VerifixConfig | None = None) -> None:
        self.config = VerifixConfig(**DEFAULT_CONFIG.model_dump()) if config is None else config

    def repair(self, bug_report: BugReport) -> RepairResult:
        start = monotonic()
        ranked_patches: list[RankedPatch] = []
        validations = 0

        try:
            assert bug_report.language in self.config.supported_languages
            assert bug_report.failing_tests, "No failing tests provided"

            suspicious_scores = localize_faults(
                project_root=bug_report.project_root,
                source_file=bug_report.file_path,
                failing_tests=bug_report.failing_tests,
                passing_tests=bug_report.passing_tests,
                algorithm=self.config.fl_algorithm,
                top_n=self.config.fl_top_n_lines,
                python_executable=self.config.python_executable,
            )
            suspicious_lines = [item.line for item in suspicious_scores]
            if not suspicious_lines:
                suspicious_lines = list(range(1, bug_report.buggy_source.count("\n") + 2))

            annotated = build_ast(
                bug_report.buggy_source,
                bug_report.file_path,
                language=bug_report.language,
            )
            edits = get_candidate_edits(
                annotated,
                suspicious_lines,
                max_edits_per_node=self.config.max_candidates_per_node,
            )
            edits = sorted(edits, key=lambda edit: operator_priority_score([edit]), reverse=True)

            for idx, edit in enumerate(edits[: self.config.max_patch_candidates], start=1):
                patched_source, success_flags = apply_edit_sequence(bug_report.buggy_source, [edit])
                if not success_flags or not success_flags[0]:
                    continue

                syntax_ok, _syntax_error = validate_syntax(patched_source, language=bug_report.language)
                if not syntax_ok:
                    continue

                state_id = hashlib.sha256(patched_source.encode("utf-8")).hexdigest()[:12]
                validation = validate_patch(
                    patched_source=patched_source,
                    bug_report=bug_report,
                    config=self.config,
                    state_id=state_id,
                )
                validations += 1

                if not validation.is_plausible:
                    continue

                ranked_patches.append(
                    RankedPatch(
                        rank=1,
                        edit_sequence=[edit],
                        patched_source=patched_source,
                        validation=validation,
                        score=1.0 - (0.01 * idx),
                        diff=generate_diff(
                            bug_report.buggy_source,
                            patched_source,
                            file_path=bug_report.file_path,
                        ),
                    )
                )
                break

            return RepairResult(
                bug_id=bug_report.bug_id,
                success=bool(ranked_patches),
                ranked_patches=ranked_patches,
                total_states_explored=min(len(edits), self.config.max_patch_candidates),
                total_validations_run=validations,
                wall_time_seconds=monotonic() - start,
                search_tree_depth=1,
                error=None,
            )
        except Exception as exc:
            return RepairResult(
                bug_id=bug_report.bug_id,
                success=False,
                ranked_patches=[],
                total_states_explored=0,
                total_validations_run=validations,
                wall_time_seconds=monotonic() - start,
                search_tree_depth=0,
                error=str(exc),
            )

    def repair_from_file(self, file_path: str, test_ids: list[str], project_root: str) -> RepairResult:
        source_path = Path(file_path)
        source = source_path.read_text(encoding="utf-8")

        project_root_path = Path(project_root).resolve()
        source_resolved = source_path.resolve()
        try:
            relative_path = source_resolved.relative_to(project_root_path).as_posix()
        except ValueError:
            relative_path = source_path.name

        report = BugReport(
            bug_id=source_path.stem,
            language="java" if source_path.suffix.lower() == ".java" else "python",
            buggy_source=source,
            file_path=relative_path,
            failing_tests=list(test_ids),
            passing_tests=[],
            project_root=str(project_root_path),
            metadata={"source": "greedy_repair_from_file"},
        )
        return self.repair(report)


def run_ablation(
    ablation: AblationConfig,
    bug_reports: list[BugReport],
    base_config: VerifixConfig,
    output_dir: str,
) -> dict:
    config = _with_overrides(base_config, ablation.config_overrides)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bugs_total = len(bug_reports)
    bugs_repaired = 0
    total_plausible = 0
    total_validations = 0
    total_time = 0.0
    high_trust_patches = 0
    fuzz_rates: list[float] = []

    per_bug: dict[str, dict[str, Any]] = {}

    for report in bug_reports:
        result = _run_single(report, config, ablation)

        if isinstance(result, RepairResultV2):
            repaired = bool(result.success)
            plausible = len(result.evidence_list)
            validations = result.v1_result.total_validations_run
            wall_time = result.total_wall_time_seconds
            high = len([item for item in result.evidence_list if item.trust_level == "HIGH"])
            fuzz_rate = float(result.funnel_stats.get("fuzz_rejection_rate", 0.0))

            high_trust_patches += high
            fuzz_rates.append(fuzz_rate)

            per_bug[report.bug_id] = {
                "success": repaired,
                "plausible_patches": plausible,
                "validations": validations,
                "wall_time": wall_time,
                "high_trust": high,
                "fuzz_rejection_rate": fuzz_rate,
            }
        else:
            repaired = bool(result.success)
            plausible = len(result.ranked_patches)
            validations = result.total_validations_run
            wall_time = result.wall_time_seconds

            per_bug[report.bug_id] = {
                "success": repaired,
                "plausible_patches": plausible,
                "validations": validations,
                "wall_time": wall_time,
                "high_trust": None,
                "fuzz_rejection_rate": None,
            }

        bugs_repaired += int(repaired)
        total_plausible += plausible
        total_validations += validations
        total_time += wall_time

    summary = {
        "ablation_name": ablation.name,
        "bugs_total": bugs_total,
        "bugs_repaired": bugs_repaired,
        "repair_rate": (bugs_repaired / bugs_total) if bugs_total else 0.0,
        "avg_plausible_patches": (total_plausible / bugs_total) if bugs_total else 0.0,
        "avg_validations_used": (total_validations / bugs_total) if bugs_total else 0.0,
        "avg_wall_time": (total_time / bugs_total) if bugs_total else 0.0,
        "high_trust_patches": high_trust_patches if ablation.use_v2_pipeline else None,
        "fuzz_rejection_rate": (sum(fuzz_rates) / len(fuzz_rates)) if fuzz_rates else None,
        "per_bug": per_bug,
    }

    (out_dir / f"{ablation.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_all_ablations(
    bug_reports: list[BugReport],
    base_config: VerifixConfig,
    ablations: list[AblationConfig] | None = None,
    output_dir: str = "./ablation_results",
    parallel: bool = False,
) -> dict:
    selected = list(STANDARD_ABLATIONS if ablations is None else ablations)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    if parallel and len(selected) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(selected))) as pool:
            future_map = {
                pool.submit(run_ablation, ablation, bug_reports, base_config, str(out_dir)): ablation.name
                for ablation in selected
            }
            for future in as_completed(future_map):
                name = future_map[future]
                summaries[name] = future.result()
    else:
        for ablation in selected:
            summaries[ablation.name] = run_ablation(ablation, bug_reports, base_config, str(out_dir))

    comparison: dict[str, dict[str, Any]] = {}
    for name, summary in summaries.items():
        comparison[name] = {
            "repair_rate": summary.get("repair_rate", 0.0),
            "avg_validations": summary.get("avg_validations_used", 0.0),
            "avg_time": summary.get("avg_wall_time", 0.0),
            "high_trust_patches": summary.get("high_trust_patches"),
            "fuzz_rejection_rate": summary.get("fuzz_rejection_rate"),
        }

    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def print_ablation_table(results: dict) -> None:
    header = "Ablation                | Repair Rate | Avg Valid. | Avg Time | Fuzz Reject"
    separator = "-" * len(header)
    print(header)
    print(separator)

    for name in sorted(results.keys()):
        row = results[name]
        repair_rate = 100.0 * float(row.get("repair_rate", 0.0))
        avg_valid = float(row.get("avg_validations", 0.0))
        avg_time = float(row.get("avg_time", 0.0))
        fuzz_reject = row.get("fuzz_rejection_rate")

        fuzz_str = "N/A" if fuzz_reject is None else f"{100.0 * float(fuzz_reject):.1f}%"
        print(f"{name:<23} | {repair_rate:9.1f}% | {avg_valid:10.1f} | {avg_time:7.1f}s | {fuzz_str:>10}")


def _with_overrides(base_config: VerifixConfig, overrides: dict[str, Any]) -> VerifixConfig:
    payload = base_config.model_dump()
    payload.update(overrides)
    return VerifixConfig(**payload)


def _run_single(report: BugReport, config: VerifixConfig, ablation: AblationConfig) -> RepairResult | RepairResultV2:
    strategy = ablation.search_strategy.lower().strip()

    if strategy == "greedy":
        return GreedySearchAgent(config).repair(report)

    if ablation.use_v2_pipeline:
        agent = RepairAgentV2(config)
        return _run_v2_with_stage_toggles(agent, report, ablation)

    return RepairAgent(config).repair(report)


def _run_v2_with_stage_toggles(
    agent: RepairAgentV2,
    bug_report: BugReport,
    ablation: AblationConfig,
) -> RepairResultV2:
    if not ablation.disable_fuzzing and not ablation.disable_smt:
        return agent.repair(bug_report)

    original_fuzz_patch = v2_pipeline_module.fuzz_patch
    original_smt_screen = v2_pipeline_module.smt_screen_patches

    def _neutral_fuzz_patch(*args, **kwargs) -> FuzzResult:
        del args
        del kwargs
        return FuzzResult(
            survived=True,
            total_inputs_tested=50,
            failing_inputs=[],
            coverage_achieved=0.0,
            fuzz_time_seconds=0.0,
            strategy_used="disabled_fuzz",
        )

    def _neutral_smt_screen(
        patches: list[tuple],
        original_source: str,
        top_k: int = 5,
        timeout_ms: float = 5000.0,
    ) -> list[tuple]:
        del original_source
        del timeout_ms
        out = []
        for edits, patched_source, validation in patches[:top_k]:
            out.append(
                (
                    edits,
                    patched_source,
                    validation,
                    SMTResult(
                        smt_applicable=False,
                        smt_passed=False,
                        counterexample=None,
                        property_checked="disabled_smt",
                        solver_time_ms=0.0,
                        verdict="NOT_APPLICABLE",
                    ),
                )
            )
        return out

    try:
        if ablation.disable_fuzzing:
            v2_pipeline_module.fuzz_patch = _neutral_fuzz_patch
        if ablation.disable_smt:
            v2_pipeline_module.smt_screen_patches = _neutral_smt_screen

        return agent.repair(bug_report)
    finally:
        v2_pipeline_module.fuzz_patch = original_fuzz_patch
        v2_pipeline_module.smt_screen_patches = original_smt_screen


__all__ = [
    "AblationConfig",
    "STANDARD_ABLATIONS",
    "GreedySearchAgent",
    "run_ablation",
    "run_all_ablations",
    "print_ablation_table",
]
