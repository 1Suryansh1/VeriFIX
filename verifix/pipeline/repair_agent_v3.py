from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.models.latent_jepa import (
    JEPATransitionPredictor,
    MultiTaskRepairGAT,
    load_checkpoint,
)
from verifix.models.pyg_converter import ASTtoPyG
from verifix.pipeline.repair_agent import RepairAgent
from verifix.search.mcts_latent import LatentSearchDiagnostics, latent_guided_search
from verifix.validator.executor import ConcreteValidator
from verifix.validator.patch_ranker import rank_patches


@dataclass(frozen=False)
class RepairResultV3:
    repair_result: RepairResult
    mode: str
    latent_diagnostics: dict[str, Any]
    checkpoint_path: str | None

    @property
    def success(self) -> bool:
        return self.repair_result.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_result": self.repair_result.to_dict(),
            "mode": self.mode,
            "latent_diagnostics": dict(self.latent_diagnostics),
            "checkpoint_path": self.checkpoint_path,
            "success": self.success,
        }


class RepairAgentV3:
    def __init__(
        self,
        config: VerifixConfig | None = None,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = VerifixConfig(**DEFAULT_CONFIG.model_dump()) if config is None else config
        self.device = device
        self.logger = logging.getLogger("verifix.pipeline.v3")

        self._baseline = RepairAgent(self.config)
        self._converter = ASTtoPyG()
        self._model: MultiTaskRepairGAT | None = None
        self._predictor: JEPATransitionPredictor | None = None
        self._checkpoint_metadata: dict[str, Any] = {}
        self._checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path else None

        if self._checkpoint_path is not None and Path(self._checkpoint_path).exists():
            self._load_checkpoint(self._checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        model, predictor, metadata = load_checkpoint(checkpoint_path, device=self.device)
        self._model = model
        self._predictor = predictor
        self._checkpoint_metadata = metadata

    def repair(self, bug_report: BugReport) -> RepairResultV3:
        start = time.monotonic()

        if self._model is None or self._predictor is None:
            fallback = self._baseline.repair(bug_report)
            diagnostics = {
                "fallback": "checkpoint_not_loaded",
                "wall_time_seconds": time.monotonic() - start,
            }
            return RepairResultV3(
                repair_result=fallback,
                mode="fallback-concrete",
                latent_diagnostics=diagnostics,
                checkpoint_path=self._checkpoint_path,
            )

        try:
            validator = ConcreteValidator(self.config)
            search_result, diag = latent_guided_search(
                bug_report=bug_report,
                validator=validator,
                config=self.config,
                model=self._model,
                predictor=self._predictor,
                converter=self._converter,
                rollout_mode=self.config.v3_rollout_mode,
                critic_threshold=self.config.v3_critic_threshold,
            )

            ranked_patches = rank_patches(
                search_result=search_result,
                original_source=bug_report.buggy_source,
                bug_report=bug_report,
                config=self.config,
                max_patches=10,
            )

            repair_result = RepairResult(
                bug_id=bug_report.bug_id,
                success=bool(ranked_patches),
                ranked_patches=ranked_patches,
                total_states_explored=search_result.total_iterations,
                total_validations_run=search_result.total_validations,
                wall_time_seconds=time.monotonic() - start,
                search_tree_depth=search_result.tree_depth_reached,
                error=None,
            )
            diagnostics = _diag_to_dict(diag)
            diagnostics["terminated_by"] = search_result.terminated_by
            diagnostics["checkpoint_epochs"] = self._checkpoint_metadata.get("epochs")

            return RepairResultV3(
                repair_result=repair_result,
                mode=self.config.v3_rollout_mode,
                latent_diagnostics=diagnostics,
                checkpoint_path=self._checkpoint_path,
            )
        except Exception as exc:
            self.logger.error("V3 repair failed (%s); falling back to concrete baseline", exc)
            fallback = self._baseline.repair(bug_report)
            diagnostics = {
                "fallback": "runtime_error",
                "error": str(exc),
                "wall_time_seconds": time.monotonic() - start,
            }
            return RepairResultV3(
                repair_result=fallback,
                mode="fallback-concrete",
                latent_diagnostics=diagnostics,
                checkpoint_path=self._checkpoint_path,
            )

    def repair_from_file(self, file_path: str, test_ids: list[str], project_root: str) -> RepairResultV3:
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
            metadata={"source": "repair_from_file_v3"},
        )
        return self.repair(report)


def _diag_to_dict(diag: LatentSearchDiagnostics) -> dict[str, Any]:
    return {
        "suspicious_lines": list(diag.suspicious_lines),
        "initial_critic_score": float(diag.initial_critic_score),
        "candidate_count": int(diag.candidate_count),
        "candidates_scored": int(diag.candidates_scored),
        "candidates_validated": int(diag.candidates_validated),
        "screened_out_by_critic": int(diag.screened_out_by_critic),
        "best_candidate_critic": float(diag.best_candidate_critic),
        "rollout_mode": diag.rollout_mode,
        "depth_floor": int(diag.depth_floor),
        "depth_reached": int(diag.depth_reached),
        "candidate_node_weight": float(diag.candidate_node_weight),
        "candidate_action_weight": float(diag.candidate_action_weight),
        "root_scored_candidates": int(diag.root_scored_candidates),
        "root_candidates_kept_after_global_beam": int(diag.root_candidates_kept_after_global_beam),
        "root_candidates_dropped_by_global_beam": int(diag.root_candidates_dropped_by_global_beam),
        "candidate_eval_limit": int(diag.candidate_eval_limit),
        "candidate_eval_truncated_by_max_patch": int(diag.candidate_eval_truncated_by_max_patch),
        "per_state_candidates_total": int(diag.per_state_candidates_total),
        "per_state_candidates_selected": int(diag.per_state_candidates_selected),
        "per_state_candidates_dropped_by_branch": int(diag.per_state_candidates_dropped_by_branch),
        "frontier_candidates_total": int(diag.frontier_candidates_total),
        "frontier_candidates_after_dedupe": int(diag.frontier_candidates_after_dedupe),
        "frontier_candidates_kept_after_global_beam": int(diag.frontier_candidates_kept_after_global_beam),
        "frontier_candidates_dropped_by_global_beam": int(diag.frontier_candidates_dropped_by_global_beam),
        "frontier_candidates_dropped_by_dedupe": int(diag.frontier_candidates_dropped_by_dedupe),
        "gate_reason_counts": dict(diag.gate_reason_counts),
        "validation_reason_counts": dict(diag.validation_reason_counts),
        "candidate_trace_top": [dict(item) for item in diag.candidate_trace_top],
        "trace_version": diag.trace_version,
    }


__all__ = ["RepairResultV3", "RepairAgentV3"]
