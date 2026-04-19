from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.pipeline.repair_agent import RepairAgent
from verifix.verifier.evidence_report import (
    PatchEvidence,
    evidence_to_json,
    evidence_to_markdown,
)
from verifix.verifier.v2_pipeline import VerificationFunnel


@dataclass(frozen=False)
class RepairResultV2:
    v1_result: RepairResult
    evidence_list: list[PatchEvidence]
    funnel_stats: dict[str, Any]
    best_evidence: PatchEvidence | None
    total_wall_time_seconds: float

    @property
    def success(self) -> bool:
        return any(item.trust_level in {"HIGH", "MEDIUM"} for item in self.evidence_list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v1_result": self.v1_result.to_dict(),
            "evidence_list": [evidence_to_json(item) for item in self.evidence_list],
            "funnel_stats": dict(self.funnel_stats),
            "best_evidence": evidence_to_json(self.best_evidence) if self.best_evidence is not None else None,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "success": self.success,
        }

    def to_markdown(self) -> str:
        lines = [
            "# VeriFix V2 Repair Report",
            f"**V1 Success**: {'YES' if self.v1_result.success else 'NO'}",
            f"**V2 Success**: {'YES' if self.success else 'NO'}",
            f"**Total Time**: {self.total_wall_time_seconds:.2f}s",
            "",
            "## Funnel Summary",
            "| Metric | Value |",
            "|---|---|",
        ]

        for key in [
            "total_patches",
            "high_trust",
            "medium_trust",
            "low_trust",
            "fuzz_rejection_rate",
            "smt_verification_rate",
        ]:
            if key in self.funnel_stats:
                value = self.funnel_stats[key]
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.3f} |")
                else:
                    lines.append(f"| {key} | {value} |")

        lines.append("")
        if self.best_evidence is None:
            lines.append("No evidence report available.")
        else:
            lines.append(evidence_to_markdown(self.best_evidence))

        return "\n".join(lines)


class RepairAgentV2:
    def __init__(self, config: VerifixConfig | None = None) -> None:
        if config is None:
            self.config = VerifixConfig(**DEFAULT_CONFIG.model_dump())
        else:
            self.config = config

        self._v1_agent = RepairAgent(self.config)
        self._funnel = VerificationFunnel(self.config)
        self.logger = logging.getLogger("verifix.pipeline.v2")

    def repair(self, bug_report: BugReport) -> RepairResultV2:
        start = time.monotonic()

        def log(message: str) -> None:
            self.logger.info(message)

        log("V2 Pipeline: Running V1 search phase")
        v1_result = self._v1_agent.repair(bug_report)

        if not v1_result.success:
            log("V1 search found no plausible patches. V2 cannot proceed.")
            return RepairResultV2(
                v1_result=v1_result,
                evidence_list=[],
                funnel_stats=self._funnel.get_summary_stats([]),
                best_evidence=None,
                total_wall_time_seconds=v1_result.wall_time_seconds,
            )

        plausible = [
            (patch.edit_sequence, patch.patched_source, patch.validation)
            for patch in v1_result.ranked_patches
            if patch.validation.is_plausible
        ]

        log(f"V2 Pipeline: Running verification funnel on {len(plausible)} patches")
        funnel_start = time.monotonic()
        evidence_list = self._funnel.run(
            plausible_patches=plausible,
            bug_report=bug_report,
            original_source=bug_report.buggy_source,
            max_fuzz_patches=min(20, len(plausible)),
            top_k_smt=min(5, len(plausible)),
        )
        funnel_time = time.monotonic() - funnel_start

        high_count = len([item for item in evidence_list if item.trust_level == "HIGH"])
        log(f"V2 Pipeline: {high_count} HIGH-trust patches")

        total = v1_result.wall_time_seconds + funnel_time
        if total <= 0.0:
            total = time.monotonic() - start

        best = evidence_list[0] if evidence_list else None
        return RepairResultV2(
            v1_result=v1_result,
            evidence_list=evidence_list,
            funnel_stats=self._funnel.get_summary_stats(evidence_list),
            best_evidence=best,
            total_wall_time_seconds=total,
        )

    def repair_from_file(self, file_path: str, test_ids: list[str], project_root: str) -> RepairResultV2:
        source_path = Path(file_path)
        source = source_path.read_text(encoding="utf-8")

        project_root_path = Path(project_root).resolve()
        source_resolved = source_path.resolve()
        try:
            relative_path = source_resolved.relative_to(project_root_path).as_posix()
        except ValueError:
            relative_path = source_path.name

        language = "java" if source_path.suffix.lower() == ".java" else "python"

        report = BugReport(
            bug_id=source_path.stem,
            language=language,
            buggy_source=source,
            file_path=relative_path,
            failing_tests=list(test_ids),
            passing_tests=[],
            project_root=str(project_root_path),
            metadata={"source": "repair_from_file_v2"},
        )
        return self.repair(report)


__all__ = ["RepairResultV2", "RepairAgentV2"]
