from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from verifix.benchmarks.quixbugs import QuixBugsLoader
from verifix.benchmarks.quixbugs_split import (
    DEFAULT_SPLIT_SEED,
    alphabetical_split,
    stratified_split,
)
from verifix.core.config import QuixBugsConfig, VerifixConfig
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v3 import RepairAgentV3


class V3QuixBugsBenchmark:
    def __init__(
        self,
        quixbugs_root: str,
        checkpoint_path: str,
        config: VerifixConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.loader = QuixBugsLoader(quixbugs_root)
        self.quixbugs_root = Path(quixbugs_root).resolve()
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.base_config = QuixBugsConfig() if config is None else config
        self.device = device

    def run_holdout(
        self,
        output_dir: str = "./results/quixbugs_v3",
        split_strategy: str = "stratified",
        seed: int = DEFAULT_SPLIT_SEED,
        max_programs: int | None = None,
        run_latent_ablation: bool = True,
    ) -> dict[str, Any]:
        reports = self.loader.load_all(language="python")
        raw_report_map = {
            str(report.metadata.get("program_name", report.bug_id)): report for report in reports
        }
        report_map = {
            name: report for name, report in raw_report_map.items() if len(report.failing_tests) > 0
        }
        available_programs = sorted(report_map.keys())

        fixed_sources = self._load_fixed_sources(available_programs)
        if len(fixed_sources) < 2:
            raise RuntimeError("Not enough runnable QuixBugs programs for benchmark")

        if len(fixed_sources) > 10:
            train_size = len(fixed_sources) - 10
        else:
            train_size = max(1, int(len(fixed_sources) * 0.7))

        if split_strategy == "stratified":
            split = stratified_split(fixed_sources, train_size=train_size, seed=seed)
        elif split_strategy == "alphabetical":
            split = alphabetical_split(fixed_sources.keys(), train_size=train_size)
        else:
            raise ValueError("split_strategy must be 'stratified' or 'alphabetical'")

        holdout = list(split.test_programs)
        if max_programs is not None:
            holdout = holdout[:max_programs]

        v1_agent = RepairAgent(self.base_config)
        hybrid_config = VerifixConfig(**self.base_config.model_dump())
        hybrid_config.v3_enabled = True
        hybrid_config.v3_rollout_mode = "hybrid"
        latent_config = VerifixConfig(**self.base_config.model_dump())
        latent_config.v3_enabled = True
        latent_config.v3_rollout_mode = "latent"

        v3_hybrid_agent = RepairAgentV3(
            config=hybrid_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        v3_latent_agent = RepairAgentV3(
            config=latent_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        per_program: dict[str, dict[str, Any]] = {}
        localization_hits: list[float] = []
        critic_brier: list[float] = []
        prescreen_rates: list[float] = []

        v1_records: list[dict[str, Any]] = []
        v3_hybrid_records: list[dict[str, Any]] = []
        v3_latent_records: list[dict[str, Any]] = []

        for program in holdout:
            bug_report = report_map.get(program)
            if bug_report is None:
                continue

            v1_start = time.monotonic()
            v1_result = v1_agent.repair(bug_report)
            v1_time = time.monotonic() - v1_start

            hybrid_start = time.monotonic()
            hybrid_result = v3_hybrid_agent.repair(bug_report)
            hybrid_time = time.monotonic() - hybrid_start

            latent_result = None
            latent_time = 0.0
            if run_latent_ablation:
                latent_start = time.monotonic()
                latent_result = v3_latent_agent.repair(bug_report)
                latent_time = time.monotonic() - latent_start

            fixed_source = fixed_sources.get(program, "")
            changed_lines = _changed_lines(bug_report.buggy_source, fixed_source)
            suspicious_lines = list(hybrid_result.latent_diagnostics.get("suspicious_lines", []))
            hit = 1.0 if changed_lines.intersection(set(suspicious_lines[:3])) else 0.0
            localization_hits.append(hit)

            initial_critic = float(hybrid_result.latent_diagnostics.get("initial_critic_score", 0.0))
            critic_brier.append((initial_critic - 0.0) ** 2)

            scored = float(hybrid_result.latent_diagnostics.get("candidates_scored", 0.0))
            validated = float(hybrid_result.latent_diagnostics.get("candidates_validated", 0.0))
            if scored > 0:
                prescreen_rates.append(validated / scored)

            v1_records.append(
                {
                    "success": bool(v1_result.success),
                    "time": float(v1_time),
                    "validations": int(v1_result.total_validations_run),
                }
            )
            v3_hybrid_records.append(
                {
                    "success": bool(hybrid_result.success),
                    "time": float(hybrid_time),
                    "validations": int(hybrid_result.repair_result.total_validations_run),
                }
            )

            per_program[program] = {
                "v1": {
                    "success": bool(v1_result.success),
                    "time_seconds": float(v1_time),
                    "validations": int(v1_result.total_validations_run),
                },
                "v3_hybrid": {
                    "success": bool(hybrid_result.success),
                    "time_seconds": float(hybrid_time),
                    "validations": int(hybrid_result.repair_result.total_validations_run),
                    "diagnostics": hybrid_result.latent_diagnostics,
                },
                "localization_top3_hit": bool(hit),
            }

            if latent_result is not None:
                v3_latent_records.append(
                    {
                        "success": bool(latent_result.success),
                        "time": float(latent_time),
                        "validations": int(latent_result.repair_result.total_validations_run),
                    }
                )
                per_program[program]["v3_latent"] = {
                    "success": bool(latent_result.success),
                    "time_seconds": float(latent_time),
                    "validations": int(latent_result.repair_result.total_validations_run),
                    "diagnostics": latent_result.latent_diagnostics,
                }

        summary = {
            "split_strategy": split_strategy,
            "seed": seed,
            "available_programs": available_programs,
            "train_size": train_size,
            "holdout_programs": holdout,
            "attempted_total": len(holdout),
            "baseline_v1": _aggregate_mode(v1_records),
            "v3_hybrid": _aggregate_mode(v3_hybrid_records),
            "v3_latent": _aggregate_mode(v3_latent_records) if run_latent_ablation else None,
            "v3_metrics": {
                "localization_top3_hit_rate": (sum(localization_hits) / len(localization_hits))
                if localization_hits
                else 0.0,
                "critic_brier_score": (sum(critic_brier) / len(critic_brier)) if critic_brier else 0.0,
                "latent_prescreen_hit_rate": (sum(prescreen_rates) / len(prescreen_rates))
                if prescreen_rates
                else 0.0,
            },
            "per_program": per_program,
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "summary.md").write_text(_to_markdown(summary), encoding="utf-8")
        return summary

    def _load_fixed_sources(self, allowed_programs: list[str]) -> dict[str, str]:
        fixed_dir = self.quixbugs_root / "correct_python_programs"
        sources: dict[str, str] = {}
        allowed = set(allowed_programs)
        for py_file in sorted(fixed_dir.glob("*.py")):
            name = py_file.stem
            if name.endswith("_test") or name == "node":
                continue
            if name not in allowed:
                continue
            sources[name] = py_file.read_text(encoding="utf-8")
        return sources


def _aggregate_mode(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    repaired = sum(1 for item in records if item.get("success"))
    avg_time = sum(float(item.get("time", 0.0)) for item in records) / total if total else 0.0
    avg_validations = (
        sum(float(item.get("validations", 0.0)) for item in records) / total if total else 0.0
    )
    return {
        "attempted": total,
        "repaired": repaired,
        "repair_rate": (repaired / total) if total else 0.0,
        "avg_time_seconds": avg_time,
        "avg_validations": avg_validations,
    }


def _changed_lines(buggy_source: str, fixed_source: str) -> set[int]:
    if not fixed_source:
        return set()
    buggy_lines = buggy_source.splitlines()
    fixed_lines = fixed_source.splitlines()
    max_len = max(len(buggy_lines), len(fixed_lines))
    changed: set[int] = set()
    for idx in range(max_len):
        buggy_line = buggy_lines[idx] if idx < len(buggy_lines) else ""
        fixed_line = fixed_lines[idx] if idx < len(fixed_lines) else ""
        if buggy_line != fixed_line:
            changed.add(idx + 1)
    return changed


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V3 QuixBugs Benchmark",
        "",
        f"- Split strategy: {summary.get('split_strategy')}",
        f"- Seed: {summary.get('seed')}",
        f"- Attempted holdout programs: {summary.get('attempted_total')}",
        "",
        "## Mode Metrics",
        "",
    ]

    for key in ["baseline_v1", "v3_hybrid", "v3_latent"]:
        payload = summary.get(key)
        if payload is None:
            continue
        lines.extend(
            [
                f"### {key}",
                f"- Repair rate: {float(payload.get('repair_rate', 0.0)):.3f}",
                f"- Avg time: {float(payload.get('avg_time_seconds', 0.0)):.3f}s",
                f"- Avg validations: {float(payload.get('avg_validations', 0.0)):.3f}",
                "",
            ]
        )

    metrics = summary.get("v3_metrics", {})
    lines.extend(
        [
            "## V3 Diagnostics",
            f"- Localization top-3 hit rate: {float(metrics.get('localization_top3_hit_rate', 0.0)):.3f}",
            f"- Critic Brier score: {float(metrics.get('critic_brier_score', 0.0)):.3f}",
            f"- Latent pre-screen hit rate: {float(metrics.get('latent_prescreen_hit_rate', 0.0)):.3f}",
            "",
        ]
    )

    return "\n".join(lines)


__all__ = ["V3QuixBugsBenchmark"]
