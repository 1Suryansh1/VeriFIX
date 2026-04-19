"""QuixBugs PoC Benchmark Runner.

Dedicated runner with aggressive QuixBugsConfig settings and rich diagnostics.
Separated from the canonical quixbugs.py to preserve historical baselines.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from verifix.benchmarks.quixbugs import QuixBugsLoader
from verifix.core.config import QuixBugsConfig
from verifix.core.models import BugReport, RepairResult
from verifix.pipeline.repair_agent import RepairAgent
from verifix.verifier.smt_layer import smt_screen_patches


class QuixBugsPoCBenchmark:
    """Runs RepairAgent V1 on QuixBugs with QuixBugsConfig, reporting SMT gap."""

    def __init__(self, quixbugs_root: str, config: QuixBugsConfig | None = None) -> None:
        self.loader = QuixBugsLoader(quixbugs_root)
        self.config = config or QuixBugsConfig()
        self.agent = RepairAgent(config=self.config)

    def run_single(self, program_name: str) -> dict:
        start = time.monotonic()
        try:
            bug_report = self.loader.load_program(program_name, language="python")
        except Exception as exc:
            return {
                "program": program_name,
                "skipped": True,
                "success": False,
                "time": time.monotonic() - start,
                "error": f"load_error: {exc}",
                "edits_applied": 0,
                "validations": 0,
                "fl_hit": False,
                "smt_verified": False,
                "smt_verdict": "NOT_APPLICABLE",
            }

        if not bug_report.failing_tests:
            return {
                "program": program_name,
                "skipped": True,
                "success": False,
                "time": time.monotonic() - start,
                "error": "no failing tests inferred for program",
                "edits_applied": 0,
                "validations": 0,
                "fl_hit": False,
                "smt_verified": False,
                "smt_verdict": "NOT_APPLICABLE",
            }

        try:
            result = self.agent.repair(bug_report)
        except Exception as exc:
            return {
                "program": program_name,
                "skipped": False,
                "success": False,
                "time": time.monotonic() - start,
                "error": f"repair_error: {exc}",
                "edits_applied": 0,
                "validations": result.total_validations_run if "result" in dir() else 0,
                "fl_hit": False,
                "smt_verified": False,
                "smt_verdict": "NOT_APPLICABLE",
            }

        # --- SMT screening ---
        smt_verified = False
        smt_verdict = "NOT_APPLICABLE"
        if result.ranked_patches:
            plausible = [
                (p.edit_sequence, p.patched_source, p.validation)
                for p in result.ranked_patches
                if p.validation.is_plausible
            ]
            if plausible:
                screened = smt_screen_patches(
                    plausible,
                    original_source=bug_report.buggy_source,
                    top_k=1,
                    timeout_ms=5000.0,
                )
                if screened:
                    smt_result = screened[0][3]
                    smt_verdict = smt_result.verdict
                    smt_verified = smt_result.smt_passed

        top_edits = len(result.ranked_patches[0].edit_sequence) if result.ranked_patches else 0
        elapsed = time.monotonic() - start

        return {
            "program": program_name,
            "skipped": False,
            "success": result.success,
            "time": elapsed,
            "error": result.error,
            "edits_applied": top_edits,
            "validations": result.total_validations_run,
            "fl_hit": top_edits > 0,
            "smt_verified": smt_verified,
            "smt_verdict": smt_verdict,
        }

    def run_all(
        self,
        max_programs: int | None = None,
        output_dir: str = "./.quixbugs_poc_results",
        selected_programs: list[str] | None = None,
    ) -> dict:
        reports = self.loader.load_all(language="python")
        program_names = [
            str(r.metadata.get("program_name", r.bug_id)) for r in reports
        ]

        if selected_programs is not None:
            available = set(program_names)
            program_names = [n for n in selected_programs if n in available]

        if max_programs is not None:
            program_names = program_names[:max_programs]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        for name in program_names:
            print(f"[PoC] Running {name} ...", flush=True)
            item = self.run_single(name)
            status = "✅" if item["success"] else "❌"
            print(f"  {status} {name} | time={item['time']:.1f}s | validations={item['validations']} | smt={item['smt_verdict']}")
            results.append(item)

            (output_path / f"{name}.json").write_text(
                json.dumps(item, indent=2, default=str), encoding="utf-8"
            )

        dataset_total = len(results)
        evaluated = [r for r in results if not r.get("skipped", False)]
        skipped = dataset_total - len(evaluated)

        total = len(evaluated)
        repaired = sum(1 for r in evaluated if r["success"])
        smt_verified_count = sum(1 for r in evaluated if r["smt_verified"])
        avg_time = sum(r["time"] for r in evaluated) / total if total else 0.0
        avg_validations = sum(r["validations"] for r in evaluated) / total if total else 0.0
        fl_hit_count = sum(1 for r in evaluated if r["fl_hit"])

        summary = {
            "dataset_total": dataset_total,
            "skipped": skipped,
            "total": total,
            "repaired": repaired,
            "repair_rate": repaired / total if total else 0.0,
            "smt_verified": smt_verified_count,
            "smt_verification_rate": smt_verified_count / total if total else 0.0,
            "smt_gap": (repaired - smt_verified_count) / total if total else 0.0,
            "fl_hit_rate": fl_hit_count / total if total else 0.0,
            "avg_time_seconds": avg_time,
            "avg_validations": avg_validations,
            "config": repr(self.config),
            "per_program": {
                r["program"]: {
                    "success": r["success"],
                    "time": r["time"],
                    "smt_verified": r["smt_verified"],
                    "smt_verdict": r["smt_verdict"],
                    "edits_applied": r["edits_applied"],
                    "validations": r["validations"],
                    "error": r["error"],
                }
                for r in results
            },
        }

        (output_path / "poc_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        self._print_report(summary)
        return summary

    def _print_report(self, summary: dict) -> None:
        print("\n" + "=" * 60)
        print("  VeriFix PoC Results — QuixBugs")
        print("=" * 60)
        print(f"  Dataset programs: {summary['dataset_total']} | skipped: {summary['skipped']}")
        print(f"  Repair rate: {summary['repaired']}/{summary['total']} ({summary['repair_rate']:.1%})")
        print(f"  SMT verified: {summary['smt_verified']}/{summary['total']} ({summary['smt_verification_rate']:.1%})")
        print(f"  SMT gap: {summary['smt_gap']:.1%}")
        print(f"  FL hit rate: {summary['fl_hit_rate']:.1%}")
        print(f"  Avg time: {summary['avg_time_seconds']:.1f}s")
        print(f"  Avg validations: {summary['avg_validations']:.0f}")
        print("-" * 60)

        per = summary.get("per_program", {})
        print(f"  {'Program':<28} {'Status':>7} {'Edits':>5} {'Val':>4} {'SMT':>12} {'Time':>7}")
        print(f"  {'-'*28} {'-'*7} {'-'*5} {'-'*4} {'-'*12} {'-'*7}")
        for name in sorted(per.keys()):
            m = per[name]
            status = "✅" if m["success"] else "❌"
            print(f"  {name:<28} {status:>7} {m['edits_applied']:>5} {m['validations']:>4} {m['smt_verdict']:>12} {m['time']:>6.1f}s")

        print("=" * 60)


__all__ = ["QuixBugsPoCBenchmark"]
