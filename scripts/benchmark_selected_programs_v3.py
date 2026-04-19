from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.quixbugs import QuixBugsLoader
from verifix.core.config import QuixBugsConfig, VerifixConfig
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v3 import RepairAgentV3


def parse_args() -> argparse.Namespace:
    qb_defaults = QuixBugsConfig()
    parser = argparse.ArgumentParser(description="Benchmark selected QuixBugs programs across V1/V3 modes")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--checkpoint", default=".v3_checkpoints/cycle_2026_04_04_opcov225_train01/v3_multitask_gat.pt")
    parser.add_argument("--matrix-json", default=".quixbugs_operator_matrix_afterfix.json")
    parser.add_argument("--programs", default="")
    parser.add_argument("--program-source", choices=["manual", "expressible"], default="expressible")
    parser.add_argument("--modes", default="v1,hybrid")
    parser.add_argument("--output-json", default=".analysis/selected_programs_benchmark.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-iterations", type=int, default=qb_defaults.mcts_iterations)
    parser.add_argument("--mcts-max-depth", type=int, default=qb_defaults.mcts_max_depth)
    parser.add_argument("--max-validations", type=int, default=qb_defaults.max_validations)
    parser.add_argument("--max-patch-candidates", type=int, default=qb_defaults.max_patch_candidates)
    parser.add_argument("--max-candidates-per-node", type=int, default=qb_defaults.max_candidates_per_node)
    parser.add_argument("--fl-top-n-lines", type=int, default=qb_defaults.fl_top_n_lines)
    parser.add_argument("--time-budget", type=float, default=qb_defaults.mcts_time_budget_seconds)
    parser.add_argument("--v3-min-rollout-depth", type=int, default=1)
    parser.add_argument("--v3-branch-per-state", type=int, default=4)
    parser.add_argument("--v3-critic-threshold", type=float, default=0.30)
    parser.add_argument("--python-executable", default="c:/Users/sunil/OneDrive/Desktop/VeriFIX/.venv/Scripts/python.exe")
    parser.add_argument("--working-dir", default="./.work_selected_bench")
    return parser.parse_args()


def _resolve_programs(args: argparse.Namespace, runnable: set[str]) -> list[str]:
    if args.program_source == "manual":
        requested = [token.strip() for token in args.programs.split(",") if token.strip()]
        return [name for name in requested if name in runnable]

    matrix_payload = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))
    expressible = [
        row["program"]
        for row in matrix_payload.get("rows", [])
        if str(row.get("status", "")).startswith("expressible")
    ]
    return [name for name in expressible if name in runnable]


def _mk_base_config(args: argparse.Namespace) -> VerifixConfig:
    return VerifixConfig(
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        mcts_time_budget_seconds=args.time_budget,
        max_validations=args.max_validations,
        max_patch_candidates=args.max_patch_candidates,
        max_candidates_per_node=args.max_candidates_per_node,
        fl_top_n_lines=args.fl_top_n_lines,
        test_timeout_seconds=8.0,
        python_executable=args.python_executable,
        working_dir=args.working_dir,
        v3_min_rollout_depth=args.v3_min_rollout_depth,
        v3_branch_per_state=args.v3_branch_per_state,
        v3_critic_threshold=args.v3_critic_threshold,
    )


def main() -> int:
    args = parse_args()
    modes = [token.strip().lower() for token in args.modes.split(",") if token.strip()]
    allowed_modes = {"v1", "hybrid", "latent"}
    for mode in modes:
        if mode not in allowed_modes:
            raise ValueError(f"Unsupported mode: {mode}")

    loader = QuixBugsLoader(args.quixbugs_root)
    reports = loader.load_all(language="python")
    report_map = {
        str(report.metadata.get("program_name", report.bug_id)): report
        for report in reports
        if report.failing_tests
    }
    selected_programs = _resolve_programs(args, set(report_map.keys()))

    base_config = _mk_base_config(args)

    agents: dict[str, object] = {}
    if "v1" in modes:
        agents["v1"] = RepairAgent(base_config)
    if "hybrid" in modes:
        cfg = VerifixConfig(**base_config.model_dump())
        cfg.v3_enabled = True
        cfg.v3_rollout_mode = "hybrid"
        agents["hybrid"] = RepairAgentV3(cfg, checkpoint_path=args.checkpoint, device=args.device)
    if "latent" in modes:
        cfg = VerifixConfig(**base_config.model_dump())
        cfg.v3_enabled = True
        cfg.v3_rollout_mode = "latent"
        agents["latent"] = RepairAgentV3(cfg, checkpoint_path=args.checkpoint, device=args.device)

    per_program: dict[str, dict[str, object]] = {name: {} for name in selected_programs}
    aggregate = {
        mode: {"attempted": 0, "repaired": 0, "times": [], "validations": []}
        for mode in modes
    }

    for program in selected_programs:
        report = report_map[program]
        for mode in modes:
            agent = agents[mode]
            started = time.monotonic()

            if mode == "v1":
                result = agent.repair(report)  # type: ignore[union-attr]
                success = bool(result.success)
                validations = int(result.total_validations_run)
                terminated_by = None
                diagnostics = {}
            else:
                result = agent.repair(report)  # type: ignore[union-attr]
                success = bool(result.success)
                validations = int(result.repair_result.total_validations_run)
                terminated_by = str(result.latent_diagnostics.get("terminated_by"))
                diagnostics = dict(result.latent_diagnostics)

            elapsed = time.monotonic() - started
            aggregate[mode]["attempted"] += 1
            aggregate[mode]["repaired"] += int(success)
            aggregate[mode]["times"].append(float(elapsed))
            aggregate[mode]["validations"].append(float(validations))

            per_program[program][mode] = {
                "success": success,
                "time_seconds": float(elapsed),
                "validations": validations,
                "terminated_by": terminated_by,
                "diagnostics": diagnostics,
            }

    mode_summary: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        attempted = int(aggregate[mode]["attempted"])
        repaired = int(aggregate[mode]["repaired"])
        times = list(aggregate[mode]["times"])
        validations = list(aggregate[mode]["validations"])
        mode_summary[mode] = {
            "attempted": attempted,
            "repaired": repaired,
            "repair_rate": (repaired / attempted) if attempted else 0.0,
            "avg_time_seconds": (sum(times) / attempted) if attempted else 0.0,
            "avg_validations": (sum(validations) / attempted) if attempted else 0.0,
        }

    payload = {
        "selected_programs": selected_programs,
        "program_source": args.program_source,
        "modes": modes,
        "config": {
            "mcts_iterations": args.mcts_iterations,
            "mcts_max_depth": args.mcts_max_depth,
            "max_validations": args.max_validations,
            "max_patch_candidates": args.max_patch_candidates,
            "max_candidates_per_node": args.max_candidates_per_node,
            "fl_top_n_lines": args.fl_top_n_lines,
            "time_budget": args.time_budget,
            "v3_min_rollout_depth": args.v3_min_rollout_depth,
            "v3_branch_per_state": args.v3_branch_per_state,
            "v3_critic_threshold": args.v3_critic_threshold,
        },
        "summary": mode_summary,
        "per_program": per_program,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "summary": mode_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
