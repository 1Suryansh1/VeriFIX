from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.quixbugs import QuixBugsLoader
from verifix.core.config import QuixBugsConfig, VerifixConfig
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v2 import RepairAgentV2
from verifix.pipeline.repair_agent_v3 import RepairAgentV3


def parse_args() -> argparse.Namespace:
    qb_defaults = QuixBugsConfig()
    parser = argparse.ArgumentParser(
        description="Benchmark QuixBugs holdout across V1, V2, V3-hybrid, and V3-latent",
    )
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-artifact", default=".analysis/quixbugs_split_20_20.json")
    parser.add_argument("--split-side", choices=["test", "train"], default="test")
    parser.add_argument("--programs", default="")
    parser.add_argument("--output-json", default=".analysis/v4_holdout_ablation.json")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--mcts-iterations", type=int, default=qb_defaults.mcts_iterations)
    parser.add_argument("--mcts-max-depth", type=int, default=qb_defaults.mcts_max_depth)
    parser.add_argument("--max-validations", type=int, default=qb_defaults.max_validations)
    parser.add_argument("--max-patch-candidates", type=int, default=qb_defaults.max_patch_candidates)
    parser.add_argument("--max-candidates-per-node", type=int, default=qb_defaults.max_candidates_per_node)
    parser.add_argument("--fl-top-n-lines", type=int, default=qb_defaults.fl_top_n_lines)
    parser.add_argument("--time-budget", type=float, default=qb_defaults.mcts_time_budget_seconds)

    parser.add_argument("--v3-min-rollout-depth", type=int, default=3)
    parser.add_argument("--v3-branch-per-state", type=int, default=3)
    parser.add_argument("--v3-critic-threshold", type=float, default=0.45)
    parser.add_argument(
        "--v3-candidate-node-weight",
        type=float,
        default=qb_defaults.v3_candidate_node_weight,
    )
    parser.add_argument(
        "--v3-candidate-action-weight",
        type=float,
        default=qb_defaults.v3_candidate_action_weight,
    )
    parser.add_argument("--search-profile-name", default="")

    parser.add_argument(
        "--python-executable",
        default="c:/Users/sunil/OneDrive/Desktop/VeriFIX/.venv/Scripts/python.exe",
    )
    parser.add_argument("--working-dir", default="./.work_holdout_ablation_v4")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--run-v1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable V1 concrete baseline",
    )
    parser.add_argument(
        "--run-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable V2 verification-funnel baseline",
    )
    parser.add_argument(
        "--run-v3-hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable V3 hybrid rollout",
    )
    parser.add_argument(
        "--run-v3-latent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable V3 latent rollout",
    )
    return parser.parse_args()


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
        v3_candidate_node_weight=args.v3_candidate_node_weight,
        v3_candidate_action_weight=args.v3_candidate_action_weight,
    )


def _load_programs(args: argparse.Namespace) -> list[str]:
    if args.programs.strip():
        return [token.strip() for token in args.programs.split(",") if token.strip()]

    payload = json.loads(Path(args.split_artifact).read_text(encoding="utf-8"))
    key = "test_programs" if args.split_side == "test" else "train_programs"
    programs = payload.get(key, [])
    if not isinstance(programs, list):
        raise ValueError(f"Split artifact field {key} must be a list")
    return [str(name).strip() for name in programs if str(name).strip()]


def _aggregate_mode(entries: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(entries)
    repaired = sum(1 for item in entries if bool(item.get("success", False)))
    avg_time = sum(float(item.get("time_seconds", 0.0)) for item in entries) / total if total else 0.0
    avg_validations = (
        sum(float(item.get("validations", 0.0)) for item in entries) / total if total else 0.0
    )
    return {
        "attempted": total,
        "repaired": repaired,
        "repair_rate": (repaired / total) if total else 0.0,
        "avg_time_seconds": avg_time,
        "avg_validations": avg_validations,
    }


def _search_profile_name(args: argparse.Namespace) -> str:
    if args.search_profile_name.strip():
        return args.search_profile_name.strip()
    return (
        f"md{args.mcts_max_depth}"
        f"_rd{args.v3_min_rollout_depth}"
        f"_br{args.v3_branch_per_state}"
        f"_it{args.mcts_iterations}"
        f"_tb{args.time_budget:g}"
    )


def _checkpoint_provenance(checkpoint_path: str) -> dict[str, Any]:
    summary_path = Path(checkpoint_path).resolve().parent / "training_summary.json"
    if not summary_path.exists():
        return {}

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return {}

    return {
        "training_summary_path": str(summary_path.resolve()),
        "checkpoint_run_id": metadata.get("run_id"),
        "checkpoint_epochs": metadata.get("epochs"),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "action_space_sha256": metadata.get("action_space_sha256"),
        "operator_space_sha256": metadata.get("operator_space_sha256"),
    }


def main() -> int:
    args = parse_args()
    if not any((args.run_v1, args.run_v2, args.run_v3_hybrid, args.run_v3_latent)):
        raise ValueError("At least one mode must be enabled (--run-v1/--run-v2/--run-v3-hybrid/--run-v3-latent)")

    selected_programs = _load_programs(args)
    requested_programs = list(selected_programs)

    loader = QuixBugsLoader(args.quixbugs_root)
    report_map: dict[str, Any] = {}

    if args.show_progress:
        print(
            f"[v4] preparing reports for {len(requested_programs)} requested programs...",
            flush=True,
        )

    for index, name in enumerate(requested_programs, start=1):
        if args.show_progress:
            print(f"[v4] [load {index}/{len(requested_programs)}] {name} ...", flush=True)

        try:
            report = loader.load_program(name, language="python")
        except Exception as exc:
            if args.show_progress:
                print(f"[v4] [load {index}/{len(requested_programs)}] {name} skipped ({exc})", flush=True)
            continue

        if not report.failing_tests:
            if args.show_progress:
                print(
                    f"[v4] [load {index}/{len(requested_programs)}] {name} skipped (no failing tests)",
                    flush=True,
                )
            continue

        report_map[name] = report
        if args.show_progress:
            test_source = str(report.metadata.get("test_source", "unknown"))
            print(
                f"[v4] [load {index}/{len(requested_programs)}] {name} ready "
                f"(failing={len(report.failing_tests)}, source={test_source})",
                flush=True,
            )

    selected_programs = [name for name in requested_programs if name in report_map]
    total_programs = len(selected_programs)
    run_start = time.monotonic()

    if args.show_progress:
        print(
            f"[v4] starting holdout ablation on {total_programs}/{len(requested_programs)} programs "
            f"(checkpoint={Path(args.checkpoint).name})",
            flush=True,
        )

    base_config = _mk_base_config(args)

    v1_agent = RepairAgent(base_config) if args.run_v1 else None
    v2_agent = RepairAgentV2(base_config) if args.run_v2 else None

    hybrid_agent = None
    if args.run_v3_hybrid:
        hybrid_cfg = VerifixConfig(**base_config.model_dump())
        hybrid_cfg.v3_enabled = True
        hybrid_cfg.v3_rollout_mode = "hybrid"
        hybrid_agent = RepairAgentV3(hybrid_cfg, checkpoint_path=args.checkpoint, device=args.device)

    latent_agent = None
    if args.run_v3_latent:
        latent_cfg = VerifixConfig(**base_config.model_dump())
        latent_cfg.v3_enabled = True
        latent_cfg.v3_rollout_mode = "latent"
        latent_agent = RepairAgentV3(latent_cfg, checkpoint_path=args.checkpoint, device=args.device)

    per_program: dict[str, dict[str, Any]] = {name: {} for name in selected_programs}

    rows_v1: list[dict[str, Any]] = []
    rows_v2: list[dict[str, Any]] = []
    rows_hybrid: list[dict[str, Any]] = []
    rows_latent: list[dict[str, Any]] = []

    v2_plausible_programs = 0
    v2_smt_verified_programs = 0

    for index, program in enumerate(selected_programs, start=1):
        program_start = time.monotonic()
        report = report_map[program]

        if args.show_progress:
            print(f"[v4] [{index}/{total_programs}] {program} ...", flush=True)

        entry_v1: dict[str, Any] | None = None
        entry_v2: dict[str, Any] | None = None
        entry_hybrid: dict[str, Any] | None = None
        entry_latent: dict[str, Any] | None = None

        if v1_agent is not None:
            start = time.monotonic()
            res_v1 = v1_agent.repair(report)
            t_v1 = time.monotonic() - start
            entry_v1 = {
                "success": bool(res_v1.success),
                "time_seconds": float(t_v1),
                "validations": int(res_v1.total_validations_run),
            }
            rows_v1.append(entry_v1)
            per_program[program]["v1"] = entry_v1

        if v2_agent is not None:
            start = time.monotonic()
            res_v2 = v2_agent.repair(report)
            t_v2 = time.monotonic() - start
            smt_verified_any = any(
                evidence.smt_result.verdict == "VERIFIED" for evidence in res_v2.evidence_list
            )
            if res_v2.v1_result.success:
                v2_plausible_programs += 1
            if smt_verified_any:
                v2_smt_verified_programs += 1

            entry_v2 = {
                "success": bool(res_v2.success),
                "time_seconds": float(t_v2),
                "validations": int(res_v2.v1_result.total_validations_run),
                "smt_verified_any": bool(smt_verified_any),
                "funnel_stats": dict(res_v2.funnel_stats),
            }
            rows_v2.append(entry_v2)
            per_program[program]["v2"] = entry_v2

        if hybrid_agent is not None:
            start = time.monotonic()
            res_hybrid = hybrid_agent.repair(report)
            t_hybrid = time.monotonic() - start
            entry_hybrid = {
                "success": bool(res_hybrid.success),
                "time_seconds": float(t_hybrid),
                "validations": int(res_hybrid.repair_result.total_validations_run),
                "diagnostics": dict(res_hybrid.latent_diagnostics),
            }
            rows_hybrid.append(entry_hybrid)
            per_program[program]["v3_hybrid"] = entry_hybrid

        if latent_agent is not None:
            start = time.monotonic()
            res_latent = latent_agent.repair(report)
            t_latent = time.monotonic() - start
            entry_latent = {
                "success": bool(res_latent.success),
                "time_seconds": float(t_latent),
                "validations": int(res_latent.repair_result.total_validations_run),
                "diagnostics": dict(res_latent.latent_diagnostics),
            }
            rows_latent.append(entry_latent)
            per_program[program]["v3_latent"] = entry_latent

        if args.show_progress:
            elapsed_program = time.monotonic() - program_start
            elapsed_total = time.monotonic() - run_start
            mode_bits: list[str] = []
            if entry_v1 is not None:
                mode_bits.append(f"v1={int(entry_v1['success'])}")
            if entry_v2 is not None:
                mode_bits.append(f"v2={int(entry_v2['success'])}")
            if entry_hybrid is not None:
                mode_bits.append(f"hybrid={int(entry_hybrid['success'])}")
            if entry_latent is not None:
                mode_bits.append(f"latent={int(entry_latent['success'])}")
            mode_bits_text = " ".join(mode_bits)
            print(
                "[v4] "
                f"[{index}/{total_programs}] {program} done "
                f"in {elapsed_program:.1f}s | "
                f"{mode_bits_text} | "
                f"elapsed={elapsed_total:.1f}s",
                flush=True,
            )

    summary: dict[str, dict[str, float | int]] = {}
    if rows_v1:
        summary["v1"] = _aggregate_mode(rows_v1)
    if rows_v2:
        summary["v2"] = _aggregate_mode(rows_v2)
    if rows_hybrid:
        summary["v3_hybrid"] = _aggregate_mode(rows_hybrid)
    if rows_latent:
        summary["v3_latent"] = _aggregate_mode(rows_latent)

    smt_summary: dict[str, int] = {}
    if rows_v2:
        smt_summary = {
            "v2_plausible_programs": v2_plausible_programs,
            "v2_smt_verified_programs": v2_smt_verified_programs,
            "v2_smt_gap": v2_plausible_programs - v2_smt_verified_programs,
        }

    search_profile = _search_profile_name(args)
    checkpoint_meta = _checkpoint_provenance(args.checkpoint)
    node_weight = float(args.v3_candidate_node_weight)
    action_weight = float(args.v3_candidate_action_weight)
    weight_total = node_weight + action_weight
    if weight_total > 0.0:
        norm_node = node_weight / weight_total
        norm_action = action_weight / weight_total
    else:
        norm_node, norm_action = 0.2, 0.8

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split_artifact": str(Path(args.split_artifact).resolve()),
        "split_side": args.split_side,
        "selected_programs": selected_programs,
        "attempted_total": len(selected_programs),
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
            "v3_candidate_node_weight": node_weight,
            "v3_candidate_action_weight": action_weight,
            "v3_candidate_node_weight_normalized": norm_node,
            "v3_candidate_action_weight_normalized": norm_action,
        },
        "provenance": {
            "search_profile_name": search_profile,
            "scoring_mix": {
                "node_weight": node_weight,
                "action_weight": action_weight,
                "node_weight_normalized": norm_node,
                "action_weight_normalized": norm_action,
            },
            "seed": base_config.random_seed,
            **checkpoint_meta,
        },
        "summary": summary,
        "smt": smt_summary,
        "per_program": per_program,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.show_progress:
        print(f"[v4] completed in {time.monotonic() - run_start:.1f}s", flush=True)

    print(json.dumps({"output_json": str(output_path), "summary": summary, "smt": smt_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
