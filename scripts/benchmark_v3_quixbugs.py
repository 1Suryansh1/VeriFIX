from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.v3_quixbugs import V3QuixBugsBenchmark
from verifix.core.config import QuixBugsConfig, VerifixConfig
from verifix.core.provenance import validate_checkpoint_provenance


def parse_args() -> argparse.Namespace:
    qb_defaults = QuixBugsConfig()
    parser = argparse.ArgumentParser(description="Benchmark VeriFIX V3 on QuixBugs holdout")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=".quixbugs_v3_benchmark")
    parser.add_argument("--split-strategy", choices=["stratified", "alphabetical"], default="stratified")
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--max-programs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-iterations", type=int, default=qb_defaults.mcts_iterations)
    parser.add_argument("--mcts-max-depth", type=int, default=qb_defaults.mcts_max_depth)
    parser.add_argument("--max-validations", type=int, default=qb_defaults.max_validations)
    parser.add_argument("--max-patch-candidates", type=int, default=qb_defaults.max_patch_candidates)
    parser.add_argument("--time-budget", type=float, default=qb_defaults.mcts_time_budget_seconds)
    parser.add_argument("--fl-top-n-lines", type=int, default=qb_defaults.fl_top_n_lines)
    parser.add_argument("--max-candidates-per-node", type=int, default=qb_defaults.max_candidates_per_node)
    parser.add_argument("--run-latent-ablation", action="store_true")
    parser.add_argument("--v3-min-rollout-depth", type=int, default=3)
    parser.add_argument("--v3-branch-per-state", type=int, default=3)
    parser.add_argument("--v3-critic-threshold", type=float, default=0.45)
    parser.add_argument("--run-id", default="cycle_2026_04_04_eval")
    parser.add_argument("--strict-run-prefix", default="cycle_2026_04_04_")
    parser.add_argument("--skip-provenance-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_provenance_check:
        payload = torch.load(args.checkpoint, map_location="cpu")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        valid, issues = validate_checkpoint_provenance(
            metadata,
            required_run_prefix=args.strict_run_prefix,
        )
        if not valid:
            joined = "; ".join(issues)
            raise ValueError(f"Checkpoint provenance validation failed: {joined}")

    run_output_dir = Path(args.output_dir).resolve() / args.run_id

    config = VerifixConfig(
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        mcts_time_budget_seconds=args.time_budget,
        max_validations=args.max_validations,
        max_patch_candidates=args.max_patch_candidates,
        max_candidates_per_node=args.max_candidates_per_node,
        fl_top_n_lines=args.fl_top_n_lines,
        test_timeout_seconds=8.0,
        working_dir="./.work_quixbugs_v3",
        v3_min_rollout_depth=args.v3_min_rollout_depth,
        v3_branch_per_state=args.v3_branch_per_state,
        v3_critic_threshold=args.v3_critic_threshold,
    )

    benchmark = V3QuixBugsBenchmark(
        quixbugs_root=args.quixbugs_root,
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )
    summary = benchmark.run_holdout(
        output_dir=str(run_output_dir),
        split_strategy=args.split_strategy,
        seed=args.seed,
        max_programs=(args.max_programs if args.max_programs > 0 else None),
        run_latent_ablation=args.run_latent_ablation,
    )

    print(json.dumps(summary, indent=2))
    print(f"Summary JSON: {run_output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
