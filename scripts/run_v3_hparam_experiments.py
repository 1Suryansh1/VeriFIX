from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.v3_quixbugs import V3QuixBugsBenchmark
from verifix.core.config import VerifixConfig
from verifix.models.trainer import train_from_jsonl


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    alpha: float
    beta_critic: float
    beta_localization: float
    beta_policy: float
    lr: float
    epochs: int
    v3_depth: int
    v3_branch: int
    v3_threshold: float


PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(
        name="baseline_critic_heavy",
        alpha=1.0,
        beta_critic=10.0,
        beta_localization=1.0,
        beta_policy=1.0,
        lr=1e-3,
        epochs=2,
        v3_depth=3,
        v3_branch=3,
        v3_threshold=0.45,
    ),
    ExperimentPreset(
        name="transition_policy_focus",
        alpha=3.0,
        beta_critic=2.0,
        beta_localization=1.2,
        beta_policy=3.0,
        lr=8e-4,
        epochs=2,
        v3_depth=3,
        v3_branch=4,
        v3_threshold=0.40,
    ),
    ExperimentPreset(
        name="balanced_curriculum",
        alpha=2.0,
        beta_critic=4.0,
        beta_localization=1.5,
        beta_policy=2.0,
        lr=1e-3,
        epochs=2,
        v3_depth=3,
        v3_branch=5,
        v3_threshold=0.42,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cohesive V3 hyperparameter experiments")
    parser.add_argument("--dataset", default="data/quixbugs_jepa_train.jsonl")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--output-dir", default=".v3_experiments")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-programs", type=int, default=10)
    parser.add_argument("--split-strategy", choices=["stratified", "alphabetical"], default="stratified")
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--mcts-iterations", type=int, default=10)
    parser.add_argument("--max-validations", type=int, default=10)
    parser.add_argument("--time-budget", type=float, default=20.0)
    parser.add_argument("--strict-run-prefix", default="cycle_2026_04_04_")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []

    for preset in PRESETS:
        print(f"=== Running preset: {preset.name} ===")
        run_dir = root / preset.name
        train_dir = run_dir / "train"
        bench_dir = run_dir / "bench"
        train_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

        train_result = train_from_jsonl(
            dataset_path=args.dataset,
            output_dir=train_dir,
            epochs=preset.epochs,
            learning_rate=preset.lr,
            device=args.device,
            alpha=preset.alpha,
            beta_critic=preset.beta_critic,
            beta_localization=preset.beta_localization,
            beta_policy=preset.beta_policy,
            max_records=(args.max_records if args.max_records > 0 else None),
            run_id=f"cycle_2026_04_04_{preset.name}",
            required_run_prefix=args.strict_run_prefix,
        )

        config = VerifixConfig(
            mcts_iterations=args.mcts_iterations,
            mcts_max_depth=1,
            mcts_time_budget_seconds=args.time_budget,
            max_validations=args.max_validations,
            max_candidates_per_node=5,
            fl_top_n_lines=5,
            test_timeout_seconds=8.0,
            working_dir=str((run_dir / "work").resolve()),
            v3_min_rollout_depth=preset.v3_depth,
            v3_branch_per_state=preset.v3_branch,
            v3_critic_threshold=preset.v3_threshold,
        )

        benchmark = V3QuixBugsBenchmark(
            quixbugs_root=args.quixbugs_root,
            checkpoint_path=train_result["checkpoint_path"],
            config=config,
            device=args.device,
        )
        summary = benchmark.run_holdout(
            output_dir=str(bench_dir),
            split_strategy=args.split_strategy,
            seed=args.seed,
            max_programs=(args.max_programs if args.max_programs > 0 else None),
            run_latent_ablation=True,
        )

        row = {
            "preset": preset.name,
            "alpha": preset.alpha,
            "beta_critic": preset.beta_critic,
            "beta_localization": preset.beta_localization,
            "beta_policy": preset.beta_policy,
            "lr": preset.lr,
            "epochs": preset.epochs,
            "v3_depth": preset.v3_depth,
            "v3_branch": preset.v3_branch,
            "v3_threshold": preset.v3_threshold,
            "records": int(train_result["metadata"]["records"]),
            "v1_repair_rate": float(summary["baseline_v1"]["repair_rate"]),
            "v3_hybrid_repair_rate": float(summary["v3_hybrid"]["repair_rate"]),
            "v3_latent_repair_rate": float(summary["v3_latent"]["repair_rate"]),
            "v3_hybrid_avg_time": float(summary["v3_hybrid"]["avg_time_seconds"]),
            "loc_top3_hit_rate": float(summary["v3_metrics"]["localization_top3_hit_rate"]),
            "critic_brier": float(summary["v3_metrics"]["critic_brier_score"]),
            "prescreen_hit_rate": float(summary["v3_metrics"]["latent_prescreen_hit_rate"]),
            "checkpoint_path": train_result["checkpoint_path"],
            "benchmark_summary": str((bench_dir / "summary.json").resolve()),
        }
        aggregate_rows.append(row)

        (run_dir / "cohesive_result_template.json").write_text(
            json.dumps(row, indent=2),
            encoding="utf-8",
        )

    aggregate = {
        "dataset": str(Path(args.dataset).resolve()),
        "quixbugs_root": str(Path(args.quixbugs_root).resolve()),
        "device": args.device,
        "presets": [preset.__dict__ for preset in PRESETS],
        "results": aggregate_rows,
    }
    (root / "results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (root / "results.md").write_text(_to_markdown(aggregate_rows), encoding="utf-8")

    print(json.dumps(aggregate, indent=2))
    print(f"Aggregate JSON: {(root / 'results.json').resolve()}")
    return 0


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# V3 Hyperparameter Experiments",
        "",
        "| Preset | alpha | beta_c | beta_l | beta_p | depth | branch | thr | v1 rr | v3 rr | loc@3 | brier | pre-screen | time(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['preset']} | "
            f"{row['alpha']:.2f} | {row['beta_critic']:.2f} | {row['beta_localization']:.2f} | {row['beta_policy']:.2f} | "
            f"{row['v3_depth']} | {row['v3_branch']} | {row['v3_threshold']:.2f} | "
            f"{row['v1_repair_rate']:.3f} | {row['v3_hybrid_repair_rate']:.3f} | "
            f"{row['loc_top3_hit_rate']:.3f} | {row['critic_brier']:.3f} | {row['prescreen_hit_rate']:.3f} | "
            f"{row['v3_hybrid_avg_time']:.3f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
