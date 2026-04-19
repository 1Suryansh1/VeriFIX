from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.models.trainer import train_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VeriFIX V3 multi-task GAT on synthetic QuixBugs")
    parser.add_argument(
        "--dataset",
        default="data/quixbugs_jepa_train.jsonl",
        help="Path to synthetic JSONL training dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=".v3_checkpoints",
        help="Directory to write model checkpoint and summary",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-critic", type=float, default=10.0)
    parser.add_argument("--beta-localization", type=float, default=1.0)
    parser.add_argument("--beta-policy", type=float, default=1.0)
    parser.add_argument("--run-id", default="cycle_2026_04_04_manual")
    parser.add_argument("--strict-run-prefix", default="cycle_2026_04_04_")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    run_output_dir = output_root / args.run_id

    result = train_from_jsonl(
        dataset_path=args.dataset,
        output_dir=run_output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        alpha=args.alpha,
        beta_critic=args.beta_critic,
        beta_localization=args.beta_localization,
        beta_policy=args.beta_policy,
        max_records=(args.max_records if args.max_records > 0 else None),
        run_id=args.run_id,
        required_run_prefix=args.strict_run_prefix,
    )

    print(json.dumps(result, indent=2))
    print(f"Checkpoint: {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
