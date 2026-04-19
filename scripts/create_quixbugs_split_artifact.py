from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.quixbugs_split import (
    DEFAULT_SPLIT_SEED,
    alphabetical_split,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic QuixBugs split artifact")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--strategy", choices=["stratified", "alphabetical"], default="stratified")
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--output", default=".analysis/quixbugs_split_20_20.json")
    return parser.parse_args()


def _load_fixed_sources(quixbugs_root: Path) -> dict[str, str]:
    fixed_dir = quixbugs_root / "correct_python_programs"
    if not fixed_dir.exists():
        raise FileNotFoundError(f"Missing directory: {fixed_dir}")

    sources: dict[str, str] = {}
    for file in sorted(fixed_dir.glob("*.py")):
        name = file.stem
        if name == "node" or name.endswith("_test"):
            continue
        sources[name] = file.read_text(encoding="utf-8")
    return sources


def main() -> int:
    args = parse_args()

    root = Path(args.quixbugs_root).resolve()
    fixed_sources = _load_fixed_sources(root)

    if args.strategy == "stratified":
        split = stratified_split(fixed_sources, train_size=args.train_size, seed=args.seed)
    else:
        split = alphabetical_split(fixed_sources.keys(), train_size=args.train_size)

    overlap = sorted(set(split.train_programs).intersection(split.test_programs))
    if overlap:
        raise RuntimeError(f"Invalid split artifact; overlap detected: {overlap}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quixbugs_root": str(root),
        "strategy": args.strategy,
        "seed": args.seed,
        "train_size": len(split.train_programs),
        "test_size": len(split.test_programs),
        "train_programs": split.train_programs,
        "test_programs": split.test_programs,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output), "train_size": len(split.train_programs), "test_size": len(split.test_programs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
