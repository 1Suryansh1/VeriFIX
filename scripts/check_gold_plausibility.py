from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.quixbugs import QuixBugsLoader
from verifix.core.config import VerifixConfig
from verifix.validator.executor import validate_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether gold fixed sources are plausible under benchmark tests")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--output-json", default=".analysis/gold_plausibility.json")
    parser.add_argument("--python-executable", default="c:/Users/sunil/OneDrive/Desktop/VeriFIX/.venv/Scripts/python.exe")
    parser.add_argument("--test-timeout", type=float, default=8.0)
    parser.add_argument("--working-dir", default="./.work_goldcheck")
    parser.add_argument("--only-programs", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.quixbugs_root).resolve()
    fixed_dir = root / "correct_python_programs"

    loader = QuixBugsLoader(str(root))
    reports = loader.load_all(language="python")
    report_map = {
        str(report.metadata.get("program_name", report.bug_id)): report
        for report in reports
        if report.failing_tests
    }

    selected = sorted(report_map.keys())
    if args.only_programs.strip():
        requested = {token.strip() for token in args.only_programs.split(",") if token.strip()}
        selected = [name for name in selected if name in requested]

    config = VerifixConfig(
        test_timeout_seconds=args.test_timeout,
        working_dir=args.working_dir,
        python_executable=args.python_executable,
        max_validations=500,
        mcts_iterations=10,
        mcts_max_depth=1,
        max_candidates_per_node=10,
        fl_top_n_lines=10,
    )

    rows: list[dict[str, object]] = []
    plausible_count = 0

    for program in selected:
        report = report_map[program]
        fixed_path = fixed_dir / f"{program}.py"
        row: dict[str, object] = {
            "program": program,
            "fixed_path": str(fixed_path),
            "plausible": False,
            "all_failing_tests_pass": False,
            "no_regression": False,
            "failed_tests_count": 0,
            "runtime_error": None,
            "compile_error": None,
            "error": None,
        }

        if not fixed_path.exists():
            row["error"] = "missing_fixed_source"
            rows.append(row)
            continue

        fixed_source = fixed_path.read_text(encoding="utf-8")
        result = validate_patch(
            patched_source=fixed_source,
            bug_report=report,
            config=config,
            state_id=f"gold_{program}",
        )

        row.update(
            {
                "plausible": bool(result.is_plausible),
                "all_failing_tests_pass": bool(result.all_failing_tests_pass),
                "no_regression": bool(result.no_regression),
                "failed_tests_count": len(result.tests_failed),
                "runtime_error": result.runtime_error,
                "compile_error": result.compile_error,
            }
        )

        if result.is_plausible:
            plausible_count += 1

        rows.append(row)

    summary = {
        "attempted": len(selected),
        "gold_plausible": plausible_count,
        "gold_plausible_rate": (plausible_count / len(selected)) if selected else 0.0,
        "rows": rows,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "attempted": summary["attempted"],
        "gold_plausible": summary["gold_plausible"],
        "gold_plausible_rate": summary["gold_plausible_rate"],
        "output_json": str(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
