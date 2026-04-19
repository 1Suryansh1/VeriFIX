from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.analysis.golden_trace import build_golden_trace_report, write_report_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict per-program Golden Trace report with root-cause buckets "
            "and prioritized fail list from V3 benchmark artifacts."
        )
    )
    parser.add_argument(
        "--run-json",
        action="append",
        required=True,
        help="Path to one benchmark artifact JSON. Repeat this flag to aggregate multiple runs.",
    )
    parser.add_argument(
        "--modes",
        default="v3_hybrid,v3_latent",
        help="Comma-separated modes to include (v3_hybrid,v3_latent).",
    )
    parser.add_argument(
        "--matrix-json",
        default=".quixbugs_operator_matrix_after_phase2_batch4c.json",
        help="Operator expressibility matrix JSON used for Stage-G labeling.",
    )
    parser.add_argument(
        "--ideal-actions-json",
        default=".analysis/_tmp_program_40_latest_with_action_guess.json",
        help="Ideal action guess JSON used for intervention hints.",
    )
    parser.add_argument(
        "--output-json",
        default=".analysis/v3_golden_trace_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=".analysis/v3_golden_trace_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-k prioritized failures to keep per mode.",
    )
    return parser.parse_args()


def _resolve_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for token in [item.strip() for item in value.split(",") if item.strip()]:
            paths.append(Path(token).resolve())
    return paths


def main() -> int:
    args = parse_args()

    run_json_paths = _resolve_paths(args.run_json)
    if not run_json_paths:
        raise ValueError("No run JSON path provided")

    for path in run_json_paths:
        if not path.exists():
            raise FileNotFoundError(f"Run artifact not found: {path}")

    modes = [token.strip() for token in args.modes.split(",") if token.strip()]
    matrix_json = Path(args.matrix_json).resolve()
    ideal_actions_json = Path(args.ideal_actions_json).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()

    report = build_golden_trace_report(
        run_json_paths=run_json_paths,
        modes=modes,
        matrix_json=matrix_json if matrix_json.exists() else None,
        ideal_actions_json=ideal_actions_json if ideal_actions_json.exists() else None,
        top_k=max(1, args.top_k),
    )

    write_report_outputs(
        report=report,
        output_json=output_json,
        output_markdown=output_md,
    )

    summary = report.get("summary", {})
    compact = {
        "total_traces": summary.get("total_traces", 0),
        "success_count": summary.get("success_count", 0),
        "failure_count": summary.get("failure_count", 0),
        "by_mode": summary.get("by_mode", {}),
        "output_json": str(output_json),
        "output_markdown": str(output_md),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
