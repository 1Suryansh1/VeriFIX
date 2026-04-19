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
from verifix.core.config import VerifixConfig
from verifix.edit_dsl.applicator import apply_edit, validate_syntax
from verifix.edit_dsl.operators import get_candidate_edits
from verifix.parser.ast_builder import ParseError, build_ast
from verifix.validator.executor import validate_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one-hop repair potential on expressible QuixBugs programs",
    )
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--matrix-json", default=".quixbugs_operator_matrix_afterfix.json")
    parser.add_argument("--output-json", default=".analysis/onehop_expressible.json")
    parser.add_argument("--max-edits-per-node", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--test-timeout", type=float, default=8.0)
    parser.add_argument("--working-dir", default="./.work_onehop")
    parser.add_argument("--python-executable", default="c:/Users/sunil/OneDrive/Desktop/VeriFIX/.venv/Scripts/python.exe")
    return parser.parse_args()


def _all_lines(source: str) -> list[int]:
    return list(range(1, source.count("\n") + 2))


def _edit_sort_key(edit: object) -> tuple[int, str, str, str]:
    line_number = int(getattr(edit, "line_number", 0))
    node_id = str(getattr(edit, "node_id", ""))
    operator_value = str(getattr(getattr(edit, "operator", ""), "value", getattr(edit, "operator", "")))
    replacement = str(getattr(edit, "replacement_text", "") or "")
    return line_number, node_id, operator_value, replacement


def main() -> int:
    args = parse_args()

    matrix_path = Path(args.matrix_json).resolve()
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    expressible = {
        row["program"]
        for row in matrix_payload.get("rows", [])
        if str(row.get("status", "")).startswith("expressible")
    }

    loader = QuixBugsLoader(args.quixbugs_root)
    reports = loader.load_all(language="python")
    report_map = {
        str(report.metadata.get("program_name", report.bug_id)): report
        for report in reports
        if report.failing_tests
    }
    runnable_expressible = sorted(name for name in expressible if name in report_map)

    config = VerifixConfig(
        test_timeout_seconds=args.test_timeout,
        working_dir=args.working_dir,
        python_executable=args.python_executable,
        max_validations=max(200, args.max_candidates),
        mcts_iterations=10,
        mcts_max_depth=1,
        max_candidates_per_node=args.max_edits_per_node,
        fl_top_n_lines=10,
    )

    start = time.monotonic()
    rows: list[dict[str, object]] = []
    total_plausible = 0

    for program in runnable_expressible:
        report = report_map[program]
        row: dict[str, object] = {
            "program": program,
            "candidate_count": 0,
            "checked": 0,
            "plausible": False,
            "first_plausible_index": None,
            "first_edit": None,
            "error": None,
        }

        try:
            annotated = build_ast(report.buggy_source, report.file_path, language="python")
        except ParseError as exc:
            row["error"] = f"parse_error: {exc}"
            rows.append(row)
            continue

        candidates = get_candidate_edits(
            annotated,
            suspicious_lines=_all_lines(report.buggy_source),
            max_edits_per_node=args.max_edits_per_node,
        )
        candidates = sorted(candidates, key=_edit_sort_key)
        row["candidate_count"] = len(candidates)

        for idx, edit in enumerate(candidates[: args.max_candidates], start=1):
            patched, ok = apply_edit(report.buggy_source, edit)
            if not ok:
                continue

            syntax_ok, _ = validate_syntax(patched, language="python")
            if not syntax_ok:
                continue

            row["checked"] = int(row["checked"]) + 1
            result = validate_patch(
                patched_source=patched,
                bug_report=report,
                config=config,
                state_id=f"{program}_{idx}",
            )
            if result.is_plausible:
                row["plausible"] = True
                row["first_plausible_index"] = idx
                row["first_edit"] = {
                    "operator": edit.operator.value,
                    "line": edit.line_number,
                    "original": edit.original_text,
                    "replacement": edit.replacement_text,
                }
                total_plausible += 1
                break

        rows.append(row)

    summary = {
        "matrix_json": str(matrix_path),
        "runnable_expressible_programs": runnable_expressible,
        "attempted": len(runnable_expressible),
        "plausible_found": total_plausible,
        "plausible_rate": (total_plausible / len(runnable_expressible)) if runnable_expressible else 0.0,
        "max_edits_per_node": args.max_edits_per_node,
        "max_candidates": args.max_candidates,
        "elapsed_seconds": time.monotonic() - start,
        "rows": rows,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "attempted": summary["attempted"],
        "plausible_found": summary["plausible_found"],
        "plausible_rate": summary["plausible_rate"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "output_json": str(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
