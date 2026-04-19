from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.quixbugs_split import CANONICAL_PROGRAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 40-program QuixBugs bug/fix dossier")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--matrix-json", default=".quixbugs_operator_matrix_afterfix.json")
    parser.add_argument("--output-json", default=".analysis/quixbugs_bug_dossier.json")
    parser.add_argument("--output-md", default=".analysis/quixbugs_bug_dossier.md")
    return parser.parse_args()


def _classify_diff_family(buggy_source: str, fixed_source: str) -> str:
    buggy_lines = buggy_source.splitlines()
    fixed_lines = fixed_source.splitlines()
    changed: list[tuple[str, str]] = []

    max_len = max(len(buggy_lines), len(fixed_lines))
    for idx in range(max_len):
        buggy = buggy_lines[idx] if idx < len(buggy_lines) else ""
        fixed = fixed_lines[idx] if idx < len(fixed_lines) else ""
        if buggy != fixed:
            changed.append((buggy, fixed))

    if not changed:
        return "no_change"

    for buggy, fixed in changed:
        if ("any(" in buggy and "all(" in fixed) or ("all(" in buggy and "any(" in fixed):
            return "aggregate_function_replacement"

    for buggy, fixed in changed:
        if re.search(r"\[[^\]]+\]", buggy) and re.search(r"\[[^\]]+\]", fixed):
            return "subscript_or_index_rewrite"

    comparison_tokens = ["<=", ">=", "==", "!=", "<", ">"]
    for buggy, fixed in changed:
        if any(token in buggy for token in comparison_tokens) and any(
            token in fixed for token in comparison_tokens
        ):
            return "comparison_operator_rewrite"

    arithmetic_tokens = [" + ", " - ", " * ", " / ", " // ", " % "]
    for buggy, fixed in changed:
        if any(token in buggy for token in arithmetic_tokens) and any(
            token in fixed for token in arithmetic_tokens
        ):
            return "arithmetic_operator_rewrite"

    for buggy, fixed in changed:
        if ("max(" in fixed and "max(" not in buggy) or ("min(" in fixed and "min(" not in buggy):
            return "expression_wrap_or_synthesis"
        if ("not " in fixed and "not " not in buggy) or ("not " in buggy and "not " not in fixed):
            return "condition_negation_or_wrap"

    call_pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
    for buggy, fixed in changed:
        buggy_calls = set(call_pattern.findall(buggy))
        fixed_calls = set(call_pattern.findall(fixed))
        if buggy_calls and fixed_calls and buggy_calls != fixed_calls:
            return "function_call_rewrite"

    return "generic_expression_or_statement"


def _diff_rows(buggy_source: str, fixed_source: str) -> list[dict[str, Any]]:
    buggy_lines = buggy_source.splitlines()
    fixed_lines = fixed_source.splitlines()

    matcher = difflib.SequenceMatcher(a=buggy_lines, b=fixed_lines)
    rows: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        before = buggy_lines[i1:i2]
        after = fixed_lines[j1:j2]
        window = max(len(before), len(after))

        for offset in range(window):
            buggy_text = before[offset] if offset < len(before) else ""
            fixed_text = after[offset] if offset < len(after) else ""
            rows.append(
                {
                    "buggy_line": (i1 + offset + 1) if offset < len(before) else None,
                    "fixed_line": (j1 + offset + 1) if offset < len(after) else None,
                    "buggy_text": buggy_text,
                    "fixed_text": fixed_text,
                    "tag": tag,
                }
            )

    return rows


def _load_matrix_rows(matrix_json: Path) -> dict[str, dict[str, Any]]:
    if not matrix_json.exists():
        return {}

    payload = json.loads(matrix_json.read_text(encoding="utf-8"))
    row_map: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        program = str(row.get("program", "")).strip()
        if not program:
            continue
        row_map[program] = row
    return row_map


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# QuixBugs Bug Dossier",
        "",
        "This file is a review checklist for all canonical 40 buggy/fixed pairs.",
        "Mark each item reviewed after manual inspection of semantic bug delta.",
        "",
    ]

    for row in rows:
        lines.append(f"## {row['program']}")
        lines.append(f"- Reviewed: [ ]")
        lines.append(f"- Gold family: {row['gold_family']}")
        lines.append(f"- Expressibility status: {row['expressibility_status']}")
        lines.append(f"- Changed lines: {row['changed_line_count']}")
        lines.append("- Manual root cause:")
        lines.append("- Minimal fix narrative:")
        lines.append("- Notes:")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    root = Path(args.quixbugs_root).resolve()
    buggy_dir = root / "python_programs"
    fixed_dir = root / "correct_python_programs"

    matrix_map = _load_matrix_rows(Path(args.matrix_json).resolve())

    rows: list[dict[str, Any]] = []

    for program in CANONICAL_PROGRAMS:
        buggy_path = buggy_dir / f"{program}.py"
        fixed_path = fixed_dir / f"{program}.py"

        if not buggy_path.exists() or not fixed_path.exists():
            rows.append(
                {
                    "program": program,
                    "buggy_path": str(buggy_path),
                    "fixed_path": str(fixed_path),
                    "error": "missing_buggy_or_fixed_file",
                    "gold_family": "unknown",
                    "changed_line_count": 0,
                    "changed_lines": [],
                    "expressibility_status": "unknown",
                    "expressibility_single": False,
                    "expressibility_composition": False,
                    "manual": {
                        "reviewed": False,
                        "root_cause": "",
                        "minimal_fix_narrative": "",
                        "notes": "",
                    },
                }
            )
            continue

        buggy_source = buggy_path.read_text(encoding="utf-8")
        fixed_source = fixed_path.read_text(encoding="utf-8")

        diff_rows = _diff_rows(buggy_source, fixed_source)
        matrix_row = matrix_map.get(program, {})

        rows.append(
            {
                "program": program,
                "buggy_path": str(buggy_path),
                "fixed_path": str(fixed_path),
                "error": None,
                "gold_family": _classify_diff_family(buggy_source, fixed_source),
                "changed_line_count": len(diff_rows),
                "changed_lines": diff_rows,
                "expressibility_status": matrix_row.get("status", "unknown"),
                "expressibility_single": bool(matrix_row.get("single_edit_reachable", False)),
                "expressibility_composition": bool(matrix_row.get("composition_reachable", False)),
                "matrix_single_edit": matrix_row.get("single_edit"),
                "manual": {
                    "reviewed": False,
                    "root_cause": "",
                    "minimal_fix_narrative": "",
                    "notes": "",
                },
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quixbugs_root": str(root),
        "total_programs": len(rows),
        "review_required": len(rows),
        "rows": rows,
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_build_markdown(rows), encoding="utf-8")

    summary = {
        "output_json": str(output_json),
        "output_md": str(output_md),
        "total_programs": len(rows),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
