from __future__ import annotations

import argparse
import ast
import csv
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
from verifix.edit_dsl.applicator import apply_edit, validate_syntax
from verifix.edit_dsl.operators import get_candidate_edits
from verifix.parser.ast_builder import ParseError, build_ast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build QuixBugs operator coverage matrix")
    parser.add_argument("--quixbugs-root", default="quixbugs")
    parser.add_argument("--output-json", default=".quixbugs_operator_matrix.json")
    parser.add_argument("--output-csv", default=".quixbugs_operator_matrix.csv")
    parser.add_argument("--max-edits-per-node", type=int, default=12)
    parser.add_argument("--max-first-level", type=int, default=80)
    parser.add_argument("--max-second-level", type=int, default=40)
    parser.add_argument("--max-programs", type=int, default=0)
    return parser.parse_args()


def normalize_source(source: str) -> str:
    lines = [line.rstrip() for line in source.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def canonicalize_executable_source(source: str) -> str:
    """
    Canonicalize source for semantic matching.

    QuixBugs files often contain trailing prose blocks encoded as top-level
    string expressions. We drop those and compare only executable statements.
    """
    tree = ast.parse(source)

    executable_nodes: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        executable_nodes.append(node)

    if not executable_nodes:
        return ""

    return "\n\n".join(ast.unparse(node).strip() for node in executable_nodes)


def safe_canonicalize(source: str) -> str | None:
    try:
        return normalize_source(canonicalize_executable_source(source))
    except SyntaxError:
        return None


def all_lines(source: str) -> list[int]:
    return list(range(1, source.count("\n") + 2))


def classify_diff_family(buggy_source: str, fixed_source: str) -> str:
    buggy_lines = buggy_source.splitlines()
    fixed_lines = fixed_source.splitlines()
    changed = []

    max_len = max(len(buggy_lines), len(fixed_lines))
    for idx in range(max_len):
        b = buggy_lines[idx] if idx < len(buggy_lines) else ""
        f = fixed_lines[idx] if idx < len(fixed_lines) else ""
        if b != f:
            changed.append((b, f))

    if not changed:
        return "no_change"

    for b, f in changed:
        if ("any(" in b and "all(" in f) or ("all(" in b and "any(" in f):
            return "aggregate_function_replacement"

    for b, f in changed:
        if re.search(r"\[[^\]]+\]", b) and re.search(r"\[[^\]]+\]", f):
            return "subscript_or_index_rewrite"

    comparison_tokens = ["<=", ">=", "==", "!=", "<", ">"]
    for b, f in changed:
        if any(tok in b for tok in comparison_tokens) and any(tok in f for tok in comparison_tokens):
            return "comparison_operator_rewrite"

    arithmetic_tokens = [" + ", " - ", " * ", " / ", " // ", " % "]
    for b, f in changed:
        if any(tok in b for tok in arithmetic_tokens) and any(tok in f for tok in arithmetic_tokens):
            return "arithmetic_operator_rewrite"

    for b, f in changed:
        if ("max(" in f and "max(" not in b) or ("min(" in f and "min(" not in b):
            return "expression_wrap_or_synthesis"
        if ("not " in f and "not " not in b) or ("not " in b and "not " not in f):
            return "condition_negation_or_wrap"

    call_pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
    for b, f in changed:
        b_calls = set(call_pattern.findall(b))
        f_calls = set(call_pattern.findall(f))
        if b_calls and f_calls and b_calls != f_calls:
            return "function_call_rewrite"

    return "generic_expression_or_statement"


def collect_candidates(source: str, file_path: str, max_edits_per_node: int) -> list:
    annotated = build_ast(source, file_path=file_path, language="python")
    return get_candidate_edits(
        annotated,
        suspicious_lines=all_lines(source),
        max_edits_per_node=max_edits_per_node,
    )


def evaluate_program(
    program: str,
    buggy_source: str,
    fixed_source: str,
    max_edits_per_node: int,
    max_first_level: int,
    max_second_level: int,
) -> dict[str, Any]:
    canonical_buggy = safe_canonicalize(buggy_source)
    canonical_fixed = safe_canonicalize(fixed_source)

    if canonical_buggy is None or canonical_fixed is None:
        return {
            "program": program,
            "status": "parse_error",
            "gold_family": classify_diff_family(buggy_source, fixed_source),
            "single_edit_reachable": False,
            "composition_reachable": False,
            "single_edit": None,
            "composition": [],
            "single_candidates": 0,
            "first_level_checked": 0,
            "second_level_checked": 0,
            "error": "unable to canonicalize buggy or fixed source",
        }

    if canonical_buggy == canonical_fixed:
        return {
            "program": program,
            "status": "already_equal",
            "gold_family": "no_change",
            "single_edit_reachable": True,
            "composition_reachable": True,
            "single_edit": None,
            "composition": [],
            "single_candidates": 0,
            "first_level_checked": 0,
            "second_level_checked": 0,
            "error": None,
        }

    try:
        first_candidates = collect_candidates(
            source=buggy_source,
            file_path=f"{program}.py",
            max_edits_per_node=max_edits_per_node,
        )
    except ParseError as exc:
        return {
            "program": program,
            "status": "parse_error",
            "gold_family": classify_diff_family(buggy_source, fixed_source),
            "single_edit_reachable": False,
            "composition_reachable": False,
            "single_edit": None,
            "composition": [],
            "single_candidates": 0,
            "first_level_checked": 0,
            "second_level_checked": 0,
            "error": str(exc),
        }

    for edit in first_candidates:
        patched, ok = apply_edit(buggy_source, edit)
        if not ok:
            continue
        patched_canonical = safe_canonicalize(patched)
        if patched_canonical is None:
            continue
        if patched_canonical == canonical_fixed:
            return {
                "program": program,
                "status": "expressible_single",
                "gold_family": classify_diff_family(buggy_source, fixed_source),
                "single_edit_reachable": True,
                "composition_reachable": True,
                "single_edit": {
                    "operator": edit.operator.value,
                    "line": edit.line_number,
                    "original": edit.original_text,
                    "replacement": edit.replacement_text,
                },
                "composition": [],
                "single_candidates": len(first_candidates),
                "first_level_checked": len(first_candidates),
                "second_level_checked": 0,
                "error": None,
            }

    checked_first = 0
    checked_second = 0

    for first_edit in first_candidates[:max_first_level]:
        patched_1, ok_1 = apply_edit(buggy_source, first_edit)
        if not ok_1:
            continue

        syntax_ok, _ = validate_syntax(patched_1, language="python")
        if not syntax_ok:
            continue

        checked_first += 1
        patched_1_canonical = safe_canonicalize(patched_1)
        if patched_1_canonical is None:
            continue
        if patched_1_canonical == canonical_fixed:
            return {
                "program": program,
                "status": "expressible_single",
                "gold_family": classify_diff_family(buggy_source, fixed_source),
                "single_edit_reachable": True,
                "composition_reachable": True,
                "single_edit": {
                    "operator": first_edit.operator.value,
                    "line": first_edit.line_number,
                    "original": first_edit.original_text,
                    "replacement": first_edit.replacement_text,
                },
                "composition": [],
                "single_candidates": len(first_candidates),
                "first_level_checked": checked_first,
                "second_level_checked": checked_second,
                "error": None,
            }

        try:
            second_candidates = collect_candidates(
                source=patched_1,
                file_path=f"{program}.py",
                max_edits_per_node=max_edits_per_node,
            )
        except ParseError:
            continue

        for second_edit in second_candidates[:max_second_level]:
            patched_2, ok_2 = apply_edit(patched_1, second_edit)
            if not ok_2:
                continue
            checked_second += 1
            patched_2_canonical = safe_canonicalize(patched_2)
            if patched_2_canonical is None:
                continue
            if patched_2_canonical == canonical_fixed:
                return {
                    "program": program,
                    "status": "expressible_composition",
                    "gold_family": classify_diff_family(buggy_source, fixed_source),
                    "single_edit_reachable": False,
                    "composition_reachable": True,
                    "single_edit": None,
                    "composition": [
                        {
                            "operator": first_edit.operator.value,
                            "line": first_edit.line_number,
                            "original": first_edit.original_text,
                            "replacement": first_edit.replacement_text,
                        },
                        {
                            "operator": second_edit.operator.value,
                            "line": second_edit.line_number,
                            "original": second_edit.original_text,
                            "replacement": second_edit.replacement_text,
                        },
                    ],
                    "single_candidates": len(first_candidates),
                    "first_level_checked": checked_first,
                    "second_level_checked": checked_second,
                    "error": None,
                }

    return {
        "program": program,
        "status": "missing_family",
        "gold_family": classify_diff_family(buggy_source, fixed_source),
        "single_edit_reachable": False,
        "composition_reachable": False,
        "single_edit": None,
        "composition": [],
        "single_candidates": len(first_candidates),
        "first_level_checked": checked_first,
        "second_level_checked": checked_second,
        "error": None,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    single = sum(1 for row in rows if row["status"] == "expressible_single")
    comp = sum(1 for row in rows if row["status"] == "expressible_composition")
    missing = sum(1 for row in rows if row["status"] == "missing_family")
    parse_errors = sum(1 for row in rows if row["status"] == "parse_error")

    family_missing_counts: dict[str, int] = {}
    for row in rows:
        if row["status"] != "missing_family":
            continue
        family = row["gold_family"]
        family_missing_counts[family] = family_missing_counts.get(family, 0) + 1

    return {
        "total_programs": total,
        "expressible_single": single,
        "expressible_composition": comp,
        "missing_family": missing,
        "parse_error": parse_errors,
        "expressibility_upper_bound": {
            "count": single + comp,
            "rate": ((single + comp) / total) if total else 0.0,
        },
        "missing_by_gold_family": dict(sorted(family_missing_counts.items())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "program",
                "status",
                "gold_family",
                "single_edit_reachable",
                "composition_reachable",
                "single_candidates",
                "first_level_checked",
                "second_level_checked",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "program": row["program"],
                    "status": row["status"],
                    "gold_family": row["gold_family"],
                    "single_edit_reachable": row["single_edit_reachable"],
                    "composition_reachable": row["composition_reachable"],
                    "single_candidates": row["single_candidates"],
                    "first_level_checked": row["first_level_checked"],
                    "second_level_checked": row["second_level_checked"],
                    "error": row["error"],
                }
            )


def main() -> int:
    args = parse_args()

    root = Path(args.quixbugs_root).resolve()
    buggy_dir = root / "python_programs"
    fixed_dir = root / "correct_python_programs"

    programs = list(CANONICAL_PROGRAMS)
    if args.max_programs > 0:
        programs = programs[: args.max_programs]

    rows: list[dict[str, Any]] = []
    for program in programs:
        buggy_path = buggy_dir / f"{program}.py"
        fixed_path = fixed_dir / f"{program}.py"

        if not buggy_path.exists() or not fixed_path.exists():
            rows.append(
                {
                    "program": program,
                    "status": "missing_program_file",
                    "gold_family": "unknown",
                    "single_edit_reachable": False,
                    "composition_reachable": False,
                    "single_edit": None,
                    "composition": [],
                    "single_candidates": 0,
                    "first_level_checked": 0,
                    "second_level_checked": 0,
                    "error": "buggy or fixed source missing",
                }
            )
            continue

        buggy_source = buggy_path.read_text(encoding="utf-8")
        fixed_source = fixed_path.read_text(encoding="utf-8")

        row = evaluate_program(
            program=program,
            buggy_source=buggy_source,
            fixed_source=fixed_source,
            max_edits_per_node=args.max_edits_per_node,
            max_first_level=args.max_first_level,
            max_second_level=args.max_second_level,
        )
        rows.append(row)
        print(f"{program:30} -> {row['status']}")

    summary = build_summary(rows)

    output_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "quixbugs_root": str(root),
            "max_edits_per_node": args.max_edits_per_node,
            "max_first_level": args.max_first_level,
            "max_second_level": args.max_second_level,
            "max_programs": args.max_programs,
        },
        "summary": summary,
        "rows": rows,
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    write_csv(Path(args.output_csv).resolve(), rows)

    print("\n=== Coverage Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"JSON: {output_json}")
    print(f"CSV: {Path(args.output_csv).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
