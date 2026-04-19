"""Clean corrupted QuixBugs correct programs.

Removes trailing triple-quoted string blocks that contain duplicate
function definitions (alternative correct solutions from upstream QuixBugs).
Keeps only the first valid function definition in each file.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CORRUPTED_PROGRAMS = [
    "breadth_first_search",
    "detect_cycle",
    "find_first_in_sorted",
    "get_factors",
    "is_valid_parenthesization",
    "lis",
    "max_sublist_sum",
    "mergesort",
    "next_permutation",
    "possible_change",
    "powerset",
    "quicksort",
    "reverse_linked_list",
    "rpn_eval",
    "sieve",
    "sqrt",
    "to_base",
]

def extract_first_definition(source: str, program_name: str) -> str:
    """Keep only code before the first triple-quoted string block."""
    # Find the first occurrence of a standalone triple-quote that starts an
    # alternative-solutions block.  These always appear as "\n"""\n" at module
    # level after the first function definition body.
    pattern = re.compile(r'^\s*"""', re.MULTILINE)
    matches = list(pattern.finditer(source))

    if not matches:
        return source  # Nothing to clean

    # The first triple-quote might be a module-level docstring at the very top.
    # If it opens and closes before any function def, skip it.
    first_match = matches[0]
    cut_pos = first_match.start()

    # Verify the truncated source still parses and contains the target function
    truncated = source[:cut_pos].rstrip() + "\n"
    try:
        tree = ast.parse(truncated)
    except SyntaxError:
        print(f"  WARNING: truncation at pos {cut_pos} caused SyntaxError, trying alternative")
        # Try cutting at each subsequent triple-quote
        for match in matches[1:]:
            truncated = source[:match.start()].rstrip() + "\n"
            try:
                tree = ast.parse(truncated)
                break
            except SyntaxError:
                continue
        else:
            print(f"  ERROR: could not find valid truncation point for {program_name}")
            return source

    # Verify the function exists in the truncated source
    has_func = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == program_name
        for n in ast.walk(tree)
    )
    if not has_func:
        print(f"  WARNING: function '{program_name}' not found after truncation")
        return source

    return truncated


def main() -> int:
    correct_dir = REPO_ROOT / "quixbugs" / "correct_python_programs"
    if not correct_dir.exists():
        print(f"ERROR: {correct_dir} does not exist")
        return 1

    cleaned_count = 0
    for name in CORRUPTED_PROGRAMS:
        path = correct_dir / f"{name}.py"
        if not path.exists():
            print(f"SKIP: {path} does not exist")
            continue

        original = path.read_text(encoding="utf-8")
        cleaned = extract_first_definition(original, name)

        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8")
            orig_lines = len(original.splitlines())
            new_lines = len(cleaned.splitlines())
            print(f"CLEANED: {name}.py  {orig_lines} -> {new_lines} lines")
            cleaned_count += 1
        else:
            print(f"OK: {name}.py (no change needed)")

    print(f"\nTotal cleaned: {cleaned_count}/{len(CORRUPTED_PROGRAMS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
