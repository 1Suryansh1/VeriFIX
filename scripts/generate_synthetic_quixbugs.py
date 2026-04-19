from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifix.benchmarks.quixbugs_split import (
    CANONICAL_PROGRAMS,
    DEFAULT_SPLIT_SEED,
    alphabetical_split,
    stratified_split,
)
from verifix.core.action_space import action_id_to_name, operator_to_action_id
from verifix.core.models import EditOperator
from verifix.edit_dsl.applicator import apply_edit, validate_syntax
from verifix.edit_dsl.operators import get_candidate_edits
from verifix.parser.ast_builder import ParseError, build_ast


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic QuixBugs JEPA training records",
    )
    parser.add_argument(
        "--quixbugs-root",
        default="quixbugs",
        help="Path to QuixBugs root directory",
    )
    parser.add_argument(
        "--output",
        default="data/quixbugs_jepa_train.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-synthetic-per-program",
        type=int,
        default=300,
        help="Maximum synthetic mutations to keep per train program",
    )
    parser.add_argument(
        "--target-synthetic-count",
        type=int,
        default=5000,
        help="Target number of synthetic transition records across all train programs",
    )
    parser.add_argument(
        "--num-mutations",
        type=int,
        default=3,
        help="Maximum number of consecutive mutations in curriculum chain",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.35,
        help="Hard-negative records per synthetic transition ratio",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["stratified", "alphabetical"],
        default="stratified",
        help="Train/test split strategy for 30/10 split",
    )
    parser.add_argument(
        "--split-artifact",
        default="",
        help="Optional JSON split artifact path with train_programs/test_programs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used for deterministic stratified split",
    )
    return parser.parse_args()


def _all_lines(source: str) -> list[int]:
    return list(range(1, source.count("\n") + 2))


def _edit_sort_key(edit: object) -> tuple[int, str, str, str]:
    line_number = int(getattr(edit, "line_number", 0))
    node_id = str(getattr(edit, "node_id", ""))
    operator_value = str(getattr(getattr(edit, "operator", ""), "value", getattr(edit, "operator", "")))
    replacement = str(getattr(edit, "replacement_text", "") or "")
    return line_number, node_id, operator_value, replacement


def _choose_edit(
    candidates: list,
    rng: random.Random,
) -> object | None:
    if not candidates:
        return None
    # Bias toward earlier sorted edits but still keep stochasticity.
    top_window = max(1, min(12, len(candidates)))
    idx = rng.randrange(top_window)
    return candidates[idx]


def _load_program_pairs(quixbugs_root: Path) -> dict[str, tuple[str, str]]:
    buggy_dir = quixbugs_root / "python_programs"
    fixed_dir = quixbugs_root / "correct_python_programs"

    pairs: dict[str, tuple[str, str]] = {}
    for name in CANONICAL_PROGRAMS:
        buggy_path = buggy_dir / f"{name}.py"
        fixed_path = fixed_dir / f"{name}.py"
        if not buggy_path.exists() or not fixed_path.exists():
            continue

        pairs[name] = (
            buggy_path.read_text(encoding="utf-8"),
            fixed_path.read_text(encoding="utf-8"),
        )
    return pairs


def _build_split(
    fixed_sources: dict[str, str],
    strategy: str,
    seed: int,
) -> tuple[list[str], list[str]]:
    if strategy == "stratified":
        split = stratified_split(fixed_sources, train_size=30, seed=seed)
    else:
        split = alphabetical_split(fixed_sources.keys(), train_size=30)
    return split.train_programs, split.test_programs


def _load_split_artifact(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Split artifact must be a JSON object")

    train_programs = payload.get("train_programs", [])
    test_programs = payload.get("test_programs", [])
    if not isinstance(train_programs, list) or not isinstance(test_programs, list):
        raise ValueError("Split artifact must define list fields: train_programs and test_programs")

    train = [str(name).strip() for name in train_programs if str(name).strip()]
    test = [str(name).strip() for name in test_programs if str(name).strip()]
    return train, test


def _normalize_source(source: str) -> str:
    return source.replace("\r\n", "\n").rstrip()


def _operator_tier_from_edit(edit: object) -> str:
    metadata = getattr(edit, "metadata", {}) or {}
    target = str(metadata.get("target", "")).strip().lower()
    tier = str(metadata.get("operator_tier", "")).strip().lower()
    if tier == "synthetic_only" or target.startswith("synthetic_"):
        return "synthetic_aux"
    return "core"


def _resolve_repair_transition(
    mutated_ast,
    mutated_source: str,
    previous_source: str,
) -> dict[str, object] | None:
    candidates = get_candidate_edits(
        mutated_ast,
        suspicious_lines=_all_lines(mutated_source),
        max_edits_per_node=3,
        operator_tier="core",
    )
    candidates = sorted(candidates, key=_edit_sort_key)
    previous_norm = _normalize_source(previous_source)

    for candidate in candidates:
        repaired_source, repaired_ok = apply_edit(mutated_source, candidate)
        if not repaired_ok:
            continue
        if _normalize_source(repaired_source) != previous_norm:
            continue

        try:
            action_id = operator_to_action_id(candidate.operator, candidate.metadata)
        except ValueError:
            continue

        return {
            "action_id": action_id,
            "action_operator": action_id_to_name(action_id),
            "operator_value": candidate.operator.value,
            "operator_tier": _operator_tier_from_edit(candidate),
            "target_node_id": candidate.node_id,
            "bug_node_id": candidate.node_id,
            "action_resolution": "direct_candidate",
        }

    return None


def _fallback_repair_transition(mutation_edit: object) -> dict[str, object] | None:
    operator_value = str(
        getattr(getattr(mutation_edit, "operator", ""), "value", getattr(mutation_edit, "operator", ""))
    )
    metadata = getattr(mutation_edit, "metadata", {}) or {}

    try:
        if operator_value in {
            EditOperator.INSERT_STMT_BEFORE.value,
            EditOperator.INSERT_STMT_AFTER.value,
        }:
            repair_operator = EditOperator.DELETE_STMT
            action_id = operator_to_action_id(repair_operator)
            repair_operator_value = repair_operator.value
        elif operator_value == EditOperator.DELETE_STMT.value:
            repair_operator = EditOperator.INSERT_STMT_BEFORE
            action_id = operator_to_action_id(
                repair_operator,
                {"target": "statement_insertion", "position": "before"},
            )
            repair_operator_value = repair_operator.value
        elif operator_value == EditOperator.UNWRAP_BLOCK.value:
            repair_operator = EditOperator.WRAP_CONDITION
            action_id = operator_to_action_id(repair_operator, {"target": "wrap_condition"})
            repair_operator_value = repair_operator.value
        elif operator_value == EditOperator.WRAP_CONDITION.value:
            repair_operator = EditOperator.REPLACE_EXPR
            action_id = operator_to_action_id(repair_operator, {"target": "variable"})
            repair_operator_value = repair_operator.value
        else:
            repair_operator = getattr(mutation_edit, "operator")
            action_id = operator_to_action_id(repair_operator, metadata)
            repair_operator_value = operator_value
    except (AttributeError, ValueError):
        return None

    return {
        "action_id": action_id,
        "action_operator": action_id_to_name(action_id),
        "operator_value": repair_operator_value,
        "operator_tier": "core",
        "target_node_id": str(getattr(mutation_edit, "node_id", "")),
        "bug_node_id": str(getattr(mutation_edit, "node_id", "")),
        "action_resolution": "fallback_inverse",
    }


def main() -> int:
    args = _parse_args()
    if args.target_synthetic_count < 0:
        raise ValueError("--target-synthetic-count must be >= 0")
    if args.num_mutations < 1:
        raise ValueError("--num-mutations must be >= 1")
    if args.max_synthetic_per_program < 1:
        raise ValueError("--max-synthetic-per-program must be >= 1")
    if args.hard_negative_ratio < 0.0:
        raise ValueError("--hard-negative-ratio must be >= 0.0")

    rng = random.Random(args.seed)

    quixbugs_root = Path(args.quixbugs_root).resolve()
    if not quixbugs_root.exists():
        raise FileNotFoundError(f"QuixBugs root does not exist: {quixbugs_root}")

    pairs = _load_program_pairs(quixbugs_root)
    if not pairs:
        raise RuntimeError("No canonical QuixBugs program pairs were found")

    fixed_sources = {name: fixed_source for name, (_buggy, fixed_source) in pairs.items()}
    if args.split_artifact:
        artifact_path = Path(args.split_artifact).resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Split artifact does not exist: {artifact_path}")
        train_programs, test_programs = _load_split_artifact(artifact_path)
    else:
        train_programs, test_programs = _build_split(
            fixed_sources=fixed_sources,
            strategy=args.split_strategy,
            seed=args.seed,
        )

    train_programs = [name for name in train_programs if name in pairs]
    test_programs = [name for name in test_programs if name in pairs]

    print(f"Split strategy: {args.split_strategy}")
    print(f"TRAIN programs ({len(train_programs)}): {train_programs}")
    print(f"TEST programs  ({len(test_programs)}): {test_programs}")

    if set(train_programs) & set(test_programs):
        raise RuntimeError("Train/test overlap detected in split")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_count = 0
    synthetic_count = 0
    hard_negative_count = 0
    repair_action_direct = 0
    repair_action_fallback = 0
    repair_action_unresolved = 0
    generated_programs: set[str] = set()
    synthetic_by_program = {name: 0 for name in train_programs}
    hard_negative_target = int(args.target_synthetic_count * args.hard_negative_ratio)

    with output_path.open("w", encoding="utf-8") as f:
        for program in train_programs:
            pair = pairs.get(program)
            if pair is None:
                continue

            buggy_source, fixed_source = pair
            try:
                buggy_ast = build_ast(buggy_source, f"{program}.py", language="python")
                fixed_ast = build_ast(fixed_source, f"{program}.py", language="python")
            except ParseError as exc:
                print(f"Skipping {program}: parse error: {exc}")
                continue

            real_record = {
                "program": program,
                "split": "train",
                "buggy_ast": buggy_ast.to_dict(),
                "fixed_ast": fixed_ast.to_dict(),
                "action_operator": "unknown",
                "action_id": -1,
                "operator_tier": "core",
                "target_node_id": None,
                "bug_node_id": None,
                "critic_buggy": 0.0,
                "critic_fixed": 1.0,
                "source": "real",
            }
            f.write(json.dumps(real_record) + "\n")
            real_count += 1
            generated_programs.add(program)

        stalled_rounds = 0
        while synthetic_count < args.target_synthetic_count and stalled_rounds < 15:
            before_round = synthetic_count

            for program in train_programs:
                if synthetic_count >= args.target_synthetic_count:
                    break
                if synthetic_by_program[program] >= args.max_synthetic_per_program:
                    continue

                pair = pairs.get(program)
                if pair is None:
                    continue
                _buggy_source, fixed_source = pair

                try:
                    current_ast = build_ast(fixed_source, f"{program}.py", language="python")
                except ParseError:
                    continue

                current_source = fixed_source
                steps = rng.randint(1, args.num_mutations)
                transition_written = False

                for _step in range(steps):
                    candidates = get_candidate_edits(
                        current_ast,
                        suspicious_lines=_all_lines(current_source),
                        max_edits_per_node=3,
                        operator_tier="all",
                    )
                    candidates = sorted(candidates, key=_edit_sort_key)
                    edit = _choose_edit(candidates, rng)
                    if edit is None:
                        break

                    mutated_source, success = apply_edit(current_source, edit)
                    if not success or mutated_source == current_source:
                        continue

                    syntax_ok, _syntax_error = validate_syntax(mutated_source, language="python")
                    if not syntax_ok:
                        continue

                    try:
                        mutated_ast = build_ast(mutated_source, f"{program}.py", language="python")
                    except ParseError:
                        continue

                    repair_transition = _resolve_repair_transition(
                        mutated_ast=mutated_ast,
                        mutated_source=mutated_source,
                        previous_source=current_source,
                    )
                    if repair_transition is None:
                        repair_transition = _fallback_repair_transition(edit)
                        if repair_transition is None:
                            repair_action_unresolved += 1
                            continue
                        repair_action_fallback += 1
                    else:
                        repair_action_direct += 1

                    synthetic_record = {
                        "program": program,
                        "split": "train",
                        "buggy_ast": mutated_ast.to_dict(),
                        "fixed_ast": current_ast.to_dict(),
                        "action_operator": repair_transition["action_operator"],
                        "action_id": repair_transition["action_id"],
                        "operator_tier": repair_transition["operator_tier"],
                        "operator_value": repair_transition["operator_value"],
                        "mutation_operator_value": edit.operator.value,
                        "mutation_operator_tier": _operator_tier_from_edit(edit),
                        "target_node_id": repair_transition["target_node_id"],
                        "bug_node_id": repair_transition["bug_node_id"],
                        "action_resolution": repair_transition["action_resolution"],
                        "critic_buggy": 0.0,
                        "critic_fixed": 1.0,
                        "source": "synthetic",
                    }
                    f.write(json.dumps(synthetic_record) + "\n")
                    synthetic_count += 1
                    synthetic_by_program[program] += 1
                    transition_written = True

                    current_source = mutated_source
                    current_ast = mutated_ast

                    if synthetic_count >= args.target_synthetic_count:
                        break

                if (
                    transition_written
                    and hard_negative_count < hard_negative_target
                    and synthetic_by_program[program] < args.max_synthetic_per_program
                ):
                    # Build the fixed AST from the current program's fixed_source
                    # (not the stale `fixed_ast` from the real-record generation loop).
                    try:
                        program_fixed_ast = build_ast(fixed_source, f"{program}.py", language="python")
                    except ParseError:
                        continue

                    hard_negative_record = {
                        "program": program,
                        "split": "train",
                        "buggy_ast": current_ast.to_dict(),
                        "fixed_ast": program_fixed_ast.to_dict(),
                        "action_operator": "hard_negative",
                        "action_id": -1,
                        "operator_tier": "synthetic_aux",
                        "operator_value": None,
                        "target_node_id": None,
                        "bug_node_id": None,
                        "critic_buggy": 0.0,
                        "critic_fixed": 1.0,
                        "source": "hard_negative",
                    }
                    f.write(json.dumps(hard_negative_record) + "\n")
                    hard_negative_count += 1
                    synthetic_by_program[program] += 1

            stalled_rounds = stalled_rounds + 1 if synthetic_count == before_round else 0

    leaked = generated_programs.intersection(set(test_programs))
    if leaked:
        raise RuntimeError(f"Leakage detected: generated train records for test programs: {sorted(leaked)}")

    print(
        f"Generated {real_count} real + {synthetic_count} synthetic + "
        f"{hard_negative_count} hard-negative records from {len(train_programs)} programs"
    )
    print(f"Target synthetic count: {args.target_synthetic_count}")
    print(f"Hard-negative target: {hard_negative_target}")
    print(
        "Repair action resolution: "
        f"direct={repair_action_direct}, "
        f"fallback={repair_action_fallback}, "
        f"unresolved={repair_action_unresolved}"
    )
    print(f"TEST programs (held out, never touched): {test_programs}")
    print(f"Wrote dataset: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
