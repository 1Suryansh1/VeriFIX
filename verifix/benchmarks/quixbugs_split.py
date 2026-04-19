from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

DEFAULT_SPLIT_SEED = 20260404

# Canonical APR set (40 algorithms). Helper/test modules are excluded.
CANONICAL_PROGRAMS: list[str] = [
    "bitcount",
    "breadth_first_search",
    "bucketsort",
    "depth_first_search",
    "detect_cycle",
    "find_first_in_sorted",
    "find_in_sorted",
    "flatten",
    "gcd",
    "get_factors",
    "hanoi",
    "is_valid_parenthesization",
    "kheapsort",
    "knapsack",
    "kth",
    "lcs_length",
    "levenshtein",
    "lis",
    "longest_common_subsequence",
    "max_sublist_sum",
    "mergesort",
    "minimum_spanning_tree",
    "next_palindrome",
    "next_permutation",
    "pascal",
    "possible_change",
    "powerset",
    "quicksort",
    "reverse_linked_list",
    "rpn_eval",
    "shortest_path_length",
    "shortest_path_lengths",
    "shortest_paths",
    "shunting_yard",
    "sieve",
    "sqrt",
    "subsequences",
    "to_base",
    "topological_ordering",
    "wrap",
]

PROGRAM_FAMILY: dict[str, str] = {
    "bitcount": "math_logic",
    "breadth_first_search": "graph_tree",
    "bucketsort": "list_array",
    "depth_first_search": "graph_tree",
    "detect_cycle": "graph_tree",
    "find_first_in_sorted": "list_array",
    "find_in_sorted": "list_array",
    "flatten": "list_array",
    "gcd": "math_logic",
    "get_factors": "math_logic",
    "hanoi": "math_logic",
    "is_valid_parenthesization": "dp_string",
    "kheapsort": "list_array",
    "knapsack": "dp_string",
    "kth": "list_array",
    "lcs_length": "dp_string",
    "levenshtein": "dp_string",
    "lis": "dp_string",
    "longest_common_subsequence": "dp_string",
    "max_sublist_sum": "dp_string",
    "mergesort": "list_array",
    "minimum_spanning_tree": "graph_tree",
    "next_palindrome": "math_logic",
    "next_permutation": "list_array",
    "pascal": "math_logic",
    "possible_change": "dp_string",
    "powerset": "math_logic",
    "quicksort": "list_array",
    "reverse_linked_list": "list_array",
    "rpn_eval": "math_logic",
    "shortest_path_length": "graph_tree",
    "shortest_path_lengths": "graph_tree",
    "shortest_paths": "graph_tree",
    "shunting_yard": "math_logic",
    "sieve": "math_logic",
    "sqrt": "math_logic",
    "subsequences": "dp_string",
    "to_base": "math_logic",
    "topological_ordering": "graph_tree",
    "wrap": "dp_string",
}


@dataclass(frozen=True)
class QuixBugsSplit:
    train_programs: list[str]
    test_programs: list[str]
    seed: int


def is_auxiliary_program(program_name: str) -> bool:
    return program_name == "node" or program_name.endswith("_test")


def canonicalize_programs(program_names: Iterable[str]) -> list[str]:
    available = set(program_names)
    return [name for name in CANONICAL_PROGRAMS if name in available]


def alphabetical_split(
    program_names: Iterable[str],
    train_size: int = 30,
) -> QuixBugsSplit:
    canonical = canonicalize_programs(program_names)
    if train_size < 0 or train_size > len(canonical):
        raise ValueError(f"train_size must be within [0, {len(canonical)}]")

    return QuixBugsSplit(
        train_programs=canonical[:train_size],
        test_programs=canonical[train_size:],
        seed=0,
    )


def stratified_split(
    program_sources: Mapping[str, str],
    train_size: int = 30,
    seed: int = DEFAULT_SPLIT_SEED,
) -> QuixBugsSplit:
    canonical = canonicalize_programs(program_sources.keys())
    total = len(canonical)
    if total == 0:
        return QuixBugsSplit(train_programs=[], test_programs=[], seed=seed)

    if train_size < 0 or train_size > total:
        raise ValueError(f"train_size must be within [0, {total}]")

    test_size = total - train_size
    if test_size == 0:
        return QuixBugsSplit(train_programs=canonical, test_programs=[], seed=seed)

    by_family: dict[str, list[str]] = defaultdict(list)
    for name in canonical:
        family = PROGRAM_FAMILY.get(name, "math_logic")
        by_family[family].append(name)

    family_counts = {family: len(names) for family, names in by_family.items()}
    family_test_quota = _allocate_quotas(
        counts=family_counts,
        total_target=test_size,
        min_per_nonempty=1,
    )

    selected_test: list[str] = []
    for family in sorted(by_family.keys()):
        names = by_family[family]
        quota = family_test_quota.get(family, 0)
        if quota <= 0:
            continue

        by_difficulty: dict[str, list[str]] = defaultdict(list)
        for name in names:
            by_difficulty[_difficulty_bucket(program_sources.get(name, ""))].append(name)

        difficulty_counts = {
            bucket: len(bucket_names) for bucket, bucket_names in by_difficulty.items()
        }
        difficulty_quota = _allocate_quotas(
            counts=difficulty_counts,
            total_target=quota,
            min_per_nonempty=0,
        )

        family_selected: list[str] = []
        for bucket in sorted(by_difficulty.keys()):
            bucket_names = by_difficulty[bucket]
            pick_count = difficulty_quota.get(bucket, 0)
            ranked = sorted(bucket_names, key=lambda item: _stable_rank(item, seed))
            family_selected.extend(ranked[:pick_count])

        if len(family_selected) < quota:
            remainder = [
                name
                for name in sorted(names, key=lambda item: _stable_rank(item, seed))
                if name not in family_selected
            ]
            family_selected.extend(remainder[: quota - len(family_selected)])

        selected_test.extend(family_selected[:quota])

    if len(selected_test) < test_size:
        remaining = [
            name
            for name in sorted(canonical, key=lambda item: _stable_rank(item, seed))
            if name not in selected_test
        ]
        selected_test.extend(remaining[: test_size - len(selected_test)])

    selected_test = sorted(
        selected_test[:test_size],
        key=lambda item: CANONICAL_PROGRAMS.index(item),
    )
    selected_set = set(selected_test)
    train_programs = [name for name in canonical if name not in selected_set]

    return QuixBugsSplit(
        train_programs=train_programs,
        test_programs=selected_test,
        seed=seed,
    )


def split_from_mode(
    mode: str,
    program_sources: Mapping[str, str],
    seed: int = DEFAULT_SPLIT_SEED,
) -> QuixBugsSplit:
    normalized = mode.strip().lower()
    all_programs = list(program_sources.keys())
    canonical = canonicalize_programs(all_programs)

    if normalized in {"all", "none", ""}:
        return QuixBugsSplit(train_programs=all_programs, test_programs=[], seed=seed)

    if normalized == "alphabetical-train":
        return alphabetical_split(canonical, train_size=30)
    if normalized == "alphabetical-test":
        split = alphabetical_split(canonical, train_size=30)
        return QuixBugsSplit(train_programs=split.test_programs, test_programs=[], seed=split.seed)

    if normalized == "stratified-train":
        split = stratified_split(program_sources, train_size=30, seed=seed)
        return QuixBugsSplit(train_programs=split.train_programs, test_programs=[], seed=seed)
    if normalized == "stratified-test":
        split = stratified_split(program_sources, train_size=30, seed=seed)
        return QuixBugsSplit(train_programs=split.test_programs, test_programs=[], seed=seed)

    raise ValueError(
        "Unsupported split mode. Use one of: all, alphabetical-train, alphabetical-test, "
        "stratified-train, stratified-test"
    )


def _difficulty_bucket(source: str) -> str:
    logical_lines = [
        line for line in source.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    count = len(logical_lines)
    if count <= 12:
        return "small"
    if count <= 24:
        return "medium"
    return "large"


def _stable_rank(name: str, seed: int) -> str:
    material = f"{seed}:{name}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _allocate_quotas(
    counts: Mapping[str, int],
    total_target: int,
    min_per_nonempty: int,
) -> dict[str, int]:
    groups = sorted([group for group, count in counts.items() if count > 0])
    quotas = {group: 0 for group in counts.keys()}
    if not groups or total_target <= 0:
        return quotas

    remaining_target = total_target
    if min_per_nonempty > 0 and total_target >= len(groups) * min_per_nonempty:
        for group in groups:
            base = min(min_per_nonempty, counts[group])
            quotas[group] = base
            remaining_target -= base

    capacities = {group: max(0, counts[group] - quotas[group]) for group in groups}
    total_capacity = sum(capacities.values())
    if remaining_target <= 0 or total_capacity <= 0:
        return quotas

    raw = {
        group: (remaining_target * capacities[group] / total_capacity) if total_capacity > 0 else 0.0
        for group in groups
    }

    floors = {
        group: min(capacities[group], int(raw[group]))
        for group in groups
    }

    for group in groups:
        quotas[group] += floors[group]

    assigned = sum(floors.values())
    remainder = max(0, remaining_target - assigned)

    if remainder > 0:
        ranked_remainders = sorted(
            groups,
            key=lambda group: (-(raw[group] - floors[group]), group),
        )
        for group in ranked_remainders:
            if remainder <= 0:
                break
            if quotas[group] >= counts[group]:
                continue
            quotas[group] += 1
            remainder -= 1

    return quotas


__all__ = [
    "DEFAULT_SPLIT_SEED",
    "CANONICAL_PROGRAMS",
    "PROGRAM_FAMILY",
    "QuixBugsSplit",
    "is_auxiliary_program",
    "canonicalize_programs",
    "alphabetical_split",
    "stratified_split",
    "split_from_mode",
]
