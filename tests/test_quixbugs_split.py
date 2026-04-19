from __future__ import annotations

from verifix.benchmarks.quixbugs_split import (
    CANONICAL_PROGRAMS,
    PROGRAM_FAMILY,
    alphabetical_split,
    canonicalize_programs,
    is_auxiliary_program,
    split_from_mode,
    stratified_split,
)


def _mock_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for idx, name in enumerate(CANONICAL_PROGRAMS):
        body_lines = 8 + (idx % 22)
        body = "\n".join("    value = value + 1" for _ in range(body_lines))
        sources[name] = f"def {name}(value):\n{body}\n    return value\n"
    return sources


def test_is_auxiliary_program_flags_helpers() -> None:
    assert is_auxiliary_program("node") is True
    assert is_auxiliary_program("depth_first_search_test") is True
    assert is_auxiliary_program("depth_first_search") is False


def test_canonicalize_programs_filters_and_orders() -> None:
    candidates = ["node", "sqrt", "bitcount", "unknown", "wrap", "foo_test"]
    canonical = canonicalize_programs(candidates)
    assert canonical == ["bitcount", "sqrt", "wrap"]


def test_alphabetical_split_sizes() -> None:
    split = alphabetical_split(CANONICAL_PROGRAMS, train_size=30)
    assert len(split.train_programs) == 30
    assert len(split.test_programs) == 10


def test_stratified_split_is_deterministic_and_disjoint() -> None:
    sources = _mock_sources()
    left = stratified_split(sources, train_size=30, seed=2026)
    right = stratified_split(sources, train_size=30, seed=2026)

    assert left.train_programs == right.train_programs
    assert left.test_programs == right.test_programs
    assert len(left.train_programs) == 30
    assert len(left.test_programs) == 10
    assert set(left.train_programs).isdisjoint(set(left.test_programs))


def test_stratified_test_contains_all_families() -> None:
    sources = _mock_sources()
    split = stratified_split(sources, train_size=30, seed=2026)
    family_set = {PROGRAM_FAMILY[name] for name in split.test_programs}
    assert {"graph_tree", "list_array", "dp_string", "math_logic"}.issubset(family_set)


def test_split_from_mode_stratified_test_returns_test_selection_only() -> None:
    sources = _mock_sources()
    mode_split = split_from_mode("stratified-test", sources, seed=2026)
    assert len(mode_split.train_programs) == 10
    assert mode_split.test_programs == []
