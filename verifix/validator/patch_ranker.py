from __future__ import annotations

import hashlib

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, RankedPatch, ValidationResult
from verifix.edit_dsl.applicator import generate_diff
from verifix.search.mcts import MCTSSearchResult
from verifix.search.scorer import score_patch_candidate


def rank_patches(
    search_result: MCTSSearchResult,
    original_source: str,
    bug_report: BugReport,
    config: VerifixConfig,
    max_patches: int = 10,
) -> list[RankedPatch]:
    del config

    if max_patches <= 0:
        return []

    plausible = [patch for patch in search_result.plausible_patches if patch[2].is_plausible]

    deduped: dict[str, tuple[list, str, ValidationResult]] = {}
    for edits, patched_source, validation in plausible:
        normalized = _normalize_source(patched_source)
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        existing = deduped.get(key)
        if existing is None or len(edits) < len(existing[0]):
            deduped[key] = (edits, patched_source, validation)

    scored: list[tuple[float, tuple[list, str, ValidationResult]]] = []
    for item in deduped.values():
        edits, patched_source, validation = item
        patch_score = score_patch_candidate(
            edits=edits,
            original_source=original_source,
            patched_source=patched_source,
            suspicious_lines=[],
        )
        score = (
            0.5 * float(validation.all_failing_tests_pass)
            + 0.3 * float(validation.no_regression)
            + 0.2 * patch_score
            - 0.05 * max(0, len(edits) - 1)
        )
        scored.append((score, item))

    scored.sort(key=lambda entry: (-entry[0], len(entry[1][0]), entry[1][1]))

    ranked: list[RankedPatch] = []
    for idx, (score, (edits, patched_source, validation)) in enumerate(scored[:max_patches], start=1):
        ranked.append(
            RankedPatch(
                rank=idx,
                edit_sequence=edits,
                patched_source=patched_source,
                validation=validation,
                score=score,
                diff=generate_diff(original_source, patched_source, file_path=bug_report.file_path),
            )
        )

    return ranked


def summarize_results(ranked_patches: list[RankedPatch], bug_id: str) -> str:
    lines: list[str] = [
        f"=== VeriFix Repair Results for {bug_id} ===",
        f"Found {len(ranked_patches)} plausible patch(es).",
        "",
    ]

    for patch in ranked_patches:
        lines.extend(
            [
                f"PATCH #{patch.rank} (score: {patch.score:.2f}, edits: {len(patch.edit_sequence)})",
                "----------------------------------",
                patch.diff.rstrip("\n"),
                f"Tests: {len(patch.validation.tests_passed)} passing, {len(patch.validation.tests_failed)} failing",
                "",
            ]
        )

    lines.append("=== END ===")
    return "\n".join(lines)


def _normalize_source(source: str) -> str:
    lines = [line.rstrip() for line in source.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


__all__ = ["rank_patches", "summarize_results"]
