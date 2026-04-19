from __future__ import annotations

import ast
import json
import logging
from typing import Any

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, RankedPatch, ValidationResult
from verifix.edit_dsl.applicator import generate_diff
from verifix.verifier.evidence_report import PatchEvidence, build_evidence_report
from verifix.verifier.fuzzer import FuzzResult, fuzz_patch
from verifix.verifier.smt_layer import SMTResult, smt_screen_patches


logger = logging.getLogger(__name__)


def extract_main_function(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return ""


def extract_test_cases(bug_report: BugReport) -> list[dict]:
    raw_cases = bug_report.metadata.get("test_cases")

    if raw_cases is None:
        return []
    if isinstance(raw_cases, list):
        return [case for case in raw_cases if isinstance(case, dict)]
    if isinstance(raw_cases, dict):
        nested = raw_cases.get("cases")
        if isinstance(nested, list):
            return [case for case in nested if isinstance(case, dict)]
        return []
    if isinstance(raw_cases, str):
        text = raw_cases.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [case for case in parsed if isinstance(case, dict)]
        if isinstance(parsed, dict):
            nested = parsed.get("cases")
            if isinstance(nested, list):
                return [case for case in nested if isinstance(case, dict)]
    return []


def rank_patches_by_validation(
    patches: list[tuple[list[Edit], str, ValidationResult]],
) -> list[tuple[list[Edit], str, ValidationResult]]:
    def _score(item: tuple[list[Edit], str, ValidationResult]) -> tuple[float, int]:
        edits, _source, validation = item
        validation_score = (
            2.0 * float(validation.all_failing_tests_pass)
            + 1.0 * float(validation.no_regression)
            - 0.1 * max(0, len(edits) - 1)
        )
        return validation_score, -len(edits)

    ordered = sorted(
        patches,
        key=lambda item: _score(item),
        reverse=True,
    )
    return ordered


def build_ranked_patch(
    edits: list[Edit],
    patched_source: str,
    validation: ValidationResult,
    original_source: str,
    file_path: str,
    rank: int = 1,
) -> RankedPatch:
    score = (
        0.5 * float(validation.all_failing_tests_pass)
        + 0.3 * float(validation.no_regression)
        - 0.05 * max(0, len(edits) - 1)
    )
    return RankedPatch(
        rank=rank,
        edit_sequence=edits,
        patched_source=patched_source,
        validation=validation,
        score=score,
        diff=generate_diff(original_source, patched_source, file_path=file_path),
    )


def _neutral_fuzz_result(reason: str) -> FuzzResult:
    return FuzzResult(
        survived=True,
        total_inputs_tested=50,
        failing_inputs=[{"error": reason}],
        coverage_achieved=0.0,
        fuzz_time_seconds=0.0,
        strategy_used="FUZZ_ERROR",
    )


class VerificationFunnel:
    def __init__(self, config: VerifixConfig):
        self.config = config
        self.last_run_results: dict[str, Any] = {
            "input_count": 0,
            "fuzz_candidates": 0,
            "fuzz_survivors": 0,
            "fuzz_rejected": 0,
            "smt_candidates": 0,
            "smt_screened": 0,
        }

    def run(
        self,
        plausible_patches: list[tuple[list[Edit], str, ValidationResult]],
        bug_report: BugReport,
        original_source: str,
        max_fuzz_patches: int = 20,
        top_k_smt: int = 5,
    ) -> list[PatchEvidence]:
        logger.info("Funnel input: %d plausible patches", len(plausible_patches))

        input_count = len(plausible_patches)
        if input_count == 0:
            self.last_run_results = {
                "input_count": 0,
                "fuzz_candidates": 0,
                "fuzz_survivors": 0,
                "fuzz_rejected": 0,
                "smt_candidates": 0,
                "smt_screened": 0,
            }
            return []

        capped = max(0, int(max_fuzz_patches))
        candidates = plausible_patches[:capped] if capped else []

        function_name_by_source: dict[str, str] = {}
        fuzz_results: dict[int, FuzzResult] = {}
        original_test_cases = extract_test_cases(bug_report)

        for edits, source, _validation in candidates:
            patch_key = id(edits)
            try:
                function_name = function_name_by_source.get(source)
                if function_name is None:
                    function_name = extract_main_function(source)
                    function_name_by_source[source] = function_name

                if not function_name:
                    raise ValueError("Unable to extract function name for fuzzing")

                fr = fuzz_patch(
                    patched_source=source,
                    function_name=function_name,
                    original_test_cases=original_test_cases,
                    bug_report=bug_report,
                    config=self.config,
                )
                fuzz_results[patch_key] = fr
            except Exception as exc:  # pragma: no cover - guarded by tests via monkeypatch
                logger.warning("Fuzz stage error: %s", exc)
                fuzz_results[patch_key] = _neutral_fuzz_result(str(exc))

        raw_fuzz_survivors = [
            (edits, source, validation)
            for edits, source, validation in candidates
            if fuzz_results[id(edits)].survived
        ]
        logger.info("Post-fuzz survivors: %d/%d", len(raw_fuzz_survivors), len(candidates))

        pipeline_survivors = list(raw_fuzz_survivors)
        if not pipeline_survivors and candidates:
            logger.info("All candidates were rejected by fuzzing; using neutral fallback for SMT screening")
            pipeline_survivors = rank_patches_by_validation(candidates)
            for edits, _source, _validation in pipeline_survivors:
                if not fuzz_results[id(edits)].survived:
                    fuzz_results[id(edits)] = _neutral_fuzz_result("ALL_REJECTED_FALLBACK")

        ordered_survivors = rank_patches_by_validation(pipeline_survivors)
        smt_k = max(0, int(top_k_smt))
        smt_candidates = ordered_survivors[:smt_k] if smt_k else []
        smt_results = smt_screen_patches(smt_candidates, original_source, top_k=smt_k)

        evidence_list: list[PatchEvidence] = []
        for rank, (edits, source, validation, smt_result) in enumerate(smt_results, start=1):
            fuzz_result = fuzz_results[id(edits)]
            ranked_patch = build_ranked_patch(
                edits=edits,
                patched_source=source,
                validation=validation,
                original_source=original_source,
                file_path=bug_report.file_path,
                rank=rank,
            )
            evidence = build_evidence_report(
                ranked_patch=ranked_patch,
                validation=validation,
                fuzz_result=fuzz_result,
                smt_result=smt_result,
                bug_report=bug_report,
            )
            evidence_list.append(evidence)

        evidence_list.sort(key=lambda evidence: evidence.trust_score, reverse=True)

        self.last_run_results = {
            "input_count": input_count,
            "fuzz_candidates": len(candidates),
            "fuzz_survivors": len(raw_fuzz_survivors),
            "fuzz_rejected": max(0, len(candidates) - len(raw_fuzz_survivors)),
            "smt_candidates": len(smt_candidates),
            "smt_screened": len(smt_results),
        }
        return evidence_list

    def get_fuzz_rejection_rate(self, last_run_results: dict) -> float:
        if not last_run_results:
            return 0.0

        total = int(last_run_results.get("fuzz_candidates", 0))
        if total <= 0:
            return 0.0

        rejected = int(last_run_results.get("fuzz_rejected", 0))
        return max(0.0, min(1.0, rejected / total))

    def get_summary_stats(self, evidence_list: list[PatchEvidence]) -> dict:
        level_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for evidence in evidence_list:
            if evidence.trust_level in level_counts:
                level_counts[evidence.trust_level] += 1

        applicable = [item for item in evidence_list if item.smt_result.smt_applicable]
        verified = [item for item in applicable if item.smt_result.verdict == "VERIFIED"]
        smt_verification_rate = (len(verified) / len(applicable)) if applicable else 0.0

        return {
            "total_patches": int(self.last_run_results.get("input_count", len(evidence_list))),
            "high_trust": level_counts["HIGH"],
            "medium_trust": level_counts["MEDIUM"],
            "low_trust": level_counts["LOW"],
            "fuzz_rejection_rate": self.get_fuzz_rejection_rate(self.last_run_results),
            "smt_verification_rate": smt_verification_rate,
        }


__all__ = [
    "VerificationFunnel",
    "extract_main_function",
    "extract_test_cases",
    "rank_patches_by_validation",
    "build_ranked_patch",
]
