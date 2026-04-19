from __future__ import annotations

import json
from pathlib import Path

from verifix.analysis.golden_trace import build_golden_trace_report, classify_root_cause_bucket


def test_classify_root_cause_bucket_expressibility_gap() -> None:
    bucket = classify_root_cause_bucket(
        success=False,
        matrix_status="missing_family",
        terminated_by="candidate_exhausted",
        candidate_count=120,
        candidates_scored=40,
        candidates_validated=10,
        screened_out_by_critic=5,
        max_patch_candidates=40,
        max_validations=25,
        candidate_eval_truncated_by_max_patch=0,
        root_candidates_dropped_by_global_beam=0,
        per_state_candidates_dropped_by_branch=0,
        frontier_candidates_dropped_by_global_beam=0,
    )

    assert bucket == "EXPRESSIBILITY_GAP"


def test_classify_root_cause_bucket_validation_budget() -> None:
    bucket = classify_root_cause_bucket(
        success=False,
        matrix_status="expressible_single",
        terminated_by="validation_cap",
        candidate_count=350,
        candidates_scored=40,
        candidates_validated=25,
        screened_out_by_critic=3,
        max_patch_candidates=40,
        max_validations=25,
        candidate_eval_truncated_by_max_patch=0,
        root_candidates_dropped_by_global_beam=0,
        per_state_candidates_dropped_by_branch=0,
        frontier_candidates_dropped_by_global_beam=0,
    )

    assert bucket == "VALIDATION_BUDGET_EXHAUSTED"


def test_build_golden_trace_report_prioritizes_failed_program(tmp_path: Path) -> None:
    run_payload = {
        "config": {
            "max_patch_candidates": 40,
            "max_validations": 25,
        },
        "per_program": {
            "alpha": {
                "v3_hybrid": {
                    "success": False,
                    "diagnostics": {
                        "candidate_count": 420,
                        "candidates_scored": 40,
                        "candidates_validated": 25,
                        "screened_out_by_critic": 2,
                        "best_candidate_critic": 0.51,
                        "terminated_by": "validation_cap",
                    },
                }
            },
            "beta": {
                "v3_hybrid": {
                    "success": True,
                    "diagnostics": {
                        "candidate_count": 90,
                        "candidates_scored": 40,
                        "candidates_validated": 10,
                        "screened_out_by_critic": 1,
                        "best_candidate_critic": 0.73,
                        "terminated_by": "candidate_exhausted",
                    },
                }
            },
        },
    }

    matrix_payload = {
        "rows": [
            {
                "program": "alpha",
                "status": "expressible_single",
                "gold_family": "comparison_operator_rewrite",
            },
            {
                "program": "beta",
                "status": "expressible_single",
                "gold_family": "generic_expression_or_statement",
            },
        ]
    }

    ideal_payload = [
        {
            "program": "alpha",
            "likely_action_name": "replace_comparison_operator",
            "action_guess_confidence": "high",
        },
        {
            "program": "beta",
            "likely_action_name": "replace_variable",
            "action_guess_confidence": "medium",
        },
    ]

    run_path = tmp_path / "run.json"
    matrix_path = tmp_path / "matrix.json"
    ideal_path = tmp_path / "ideal.json"

    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    matrix_path.write_text(json.dumps(matrix_payload), encoding="utf-8")
    ideal_path.write_text(json.dumps(ideal_payload), encoding="utf-8")

    report = build_golden_trace_report(
        run_json_paths=[run_path],
        modes=["v3_hybrid"],
        matrix_json=matrix_path,
        ideal_actions_json=ideal_path,
        top_k=10,
    )

    assert report["summary"]["total_traces"] == 2
    assert report["summary"]["failure_count"] == 1

    traces = report["program_traces"]
    alpha_trace = next(item for item in traces if item["program"] == "alpha")
    assert alpha_trace["root_cause_bucket"] == "VALIDATION_BUDGET_EXHAUSTED"

    prioritized = report["prioritized_failures"]["v3_hybrid"]
    assert prioritized[0]["program"] == "alpha"
    assert prioritized[0]["dominant_bucket"] == "VALIDATION_BUDGET_EXHAUSTED"
