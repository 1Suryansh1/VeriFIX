from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_CAUSE_BUCKETS = {
    "SUCCESS",
    "EXPRESSIBILITY_GAP",
    "VALIDATION_BUDGET_EXHAUSTED",
    "CRITIC_GATE_OVERFILTERING",
    "TRUNCATION_PRESSURE",
    "VALIDATED_NOT_PLAUSIBLE",
    "NO_VALIDATION_ATTEMPT",
    "NO_CANDIDATES_GENERATED",
    "UNCERTAIN",
}

_BUCKET_INTERVENTIONS: dict[str, tuple[str, str]] = {
    "SUCCESS": ("No intervention needed.", "none"),
    "EXPRESSIBILITY_GAP": (
        "Add missing operator family/action mapping for the gold edit and retrain with balanced supervision.",
        "high",
    ),
    "VALIDATION_BUDGET_EXHAUSTED": (
        "Use adaptive per-program validation budget and always reserve slots for top-ranked diverse candidates.",
        "high",
    ),
    "CRITIC_GATE_OVERFILTERING": (
        "Calibrate critic threshold by mode and keep a fixed rank-based safety valve before strict gating.",
        "high",
    ),
    "TRUNCATION_PRESSURE": (
        "Increase root recall selectively for high-candidate programs and improve ranking toward ideal-action edits.",
        "high",
    ),
    "VALIDATED_NOT_PLAUSIBLE": (
        "Improve suspicious-line localization and candidate quality around the ideal action family.",
        "medium",
    ),
    "NO_VALIDATION_ATTEMPT": (
        "Guarantee minimum validation probes for top ranks before gate pruning.",
        "medium",
    ),
    "NO_CANDIDATES_GENERATED": (
        "Expand candidate generation around suspicious lines and check AST/operator applicability constraints.",
        "medium",
    ),
    "UNCERTAIN": (
        "Inspect candidate_trace_top and gate_reason_counts for fine-grained failure mode.",
        "medium",
    ),
}

_BUCKET_BASE_PRIORITY: dict[str, float] = {
    "SUCCESS": 0.0,
    "EXPRESSIBILITY_GAP": 95.0,
    "VALIDATION_BUDGET_EXHAUSTED": 100.0,
    "CRITIC_GATE_OVERFILTERING": 90.0,
    "TRUNCATION_PRESSURE": 92.0,
    "VALIDATED_NOT_PLAUSIBLE": 75.0,
    "NO_VALIDATION_ATTEMPT": 78.0,
    "NO_CANDIDATES_GENERATED": 70.0,
    "UNCERTAIN": 60.0,
}


@dataclass(frozen=True)
class ProgramTrace:
    run_id: str
    mode: str
    program: str
    success: bool
    root_cause_bucket: str
    priority_score: float
    predicted_intervention: str
    predicted_roi: str
    matrix_status: str
    gold_family: str
    ideal_action_name: str
    action_guess_confidence: str
    terminated_by: str
    candidate_count: int
    candidates_scored: int
    candidates_validated: int
    screened_out_by_critic: int
    best_candidate_critic: float
    candidate_eval_truncated_by_max_patch: int
    root_candidates_dropped_by_global_beam: int
    per_state_candidates_dropped_by_branch: int
    frontier_candidates_dropped_by_global_beam: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "program": self.program,
            "success": self.success,
            "root_cause_bucket": self.root_cause_bucket,
            "priority_score": self.priority_score,
            "predicted_intervention": self.predicted_intervention,
            "predicted_roi": self.predicted_roi,
            "matrix_status": self.matrix_status,
            "gold_family": self.gold_family,
            "ideal_action_name": self.ideal_action_name,
            "action_guess_confidence": self.action_guess_confidence,
            "terminated_by": self.terminated_by,
            "candidate_count": self.candidate_count,
            "candidates_scored": self.candidates_scored,
            "candidates_validated": self.candidates_validated,
            "screened_out_by_critic": self.screened_out_by_critic,
            "best_candidate_critic": self.best_candidate_critic,
            "candidate_eval_truncated_by_max_patch": self.candidate_eval_truncated_by_max_patch,
            "root_candidates_dropped_by_global_beam": self.root_candidates_dropped_by_global_beam,
            "per_state_candidates_dropped_by_branch": self.per_state_candidates_dropped_by_branch,
            "frontier_candidates_dropped_by_global_beam": self.frontier_candidates_dropped_by_global_beam,
        }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json_file(path: Path) -> Any:
    # Some artifacts are written with UTF-8 BOM on Windows.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_matrix_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    payload = _load_json_file(path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        program = str(row.get("program", "")).strip()
        if not program:
            continue
        status = str(row.get("status", "unknown")).strip() or "unknown"
        family = str(row.get("gold_family", "unknown")).strip() or "unknown"
        index[program] = {
            "status": status,
            "gold_family": family,
        }
    return index


def load_ideal_action_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    payload = _load_json_file(path)
    if not isinstance(payload, list):
        return {}

    index: dict[str, dict[str, str]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        program = str(row.get("program", "")).strip()
        if not program:
            continue
        index[program] = {
            "ideal_action_name": str(row.get("likely_action_name", "unknown")).strip() or "unknown",
            "action_guess_confidence": str(row.get("action_guess_confidence", "unknown")).strip() or "unknown",
        }
    return index


def _compute_truncation_pressure(
    candidate_count: int,
    max_patch_candidates: int,
    candidate_eval_truncated_by_max_patch: int,
    root_candidates_dropped_by_global_beam: int,
    per_state_candidates_dropped_by_branch: int,
    frontier_candidates_dropped_by_global_beam: int,
) -> int:
    pressure = 0
    pressure += max(0, candidate_eval_truncated_by_max_patch)
    pressure += max(0, root_candidates_dropped_by_global_beam)
    pressure += max(0, per_state_candidates_dropped_by_branch)
    pressure += max(0, frontier_candidates_dropped_by_global_beam)

    if max_patch_candidates > 0 and candidate_count >= max_patch_candidates * 3:
        pressure += candidate_count - (max_patch_candidates * 2)

    return pressure


def classify_root_cause_bucket(
    *,
    success: bool,
    matrix_status: str,
    terminated_by: str,
    candidate_count: int,
    candidates_scored: int,
    candidates_validated: int,
    screened_out_by_critic: int,
    max_patch_candidates: int,
    max_validations: int,
    candidate_eval_truncated_by_max_patch: int,
    root_candidates_dropped_by_global_beam: int,
    per_state_candidates_dropped_by_branch: int,
    frontier_candidates_dropped_by_global_beam: int,
) -> str:
    if success:
        return "SUCCESS"

    matrix_state = matrix_status.strip().lower()
    if matrix_state in {"missing_family", "parse_error", "missing_program_file"}:
        return "EXPRESSIBILITY_GAP"

    if terminated_by == "validation_cap":
        return "VALIDATION_BUDGET_EXHAUSTED"

    gate_block_cutoff = max(5, int(0.25 * max(1, candidates_scored)))
    low_validation_cutoff = max(2, int(0.20 * max(1, max_validations)))
    if screened_out_by_critic >= gate_block_cutoff and candidates_validated <= low_validation_cutoff:
        return "CRITIC_GATE_OVERFILTERING"

    truncation_pressure = _compute_truncation_pressure(
        candidate_count=candidate_count,
        max_patch_candidates=max_patch_candidates,
        candidate_eval_truncated_by_max_patch=candidate_eval_truncated_by_max_patch,
        root_candidates_dropped_by_global_beam=root_candidates_dropped_by_global_beam,
        per_state_candidates_dropped_by_branch=per_state_candidates_dropped_by_branch,
        frontier_candidates_dropped_by_global_beam=frontier_candidates_dropped_by_global_beam,
    )
    if truncation_pressure > 0:
        return "TRUNCATION_PRESSURE"

    if candidates_validated > 0 and terminated_by == "candidate_exhausted":
        return "VALIDATED_NOT_PLAUSIBLE"

    if candidates_validated == 0 and candidates_scored > 0:
        return "NO_VALIDATION_ATTEMPT"

    if candidate_count == 0:
        return "NO_CANDIDATES_GENERATED"

    return "UNCERTAIN"


def _confidence_bonus(confidence: str) -> float:
    lowered = confidence.strip().lower()
    if lowered == "high":
        return 12.0
    if lowered in {"medium", "medium-high", "medium_high"}:
        return 6.0
    if lowered == "low":
        return 2.0
    return 0.0


def compute_priority_score(
    *,
    root_cause_bucket: str,
    candidate_count: int,
    terminated_by: str,
    action_guess_confidence: str,
    screened_out_by_critic: int,
    truncation_pressure: int,
) -> float:
    base = _BUCKET_BASE_PRIORITY.get(root_cause_bucket, 50.0)
    score = base
    score += min(60.0, max(0.0, candidate_count / 20.0))
    score += _confidence_bonus(action_guess_confidence)

    if terminated_by == "validation_cap":
        score += 8.0

    if root_cause_bucket == "CRITIC_GATE_OVERFILTERING":
        score += min(30.0, max(0.0, screened_out_by_critic / 2.0))

    if root_cause_bucket == "TRUNCATION_PRESSURE":
        score += min(30.0, max(0.0, truncation_pressure / 4.0))

    return round(score, 3)


def _intervention_for_bucket(bucket: str) -> tuple[str, str]:
    return _BUCKET_INTERVENTIONS.get(
        bucket,
        ("Inspect diagnostics and candidate trace to identify true bottleneck.", "medium"),
    )


def _normalized_mode_list(modes: list[str]) -> list[str]:
    allowed = {"v3_hybrid", "v3_latent"}
    normalized: list[str] = []
    for mode in modes:
        key = mode.strip().lower()
        if key in allowed and key not in normalized:
            normalized.append(key)
    if not normalized:
        normalized = ["v3_hybrid", "v3_latent"]
    return normalized


def extract_program_traces(
    *,
    run_payload: dict[str, Any],
    run_id: str,
    modes: list[str],
    matrix_index: dict[str, dict[str, str]],
    ideal_action_index: dict[str, dict[str, str]],
) -> list[ProgramTrace]:
    traces: list[ProgramTrace] = []
    per_program = run_payload.get("per_program", {})
    if not isinstance(per_program, dict):
        return traces

    config = run_payload.get("config", {})
    max_patch_candidates = _safe_int(config.get("max_patch_candidates"), default=40)
    max_validations = _safe_int(config.get("max_validations"), default=25)

    for program, program_entry in per_program.items():
        if not isinstance(program_entry, dict):
            continue

        matrix_meta = matrix_index.get(program, {"status": "unknown", "gold_family": "unknown"})
        ideal_meta = ideal_action_index.get(
            program,
            {"ideal_action_name": "unknown", "action_guess_confidence": "unknown"},
        )

        for mode in modes:
            mode_entry = program_entry.get(mode)
            if not isinstance(mode_entry, dict):
                continue

            diagnostics = mode_entry.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                diagnostics = {}

            success = bool(mode_entry.get("success", False))
            terminated_by = str(diagnostics.get("terminated_by", "unknown")).strip() or "unknown"
            candidate_count = _safe_int(diagnostics.get("candidate_count"))
            candidates_scored = _safe_int(diagnostics.get("candidates_scored"))
            candidates_validated = _safe_int(diagnostics.get("candidates_validated"))
            screened_out_by_critic = _safe_int(diagnostics.get("screened_out_by_critic"))
            best_candidate_critic = _safe_float(diagnostics.get("best_candidate_critic"))
            candidate_eval_truncated_by_max_patch = _safe_int(
                diagnostics.get("candidate_eval_truncated_by_max_patch")
            )
            root_candidates_dropped_by_global_beam = _safe_int(
                diagnostics.get("root_candidates_dropped_by_global_beam")
            )
            per_state_candidates_dropped_by_branch = _safe_int(
                diagnostics.get("per_state_candidates_dropped_by_branch")
            )
            frontier_candidates_dropped_by_global_beam = _safe_int(
                diagnostics.get("frontier_candidates_dropped_by_global_beam")
            )

            root_cause_bucket = classify_root_cause_bucket(
                success=success,
                matrix_status=matrix_meta["status"],
                terminated_by=terminated_by,
                candidate_count=candidate_count,
                candidates_scored=candidates_scored,
                candidates_validated=candidates_validated,
                screened_out_by_critic=screened_out_by_critic,
                max_patch_candidates=max_patch_candidates,
                max_validations=max_validations,
                candidate_eval_truncated_by_max_patch=candidate_eval_truncated_by_max_patch,
                root_candidates_dropped_by_global_beam=root_candidates_dropped_by_global_beam,
                per_state_candidates_dropped_by_branch=per_state_candidates_dropped_by_branch,
                frontier_candidates_dropped_by_global_beam=frontier_candidates_dropped_by_global_beam,
            )

            truncation_pressure = _compute_truncation_pressure(
                candidate_count=candidate_count,
                max_patch_candidates=max_patch_candidates,
                candidate_eval_truncated_by_max_patch=candidate_eval_truncated_by_max_patch,
                root_candidates_dropped_by_global_beam=root_candidates_dropped_by_global_beam,
                per_state_candidates_dropped_by_branch=per_state_candidates_dropped_by_branch,
                frontier_candidates_dropped_by_global_beam=frontier_candidates_dropped_by_global_beam,
            )

            priority_score = compute_priority_score(
                root_cause_bucket=root_cause_bucket,
                candidate_count=candidate_count,
                terminated_by=terminated_by,
                action_guess_confidence=ideal_meta["action_guess_confidence"],
                screened_out_by_critic=screened_out_by_critic,
                truncation_pressure=truncation_pressure,
            )

            intervention, roi = _intervention_for_bucket(root_cause_bucket)
            traces.append(
                ProgramTrace(
                    run_id=run_id,
                    mode=mode,
                    program=str(program),
                    success=success,
                    root_cause_bucket=root_cause_bucket,
                    priority_score=priority_score,
                    predicted_intervention=intervention,
                    predicted_roi=roi,
                    matrix_status=matrix_meta["status"],
                    gold_family=matrix_meta["gold_family"],
                    ideal_action_name=ideal_meta["ideal_action_name"],
                    action_guess_confidence=ideal_meta["action_guess_confidence"],
                    terminated_by=terminated_by,
                    candidate_count=candidate_count,
                    candidates_scored=candidates_scored,
                    candidates_validated=candidates_validated,
                    screened_out_by_critic=screened_out_by_critic,
                    best_candidate_critic=best_candidate_critic,
                    candidate_eval_truncated_by_max_patch=candidate_eval_truncated_by_max_patch,
                    root_candidates_dropped_by_global_beam=root_candidates_dropped_by_global_beam,
                    per_state_candidates_dropped_by_branch=per_state_candidates_dropped_by_branch,
                    frontier_candidates_dropped_by_global_beam=frontier_candidates_dropped_by_global_beam,
                )
            )

    return traces


def aggregate_prioritized_failures(
    traces: list[ProgramTrace],
    *,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    mode_key = mode.strip().lower()
    failures = [item for item in traces if item.mode == mode_key and not item.success]
    grouped: dict[str, list[ProgramTrace]] = {}
    for item in failures:
        grouped.setdefault(item.program, []).append(item)

    rows: list[dict[str, Any]] = []
    for program, items in grouped.items():
        failure_count = len(items)
        run_count = len({item.run_id for item in items})

        bucket_counts: dict[str, int] = {}
        for item in items:
            bucket_counts[item.root_cause_bucket] = bucket_counts.get(item.root_cause_bucket, 0) + 1

        dominant_bucket = sorted(
            bucket_counts.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )[0][0]

        avg_priority = sum(item.priority_score for item in items) / failure_count if failure_count else 0.0
        avg_candidates = sum(item.candidate_count for item in items) / failure_count if failure_count else 0.0
        avg_validated = sum(item.candidates_validated for item in items) / failure_count if failure_count else 0.0

        top_item = sorted(items, key=lambda item: item.priority_score, reverse=True)[0]
        intervention, roi = _intervention_for_bucket(dominant_bucket)

        rows.append(
            {
                "program": program,
                "mode": mode_key,
                "failure_count": failure_count,
                "run_count": run_count,
                "dominant_bucket": dominant_bucket,
                "priority_score": round(avg_priority, 3),
                "avg_candidate_count": round(avg_candidates, 3),
                "avg_candidates_validated": round(avg_validated, 3),
                "terminated_by": top_item.terminated_by,
                "ideal_action_name": top_item.ideal_action_name,
                "action_guess_confidence": top_item.action_guess_confidence,
                "gold_family": top_item.gold_family,
                "predicted_intervention": intervention,
                "predicted_roi": roi,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["failure_count"],
            -row["priority_score"],
            -row["avg_candidate_count"],
            row["program"],
        )
    )
    return rows[: max(1, top_k)]


def strict_template_markdown() -> str:
    return "\n".join(
        [
            "## Strict Per-Program Golden Trace Template",
            "",
            "1. Program Metadata:",
            "Program name, run id, mode, matrix status, gold family, ideal action, confidence.",
            "2. Stage G Expressibility:",
            "Matrix status and whether gold patch is within current operator/action family.",
            "3. Stage R Ranking and Truncation:",
            "candidate_count, candidates_scored, root/global-beam truncation, per-state branch truncation.",
            "4. Stage C Gate:",
            "screened_out_by_critic, gate_reason_counts, critic score profile.",
            "5. Stage V Validation:",
            "candidates_validated, terminated_by, validation_reason_counts.",
            "6. Root Cause Bucket:",
            "Single deterministic bucket chosen by precedence rules.",
            "7. Predicted Intervention:",
            "One concrete highest-ROI action tied to the bucket.",
            "",
            "Bucket precedence for failed cases:",
            "EXPRESSIBILITY_GAP > VALIDATION_BUDGET_EXHAUSTED > CRITIC_GATE_OVERFILTERING > TRUNCATION_PRESSURE > VALIDATED_NOT_PLAUSIBLE > NO_VALIDATION_ATTEMPT > NO_CANDIDATES_GENERATED > UNCERTAIN.",
        ]
    )


def render_markdown_report(
    *,
    traces: list[ProgramTrace],
    prioritized_by_mode: dict[str, list[dict[str, Any]]],
    run_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# V3 Golden Trace Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for run_path in run_paths:
        lines.append(f"- {run_path}")

    lines.append("")
    lines.append(strict_template_markdown())

    for mode, rows in prioritized_by_mode.items():
        lines.append("")
        lines.append(f"## Prioritized Fail List {mode}")
        lines.append("")
        lines.append("| Program | Bucket | Priority | Avg Candidates | Avg Validated | Ideal Action | ROI |")
        lines.append("|---|---|---:|---:|---:|---|---|")
        for row in rows:
            lines.append(
                "| "
                f"{row['program']} | {row['dominant_bucket']} | {row['priority_score']:.3f} | "
                f"{row['avg_candidate_count']:.3f} | {row['avg_candidates_validated']:.3f} | "
                f"{row['ideal_action_name']} ({row['action_guess_confidence']}) | {row['predicted_roi']} |"
            )

    failure_traces = [trace for trace in traces if not trace.success]
    failure_traces.sort(key=lambda trace: (-trace.priority_score, trace.program, trace.mode))

    lines.append("")
    lines.append("## Per-Program Golden Traces")
    lines.append("")
    for trace in failure_traces:
        lines.append(f"### {trace.program} ({trace.mode})")
        lines.append("")
        lines.append(f"- run_id: {trace.run_id}")
        lines.append(f"- root_cause_bucket: {trace.root_cause_bucket}")
        lines.append(f"- terminated_by: {trace.terminated_by}")
        lines.append(f"- matrix_status: {trace.matrix_status}")
        lines.append(f"- gold_family: {trace.gold_family}")
        lines.append(f"- ideal_action: {trace.ideal_action_name} ({trace.action_guess_confidence})")
        lines.append(f"- candidate_count: {trace.candidate_count}")
        lines.append(f"- candidates_scored: {trace.candidates_scored}")
        lines.append(f"- candidates_validated: {trace.candidates_validated}")
        lines.append(f"- screened_out_by_critic: {trace.screened_out_by_critic}")
        lines.append(f"- trunc_by_max_patch: {trace.candidate_eval_truncated_by_max_patch}")
        lines.append(f"- trunc_by_global_beam_root: {trace.root_candidates_dropped_by_global_beam}")
        lines.append(f"- trunc_by_branch: {trace.per_state_candidates_dropped_by_branch}")
        lines.append(f"- trunc_by_global_beam_frontier: {trace.frontier_candidates_dropped_by_global_beam}")
        lines.append(f"- priority_score: {trace.priority_score:.3f}")
        lines.append(f"- predicted_intervention: {trace.predicted_intervention}")
        lines.append("")

    return "\n".join(lines)


def build_golden_trace_report(
    *,
    run_json_paths: list[Path],
    modes: list[str],
    matrix_json: Path | None,
    ideal_actions_json: Path | None,
    top_k: int,
) -> dict[str, Any]:
    normalized_modes = _normalized_mode_list(modes)
    matrix_index = load_matrix_index(matrix_json)
    ideal_action_index = load_ideal_action_index(ideal_actions_json)

    traces: list[ProgramTrace] = []
    run_paths_for_report: list[str] = []

    for run_path in run_json_paths:
        payload = _load_json_file(run_path)
        run_id = run_path.stem
        run_paths_for_report.append(str(run_path))
        traces.extend(
            extract_program_traces(
                run_payload=payload,
                run_id=run_id,
                modes=normalized_modes,
                matrix_index=matrix_index,
                ideal_action_index=ideal_action_index,
            )
        )

    prioritized_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in normalized_modes:
        prioritized_by_mode[mode] = aggregate_prioritized_failures(
            traces,
            mode=mode,
            top_k=top_k,
        )

    summary = {
        "total_traces": len(traces),
        "success_count": sum(1 for item in traces if item.success),
        "failure_count": sum(1 for item in traces if not item.success),
        "by_mode": {
            mode: {
                "total": sum(1 for item in traces if item.mode == mode),
                "success": sum(1 for item in traces if item.mode == mode and item.success),
                "failure": sum(1 for item in traces if item.mode == mode and not item.success),
            }
            for mode in normalized_modes
        },
    }

    report = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_json_paths": [str(path) for path in run_json_paths],
            "modes": normalized_modes,
            "matrix_json": str(matrix_json) if matrix_json is not None else None,
            "ideal_actions_json": str(ideal_actions_json) if ideal_actions_json is not None else None,
            "template_version": "golden-trace-v1",
        },
        "summary": summary,
        "strict_template": strict_template_markdown(),
        "program_traces": [item.to_dict() for item in traces],
        "prioritized_failures": prioritized_by_mode,
    }

    report["markdown"] = render_markdown_report(
        traces=traces,
        prioritized_by_mode=prioritized_by_mode,
        run_paths=run_paths_for_report,
    )
    return report


def write_report_outputs(
    *,
    report: dict[str, Any],
    output_json: Path,
    output_markdown: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = str(report.get("markdown", ""))
    output_markdown.write_text(markdown, encoding="utf-8")
