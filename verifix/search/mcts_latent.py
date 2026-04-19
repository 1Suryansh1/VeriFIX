from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from verifix.core.action_space import operator_to_action_id
from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, Edit, ValidationResult
from verifix.edit_dsl.applicator import apply_edit_sequence, validate_syntax
from verifix.edit_dsl.operators import get_candidate_edits
from verifix.models.latent_jepa import JEPATransitionPredictor, MultiTaskRepairGAT
from verifix.models.pyg_converter import ASTtoPyG
from verifix.parser.ast_builder import ParseError, build_ast
from verifix.search.mcts import MCTSSearchResult, ValidatorProtocol


@dataclass(frozen=False)
class LatentSearchDiagnostics:
    suspicious_lines: list[int]
    initial_critic_score: float
    candidate_count: int
    candidates_scored: int
    candidates_validated: int
    screened_out_by_critic: int
    best_candidate_critic: float
    rollout_mode: str
    depth_floor: int
    depth_reached: int
    candidate_node_weight: float
    candidate_action_weight: float
    root_scored_candidates: int
    root_candidates_kept_after_global_beam: int
    root_candidates_dropped_by_global_beam: int
    candidate_eval_limit: int
    candidate_eval_truncated_by_max_patch: int
    per_state_candidates_total: int
    per_state_candidates_selected: int
    per_state_candidates_dropped_by_branch: int
    frontier_candidates_total: int
    frontier_candidates_after_dedupe: int
    frontier_candidates_kept_after_global_beam: int
    frontier_candidates_dropped_by_global_beam: int
    frontier_candidates_dropped_by_dedupe: int
    gate_reason_counts: dict[str, int]
    validation_reason_counts: dict[str, int]
    candidate_trace_top: list[dict[str, object]]
    trace_version: str


@dataclass(frozen=True)
class _CandidatePath:
    edits: tuple[Edit, ...]
    source: str
    score: float
    depth: int
    latent_z: Tensor | None = None


@dataclass(frozen=False)
class _ExpansionDiagnostics:
    root_candidates_total: int = 0
    root_candidates_considered: int = 0
    root_candidates_kept: int = 0
    root_candidates_dropped_by_global_beam: int = 0
    per_state_candidates_total: int = 0
    per_state_candidates_selected: int = 0
    per_state_candidates_dropped_by_branch: int = 0
    frontier_candidates_total: int = 0
    frontier_candidates_after_dedupe: int = 0
    frontier_candidates_kept_after_global_beam: int = 0
    frontier_candidates_dropped_by_global_beam: int = 0
    frontier_candidates_dropped_by_dedupe: int = 0


def latent_guided_search(
    bug_report: BugReport,
    validator: ValidatorProtocol,
    config: VerifixConfig,
    model: MultiTaskRepairGAT,
    predictor: JEPATransitionPredictor,
    converter: ASTtoPyG,
    rollout_mode: str = "hybrid",
    critic_threshold: float = 0.45,
) -> tuple[MCTSSearchResult, LatentSearchDiagnostics]:
    start = time.monotonic()
    plausible_patches: list[tuple[list[Edit], str, ValidationResult]] = []
    validations_used = 0
    iterations = 0
    screened_out = 0
    best_candidate_critic = 0.0
    depth_floor = max(1, int(getattr(config, "v3_min_rollout_depth", 3)))
    node_weight, action_weight = _resolve_candidate_weights(config)
    expansion_stats = _ExpansionDiagnostics()
    gate_reason_counts: dict[str, int] = {}
    validation_reason_counts: dict[str, int] = {
        "validated": 0,
        "plausible": 0,
        "non_plausible": 0,
        "skipped": 0,
    }
    candidate_trace_top: list[dict[str, object]] = []
    trace_limit = 25

    try:
        annotated = build_ast(
            source=bug_report.buggy_source,
            file_path=bug_report.file_path,
            language=bug_report.language,
        )
    except ParseError as exc:
        diagnostics = LatentSearchDiagnostics(
            suspicious_lines=[],
            initial_critic_score=0.0,
            candidate_count=0,
            candidates_scored=0,
            candidates_validated=0,
            screened_out_by_critic=0,
            best_candidate_critic=0.0,
            rollout_mode=rollout_mode,
            depth_floor=depth_floor,
            depth_reached=0,
            candidate_node_weight=node_weight,
            candidate_action_weight=action_weight,
            root_scored_candidates=0,
            root_candidates_kept_after_global_beam=0,
            root_candidates_dropped_by_global_beam=0,
            candidate_eval_limit=0,
            candidate_eval_truncated_by_max_patch=0,
            per_state_candidates_total=0,
            per_state_candidates_selected=0,
            per_state_candidates_dropped_by_branch=0,
            frontier_candidates_total=0,
            frontier_candidates_after_dedupe=0,
            frontier_candidates_kept_after_global_beam=0,
            frontier_candidates_dropped_by_global_beam=0,
            frontier_candidates_dropped_by_dedupe=0,
            gate_reason_counts={},
            validation_reason_counts=dict(validation_reason_counts),
            candidate_trace_top=[],
            trace_version="v1",
        )
        return _empty_result(time.monotonic() - start, terminated_by=f"parse_error:{exc}"), diagnostics

    graph, node_id_to_idx, _labels = converter.convert(annotated)
    model_device = _infer_model_device(model)
    graph = graph.to(model_device)
    predictor.eval()

    model.eval()
    with torch.no_grad():
        outputs = model(graph)

    node_fault_probs = outputs["fault_probs"].view(-1)
    initial_critic_score = float(outputs["critic_scores"].mean().item())
    suspicious_lines = _top_lines_by_fault_prob(
        annotated=annotated,
        node_id_to_idx=node_id_to_idx,
        node_fault_probs=node_fault_probs,
        top_n=config.fl_top_n_lines,
    )
    if not suspicious_lines:
        suspicious_lines = list(range(1, bug_report.buggy_source.count("\n") + 2))

    candidates = get_candidate_edits(
        annotated,
        suspicious_lines=suspicious_lines,
        max_edits_per_node=config.max_candidates_per_node,
    )

    scored_candidates = _score_candidates(
        candidates=candidates,
        node_id_to_idx=node_id_to_idx,
        node_fault_probs=node_fault_probs,
        node_policy_logits=outputs["policy_logits"],
        node_weight=node_weight,
        action_weight=action_weight,
    )

    expanded_paths, expansion_stats = _expand_candidate_paths(
        root_source=bug_report.buggy_source,
        root_scored_candidates=scored_candidates,
        bug_report=bug_report,
        model=model,
        predictor=predictor,
        converter=converter,
        config=config,
        depth_floor=depth_floor,
        suspicious_lines_hint=suspicious_lines,
        rollout_mode=rollout_mode,
        root_latent_z=outputs["z_graph"],
        node_weight=node_weight,
        action_weight=action_weight,
    )

    if not expanded_paths:
        fallback_limit = min(config.max_patch_candidates, len(scored_candidates))
        expanded_paths = _paths_from_scored_candidates(
            root_source=bug_report.buggy_source,
            scored_candidates=scored_candidates,
            bug_report=bug_report,
            max_candidates=fallback_limit,
            rollout_mode=rollout_mode,
            model=model,
            predictor=predictor,
            root_latent_z=outputs["z_graph"],
        )
        if expansion_stats.root_candidates_total == 0:
            expansion_stats.root_candidates_total = len(scored_candidates)
            expansion_stats.root_candidates_considered = fallback_limit
            expansion_stats.root_candidates_dropped_by_global_beam = max(
                0,
                len(scored_candidates) - fallback_limit,
            )
        expansion_stats.root_candidates_kept = max(
            expansion_stats.root_candidates_kept,
            len(expanded_paths),
        )

    max_candidates = min(len(expanded_paths), config.max_patch_candidates)
    candidate_eval_truncated = max(0, len(expanded_paths) - max_candidates)
    terminated_by = "candidate_exhausted"

    depth_reached = max((path.depth for path in expanded_paths), default=0)

    for rank, candidate in enumerate(expanded_paths[:max_candidates], start=1):
        iterations += 1

        patched_source = candidate.source
        edit_sequence = list(candidate.edits)

        if rollout_mode.strip().lower() == "latent" and candidate.latent_z is not None:
            candidate_critic = _score_latent_with_critic(candidate.latent_z, model)
        else:
            candidate_critic = _score_candidate_with_critic(
                patched_source=patched_source,
                bug_report=bug_report,
                converter=converter,
                model=model,
            )
        best_candidate_critic = max(best_candidate_critic, candidate_critic)

        should_validate, gate_reason = _validation_gate_decision(
            rollout_mode=rollout_mode,
            critic_score=candidate_critic,
            critic_threshold=critic_threshold,
            validations_used=validations_used,
            max_validations=config.max_validations,
            candidate_rank=rank,
            candidate_depth=candidate.depth,
            depth_floor=depth_floor,
        )
        _increment_counter(gate_reason_counts, gate_reason)

        if not should_validate:
            screened_out += 1
            validation_reason_counts["skipped"] += 1
            if len(candidate_trace_top) < trace_limit:
                candidate_trace_top.append(
                    {
                        "rank": rank,
                        "depth": candidate.depth,
                        "candidate_score": float(candidate.score),
                        "critic_score": float(candidate_critic),
                        "gate_reason": gate_reason,
                        "validated": False,
                        "plausible": False,
                    }
                )
            continue

        result = validator.validate(patched_source, bug_report)
        validations_used += 1
        validation_reason_counts["validated"] += 1
        if result.is_plausible:
            plausible_patches.append((edit_sequence, patched_source, result))
            validation_reason_counts["plausible"] += 1
        else:
            validation_reason_counts["non_plausible"] += 1

        if len(candidate_trace_top) < trace_limit:
            candidate_trace_top.append(
                {
                    "rank": rank,
                    "depth": candidate.depth,
                    "candidate_score": float(candidate.score),
                    "critic_score": float(candidate_critic),
                    "gate_reason": gate_reason,
                    "validated": True,
                    "plausible": bool(result.is_plausible),
                }
            )

        if validations_used >= config.max_validations:
            terminated_by = "validation_cap"
            break

    wall_time = time.monotonic() - start
    search_result = MCTSSearchResult(
        plausible_patches=plausible_patches,
        total_iterations=iterations,
        total_validations=validations_used,
        wall_time_seconds=wall_time,
        tree_depth_reached=1,
        terminated_by=terminated_by,
    )
    diagnostics = LatentSearchDiagnostics(
        suspicious_lines=suspicious_lines,
        initial_critic_score=initial_critic_score,
        candidate_count=len(candidates),
        candidates_scored=len(expanded_paths[:max_candidates]),
        candidates_validated=validations_used,
        screened_out_by_critic=screened_out,
        best_candidate_critic=best_candidate_critic,
        rollout_mode=rollout_mode,
        depth_floor=depth_floor,
        depth_reached=depth_reached,
        candidate_node_weight=node_weight,
        candidate_action_weight=action_weight,
        root_scored_candidates=len(scored_candidates),
        root_candidates_kept_after_global_beam=expansion_stats.root_candidates_kept,
        root_candidates_dropped_by_global_beam=expansion_stats.root_candidates_dropped_by_global_beam,
        candidate_eval_limit=max_candidates,
        candidate_eval_truncated_by_max_patch=candidate_eval_truncated,
        per_state_candidates_total=expansion_stats.per_state_candidates_total,
        per_state_candidates_selected=expansion_stats.per_state_candidates_selected,
        per_state_candidates_dropped_by_branch=expansion_stats.per_state_candidates_dropped_by_branch,
        frontier_candidates_total=expansion_stats.frontier_candidates_total,
        frontier_candidates_after_dedupe=expansion_stats.frontier_candidates_after_dedupe,
        frontier_candidates_kept_after_global_beam=expansion_stats.frontier_candidates_kept_after_global_beam,
        frontier_candidates_dropped_by_global_beam=expansion_stats.frontier_candidates_dropped_by_global_beam,
        frontier_candidates_dropped_by_dedupe=expansion_stats.frontier_candidates_dropped_by_dedupe,
        gate_reason_counts=dict(sorted(gate_reason_counts.items())),
        validation_reason_counts=dict(validation_reason_counts),
        candidate_trace_top=candidate_trace_top,
        trace_version="v1",
    )
    return search_result, diagnostics


def _empty_result(wall_time: float, terminated_by: str) -> MCTSSearchResult:
    return MCTSSearchResult(
        plausible_patches=[],
        total_iterations=0,
        total_validations=0,
        wall_time_seconds=wall_time,
        tree_depth_reached=0,
        terminated_by=terminated_by,
    )


def _infer_model_device(model: MultiTaskRepairGAT) -> torch.device:
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device


def _top_lines_by_fault_prob(
    annotated,
    node_id_to_idx: dict[str, int],
    node_fault_probs: Tensor,
    top_n: int,
) -> list[int]:
    line_scores: dict[int, float] = {}
    for node_id, idx in node_id_to_idx.items():
        node = annotated.nodes[node_id]
        if node.lineno <= 0:
            continue
        score = float(node_fault_probs[idx].item())
        line_scores[node.lineno] = max(score, line_scores.get(node.lineno, 0.0))

    ranked = sorted(line_scores.items(), key=lambda item: (-item[1], item[0]))
    if top_n <= 0:
        return [line for line, _score in ranked]
    return [line for line, _score in ranked[:top_n]]


def _score_candidates(
    candidates: list[Edit],
    node_id_to_idx: dict[str, int],
    node_fault_probs: Tensor,
    node_policy_logits: Tensor,
    node_weight: float,
    action_weight: float,
) -> list[tuple[float, int, Edit]]:
    scored: list[tuple[float, int, Edit]] = []
    for edit in candidates:
        node_idx = node_id_to_idx.get(edit.node_id)
        if node_idx is None:
            continue

        try:
            action_id = operator_to_action_id(edit.operator, edit.metadata)
        except ValueError:
            continue
        policy_probs = torch.softmax(node_policy_logits[node_idx], dim=-1)
        action_prob = float(policy_probs[action_id].item())
        node_prob = float(node_fault_probs[node_idx].item())
        combined = node_weight * node_prob + action_weight * action_prob
        scored.append((combined, action_id, edit))

    scored.sort(key=lambda item: (-item[0], item[2].line_number, item[2].node_id))
    return scored


def _score_candidate_with_critic(
    patched_source: str,
    bug_report: BugReport,
    converter: ASTtoPyG,
    model: MultiTaskRepairGAT,
) -> float:
    try:
        patched_ast = build_ast(
            patched_source,
            bug_report.file_path,
            language=bug_report.language,
        )
    except ParseError:
        return 0.0

    graph, _map, _labels = converter.convert(patched_ast)
    graph = graph.to(_infer_model_device(model))
    model.eval()
    with torch.no_grad():
        critic_score = model(graph)["critic_scores"].mean().item()
    return float(critic_score)


def _paths_from_scored_candidates(
    root_source: str,
    scored_candidates: list[tuple[float, int, Edit]],
    bug_report: BugReport,
    max_candidates: int,
    rollout_mode: str,
    model: MultiTaskRepairGAT,
    predictor: JEPATransitionPredictor,
    root_latent_z: Tensor,
) -> list[_CandidatePath]:
    paths: list[_CandidatePath] = []
    latent_mode = rollout_mode.strip().lower() == "latent"

    for score, action_id, edit in scored_candidates[:max_candidates]:
        patched_source, success_flags = apply_edit_sequence(root_source, [edit])
        if not success_flags or not success_flags[0]:
            continue
        syntax_ok, _syntax_error = validate_syntax(patched_source, language=bug_report.language)
        if not syntax_ok:
            continue

        candidate_score = float(score)
        latent_z: Tensor | None = None
        if latent_mode:
            latent_z, latent_critic = _predict_latent_transition(
                predictor=predictor,
                model=model,
                current_latent=root_latent_z,
                action_id=action_id,
            )
            candidate_score = 0.5 * candidate_score + 0.5 * latent_critic

        paths.append(
            _CandidatePath(
                edits=(edit,),
                source=patched_source,
                score=candidate_score,
                depth=1,
                latent_z=latent_z,
            )
        )
    return paths


def _expand_candidate_paths(
    root_source: str,
    root_scored_candidates: list[tuple[float, int, Edit]],
    bug_report: BugReport,
    model: MultiTaskRepairGAT,
    predictor: JEPATransitionPredictor,
    converter: ASTtoPyG,
    config: VerifixConfig,
    depth_floor: int,
    suspicious_lines_hint: list[int],
    rollout_mode: str,
    root_latent_z: Tensor,
    node_weight: float,
    action_weight: float,
) -> tuple[list[_CandidatePath], _ExpansionDiagnostics]:
    per_state_branch = max(1, int(getattr(config, "v3_branch_per_state", config.beam_fallback_k)))
    global_beam = max(per_state_branch * 4, config.max_patch_candidates)
    latent_mode = rollout_mode.strip().lower() == "latent"
    stats = _ExpansionDiagnostics()

    root_limit = min(global_beam, len(root_scored_candidates))
    stats.root_candidates_total = len(root_scored_candidates)
    stats.root_candidates_considered = root_limit
    stats.root_candidates_dropped_by_global_beam = max(0, len(root_scored_candidates) - root_limit)

    frontier = _paths_from_scored_candidates(
        root_source=root_source,
        scored_candidates=root_scored_candidates,
        bug_report=bug_report,
        max_candidates=root_limit,
        rollout_mode=rollout_mode,
        model=model,
        predictor=predictor,
        root_latent_z=root_latent_z,
    )
    stats.root_candidates_kept = len(frontier)
    if not frontier:
        return [], stats

    depth_reached = 1
    while depth_reached < depth_floor:
        next_frontier: list[_CandidatePath] = []

        for path in frontier:
            try:
                annotated = build_ast(path.source, bug_report.file_path, language=bug_report.language)
            except ParseError:
                continue

            if latent_mode:
                local_candidates = get_candidate_edits(
                    annotated,
                    suspicious_lines=suspicious_lines_hint,
                    max_edits_per_node=config.max_candidates_per_node,
                )
                local_scored: list[tuple[float, int, Edit]] = []
                for edit in local_candidates:
                    try:
                        action_id = operator_to_action_id(edit.operator, edit.metadata)
                    except ValueError:
                        continue
                    local_scored.append((0.0, action_id, edit))
            else:
                graph, node_id_to_idx, _labels = converter.convert(annotated)
                graph = graph.to(_infer_model_device(model))
                model.eval()
                with torch.no_grad():
                    outputs = model(graph)

                node_fault_probs = outputs["fault_probs"].view(-1)
                local_suspicious_lines = _top_lines_by_fault_prob(
                    annotated=annotated,
                    node_id_to_idx=node_id_to_idx,
                    node_fault_probs=node_fault_probs,
                    top_n=max(config.fl_top_n_lines, len(suspicious_lines_hint)),
                )
                if not local_suspicious_lines:
                    local_suspicious_lines = suspicious_lines_hint

                local_candidates = get_candidate_edits(
                    annotated,
                    suspicious_lines=local_suspicious_lines,
                    max_edits_per_node=config.max_candidates_per_node,
                )
                local_scored = _score_candidates(
                    candidates=local_candidates,
                    node_id_to_idx=node_id_to_idx,
                    node_fault_probs=node_fault_probs,
                    node_policy_logits=outputs["policy_logits"],
                    node_weight=node_weight,
                    action_weight=action_weight,
                )

            selected_local = local_scored[:per_state_branch]
            stats.per_state_candidates_total += len(local_scored)
            stats.per_state_candidates_selected += len(selected_local)
            stats.per_state_candidates_dropped_by_branch += max(0, len(local_scored) - len(selected_local))

            for local_score, action_id, edit in selected_local:
                patched_source, success_flags = apply_edit_sequence(path.source, [edit])
                if not success_flags or not success_flags[0]:
                    continue
                syntax_ok, _syntax_error = validate_syntax(patched_source, language=bug_report.language)
                if not syntax_ok:
                    continue

                candidate_score = path.score + float(local_score)
                next_latent_z: Tensor | None = None
                if latent_mode:
                    current_latent = path.latent_z if path.latent_z is not None else root_latent_z
                    next_latent_z, predicted_critic = _predict_latent_transition(
                        predictor=predictor,
                        model=model,
                        current_latent=current_latent,
                        action_id=action_id,
                    )
                    candidate_score = path.score + predicted_critic

                next_frontier.append(
                    _CandidatePath(
                        edits=(*path.edits, edit),
                        source=patched_source,
                        score=candidate_score,
                        depth=path.depth + 1,
                        latent_z=next_latent_z,
                    )
                )

        if not next_frontier:
            break

        stats.frontier_candidates_total += len(next_frontier)

        # Deduplicate by source and keep best-scored path for each unique source.
        best_by_source: dict[str, _CandidatePath] = {}
        for path in next_frontier:
            existing = best_by_source.get(path.source)
            if existing is None or path.score > existing.score:
                best_by_source[path.source] = path

        deduped_count = len(best_by_source)
        stats.frontier_candidates_after_dedupe += deduped_count
        stats.frontier_candidates_dropped_by_dedupe += max(0, len(next_frontier) - deduped_count)
        stats.frontier_candidates_kept_after_global_beam += min(deduped_count, global_beam)
        stats.frontier_candidates_dropped_by_global_beam += max(0, deduped_count - global_beam)

        frontier = sorted(
            best_by_source.values(),
            key=lambda item: (-item.score, len(item.edits), item.source),
        )[:global_beam]
        depth_reached = max((item.depth for item in frontier), default=depth_reached)

    return frontier, stats


def _increment_counter(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _validation_gate_decision(
    rollout_mode: str,
    critic_score: float,
    critic_threshold: float,
    validations_used: int,
    max_validations: int,
    candidate_rank: int,
    candidate_depth: int,
    depth_floor: int,
) -> tuple[bool, str]:
    if validations_used >= max_validations:
        return False, "budget_exhausted"

    mode = rollout_mode.strip().lower()
    if mode == "concrete":
        return True, "concrete_always"
    if mode == "hybrid":
        if candidate_depth < depth_floor:
            if candidate_rank <= 2:
                return True, "hybrid_depth_rank_allow"
            return False, "hybrid_depth_rank_block"
        if critic_score >= critic_threshold:
            return True, "hybrid_critic_allow"
        return False, "hybrid_critic_block"
    if mode == "latent":
        if candidate_depth < depth_floor:
            if candidate_rank <= 4:
                return True, "latent_depth_rank_allow"
            return False, "latent_depth_rank_block"
        if candidate_rank <= 3:
            return True, "latent_top3_allow"
        if critic_score >= (critic_threshold * 0.8):
            return True, "latent_critic_allow"
        return False, "latent_critic_block"
    return True, "fallback_allow"


def _should_validate(
    rollout_mode: str,
    critic_score: float,
    critic_threshold: float,
    validations_used: int,
    max_validations: int,
    candidate_rank: int,
    candidate_depth: int,
    depth_floor: int,
) -> bool:
    should_validate, _reason = _validation_gate_decision(
        rollout_mode=rollout_mode,
        critic_score=critic_score,
        critic_threshold=critic_threshold,
        validations_used=validations_used,
        max_validations=max_validations,
        candidate_rank=candidate_rank,
        candidate_depth=candidate_depth,
        depth_floor=depth_floor,
    )
    return should_validate


def _score_latent_with_critic(latent_z: Tensor, model: MultiTaskRepairGAT) -> float:
    model_device = _infer_model_device(model)
    candidate = latent_z
    if candidate.dim() == 1:
        candidate = candidate.unsqueeze(0)
    candidate = candidate.to(model_device)

    model.eval()
    with torch.no_grad():
        critic_score = model.critic_head(candidate).mean().item()
    return float(critic_score)


def _predict_latent_transition(
    predictor: JEPATransitionPredictor,
    model: MultiTaskRepairGAT,
    current_latent: Tensor,
    action_id: int,
) -> tuple[Tensor, float]:
    predictor_device = _infer_predictor_device(predictor)

    current = current_latent
    if current.dim() == 1:
        current = current.unsqueeze(0)
    current = current.to(predictor_device)
    action_tensor = torch.tensor([action_id], dtype=torch.long, device=predictor_device)

    predictor.eval()
    with torch.no_grad():
        next_latent = predictor(current, action_tensor, node_context=current)

    critic = _score_latent_with_critic(next_latent, model)
    return next_latent.detach(), critic


def _infer_predictor_device(predictor: JEPATransitionPredictor) -> torch.device:
    param = next(predictor.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device


def _resolve_candidate_weights(config: VerifixConfig) -> tuple[float, float]:
    node_weight = float(getattr(config, "v3_candidate_node_weight", 0.2))
    action_weight = float(getattr(config, "v3_candidate_action_weight", 0.8))
    total = node_weight + action_weight
    if total <= 0.0:
        return 0.2, 0.8
    return node_weight / total, action_weight / total


__all__ = [
    "LatentSearchDiagnostics",
    "latent_guided_search",
]
