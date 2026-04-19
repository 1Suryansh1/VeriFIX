# VeriFIX Speaking Companion (Grounded, 2026-04-14)
---

## TOPIC 1: Core Intuition + Why This Way (Keywords Only)

### Problem pressure cues
- trust gap
- plausible != correct
- auditability missing
- budgeted verification
- failure transparency

### Why not other routes (keywords)
- LLM-central dependence
- language-locked proofs
- theorem-only mismatch
- enterprise FM overhead
- no white-box trail

### Why this route (keywords)
- verification-first
- neuro-symbolic split
- constrained edit DSL
- staged trust funnel
- latent-guided search
- ablation-ready layers
- root-cause telemetry

### Version cues
- V1 deterministic search
- V2 trust filtering
- V3 learned guidance
- V4 controlled orchestration

### Evidence cues
- 20-program holdout
- 11 ablations
- 3 -> 9 repaired
- budget bottleneck
- plausibility wall

### Faculty-fit cues
- geometry-aware learning
- optimization rigor
- manifold methods
- scalable efficiency
- theory-to-system bridge

Grounding anchors: [SOP_my.md](SOP_my.md#L5), [SOP_my.md](SOP_my.md#L8), [SOP_my.md](SOP_my.md#L10), [SOP_my.md](SOP_my.md#L14), [SOP_my.md](SOP_my.md#L15)

---

## TOPIC 2: README Brief + Repo Map (Relevant, Non-Noise)

### Root-level essentials
- [README.md](README.md#L3): project intent, architecture overview, quick-start, V1/V2/V3 workflow.
- [SOP_my.md](SOP_my.md#L1): final narrative arc used for speaking.
- [sop_grounded_revision_2026_04_12.md](sop_grounded_revision_2026_04_12.md#L6): grounded wording for claims.

### Core package: [verifix](verifix)

#### Entry
- [verifix/cli.py](verifix/cli.py#L1): CLI entrypoints for repair, repair-v2, repair-v3, trust filtering.

#### Core contracts: [verifix/core](verifix/core)
- [verifix/core/models.py](verifix/core/models.py#L1): BugReport/Edit/Validation/Repair result contracts.
- [verifix/core/config.py](verifix/core/config.py#L1): runtime and V3 hyper-parameter contract.
- [verifix/core/action_space.py](verifix/core/action_space.py#L1): action-id bridge between edit ops and model policy.
- [verifix/core/provenance.py](verifix/core/provenance.py#L1): checkpoint/run provenance validation utilities.

#### Parsing + localization: [verifix/parser](verifix/parser)
- [verifix/parser/ast_builder.py](verifix/parser/ast_builder.py#L1): source to annotated AST.
- [verifix/parser/fault_localizer.py](verifix/parser/fault_localizer.py#L168): suspicious-line scoring (Ochiai/Tarantula + fallback).

#### Edit DSL: [verifix/edit_dsl](verifix/edit_dsl)
- [verifix/edit_dsl/operators.py](verifix/edit_dsl/operators.py#L1208): candidate edit generation from operator families.
- [verifix/edit_dsl/applicator.py](verifix/edit_dsl/applicator.py#L1): apply edits, validate syntax, produce diffs.

#### Search core: [verifix/search](verifix/search)
- [verifix/search/state.py](verifix/search/state.py#L1): node state, expansion primitives.
- [verifix/search/mcts.py](verifix/search/mcts.py#L36): baseline MCTS with budget/time gates.
- [verifix/search/scorer.py](verifix/search/scorer.py#L1): scoring heuristics.
- [verifix/search/mcts_latent.py](verifix/search/mcts_latent.py#L78): latent-guided search and gating telemetry.

#### Validation + ranking: [verifix/validator](verifix/validator)
- [verifix/validator/executor.py](verifix/validator/executor.py#L1): isolated test execution.
- [verifix/validator/patch_ranker.py](verifix/validator/patch_ranker.py#L1): dedupe + ranking of plausible patches.

#### Verifier funnel: [verifix/verifier](verifix/verifier)
- [verifix/verifier/v2_pipeline.py](verifix/verifier/v2_pipeline.py#L114): fuzz + SMT funnel orchestration.
- [verifix/verifier/fuzzer.py](verifix/verifier/fuzzer.py#L1): deterministic fuzz checks.
- [verifix/verifier/smt_layer.py](verifix/verifier/smt_layer.py#L1): SMT semantic screening.
- [verifix/verifier/evidence_report.py](verifix/verifier/evidence_report.py#L23): trust score, trust level, risk flags.

#### Version orchestrators: [verifix/pipeline](verifix/pipeline)
- [verifix/pipeline/repair_agent.py](verifix/pipeline/repair_agent.py#L25): V1 end-to-end agent.
- [verifix/pipeline/repair_agent_v2.py](verifix/pipeline/repair_agent_v2.py#L78): V2 agent wrapping V1 + funnel.
- [verifix/pipeline/repair_agent_v3.py](verifix/pipeline/repair_agent_v3.py#L44): V3 latent/hybrid with fallback.

#### Neural modules: [verifix/models](verifix/models)
- [verifix/models/latent_jepa.py](verifix/models/latent_jepa.py#L28): MultiTaskRepairGAT + JEPA predictor + training losses.
- [verifix/models/pyg_converter.py](verifix/models/pyg_converter.py#L1): AST to graph conversion for model input.
- [verifix/models/trainer.py](verifix/models/trainer.py#L1): dataset-driven checkpoint training.

#### Benchmarks: [verifix/benchmarks](verifix/benchmarks)
- [verifix/benchmarks/quixbugs.py](verifix/benchmarks/quixbugs.py#L1): QuixBugs loader/runner.
- [verifix/benchmarks/v3_quixbugs.py](verifix/benchmarks/v3_quixbugs.py#L1): V3 holdout benchmark path.
- [verifix/benchmarks/quixbugs_split.py](verifix/benchmarks/quixbugs_split.py#L1): canonical split utilities.
- [verifix/benchmarks/quixbugs_poc.py](verifix/benchmarks/quixbugs_poc.py#L1): PoC harness.
- [verifix/benchmarks/defects4j.py](verifix/benchmarks/defects4j.py#L1): Defects4J-compatible runner.
- [verifix/benchmarks/bugsinpy.py](verifix/benchmarks/bugsinpy.py#L1): BugsInPy-compatible runner.
- [verifix/benchmarks/ablation_runner.py](verifix/benchmarks/ablation_runner.py#L1): ablation evaluation helper.

#### Analysis: [verifix/analysis](verifix/analysis)
- [verifix/analysis/golden_trace.py](verifix/analysis/golden_trace.py#L206): root-cause bucket classifier + priority scoring.
- [verifix/analysis/overfit_detector.py](verifix/analysis/overfit_detector.py#L1): overfit diagnostics.

### Active scripts used in this SOP path: [scripts](scripts)
- [scripts/generate_synthetic_quixbugs.py](scripts/generate_synthetic_quixbugs.py#L26): synthetic train data generation.
- [scripts/train_v3_quixbugs.py](scripts/train_v3_quixbugs.py#L12): V3 checkpoint training launcher.
- [scripts/benchmark_v3_quixbugs.py](scripts/benchmark_v3_quixbugs.py#L19): V3 holdout benchmarking launcher.
- [scripts/benchmark_holdout_ablation_v4.py](scripts/benchmark_holdout_ablation_v4.py#L21): V1/V2/V3/V3-latent unified ablation harness.
- [scripts/extract_v3_golden_trace.py](scripts/extract_v3_golden_trace.py#L12): golden trace report generation from run artifacts.
- [scripts/create_quixbugs_split_artifact.py](scripts/create_quixbugs_split_artifact.py#L1): split artifact generation.
- [scripts/quixbugs_operator_coverage_matrix.py](scripts/quixbugs_operator_coverage_matrix.py#L1): operator expressibility matrix.
- [scripts/analyze_expressible_onehop.py](scripts/analyze_expressible_onehop.py#L1): one-hop expressibility analysis.

### Benchmark corpus (kept concise to avoid noise)
- [quixbugs/python_programs](quixbugs/python_programs): buggy program corpus consumed by loader.
- [quixbugs/correct_python_programs](quixbugs/correct_python_programs): gold reference programs for synthetic/coverage analysis.
- [quixbugs/python_testcases](quixbugs/python_testcases): per-program pytest modules.
- [quixbugs/conftest.py](quixbugs/conftest.py#L1): test configuration.
- [quixbugs/tester.py](quixbugs/tester.py#L1): benchmark helper runner.

Canonical 40 algorithm code files (explicitly tracked by split contract):
- Source list contract: [verifix/benchmarks/quixbugs_split.py](verifix/benchmarks/quixbugs_split.py#L10)
- Names: bitcount.py, breadth_first_search.py, bucketsort.py, depth_first_search.py, detect_cycle.py, find_first_in_sorted.py, find_in_sorted.py, flatten.py, gcd.py, get_factors.py, hanoi.py, is_valid_parenthesization.py, kheapsort.py, knapsack.py, kth.py, lcs_length.py, levenshtein.py, lis.py, longest_common_subsequence.py, max_sublist_sum.py, mergesort.py, minimum_spanning_tree.py, next_palindrome.py, next_permutation.py, pascal.py, possible_change.py, powerset.py, quicksort.py, reverse_linked_list.py, rpn_eval.py, shortest_path_length.py, shortest_path_lengths.py, shortest_paths.py, shunting_yard.py, sieve.py, sqrt.py, subsequences.py, to_base.py, topological_ordering.py, wrap.py.
- These names are mirrored across: [quixbugs/python_programs](quixbugs/python_programs), [quixbugs/correct_python_programs](quixbugs/correct_python_programs), [quixbugs/python_testcases](quixbugs/python_testcases).
- Helper code files in corpus folders: [quixbugs/python_programs/node.py](quixbugs/python_programs/node.py#L1), [quixbugs/python_testcases/node.py](quixbugs/python_testcases/node.py#L1), [quixbugs/python_testcases/load_testdata.py](quixbugs/python_testcases/load_testdata.py#L1).

---

## TOPIC 3: Core Walkthrough with Code Windows (Intuition + Proof)

### V1: Symbolic baseline

Intuition hit: localize -> search -> validate -> rank.

Evidence: [verifix/pipeline/repair_agent.py](verifix/pipeline/repair_agent.py#L49)

```python
suspiciousness_scores = localize_faults(...)
suspicious_lines = [score.line for score in suspiciousness_scores]

search_result = mcts_search(
	bug_report=bug_report,
	suspicious_lines=suspicious_lines,
	validator=validator,
	config=self.config,
)

ranked_patches = rank_patches(...)
```

Evidence: [verifix/search/mcts.py](verifix/search/mcts.py#L36)

```python
for iteration in range(config.mcts_iterations):
	if elapsed > config.mcts_time_budget_seconds:
		terminated_by = "time_limit"
		break
	if validations_used >= config.max_validations:
		terminated_by = "validation_cap"
		break

	node = select(root, config.mcts_exploration_constant)
	...
	result = validator.validate(node.state.current_source, bug_report)
	if result.is_plausible:
		plausible_patches.append((node.state.edit_sequence, node.state.current_source, result))
```

Evidence: [verifix/edit_dsl/operators.py](verifix/edit_dsl/operators.py#L1208)

```python
def get_candidate_edits(..., operator_tier: Literal["core", "all", "synthetic_only"] = "core"):
	operator_names = enabled_operators or _resolve_operator_names(operator_tier)
	...
	if node.lineno not in suspicious_line_set:
		continue
	...
	dedup_key = (edit.node_id, edit.operator.value, edit.replacement_text)
```

### V2: Trust funnel on top of V1

Intuition hit: plausible-first, then trust calibration via fuzz + SMT + evidence.

Evidence: [verifix/pipeline/repair_agent_v2.py](verifix/pipeline/repair_agent_v2.py#L114)

```python
log(f"V2 Pipeline: Running verification funnel on {len(plausible)} patches")
evidence_list = self._funnel.run(...)

high_count = len([item for item in evidence_list if item.trust_level == "HIGH"])
```

Evidence: [verifix/verifier/v2_pipeline.py](verifix/verifier/v2_pipeline.py#L126)

```python
fr = fuzz_patch(...)
...
smt_results = smt_screen_patches(smt_candidates, original_source, top_k=smt_k)
...
evidence_list.sort(key=lambda evidence: evidence.trust_score, reverse=True)
```

Evidence: [verifix/verifier/evidence_report.py](verifix/verifier/evidence_report.py#L23)

```python
if validation.all_failing_tests_pass:
	base += 25
if validation.no_regression:
	base += 15
...
if smt_result.verdict == "VERIFIED":
	base += 25
...
if score >= 75:
	level = "HIGH"
```

### V3: Neuro-symbolic guidance (MultiTask GAT + JEPA)

Intuition hit: learned priors rank and gate; concrete checks remain for trust.

Evidence: [verifix/models/latent_jepa.py](verifix/models/latent_jepa.py#L28)

```python
class MultiTaskRepairGAT(nn.Module):
	self.gat1 = GATConv(FEATURE_DIM, 64, heads=4, concat=True)
	self.gat2 = GATConv(256, 64, heads=4, concat=True)
	self.gat3 = GATConv(256, LATENT_DIM, heads=1, concat=False)
	...
	fault_probs = self.fault_head(node_embeddings)
	policy_logits = self.policy_head(node_embeddings)
	critic_scores = self.critic_head(z_graph)
```

Evidence: [verifix/models/latent_jepa.py](verifix/models/latent_jepa.py#L96)

```python
class JEPATransitionPredictor(nn.Module):
	self.action_embedding = nn.Embedding(NUM_ACTIONS, ACTION_DIM)
	...
	features = torch.cat([z_graph, action_emb, node_context], dim=-1)
	return self.mlp(features)
```

Evidence: [verifix/search/mcts_latent.py](verifix/search/mcts_latent.py#L402)

```python
combined = node_weight * node_prob + action_weight * action_prob
```

Evidence: [verifix/search/mcts_latent.py](verifix/search/mcts_latent.py#L635)

```python
def _validation_gate_decision(...):
	if mode == "hybrid":
		if candidate_depth < depth_floor:
			if candidate_rank <= 2:
				return True, "hybrid_depth_rank_allow"
			return False, "hybrid_depth_rank_block"
		if critic_score >= critic_threshold:
			return True, "hybrid_critic_allow"
		return False, "hybrid_critic_block"

	if mode == "latent":
		if candidate_rank <= 3:
			return True, "latent_top3_allow"
		if critic_score >= (critic_threshold * 0.8):
			return True, "latent_critic_allow"
```

Evidence: [verifix/search/mcts_latent.py](verifix/search/mcts_latent.py#L97)

```python
gate_reason_counts: dict[str, int] = {}
validation_reason_counts: dict[str, int] = {
	"validated": 0,
	"plausible": 0,
	"non_plausible": 0,
	"skipped": 0,
}
candidate_trace_top: list[dict[str, object]] = []
```

### V4: Unified ablation orchestration

Intuition hit: one harness, same split, mode toggles, comparable outputs.

Evidence: [scripts/benchmark_holdout_ablation_v4.py](scripts/benchmark_holdout_ablation_v4.py#L64)

```python
--run-v1
--run-v2
--run-v3-hybrid
--run-v3-latent
```

Evidence: [scripts/benchmark_holdout_ablation_v4.py](scripts/benchmark_holdout_ablation_v4.py#L390)

```python
payload = {
	"config": {
		"max_validations": args.max_validations,
		"v3_critic_threshold": args.v3_critic_threshold,
		"v3_candidate_node_weight": node_weight,
		"v3_candidate_action_weight": action_weight,
	},
	"summary": summary,
	"smt": smt_summary,
	"per_program": per_program,
}
```

---

## TOPIC 4: Golden Telemetry System (Isolation Logic + Snippets)

### Telemetry capture at search-time

Evidence: [verifix/search/mcts_latent.py](verifix/search/mcts_latent.py#L255)

```python
_increment_counter(gate_reason_counts, gate_reason)
...
validation_reason_counts["validated"] += 1
...
candidate_trace_top.append({...})
```

What this isolates (1-2 lines):
- It isolates gate behavior, validation behavior, and top candidate trajectories in one trace object.
- This separates "generation failed" from "gating failed" from "validation failed".

### Root-cause bucket assignment

Evidence: [verifix/analysis/golden_trace.py](verifix/analysis/golden_trace.py#L206)

```python
if terminated_by == "validation_cap":
	return "VALIDATION_BUDGET_EXHAUSTED"
...
if screened_out_by_critic >= gate_block_cutoff and candidates_validated <= low_validation_cutoff:
	return "CRITIC_GATE_OVERFILTERING"
...
if truncation_pressure > 0:
	return "TRUNCATION_PRESSURE"
...
if candidates_validated > 0 and terminated_by == "candidate_exhausted":
	return "VALIDATED_NOT_PLAUSIBLE"
```

What this isolates (1-2 lines):
- It maps each failed program to a single actionable failure family.
- It converts anecdotal debugging into intervention-level categories.

### Priority ranking for interventions

Evidence: [verifix/analysis/golden_trace.py](verifix/analysis/golden_trace.py#L271)

```python
base = _BUCKET_BASE_PRIORITY.get(root_cause_bucket, 50.0)
score += min(60.0, max(0.0, candidate_count / 20.0))
score += _confidence_bonus(action_guess_confidence)
```

What this isolates (1-2 lines):
- It isolates what to fix first under limited research time.
- Programs with high recoverability and high-impact failure modes rise to top.

### End-to-end report generation

Evidence: [scripts/extract_v3_golden_trace.py](scripts/extract_v3_golden_trace.py#L87)

```python
report = build_golden_trace_report(...)
write_report_outputs(report=report, ...)
```

What this isolates (1-2 lines):
- It isolates run artifacts into a reproducible JSON + markdown root-cause report.
- Same pipeline can be rerun across checkpoints and splits for apples-to-apples diagnosis.

---

## TOPIC 5: All-11 Ablations Table + 2-3 Word Recall Cues

Primary source: [.analysis/all11_ablation_summary_2026_04_11.md](.analysis/all11_ablation_summary_2026_04_11.md#L1)

| # | Run (short) | Max validations | Repaired (H/L) | Top bucket | Recall cue |
|---|---|---:|---:|---|---|
| 1 | full_n06_ct045 | 25 | 3/20, 3/20 | VALIDATION_BUDGET_EXHAUSTED | budget wall |
| 2 | full_n06_ct000 | 25 | 3/20, 3/20 | VALIDATION_BUDGET_EXHAUSTED | no gain |
| 3 | full_n02_ct045 | 25 | 3/20, 3/20 | VALIDATION_BUDGET_EXHAUSTED | weight neutral |
| 4 | full_n02_ct000 | 25 | 3/20, 3/20 | VALIDATION_BUDGET_EXHAUSTED | budget locked |
| 5 | v3only_policy500 | 25 | 3/20, 3/20 | VALIDATION_BUDGET_EXHAUSTED | breadth alone |
| 6 | v3only_critic500_ct060 | 25 | 2/20, 3/20 | CRITIC_GATE_OVERFILTERING | critic choke |
| 7 | sniper_V1_val025 | 25 | 3/20, 3/20 | VALIDATION_BUDGET_EXHAUSTED | control repeat |
| 8 | sniper_V2_val050 | 50 | 5/20, 5/20 | VALIDATION_BUDGET_EXHAUSTED | first breakout |
| 9 | sniper_V3_val100 | 100 | 6/20, 6/20 | VALIDATION_BUDGET_EXHAUSTED | steady climb |
| 10 | sniper_V4_val150 | 150 | 8/20, 8/20 | VALIDATED_NOT_PLAUSIBLE | quality wall |
| 11 | sniper_V5_val200 | 200 | 9/20, 9/20 | VALIDATED_NOT_PLAUSIBLE | best point |

Quick speaking chain:
- 25 -> 50 -> 100 -> 150 -> 200 validations
- 3 -> 5 -> 6 -> 8 -> 9 repaired
- bucket shift: budget exhausted -> validated not plausible

Evidence anchors: [.analysis/all11_ablation_summary_2026_04_11.md](.analysis/all11_ablation_summary_2026_04_11.md#L11), [.analysis/all11_ablation_summary_2026_04_11.md](.analysis/all11_ablation_summary_2026_04_11.md#L15), [.analysis/all11_ablation_summary_2026_04_11.md](.analysis/all11_ablation_summary_2026_04_11.md#L30), [.analysis/all11_ablation_summary_2026_04_11.json](.analysis/all11_ablation_summary_2026_04_11.json#L274), [.analysis/all11_ablation_summary_2026_04_11.json](.analysis/all11_ablation_summary_2026_04_11.json#L280), [.analysis/all11_ablation_summary_2026_04_11.json](.analysis/all11_ablation_summary_2026_04_11.json#L287)

---



---

## Fast Recall Close (10-second summary)
- V1 finds, V2 verifies, V3 prioritizes, V4 proves.
- All-11 says budget first, then quality wall.
- Golden telemetry tells exactly what failed and why.
- Path forward needs geometry-aware optimization leadership.
- That is why Prof. Jawanpuria is the precise fit.

