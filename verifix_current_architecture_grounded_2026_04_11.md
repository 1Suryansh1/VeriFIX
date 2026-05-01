# VeriFIX Current Architecture (Grounded, In-Depth)

## What It Does
VeriFix is an automated program repair system for Python-first workflows with benchmark-ready interfaces for broader datasets. It localizes likely bug lines, searches for patch candidates with constrained edit operators, validates patches against failing and regression tests, and in V2 applies additional verification stages (fuzzing and SMT screening) to reduce overfitted fixes and produce trust-scored evidence reports.

## Purpose & Scope
The description is based on active modules and scripts in this repository, especially:
- verifix/cli.py
- verifix/pipeline/repair_agent.py
- verifix/pipeline/repair_agent_v2.py
- verifix/pipeline/repair_agent_v3.py
- verifix/search/mcts.py
- verifix/search/mcts_latent.py
- verifix/verifier/v2_pipeline.py
- verifix/benchmarks/quixbugs.py
- verifix/benchmarks/v3_quixbugs.py
- scripts/benchmark_holdout_ablation_v4.py
- scripts/train_v3_quixbugs.py
- scripts/benchmark_v3_quixbugs.py
- .analysis/all11_ablation_summary_2026_04_11.md

## Setup & Implementation Guidance

### 1. Requirements
*   **Python:** 3.11 or newer (enforced in `pyproject.toml`)
*   **Git**
*   **Core Dependencies:** `typer`, `pydantic`, `pyyaml`, `asttokens`, `coverage`

### 2. Basic Setup (V1 & V2)
Clone the repository and install the standard dependencies:
```bash
git clone https://github.com/your-username/VeriFIX.git
cd VeriFIX

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/MacOS:
# source .venv/bin/activate

# Install core and development dependencies (brings in pytest, pytest-cov)
pip install -e ".[dev]"
```

### 3. V3 Neural Engine Setup (PyTorch & PyTorch Geometric)
To run **V3 (Neuro-Symbolic Guidance)** experiments, you need `torch>=2.11` and `torch-geometric>=2.7` as defined in the `[dev,v3]` extras:
```bash
pip install -e ".[dev,v3]"
```
*(Note: If you encounter specific CUDA compatibilities, please install PyTorch according to the official PyTorch instructions before running the command).*

### 4. Running the Tests
To ensure everything is set up correctly:
```bash
# Run basic unit tests
pytest tests/ -v

# Run integration tests
pytest tests/ -m integration -v
```

## Versioned System View & CLI Usage

### V1 (Symbolic APR Baseline)
Main idea:
- AST parse and fault localization identify suspicious lines.
- MCTS over constrained edit operators searches candidate patches.
- Concrete test validation determines plausibility.
- Patch ranker deduplicates and ranks plausible candidates.

Key modules:
- verifix/pipeline/repair_agent.py
- verifix/parser/ast_builder.py
- verifix/parser/fault_localizer.py
- verifix/search/mcts.py
- verifix/search/state.py
- verifix/edit_dsl/operators.py
- verifix/edit_dsl/applicator.py
- verifix/validator/executor.py
- verifix/validator/patch_ranker.py

**V1 CLI Usage:**
Standard constrained edit generation using AST parsing, MCTS, and concrete test-suite validation.

```bash
verifix repair --file path/to/buggy.py \
               --project-root . \
               --failing-tests "tests/test_buggy.py::test_failing" \
               --passing-tests "tests/test_buggy.py::test_passing" \
               --config "config.yml" \
               --verbose
```
**Key toggles:**
*   `mcts_max_depth`: Depth of search tree.
*   `mcts_iterations`: Number of rollout iterations.
*   `max_validations`: Caps the concrete test budget (highest impact parameter on performance).
*   `max_patch_candidates`: Bounding box on patch truncation size per node.

### V2 (Verification Funnel on Top of V1)
Main idea:
- Reuse V1 to get plausible patches.
- Run post-search trust pipeline: fuzzing -> SMT screening -> evidence report.
- Return trust-scored evidence, not only pass/fail test outcomes.

Key modules:
- verifix/pipeline/repair_agent_v2.py
- verifix/verifier/v2_pipeline.py
- verifix/verifier/fuzzer.py
- verifix/verifier/smt_layer.py
- verifix/verifier/evidence_report.py

**V2 CLI Usage:**
```bash
verifix repair-v2 --file path/to/buggy.py \
                  --project-root . \
                  --failing-tests "tests/test_buggy.py::test_failing" \
                  --trust-level "HIGH"  
```
**Specific V2 Toggles:**
*   `--trust-level`: Hard filter on evidence output. Accepts `UNVERIFIED` | `LOW` | `MEDIUM` | `HIGH`. The CLI will return exit code 2 if it cannot find a patch matching your required trust level.

### V3 (Neuro-Symbolic Guidance)
Main idea:
- Keep symbolic Edit DSL and concrete validator.
- Add neural guidance for search using MultiTaskRepairGAT and JEPA transition predictor.
- Support rollout modes: concrete, hybrid, latent.
- Add rich latent diagnostics and gate-reason tracing.

Key modules:
- verifix/pipeline/repair_agent_v3.py
- verifix/models/latent_jepa.py
- verifix/models/pyg_converter.py
- verifix/search/mcts_latent.py

**V3 CLI Usage:**
```bash
verifix repair-v3 --file path/to/buggy.py \
                  --project-root . \
                  --failing-tests "tests/test_buggy.py::test_failing" \
                  --rollout-mode hybrid \   
                  --device cpu \            
                  --checkpoint path/to/v3_multitask_gat.pt
```
**V3 Critical Toggles:**
*   `--rollout-mode`: Accepts `concrete`, `latent`, or `hybrid`. Dictates how rollouts process unseen graph nodes.
*   `--checkpoint`: Path to the `.pt` multi-task GAT model. Without this, V3 naturally degrades/falls back to V1 configurations.
*   `--device`: PyTorch backend target (`cpu` or `cuda`).

### V4 (Experiment Orchestration Layer, Not a Separate Core Engine)
Main idea:
- Script-level benchmark harness across V1, V2, V3-hybrid, and V3-latent.
- Unified holdout evaluation and JSON artifact generation.
- Used for ablations and mode-comparison studies.

Key module:
- scripts/benchmark_holdout_ablation_v4.py

## Core Data Contracts
Primary shared contracts are in verifix/core/models.py:
- BugReport: bug id, source, test ids, project root, metadata.
- Edit and EditOperator: structured patch atom with operator semantics.
- ValidationResult: compile/test outcomes and plausibility flags.
- RankedPatch and RepairResult: ranked patch outputs and global metrics.

Config contract is in verifix/core/config.py with:
- MCTS settings.
- Validation budget settings.
- Fault localization settings.
- V3-specific controls (rollout mode, depth floor, branch cap, critic threshold, candidate score mixing).

Action-space bridge is in verifix/core/action_space.py:
- 15 research-facing action ids.
- Mapping between symbolic EditOperator/metadata and action ids used by V3 model heads.

## End-to-End Runtime Workflow (Single Bug)

### Step 1: CLI Ingress
- Commands: repair, repair-v2, repair-v3.
- CLI builds BugReport from file path, tests, and project root.
- repair-v3 sets rollout mode and enables V3 flags.

### Step 2: Parse and Localization
- AST builder creates AnnotatedAST from source.
- Fault localizer computes suspicious lines using coverage when available.
- For failing-only workloads, localizer has heuristic fallback paths.

### Step 3: Candidate Generation
- Edit DSL operators generate candidate edits near suspicious lines.
- Operator set includes replace, swap, delete, insert-before/after, unwrap, and condition-wrapping classes.
- Applicator applies edits and validates syntax before search expansion proceeds.

### Step 4A: V1 Search
- MCTS loop: select -> expand -> rollout/validate -> backpropagate.
- Concrete validator executes tests in isolated workspace copies.
- Validation is budget-limited and can short-circuit by mode constraints.

### Step 4B: V3 Search
- Build graph via ASTtoPyG.
- MultiTaskRepairGAT produces fault probs, policy logits, critic signal.
- Candidate score mixes node fault and action policy probabilities.
- Depth-floor and branch-per-state control multi-step expansion.
- Validation gate uses rollout mode + depth + rank + critic threshold.
- Diagnostics log gate reason counts, validation reason counts, and candidate traces.

### Step 5: Ranking and Output
- Rank plausible patches using combined validation and patch-quality terms.
- Deduplicate by normalized patched source.
- Output includes diffs and ranked candidates.

### Step 6 (V2 Only): Trust Funnel
- Re-rank plausible candidates for funnel input.
- Fuzz stage rejects brittle/overfit patches when possible.
- SMT stage screens semantic properties for top candidates.
- Evidence report emits trust level and structured rationale.

## QUIXBUGS Workflow (Current Main Benchmark)

### Loader and test adaptation
- QuixBugsLoader supports both JSON testcase sources and python testcase modules.
- It creates temporary runnable workspaces and infers passing/failing tests via pytest + junit XML.

### V1 benchmark path
- QuixBugsBenchmark loads reports, runs repair agent per program, and writes per-program plus summary artifacts.

### V3 benchmark path
- `V3QuixBugsBenchmark` runs holdout split evaluation with `baseline_v1`, `v3_hybrid`, and optional `v3_latent`.
- It computes diagnostics such as localization top-3 hit rate, critic Brier score, and latent pre-screen hit rate.

**V3 QuixBugs Benchmarking (Single Run)**  
Evaluates V3 across a sampled set of the QuixBugs suite.
```bash
python scripts/benchmark_v3_quixbugs.py \
    --checkpoint path/to/v3_multitask_gat.pt \
    --split-strategy stratified \
    --max-programs 10 \
    --run-latent-ablation \
    --max-validations 50
```

### V4 ablation path
- benchmark_holdout_ablation_v4.py orchestrates multi-mode runs over selected holdout programs.
- Collects per-program outcomes and summary metrics for V1, V2, V3-hybrid, V3-latent.

## Rigorous QuixBugs Execution Pipeline

To create the 20/20 train-test split for the QuixBugs dataset, the repository uses the **scripts/create_quixbugs_split_artifact.py** script. This artifact is a critical dependency used across the data generation, training, and testing workloads.

Here is the exact progression of steps and CLI parameters:

### Step 1: Create the QuixBugs Split (20 Train / 20 Test)
Before training or benchmarking, you must allocate which programs belong to the holdout set. The script reads the correct Python programs and enforces a deterministic separation.

```bash
python scripts/create_quixbugs_split_artifact.py \
    --quixbugs-root quixbugs \
    --strategy stratified \
    --train-size 20 \
    --seed 20260404 \
    --output .analysis/quixbugs_split_20_20.json
```
**Toggles**:
*   `--strategy`: Controls how the split is sampled (`stratified` or `alphabetical`). Stratified ensures equal distribution of program complexity.
*   `--train-size`: How many programs stay in the training set (default: `20`, leaving roughly 20 for testing).
*   `--output`: Generates the exact `<filename>.json` artifact referenced by downstream steps.

### Step 2: Generate Synthetic JEPA Training Data
Once the split JSON is generated, use it to synthesize training records. This ensures mutations are *only* sourced from the 20 training programs, preserving the integrity of the 20 test programs.

```bash
python scripts/generate_synthetic_quixbugs.py \
    --quixbugs-root quixbugs \
    --split-artifact .analysis/quixbugs_split_20_20.json \
    --split-strategy stratified \
    --output data/quixbugs_jepa_train.jsonl \
    --target-synthetic-count 5000 \
    --max-synthetic-per-program 300 \
    --num-mutations 3 \
    --hard-negative-ratio 0.35 
```
**Toggles**:
*   `--split-artifact`: Point to the split created in Step 1 to safely boundary the generative mutations.
*   `--target-synthetic-count`: The total number of state transitions required. 
*   `--max-synthetic-per-program`: Prevents any single program from over-dominating the GAT's bias.
*   `--hard-negative-ratio`: Controls the injection of non-constructive mutations to harden the Critic.

### Step 3: Train the Multi-Task GAT Checkpoint
With the synthetic `.jsonl` dataset generated from your specific training split, you can now feed it into the PyTorch trainer to export your `.pt` model checkpoint.

```bash
python scripts/train_v3_quixbugs.py \
    --dataset data/quixbugs_jepa_train.jsonl \
    --output-dir .v3_checkpoints \
    --run-id cycle_2026_04_manual \
    --epochs 3 \
    --lr 1e-3 \
    --device cpu \
    --beta-critic 10.0 \
    --beta-localization 1.0 \
    --beta-policy 1.0
```
*Outputs to:* `.v3_checkpoints/cycle_2026_04_manual/v3_multitask_gat.pt`

### Step 4: Run the V4 Holdout Ablations (Testing)
Finally, run your newly minted model against the exact holdout programs determined back in Step 1. The ablation tool parses the split artifact to specifically target the `test` split.

```bash
python scripts/benchmark_holdout_ablation_v4.py \
    --checkpoint .v3_checkpoints/cycle_2026_04_manual/v3_multitask_gat.pt \
    --split-artifact .analysis/quixbugs_split_20_20.json \
    --split-side test \
    --run-v1 \
    --run-v3-hybrid \
    --no-run-v3-latent \
    --mcts-iterations 40 \
    --max-validations 100 \
    --time-budget 20.0 \
    --v3-critic-threshold 0.45
```
**Toggles**:
*   `--split-artifact`: Passing the same file from Step 1 ensures the engine only benchmarks the remaining unseen 20 programs.
*   `--split-side`: Instructs the benchmark to explicitly run the `test` array from the JSON artifact.

## Current Empirical Signal from All-11 Ablation Summary
From .analysis/all11_ablation_summary_2026_04_11.md:
- Validation budget is the strongest lever in observed runs.
- Hybrid and latent move from 3/20 to 9/20 as max_validations scales 25 -> 200.
- Failure mode transitions from VALIDATION_BUDGET_EXHAUSTED to VALIDATED_NOT_PLAUSIBLE at higher budgets.
- This implies current bottleneck is not only candidate generation but also candidate quality and post-gate plausibility.

## Current Limit Boundaries (Important for Grounded Communication)
- The active production parser path is AST-based, not CPG-based.
- Java language support in parser/ast_builder.py is marked not implemented for AST build.
- V3 is checkpoint-gated and falls back to concrete baseline if model is absent or runtime fails.
- V4 is an experiment harness layer rather than a separate repair engine.

