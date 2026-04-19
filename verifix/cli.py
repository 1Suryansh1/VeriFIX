from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
import yaml

from verifix.benchmarks.defects4j import Defects4JBenchmark
from verifix.benchmarks.bugsinpy import BugsInPyBenchmark
from verifix.benchmarks.quixbugs import QuixBugsBenchmark
from verifix.benchmarks.v3_quixbugs import V3QuixBugsBenchmark
from verifix.core.config import DEFAULT_CONFIG, QuixBugsConfig, VerifixConfig
from verifix.core.models import BugReport
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v2 import RepairAgentV2
from verifix.pipeline.repair_agent_v3 import RepairAgentV3
from verifix.verifier.evidence_report import evidence_to_markdown
from verifix.validator.patch_ranker import summarize_results


app = typer.Typer(help="VeriFix automated program repair CLI")
benchmark_app = typer.Typer(help="Run benchmark suites")
config_app = typer.Typer(help="Inspect and initialize configuration")

app.add_typer(benchmark_app, name="benchmark")
app.add_typer(config_app, name="config")


@app.command("repair")
def repair_command(
    file: str = typer.Option(..., help="Path to buggy source file"),
    project_root: str = typer.Option(..., help="Project root containing tests"),
    failing_tests: str = typer.Option(..., help="Comma-separated failing test ids"),
    passing_tests: str = typer.Option("", help="Comma-separated passing test ids"),
    output: str = typer.Option("", help="Optional JSON output file"),
    config: str = typer.Option("", help="Optional YAML config file"),
    verbose: bool = typer.Option(False, help="Enable verbose progress logging"),
) -> None:
    config_obj = VerifixConfig.from_yaml(config) if config else VerifixConfig()

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout, force=True)

    source_path = Path(file)
    source_text = source_path.read_text(encoding="utf-8")
    project_path = Path(project_root).resolve()

    try:
        relative_file_path = source_path.resolve().relative_to(project_path).as_posix()
    except ValueError:
        relative_file_path = source_path.name

    report = BugReport(
        bug_id=source_path.stem,
        language="java" if source_path.suffix.lower() == ".java" else "python",
        buggy_source=source_text,
        file_path=relative_file_path,
        failing_tests=_parse_test_ids(failing_tests),
        passing_tests=_parse_test_ids(passing_tests),
        project_root=str(project_path),
        metadata={"source": "cli.repair"},
    )

    agent = RepairAgent(config_obj)
    result = agent.repair(report)

    typer.echo(summarize_results(result.ranked_patches, result.bug_id))

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if result.success:
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


@app.command("repair-v2")
def repair_v2_command(
    file: str = typer.Option(..., help="Path to buggy source file"),
    project_root: str = typer.Option(..., help="Project root containing tests"),
    failing_tests: str = typer.Option(..., help="Comma-separated failing test ids"),
    passing_tests: str = typer.Option("", help="Comma-separated passing test ids"),
    output: str = typer.Option("", help="Optional JSON output file"),
    trust_level: str = typer.Option("LOW", help="Minimum trust level: UNVERIFIED|LOW|MEDIUM|HIGH"),
    config: str = typer.Option("", help="Optional YAML config file"),
    verbose: bool = typer.Option(False, help="Enable verbose progress logging"),
) -> None:
    config_obj = VerifixConfig.from_yaml(config) if config else VerifixConfig()

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout, force=True)

    source_path = Path(file)
    source_text = source_path.read_text(encoding="utf-8")
    project_path = Path(project_root).resolve()

    try:
        relative_file_path = source_path.resolve().relative_to(project_path).as_posix()
    except ValueError:
        relative_file_path = source_path.name

    report = BugReport(
        bug_id=source_path.stem,
        language="java" if source_path.suffix.lower() == ".java" else "python",
        buggy_source=source_text,
        file_path=relative_file_path,
        failing_tests=_parse_test_ids(failing_tests),
        passing_tests=_parse_test_ids(passing_tests),
        project_root=str(project_path),
        metadata={"source": "cli.repair_v2"},
    )

    normalized_level = trust_level.strip().upper()
    if normalized_level not in _TRUST_LEVEL_ORDER:
        raise typer.BadParameter(
            f"Invalid trust level '{trust_level}'. Use one of: {', '.join(_TRUST_LEVEL_ORDER.keys())}"
        )

    agent = RepairAgentV2(config_obj)
    result = agent.repair(report)

    filtered = _filter_evidence_by_trust(result.evidence_list, normalized_level)
    typer.echo(
        f"V2 summary: total={len(result.evidence_list)}, "
        f"filtered={len(filtered)}, min_trust={normalized_level}"
    )
    if filtered:
        typer.echo(evidence_to_markdown(filtered[0]))

    if output:
        payload = result.to_dict()
        payload["applied_trust_filter"] = normalized_level
        payload["evidence_list"] = [
            item for item in payload["evidence_list"] if _trust_rank(item.get("trust_level", "UNVERIFIED")) >= _trust_rank(normalized_level)
        ]
        payload["best_evidence"] = payload["evidence_list"][0] if payload["evidence_list"] else None
        payload["success"] = any(
            item.get("trust_level") in {"HIGH", "MEDIUM"} for item in payload["evidence_list"]
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not result.evidence_list:
        raise typer.Exit(code=1)
    if normalized_level == "HIGH" and not any(item.trust_level == "HIGH" for item in result.evidence_list):
        raise typer.Exit(code=2)
    if not filtered:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command("repair-v3")
def repair_v3_command(
    file: str = typer.Option(..., help="Path to buggy source file"),
    project_root: str = typer.Option(..., help="Project root containing tests"),
    failing_tests: str = typer.Option(..., help="Comma-separated failing test ids"),
    passing_tests: str = typer.Option("", help="Comma-separated passing test ids"),
    output: str = typer.Option("", help="Optional JSON output file"),
    checkpoint: str = typer.Option("", help="Optional V3 checkpoint path (falls back to V1 when omitted)"),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda"),
    rollout_mode: str = typer.Option(
        "hybrid",
        help="V3 rollout mode: concrete|latent|hybrid",
    ),
    config: str = typer.Option("", help="Optional YAML config file"),
    verbose: bool = typer.Option(False, help="Enable verbose progress logging"),
) -> None:
    config_obj = VerifixConfig.from_yaml(config) if config else VerifixConfig()
    normalized_rollout_mode = rollout_mode.strip().lower()
    allowed_modes = {"concrete", "latent", "hybrid"}
    if normalized_rollout_mode not in allowed_modes:
        raise typer.BadParameter(
            "Invalid rollout mode. Use one of: concrete, latent, hybrid"
        )

    config_obj.v3_enabled = True
    config_obj.v3_rollout_mode = normalized_rollout_mode

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout, force=True)

    source_path = Path(file)
    source_text = source_path.read_text(encoding="utf-8")
    project_path = Path(project_root).resolve()

    try:
        relative_file_path = source_path.resolve().relative_to(project_path).as_posix()
    except ValueError:
        relative_file_path = source_path.name

    report = BugReport(
        bug_id=source_path.stem,
        language="java" if source_path.suffix.lower() == ".java" else "python",
        buggy_source=source_text,
        file_path=relative_file_path,
        failing_tests=_parse_test_ids(failing_tests),
        passing_tests=_parse_test_ids(passing_tests),
        project_root=str(project_path),
        metadata={"source": "cli.repair_v3"},
    )

    checkpoint_path = checkpoint if checkpoint else None
    agent = RepairAgentV3(config=config_obj, checkpoint_path=checkpoint_path, device=device)
    result = agent.repair(report)

    typer.echo(summarize_results(result.repair_result.ranked_patches, result.repair_result.bug_id))

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if result.success:
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


@benchmark_app.command("quixbugs")
def benchmark_quixbugs(
    quixbugs_root: str = typer.Option(..., help="Path to QuixBugs repository"),
    max_programs: int = typer.Option(0, help="Maximum number of programs to run (0 = all)"),
    output_dir: str = typer.Option("./results/quixbugs", help="Directory for benchmark outputs"),
    parallel: bool = typer.Option(False, help="Run benchmark programs in parallel"),
    split_mode: str = typer.Option(
        "all",
        help=(
            "Program split mode: all | stratified-train | stratified-test | "
            "alphabetical-train | alphabetical-test"
        ),
    ),
    split_seed: int = typer.Option(20260404, help="Seed used for deterministic stratified split"),
) -> None:
    benchmark = QuixBugsBenchmark(quixbugs_root, config=QuixBugsConfig())
    summary = benchmark.run_all(
        max_programs=(max_programs if max_programs > 0 else None),
        parallel=parallel,
        output_dir=output_dir,
        split_mode=split_mode,
        split_seed=split_seed,
    )
    benchmark.print_leaderboard(summary)
    typer.echo(json.dumps(summary, indent=2))


@benchmark_app.command("defects4j")
def benchmark_defects4j(
    d4j_root: str = typer.Option(..., help="Path to Defects4J/D4J-PY dataset"),
    output_dir: str = typer.Option("./results/defects4j", help="Directory for benchmark outputs"),
    bug_ids: str = typer.Option("", help="Optional comma-separated bug id filter"),
) -> None:
    benchmark = Defects4JBenchmark(d4j_root)
    ids = _parse_test_ids(bug_ids)
    summary = benchmark.run(bug_ids=(ids if ids else None), output_dir=output_dir)
    typer.echo(json.dumps(summary, indent=2))


@benchmark_app.command("bugsinpy")
def benchmark_bugsinpy(
    bugsinpy_root: str = typer.Option(
        ".data/BugsInPy", "--bugsinpy-root"
    ),
    projects: str = typer.Option(
        None, "--projects", help="Comma-separated: ansible,black,scrapy"
    ),
    max_bugs: int = typer.Option(None, "--max-bugs"),
    output_dir: str = typer.Option(
        ".bugsinpy_results", "--output-dir"
    ),
    use_v2: bool = typer.Option(True, "--use-v2/--no-v2"),
) -> None:
    benchmark = BugsInPyBenchmark(bugsinpy_root)
    project_filter = _parse_test_ids(projects) if projects else None
    summary = benchmark.run(
        project_filter=project_filter,
        max_bugs=max_bugs,
        output_dir=output_dir,
        use_v2=use_v2,
    )
    benchmark.print_leaderboard(summary)
    typer.echo(json.dumps(summary, indent=2))
    raise typer.Exit(code=0)


@benchmark_app.command("quixbugs-v3")
def benchmark_quixbugs_v3(
    quixbugs_root: str = typer.Option(..., help="Path to QuixBugs repository"),
    checkpoint: str = typer.Option(..., help="Path to trained V3 checkpoint"),
    output_dir: str = typer.Option("./results/quixbugs_v3", help="Directory for benchmark outputs"),
    split_strategy: str = typer.Option("stratified", help="Split strategy: stratified|alphabetical"),
    split_seed: int = typer.Option(20260404, help="Seed used for deterministic split"),
    max_programs: int = typer.Option(10, help="Maximum holdout programs to evaluate"),
    run_latent_ablation: bool = typer.Option(True, help="Also run latent-only V3 ablation mode"),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda"),
) -> None:
    benchmark = V3QuixBugsBenchmark(
        quixbugs_root=quixbugs_root,
        checkpoint_path=checkpoint,
        config=QuixBugsConfig(),
        device=device,
    )
    summary = benchmark.run_holdout(
        output_dir=output_dir,
        split_strategy=split_strategy,
        seed=split_seed,
        max_programs=(max_programs if max_programs > 0 else None),
        run_latent_ablation=run_latent_ablation,
    )
    typer.echo(json.dumps(summary, indent=2))


@config_app.command("show")
def config_show() -> None:
    config_obj = VerifixConfig()
    typer.echo(yaml.safe_dump(config_obj.model_dump(), sort_keys=True))


@config_app.command("init")
def config_init() -> None:
    path = Path("config.yaml")
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG.model_dump(), sort_keys=True), encoding="utf-8")
    typer.echo(f"Wrote default configuration to {path}")


def _parse_test_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


_TRUST_LEVEL_ORDER = {
    "UNVERIFIED": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def _trust_rank(level: str) -> int:
    return _TRUST_LEVEL_ORDER.get(level.strip().upper(), -1)


def _filter_evidence_by_trust(evidence_list: list, minimum_level: str) -> list:
    threshold = _trust_rank(minimum_level)
    return [item for item in evidence_list if _trust_rank(item.trust_level) >= threshold]


if __name__ == "__main__":
    app()
