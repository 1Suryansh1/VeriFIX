from __future__ import annotations

import logging
import time
from pathlib import Path

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.parser.ast_builder import build_ast
from verifix.parser.fault_localizer import localize_faults
from verifix.search.mcts import mcts_search
from verifix.validator.executor import ConcreteValidator
from verifix.validator.patch_ranker import rank_patches


class RepairAgent:
    def __init__(self, config: VerifixConfig | None = None) -> None:
        if config is None:
            self.config = VerifixConfig(**DEFAULT_CONFIG.model_dump())
        else:
            self.config = config

        self.logger = logging.getLogger("verifix.pipeline")

    def repair(self, bug_report: BugReport) -> RepairResult:
        start_time = time.monotonic()

        def log(message: str, level: str = "INFO") -> None:
            if level == "ERROR":
                self.logger.error(message)
            else:
                self.logger.info(message)

        assert bug_report.language in self.config.supported_languages, (
            f"Unsupported language '{bug_report.language}'. "
            f"Supported languages: {self.config.supported_languages}"
        )
        assert len(bug_report.failing_tests) > 0, "No failing tests provided"

        try:
            log(f"Building AST for {bug_report.file_path}")
            _annotated_ast = build_ast(
                bug_report.buggy_source,
                bug_report.file_path,
                language=bug_report.language,
            )

            log(f"Running fault localization ({self.config.fl_algorithm})")
            suspiciousness_scores = localize_faults(
                project_root=bug_report.project_root,
                source_file=bug_report.file_path,
                failing_tests=bug_report.failing_tests,
                passing_tests=bug_report.passing_tests,
                algorithm=self.config.fl_algorithm,
                top_n=self.config.fl_top_n_lines,
                python_executable=self.config.python_executable,
            )
            suspicious_lines = [score.line for score in suspiciousness_scores]

            if not suspicious_lines:
                log("WARNING: Fault localization returned no lines. Using all lines.")
                suspicious_lines = list(range(1, bug_report.buggy_source.count("\n") + 2))

            log(f"Top suspicious lines: {suspicious_lines[:5]}")

            validator = ConcreteValidator(self.config)

            log(f"Starting MCTS search ({self.config.mcts_iterations} iterations)")
            search_result = mcts_search(
                bug_report=bug_report,
                suspicious_lines=suspicious_lines,
                validator=validator,
                config=self.config,
                progress_callback=(
                    lambda i, v, p: log(f"  iter={i} validations={v} patches={p}") if i % 50 == 0 else None
                ),
            )

            log(f"Ranking {len(search_result.plausible_patches)} plausible patches")
            ranked_patches = rank_patches(
                search_result=search_result,
                original_source=bug_report.buggy_source,
                bug_report=bug_report,
                config=self.config,
                max_patches=10,
            )

            return RepairResult(
                bug_id=bug_report.bug_id,
                success=len(ranked_patches) > 0,
                ranked_patches=ranked_patches,
                total_states_explored=search_result.total_iterations,
                total_validations_run=search_result.total_validations,
                wall_time_seconds=time.monotonic() - start_time,
                search_tree_depth=search_result.tree_depth_reached,
                error=None,
            )

        except Exception as exc:
            log(f"ERROR: {exc}", level="ERROR")
            return RepairResult(
                bug_id=bug_report.bug_id,
                success=False,
                ranked_patches=[],
                total_states_explored=0,
                total_validations_run=0,
                wall_time_seconds=time.monotonic() - start_time,
                search_tree_depth=0,
                error=str(exc),
            )

    def repair_from_file(self, file_path: str, test_ids: list[str], project_root: str) -> RepairResult:
        source_path = Path(file_path)
        source = source_path.read_text(encoding="utf-8")

        project_root_path = Path(project_root).resolve()
        source_resolved = source_path.resolve()
        try:
            relative_path = source_resolved.relative_to(project_root_path).as_posix()
        except ValueError:
            relative_path = source_path.name

        language = "java" if source_path.suffix.lower() == ".java" else "python"

        report = BugReport(
            bug_id=source_path.stem,
            language=language,
            buggy_source=source,
            file_path=relative_path,
            failing_tests=list(test_ids),
            passing_tests=[],
            project_root=str(project_root_path),
            metadata={"source": "repair_from_file"},
        )
        return self.repair(report)


__all__ = ["RepairAgent"]
