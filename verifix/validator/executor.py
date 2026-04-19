from __future__ import annotations

import ast
import hashlib
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport, ValidationResult
from verifix.search.mcts import ValidatorProtocol


class ValidationBudgetExceeded(Exception):
    pass


class ExecutionSandbox:
    def __init__(self, project_root: str, working_dir: str, python_executable: str = sys.executable) -> None:
        self.project_root = Path(project_root)
        self.working_dir = Path(working_dir)
        self.python_executable = python_executable
        self.state_id = "state"

    def setup_workspace(self, bug_report: BugReport) -> str:
        del bug_report

        self.working_dir.mkdir(parents=True, exist_ok=True)
        workspace_name = f"verifix_{self.state_id}_{uuid4().hex[:8]}"
        workspace_path = self.working_dir / workspace_name

        ignore = shutil.ignore_patterns("__pycache__", ".git", "*.pyc")
        shutil.copytree(self.project_root, workspace_path, ignore=ignore)
        return str(workspace_path)

    def write_patched_file(self, workspace_path: str, file_path: str, source: str) -> None:
        target = Path(workspace_path) / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    def run_tests(
        self,
        workspace_path: str,
        test_ids: list[str],
        timeout_seconds: float = 30.0,
    ) -> dict[str, tuple[bool, str]]:
        if not test_ids:
            return {}

        results: dict[str, tuple[bool, str]] = {}

        for test_id in test_ids:
            command = [
                self.python_executable,
                "-m",
                "pytest",
                test_id,
                "--tb=short",
                "-q",
                "--no-header",
                f"--timeout={timeout_seconds}",
                "-x",
            ]

            try:
                proc = subprocess.run(
                    command,
                    cwd=workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                timeout_msg = "timeout"
                return {tid: (False, timeout_msg) for tid in test_ids}
            except FileNotFoundError as exc:
                raise RuntimeError("pytest not installed in project environment") from exc

            output = f"{proc.stdout}\n{proc.stderr}".strip()
            if "No module named pytest" in output or "unrecognized arguments: --timeout" in output:
                raise RuntimeError("pytest not installed in project environment")

            results[test_id] = (proc.returncode == 0, output)

        return results

    def cleanup_workspace(self, workspace_path: str) -> None:
        shutil.rmtree(workspace_path, ignore_errors=True)


def validate_patch(
    patched_source: str,
    bug_report: BugReport,
    config: VerifixConfig,
    state_id: str = "",
) -> ValidationResult:
    start = time.monotonic()

    try:
        ast.parse(patched_source)
    except SyntaxError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return ValidationResult(
            state_id=state_id,
            compiled=False,
            tests_passed=[],
            tests_failed=[*bug_report.failing_tests, *bug_report.passing_tests],
            all_failing_tests_pass=False,
            no_regression=False,
            is_plausible=False,
            compile_error=str(exc),
            runtime_error=None,
            execution_time_ms=elapsed_ms,
        )

    sandbox = ExecutionSandbox(
        project_root=bug_report.project_root,
        working_dir=config.working_dir,
        python_executable=config.python_executable,
    )
    sandbox.state_id = state_id or hashlib.sha256(patched_source.encode("utf-8")).hexdigest()[:8]

    workspace_path: str | None = None
    runtime_error: str | None = None
    collected: dict[str, tuple[bool, str]] = {}

    try:
        workspace_path = sandbox.setup_workspace(bug_report)
        sandbox.write_patched_file(workspace_path, bug_report.file_path, patched_source)

        failing_results: dict[str, tuple[bool, str]] = {}
        for index, test_id in enumerate(bug_report.failing_tests):
            result = sandbox.run_tests(
                workspace_path,
                [test_id],
                timeout_seconds=config.test_timeout_seconds,
            )
            failing_results.update(result)

            passed = result[test_id][0]
            if not passed:
                for pending in bug_report.failing_tests[index + 1 :]:
                    failing_results[pending] = (False, "short-circuited after failing test")
                break

        collected.update(failing_results)

        sampled_passing = _sample_passing_tests(
            passing_tests=bug_report.passing_tests,
            patched_source=patched_source,
            sample_cap=50,
        )

        passing_results: dict[str, tuple[bool, str]] = {}
        if sampled_passing:
            max_workers = min(
                max(1, config.parallel_validation_workers),
                len(sampled_passing),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(
                        sandbox.run_tests,
                        workspace_path,
                        [test_id],
                        config.test_timeout_seconds,
                    ): test_id
                    for test_id in sampled_passing
                }

                for future in as_completed(future_map):
                    test_id = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        passing_results[test_id] = (False, f"runtime_error: {exc}")
                        continue

                    passing_results.update(result)

        collected.update(passing_results)

    except RuntimeError as exc:
        runtime_error = str(exc)
    except Exception as exc:
        runtime_error = f"validation runtime error: {exc}"
    finally:
        if workspace_path is not None:
            sandbox.cleanup_workspace(workspace_path)

    tests_passed = sorted([test_id for test_id, (passed, _) in collected.items() if passed])
    tests_failed = sorted([test_id for test_id, (passed, _) in collected.items() if not passed])

    sampled_passing = _sample_passing_tests(
        passing_tests=bug_report.passing_tests,
        patched_source=patched_source,
        sample_cap=50,
    )

    all_failing_tests_pass = all(test in tests_passed for test in bug_report.failing_tests)
    no_regression = all(test in tests_passed for test in sampled_passing)
    plausible = all_failing_tests_pass and no_regression and runtime_error is None

    elapsed_ms = (time.monotonic() - start) * 1000.0

    return ValidationResult(
        state_id=state_id,
        compiled=True,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        all_failing_tests_pass=all_failing_tests_pass,
        no_regression=no_regression,
        is_plausible=plausible,
        compile_error=None,
        runtime_error=runtime_error,
        execution_time_ms=elapsed_ms,
    )


class ConcreteValidator(ValidatorProtocol):
    def __init__(self, config: VerifixConfig) -> None:
        self.config = config
        self.total_validations = 0

    def validate(self, source: str, bug_report: BugReport) -> ValidationResult:
        if self.total_validations >= self.config.max_validations:
            raise ValidationBudgetExceeded("Validation budget exceeded")

        self.total_validations += 1
        state_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        return validate_patch(
            patched_source=source,
            bug_report=bug_report,
            config=self.config,
            state_id=state_id,
        )


def _sample_passing_tests(
    passing_tests: list[str],
    patched_source: str,
    sample_cap: int,
) -> list[str]:
    if len(passing_tests) <= sample_cap:
        return list(passing_tests)

    seed = int(hashlib.sha256(patched_source.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return sorted(rng.sample(passing_tests, sample_cap))


__all__ = [
    "ExecutionSandbox",
    "validate_patch",
    "ConcreteValidator",
    "ValidationBudgetExceeded",
]
