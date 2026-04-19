from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.pipeline.repair_agent import RepairAgent
from verifix.pipeline.repair_agent_v2 import RepairAgentV2


logger = logging.getLogger("verifix.bugsinpy")


class BugsInPyLoader:
    def __init__(self, bugsinpy_root: str = ".data/BugsInPy") -> None:
        self.bugsinpy_root = Path(bugsinpy_root).resolve()
        self._checkout_cache: dict[tuple[str, int, str], str] = {}
        self._bug_cache: dict[tuple[str, int], BugReport] = {}

    def _parse_bug_info(self, info_path: str) -> dict[str, str]:
        path = Path(info_path)
        if not path.exists():
            raise FileNotFoundError(f"BugsInPy info file not found: {path}")

        data: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            data[key] = value

        return data

    def _checkout_bug(self, project: str, bug_id: int, workdir: str) -> str:
        cache_key = (project, int(bug_id), str(workdir))
        cached = self._checkout_cache.get(cache_key)
        if cached and Path(cached).exists():
            return cached

        workdir_path = Path(workdir)
        workdir_path.mkdir(parents=True, exist_ok=True)

        checkout_args = ["-p", project, "-i", str(bug_id), "-v", "0", "-w", str(workdir_path)]
        script_path = self.bugsinpy_root / "framework" / "bin" / "bugsinpy-checkout"

        candidate_commands: list[list[str]] = [["bugsinpy-checkout", *checkout_args]]
        if script_path.exists():
            candidate_commands.append([sys.executable, str(script_path), *checkout_args])
            candidate_commands.append(["bash", str(script_path), *checkout_args])

        failures: list[str] = []
        for command in candidate_commands:
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(self.bugsinpy_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except FileNotFoundError as exc:
                failures.append(f"{command[0]} not found: {exc}")
                continue
            except subprocess.TimeoutExpired:
                failures.append(f"Command timed out: {' '.join(command)}")
                continue

            if proc.returncode == 0:
                checkout_root = workdir_path / project
                if checkout_root.exists():
                    result = str(checkout_root.resolve())
                else:
                    result = str(workdir_path.resolve())
                self._checkout_cache[cache_key] = result
                return result

            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            failures.append(f"{' '.join(command)} failed: {stderr or stdout}")

        try:
            manual_checkout = self._checkout_via_git_metadata(project, bug_id, workdir_path)
            self._checkout_cache[cache_key] = manual_checkout
            return manual_checkout
        except Exception as exc:
            failures.append(f"metadata checkout fallback failed: {exc}")

        message = "BugsInPy checkout failed. Ensure framework tools are installed and callable.\n"
        message += "\n".join(failures)
        raise EnvironmentError(message)

    def _checkout_via_git_metadata(self, project: str, bug_id: int, workdir_path: Path) -> str:
        project_info_path = self.bugsinpy_root / "projects" / project / "project.info"
        bug_info_path = self._get_info_path(project, bug_id)

        project_info = self._parse_bug_info(str(project_info_path))
        bug_info = self._parse_bug_info(str(bug_info_path))

        github_url = project_info.get("github_url", "").strip()
        buggy_commit = bug_info.get("buggy_commit_id", "").strip()

        if not github_url:
            raise EnvironmentError("project.info missing github_url")
        if not buggy_commit:
            raise EnvironmentError("bug info missing buggy_commit_id")

        checkout_root = workdir_path / project
        if checkout_root.exists():
            shutil.rmtree(checkout_root, ignore_errors=True)
        workdir_path.mkdir(parents=True, exist_ok=True)

        repo_cache_root = Path(tempfile.gettempdir()) / "verifix_bugsinpy_repo_cache"
        repo_cache_root.mkdir(parents=True, exist_ok=True)
        cached_repo = repo_cache_root / project

        if not cached_repo.exists():
            cache_clone_proc = subprocess.run(
                ["git", "clone", github_url, str(cached_repo)],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if cache_clone_proc.returncode != 0:
                stderr = (cache_clone_proc.stderr or "").strip()
                stdout = (cache_clone_proc.stdout or "").strip()
                raise EnvironmentError(f"git cache clone failed: {stderr or stdout}")

        clone_proc = subprocess.run(
            ["git", "clone", "--shared", str(cached_repo), str(checkout_root)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if clone_proc.returncode != 0:
            stderr = (clone_proc.stderr or "").strip()
            stdout = (clone_proc.stdout or "").strip()
            raise EnvironmentError(f"git clone failed: {stderr or stdout}")

        checkout_proc = subprocess.run(
            ["git", "-C", str(checkout_root), "checkout", buggy_commit],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if checkout_proc.returncode != 0:
            stderr = (checkout_proc.stderr or "").strip()
            stdout = (checkout_proc.stdout or "").strip()
            raise EnvironmentError(f"git checkout failed: {stderr or stdout}")

        return str(checkout_root.resolve())

    def _get_failing_tests(self, project: str, bug_id: int) -> list[str]:
        info_path = self._get_info_path(project, bug_id)
        info = self._parse_bug_info(str(info_path))

        raw_cases = info.get("test_cases", "").strip()
        if raw_cases:
            normalized = raw_cases.replace(",", " ")
            tests = [token.strip() for token in normalized.split() if token.strip()]
            tests = [self._normalize_test_id(test) for test in tests]
            return [test for test in tests if test]

        run_test_path = info_path.parent / "run_test.sh"
        if run_test_path.exists():
            run_text = run_test_path.read_text(encoding="utf-8")
            tests = self._extract_tests_from_run_script(run_text)
            if tests:
                return tests

        test_file = info.get("test_file", "").strip()
        if test_file:
            return [f"{test_file}::test_placeholder"]

        return []

    def _get_passing_tests(
        self,
        project_root: str,
        failing_tests: list[str],
        python_exec: str = "python3",
    ) -> list[str]:
        try:
            proc = subprocess.run(
                [python_exec, "-m", "pytest", "--collect-only", "-q"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception:
            return []

        if proc.returncode not in {0, 1, 2, 5}:
            return []

        collected: list[str] = []
        for line in proc.stdout.splitlines():
            candidate = line.strip()
            if "::" not in candidate:
                continue
            if candidate.startswith("="):
                continue
            if "collected" in candidate.lower():
                continue
            collected.append(candidate)

        failing_set = {self._normalize_test_id(test) for test in failing_tests}
        output = [test for test in collected if self._normalize_test_id(test) not in failing_set]
        if len(output) > 200:
            output = output[:200]
        return output

    def load_bug(self, project: str, bug_id: int, workdir: str | None = None) -> BugReport:
        cache_key = (project, int(bug_id))
        if cache_key in self._bug_cache:
            return self._bug_cache[cache_key]

        if workdir is None:
            base = Path(tempfile.gettempdir()) / "verifix_bugsinpy" / f"{project}-{bug_id}"
            workdir = str(base)

        info_path = self._get_info_path(project, bug_id)
        info = self._parse_bug_info(str(info_path))

        checkout_path = self._checkout_bug(project, bug_id, workdir)

        patch_path = self.bugsinpy_root / "projects" / project / "bugs" / str(int(bug_id)) / "bug_patch.txt"
        python_path = self._resolve_python_path(info, patch_path)
        source_path = Path(checkout_path) / python_path
        if not source_path.exists():
            raise FileNotFoundError(f"Buggy source path not found in checkout: {source_path}")

        buggy_source = source_path.read_text(encoding="utf-8")
        failing_tests = self._get_failing_tests(project, bug_id)
        passing_tests = self._get_passing_tests(checkout_path, failing_tests, python_exec=sys.executable)

        report = BugReport(
            bug_id=f"BugsInPy-{project}-{int(bug_id)}",
            language="python",
            buggy_source=buggy_source,
            file_path=python_path,
            failing_tests=failing_tests,
            passing_tests=passing_tests,
            project_root=checkout_path,
            metadata={
                "project": project,
                "bug_id": int(bug_id),
                "bug_patch_path": str(patch_path),
                "buggy_commit": info.get("buggy_commit_id", ""),
                "fixed_commit": info.get("fixed_commit_id", ""),
                "bugsinpy_root": str(self.bugsinpy_root),
            },
        )
        self._bug_cache[cache_key] = report
        return report

    def load_project(self, project: str) -> list[BugReport]:
        bugs_root = self.bugsinpy_root / "projects" / project / "bugs"
        if not bugs_root.exists():
            return []

        reports: list[BugReport] = []
        for bug_dir in sorted(bugs_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
            if not bug_dir.is_dir() or not bug_dir.name.isdigit():
                continue
            bug_id = int(bug_dir.name)
            try:
                reports.append(self.load_bug(project, bug_id))
            except Exception as exc:
                logger.warning("Skipping BugsInPy bug %s-%s due to: %s", project, bug_id, exc)
                continue

        return sorted(reports, key=lambda report: int(report.metadata.get("bug_id", 0)))

    def load_all(
        self,
        projects: list[str] | None = None,
        max_bugs: int | None = None,
    ) -> list[BugReport]:
        available = self.get_available_projects()
        if projects is not None:
            project_filter = {item.strip() for item in projects if item.strip()}
            available = [project for project in available if project in project_filter]

        reports: list[BugReport] = []
        for project in available:
            reports.extend(self.load_project(project))
            if max_bugs is not None and len(reports) >= max_bugs:
                reports = reports[:max_bugs]
                break

        reports.sort(key=lambda report: report.bug_id)
        return reports

    def get_available_projects(self) -> list[str]:
        projects_root = self.bugsinpy_root / "projects"
        if not projects_root.exists():
            return []
        return sorted(path.name for path in projects_root.iterdir() if path.is_dir())

    def get_bug_count(self, project: str) -> int:
        bugs_root = self.bugsinpy_root / "projects" / project / "bugs"
        if not bugs_root.exists():
            return 0
        return sum(1 for path in bugs_root.iterdir() if path.is_dir() and path.name.isdigit())

    def _get_info_path(self, project: str, bug_id: int) -> Path:
        bug_root = self.bugsinpy_root / "projects" / project / "bugs" / str(int(bug_id))
        preferred = bug_root / "bugsinpy_bug.info"
        legacy = bug_root / "bug.info"
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
        raise FileNotFoundError(f"No BugsInPy info file found in {bug_root}")

    def _resolve_python_path(self, info: dict[str, str], patch_path: Path) -> str:
        direct = info.get("python_path", "").strip()
        if direct and direct.endswith(".py"):
            return direct

        maybe_path = info.get("pythonpath", "").strip()
        if maybe_path and maybe_path.endswith(".py"):
            return maybe_path

        if patch_path.exists():
            for line in patch_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if stripped.startswith("+++ b/"):
                    rel = stripped[len("+++ b/") :].strip()
                    if rel and rel != "/dev/null" and rel.endswith(".py"):
                        return rel
                if stripped.startswith("--- a/"):
                    rel = stripped[len("--- a/") :].strip()
                    if rel and rel != "/dev/null" and rel.endswith(".py"):
                        return rel

        raise ValueError("Unable to infer buggy Python source path from bug metadata")

    def _extract_tests_from_run_script(self, run_text: str) -> list[str]:
        tests: list[str] = []

        for token in run_text.replace("\n", " ").split():
            token = token.strip()
            if not token:
                continue

            if "::" in token and token.endswith(")"):
                token = token.rstrip(")")

            normalized = self._normalize_test_id(token)
            if "::" in normalized:
                tests.append(normalized)

        if tests:
            return sorted(dict.fromkeys(tests))

        unittest_matches = re.findall(r"[A-Za-z0-9_\.]+\.test_[A-Za-z0-9_]+", run_text)
        for match in unittest_matches:
            normalized = self._normalize_test_id(match)
            if "::" in normalized:
                tests.append(normalized)

        return sorted(dict.fromkeys(tests))

    def _normalize_test_id(self, raw: str) -> str:
        token = raw.strip().strip('"').strip("'")
        token = token.rstrip(";")
        token = token.strip()

        if token.startswith("pytest") or token.startswith("python"):
            return ""

        if "::" in token:
            return token

        if token.endswith(".py"):
            return f"{token}::test_placeholder"

        if "/" in token:
            return token

        if "." in token and "test" in token:
            parts = token.split(".")
            if len(parts) >= 3:
                method = parts[-1]
                owner = parts[-2]
                module_parts = parts[:-2]
                file_path = "/".join(module_parts) + ".py"
                if owner.startswith("Test"):
                    return f"{file_path}::{owner}::{method}"
                return f"{file_path}::{owner}"
            if len(parts) == 2:
                return f"{parts[0]}.py::{parts[1]}"

        return token


class BugsInPyBenchmark:
    def __init__(self, bugsinpy_root: str = ".data/BugsInPy", config: VerifixConfig | None = None) -> None:
        self.loader = BugsInPyLoader(bugsinpy_root)
        self.config = VerifixConfig(**DEFAULT_CONFIG.model_dump()) if config is None else config

    def run(
        self,
        project_filter: list[str] | None = None,
        max_bugs: int | None = None,
        output_dir: str = ".bugsinpy_results",
        use_v2: bool = True,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        bugs = self.loader.load_all(projects=project_filter, max_bugs=max_bugs)

        v1_agent = RepairAgent(self.config)
        v2_agent = RepairAgentV2(self.config)

        total = len(bugs)
        plausible_count = 0
        correct_count = 0
        time_sum = 0.0
        validation_sum = 0
        terminated_by_budget = 0

        by_project: dict[str, dict[str, Any]] = {}
        per_bug: dict[str, dict[str, Any]] = {}

        for bug in bugs:
            project_name = str(bug.metadata.get("project", "unknown"))
            by_project.setdefault(project_name, {"total": 0, "plausible": 0, "correct": 0, "repair_rate": 0.0})
            by_project[project_name]["total"] += 1

            if use_v2:
                result_v2 = v2_agent.repair(bug)
                repair_result = result_v2.v1_result
                success = bool(result_v2.success)
                plausible = len(repair_result.ranked_patches) > 0
                time_seconds = float(result_v2.total_wall_time_seconds)
                validations = int(repair_result.total_validations_run)
                top_patch_score = result_v2.best_evidence.trust_score if result_v2.best_evidence else None
                payload: dict[str, Any] = result_v2.to_dict()
            else:
                repair_result = v1_agent.repair(bug)
                success = bool(repair_result.success)
                plausible = len(repair_result.ranked_patches) > 0
                time_seconds = float(repair_result.wall_time_seconds)
                validations = int(repair_result.total_validations_run)
                top_patch_score = repair_result.ranked_patches[0].score if repair_result.ranked_patches else None
                payload = repair_result.to_dict()

            correct = self.compute_correct(repair_result, str(bug.metadata.get("bug_patch_path", "")))

            if plausible:
                plausible_count += 1
                by_project[project_name]["plausible"] += 1
            if correct:
                correct_count += 1
                by_project[project_name]["correct"] += 1

            if validations >= self.config.max_validations:
                terminated_by_budget += 1

            time_sum += time_seconds
            validation_sum += validations

            record = {
                "success": success,
                "plausible": plausible,
                "correct": correct,
                "time_seconds": time_seconds,
                "validations": validations,
                "top_patch_score": top_patch_score,
            }
            per_bug[bug.bug_id] = record

            save_payload = {
                "bug_id": bug.bug_id,
                "project": project_name,
                "result": payload,
                "metrics": record,
            }
            (output_path / f"{bug.bug_id}.json").write_text(json.dumps(save_payload, indent=2), encoding="utf-8")

        for project_name, stats in by_project.items():
            total_project = int(stats["total"])
            stats["repair_rate"] = (stats["correct"] / total_project) if total_project else 0.0

        summary = {
            "total_bugs": total,
            "plausible_patches": plausible_count,
            "correct_patches": correct_count,
            "plausible_rate": (plausible_count / total) if total else 0.0,
            "correct_rate": (correct_count / total) if total else 0.0,
            "avg_time_seconds": (time_sum / total) if total else 0.0,
            "avg_validations": (validation_sum / total) if total else 0.0,
            "terminated_by_budget": terminated_by_budget,
            "by_project": by_project,
            "per_bug": per_bug,
        }

        (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def compute_correct(self, result: RepairResult, bug_patch_path: str) -> bool:
        if not result.ranked_patches:
            return False

        patch_file = Path(bug_patch_path)
        if not patch_file.exists():
            return False

        added_lines: list[str] = []
        for line in patch_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("+++"):
                continue
            if not line.startswith("+"):
                continue
            normalized = _normalize_ws(line[1:])
            if normalized:
                added_lines.append(normalized)

        if not added_lines:
            return False

        patched_source = _normalize_ws(result.ranked_patches[0].patched_source)
        return all(fragment in patched_source for fragment in added_lines)

    def print_leaderboard(self, summary: dict[str, Any]) -> None:
        rows = sorted(
            summary.get("by_project", {}).items(),
            key=lambda item: float(item[1].get("repair_rate", 0.0)),
            reverse=True,
        )

        print("Project      | Bugs | Plausible | Correct | Rate")
        print("-------------|------|-----------|---------|------")

        for project, stats in rows:
            total = int(stats.get("total", 0))
            plausible = int(stats.get("plausible", 0))
            correct = int(stats.get("correct", 0))
            rate = float(stats.get("repair_rate", 0.0)) * 100.0
            print(f"{project:<12} | {total:4d} | {plausible:9d} | {correct:7d} | {rate:4.1f}%")

        total_bugs = int(summary.get("total_bugs", 0))
        plausible_total = int(summary.get("plausible_patches", 0))
        correct_total = int(summary.get("correct_patches", 0))
        total_rate = float(summary.get("correct_rate", 0.0)) * 100.0
        print(f"TOTAL        | {total_bugs:4d} | {plausible_total:9d} | {correct_total:7d} | {total_rate:4.1f}%")


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


__all__ = ["BugsInPyLoader", "BugsInPyBenchmark"]
