from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from verifix.core.config import QuixBugsConfig, VerifixConfig
from verifix.core.models import BugReport
from verifix.benchmarks.quixbugs_split import DEFAULT_SPLIT_SEED, split_from_mode
from verifix.pipeline.repair_agent import RepairAgent


class QuixBugsLoader:
    def __init__(self, quixbugs_root: str) -> None:
        self.root = Path(quixbugs_root).resolve()
        self.python_programs = self.root / "python_programs"
        self.python_testcases = self.root / "python_testcases"
        self.json_testcases = self.root / "json_testcases"
        self._workspaces: list[Path] = []

    def load_program(self, program_name: str, language: str = "python") -> BugReport:
        if language.lower() != "python":
            raise NotImplementedError("QuixBugsLoader currently supports only Python programs")

        buggy_file = self.python_programs / f"{program_name}.py"
        testcase_file = self._resolve_testcase_file(program_name)
        python_test_module = self._resolve_python_test_module(program_name)

        if not buggy_file.exists():
            raise FileNotFoundError(f"Buggy program not found: {buggy_file}")
        if testcase_file is None and python_test_module is None:
            raise FileNotFoundError(f"No test case source found for: {program_name}")

        buggy_source = buggy_file.read_text(encoding="utf-8")

        json_report: BugReport | None = None
        if testcase_file is not None and testcase_file.exists():
            testcase_text = testcase_file.read_text(encoding="utf-8")
            workspace = Path(tempfile.mkdtemp(prefix=f"verifix_quixbugs_{program_name}_"))
            self._workspaces.append(workspace)

            workspace_program = workspace / f"{program_name}.py"
            workspace_program.write_text(buggy_source, encoding="utf-8")

            generated_test_file = Path(
                self.generate_test_file(
                    program_name=program_name,
                    testcases_json=testcase_text,
                    output_dir=str(workspace),
                )
            )

            passing, failing = self._infer_test_outcomes(
                project_root=workspace,
                test_file=generated_test_file,
            )

            json_report = BugReport(
                bug_id=f"QuixBugs-{program_name}",
                language="python",
                buggy_source=buggy_source,
                file_path=f"{program_name}.py",
                failing_tests=failing,
                passing_tests=passing,
                project_root=str(workspace),
                metadata={
                    "benchmark": "QuixBugs",
                    "program_name": program_name,
                    "generated_test_file": generated_test_file.name,
                    "test_source": "json",
                    "quixbugs_root": str(self.root),
                },
            )

            if failing:
                return json_report

        if python_test_module is not None and python_test_module.exists():
            workspace = Path(tempfile.mkdtemp(prefix=f"verifix_quixbugs_{program_name}_pytests_"))
            self._workspaces.append(workspace)

            self._prepare_python_test_workspace(
                workspace=workspace,
                program_name=program_name,
                buggy_source=buggy_source,
                python_test_module=python_test_module,
            )

            rel_test_file = Path("python_testcases") / python_test_module.name
            passing, failing = self._infer_test_outcomes(
                project_root=workspace,
                test_file=rel_test_file,
            )

            return BugReport(
                bug_id=f"QuixBugs-{program_name}",
                language="python",
                buggy_source=buggy_source,
                file_path=f"python_programs/{program_name}.py",
                failing_tests=failing,
                passing_tests=passing,
                project_root=str(workspace),
                metadata={
                    "benchmark": "QuixBugs",
                    "program_name": program_name,
                    "generated_test_file": str(rel_test_file).replace("\\", "/"),
                    "test_source": "python_test_module",
                    "quixbugs_root": str(self.root),
                },
            )

        if json_report is not None:
            return json_report

        raise FileNotFoundError(f"No runnable test source found for: {program_name}")

    def load_all(self, language: str = "python") -> list[BugReport]:
        if language.lower() != "python":
            raise NotImplementedError("QuixBugsLoader currently supports only Python programs")

        reports: list[BugReport] = []
        if not self.python_programs.exists():
            return reports

        for program_file in sorted(self.python_programs.glob("*.py")):
            program_name = program_file.stem
            testcase_file = self._resolve_testcase_file(program_name)
            python_test_module = self._resolve_python_test_module(program_name)
            if testcase_file is None and python_test_module is None:
                continue

            try:
                reports.append(self.load_program(program_name, language=language))
            except Exception:
                continue

        return reports

    def generate_test_file(self, program_name: str, testcases_json: str, output_dir: str) -> str:
        json_path = Path(testcases_json)
        if json_path.exists():
            raw_json = json_path.read_text(encoding="utf-8")
        else:
            raw_json = testcases_json

        testcases = self._parse_testcases(raw_json)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        test_file = output / f"test_{program_name}_auto.py"

        params: list[str] = []
        for idx, case in enumerate(testcases):
            if not isinstance(case, dict) or "input" not in case or "output" not in case:
                continue
            params.append(
                f"    pytest.param({case['input']!r}, {case['output']!r}, id='case_{idx}')"
            )

        if not params:
            params.append("    pytest.param([], None, id='case_empty')")

        content = "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "sys.path.insert(0, str(Path(__file__).resolve().parent))",
                f"from {program_name} import {program_name}",
                "import pytest",
                "",
                "@pytest.mark.parametrize(\"inputs, expected\", [",
                ",\n".join(params),
                "])",
                f"def test_{program_name}(inputs, expected):",
                f"    result = {program_name}(*inputs)",
                "    assert result == expected",
                "",
            ]
        )

        ast.parse(content)
        test_file.write_text(content, encoding="utf-8")
        return str(test_file)

    def _parse_testcases(self, raw_json: str) -> list[dict[str, object]]:
        parsed_cases: list[dict[str, object]] = []

        try:
            payload = json.loads(raw_json)
            if isinstance(payload, list):
                parsed_cases.extend(self._normalize_cases(payload))
                if parsed_cases:
                    return parsed_cases
        except json.JSONDecodeError:
            pass

        for line in raw_json.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            parsed_cases.extend(self._normalize_cases([entry]))

        if not parsed_cases:
            raise ValueError("Unable to parse QuixBugs testcases JSON format")

        return parsed_cases

    def _normalize_cases(self, cases: list[object]) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for case in cases:
            if isinstance(case, dict) and "input" in case and "output" in case:
                inputs = case["input"]
                output = case["output"]
                normalized.append({"input": inputs, "output": output})
                continue

            if isinstance(case, list) and len(case) == 2:
                inputs = case[0]
                output = case[1]
                normalized.append({"input": inputs, "output": output})

        return normalized

    def _infer_test_outcomes(self, project_root: Path, test_file: Path) -> tuple[list[str], list[str]]:
        normalized_test_file = test_file
        if normalized_test_file.is_absolute():
            try:
                normalized_test_file = normalized_test_file.relative_to(project_root)
            except ValueError:
                normalized_test_file = Path(normalized_test_file.name)

        test_file_arg = str(normalized_test_file).replace("\\", "/")
        junit_xml = project_root / "quixbugs_test_report.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            test_file_arg,
            "-q",
            "--tb=short",
            "--no-header",
            f"--junitxml={junit_xml}",
        ]

        try:
            subprocess.run(
                command,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception:
            return [], []

        if not junit_xml.exists():
            return [], []

        tree = ET.parse(junit_xml)
        root = tree.getroot()

        passing: list[str] = []
        failing: list[str] = []

        for testcase in root.iter("testcase"):
            test_name = testcase.attrib.get("name", "")
            if not test_name:
                continue
            node_id = f"{test_file_arg}::{test_name}"
            failed = testcase.find("failure") is not None or testcase.find("error") is not None
            if failed:
                failing.append(node_id)
            else:
                passing.append(node_id)

        return sorted(passing), sorted(failing)

    def _resolve_testcase_file(self, program_name: str) -> Path | None:
        candidates = [
            self.python_testcases / f"{program_name}.json",
            self.json_testcases / f"{program_name}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_python_test_module(self, program_name: str) -> Path | None:
        candidate = self.python_testcases / f"test_{program_name}.py"
        if candidate.exists():
            return candidate
        return None

    def _prepare_python_test_workspace(
        self,
        workspace: Path,
        program_name: str,
        buggy_source: str,
        python_test_module: Path,
    ) -> None:
        python_programs_dst = workspace / "python_programs"
        correct_programs_dst = workspace / "correct_python_programs"
        json_testcases_dst = workspace / "json_testcases"
        python_testcases_dst = workspace / "python_testcases"

        shutil.copytree(self.python_programs, python_programs_dst)
        if (self.root / "correct_python_programs").exists():
            shutil.copytree(self.root / "correct_python_programs", correct_programs_dst)
        if self.json_testcases.exists():
            shutil.copytree(self.json_testcases, json_testcases_dst)

        python_testcases_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(python_test_module, python_testcases_dst / python_test_module.name)

        helper_files = [
            path
            for path in self.python_testcases.glob("*.py")
            if not path.name.startswith("test_")
        ]
        for helper in helper_files:
            shutil.copy2(helper, workspace / helper.name)

        root_conftest = self.root / "conftest.py"
        if root_conftest.exists():
            shutil.copy2(root_conftest, workspace / "conftest.py")

        (python_programs_dst / f"{program_name}.py").write_text(buggy_source, encoding="utf-8")


class QuixBugsBenchmark:
    def __init__(self, quixbugs_root: str, config: VerifixConfig | None = None) -> None:
        self.loader = QuixBugsLoader(quixbugs_root)
        self.config = QuixBugsConfig() if config is None else config
        self.agent = RepairAgent(config=self.config)

    def run_single(self, program_name: str) -> dict:
        start = time.monotonic()
        try:
            bug_report = self.loader.load_program(program_name, language="python")
            repair_result = self.agent.repair(bug_report)
            top_score = repair_result.ranked_patches[0].score if repair_result.ranked_patches else 0.0
            return {
                "program": program_name,
                "bug_id": repair_result.bug_id,
                "success": repair_result.success,
                "time": repair_result.wall_time_seconds,
                "top_patch_score": top_score,
                "validations": repair_result.total_validations_run,
                "result": repair_result.to_dict(),
                "error": repair_result.error,
            }
        except Exception as exc:
            return {
                "program": program_name,
                "bug_id": f"QuixBugs-{program_name}",
                "success": False,
                "time": time.monotonic() - start,
                "top_patch_score": 0.0,
                "validations": 0,
                "result": None,
                "error": str(exc),
            }

    def run_all(
        self,
        max_programs: int | None = None,
        parallel: bool = False,
        output_dir: str = "./quixbugs_results",
        split_mode: str = "all",
        split_seed: int = DEFAULT_SPLIT_SEED,
        selected_programs: list[str] | None = None,
    ) -> dict:
        reports = self.loader.load_all(language="python")
        dataset_program_names = [
            str(report.metadata.get("program_name", report.bug_id)) for report in reports
        ]

        if selected_programs is not None:
            available = set(dataset_program_names)
            selected = [name for name in selected_programs if name in available]
        else:
            sources = {
                str(report.metadata.get("program_name", report.bug_id)): report.buggy_source
                for report in reports
            }
            split_selection = split_from_mode(split_mode, sources, seed=split_seed)
            selected = list(split_selection.train_programs)

        if max_programs is not None:
            selected = selected[:max_programs]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []

        if parallel and selected:
            max_workers = min(4, len(selected))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.run_single, name): name for name in selected}
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for name in selected:
                results.append(self.run_single(name))

        for item in results:
            file_name = f"{item['program']}.json"
            (output_path / file_name).write_text(json.dumps(item, indent=2), encoding="utf-8")

        repaired = sum(1 for item in results if item.get("success"))
        attempted_total = len(results)
        dataset_total = len(dataset_program_names)
        avg_time = sum(float(item.get("time", 0.0)) for item in results) / attempted_total if results else 0.0
        avg_validations = (
            sum(float(item.get("validations", 0.0)) for item in results) / attempted_total if results else 0.0
        )

        per_program = {
            item["program"]: {
                "success": bool(item.get("success", False)),
                "time": float(item.get("time", 0.0)),
                "top_patch_score": float(item.get("top_patch_score", 0.0)),
            }
            for item in results
        }

        summary = {
            "total": attempted_total,
            "attempted_total": attempted_total,
            "dataset_total": dataset_total,
            "repaired": repaired,
            "repair_rate": (repaired / attempted_total) if attempted_total else 0.0,
            "dataset_repair_rate": (repaired / dataset_total) if dataset_total else 0.0,
            "avg_time_seconds": avg_time,
            "avg_validations": avg_validations,
            "split_mode": split_mode,
            "split_seed": split_seed,
            "per_program": per_program,
        }

        (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def print_leaderboard(self, summary: dict) -> None:
        per_program = summary.get("per_program", {})
        rows = sorted(
            per_program.items(),
            key=lambda item: (
                not item[1].get("success", False),
                -float(item[1].get("top_patch_score", 0.0)),
                item[0],
            ),
        )

        print("Program                     Success   Score   Time(s)")
        print("------------------------------------------------------")
        for name, metrics in rows:
            print(
                f"{name:26} {str(metrics.get('success', False)):7} "
                f"{metrics.get('top_patch_score', 0.0):6.2f} {metrics.get('time', 0.0):8.2f}"
            )


__all__ = ["QuixBugsLoader", "QuixBugsBenchmark"]
