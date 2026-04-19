from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig
from verifix.core.models import BugReport, RepairResult
from verifix.pipeline.repair_agent import RepairAgent


class Defects4JLoader:
    def __init__(self, d4j_root: str, format: str = "d4j_python") -> None:
        normalized = format.lower()
        if normalized not in {"d4j_python", "d4j_java"}:
            raise ValueError("format must be one of: d4j_python, d4j_java")

        self.root = Path(d4j_root).resolve()
        self.format = normalized

    def load_bug(self, bug_id: str) -> BugReport:
        if self.format == "d4j_python":
            return self._load_d4py_bug(bug_id)
        return self._load_d4j_java_bug(bug_id)

    def load_all(self, projects: list[str] | None = None) -> list[BugReport]:
        project_filter = {p.lower() for p in projects} if projects else None
        reports: list[BugReport] = []

        if not self.root.exists():
            return reports

        if self.format == "d4j_python":
            candidates = sorted(path for path in self.root.iterdir() if path.is_dir())
        else:
            candidates = sorted(path for path in self.root.iterdir() if path.is_dir() and (path / "metadata.json").exists())

        for bug_dir in candidates:
            bug_id = bug_dir.name
            try:
                report = self.load_bug(bug_id)
            except Exception:
                continue

            project_name = str(report.metadata.get("project", "")).lower()
            if project_filter and project_name not in project_filter:
                continue
            reports.append(report)

        return sorted(reports, key=lambda item: item.bug_id)

    def _load_d4py_bug(self, bug_id: str) -> BugReport:
        bug_dir = self.root / bug_id
        if not bug_dir.exists():
            raise FileNotFoundError(f"Bug directory not found: {bug_dir}")

        buggy_file = bug_dir / "buggy.py"
        failing_file = bug_dir / "failing_tests.txt"
        passing_file = bug_dir / "passing_tests.txt"
        metadata_file = bug_dir / "metadata.json"

        for required in [buggy_file, failing_file, passing_file, metadata_file]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required file: {required}")

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata JSON for bug {bug_id}")

        fixed_path = bug_dir / "fixed.py"
        metadata_payload = dict(metadata)
        metadata_payload["format"] = "d4j_python"
        metadata_payload["bug_dir"] = str(bug_dir)
        if fixed_path.exists():
            metadata_payload["fixed_source_path"] = str(fixed_path)

        return BugReport(
            bug_id=str(metadata_payload.get("bug_id", bug_id)),
            language="python",
            buggy_source=buggy_file.read_text(encoding="utf-8"),
            file_path=str(metadata_payload.get("file_path", "buggy.py")),
            failing_tests=_read_test_list(failing_file),
            passing_tests=_read_test_list(passing_file),
            project_root=str(bug_dir),
            metadata=metadata_payload,
        )

    def _load_d4j_java_bug(self, bug_id: str) -> BugReport:
        metadata = self._load_java_metadata(bug_id)
        project = str(metadata.get("project") or bug_id.split("-")[0])
        version = str(metadata.get("version") or bug_id.split("-")[-1])

        workspace = Path(tempfile.mkdtemp(prefix=f"verifix_d4j_{bug_id}_"))

        checkout_cmd = ["defects4j", "checkout", "-p", project, "-v", f"{version}b", "-w", str(workspace)]
        proc = subprocess.run(checkout_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            shutil.rmtree(workspace, ignore_errors=True)
            raise RuntimeError(f"defects4j checkout failed: {proc.stderr.strip()}")

        source_rel = str(metadata.get("file_path", ""))
        source_text = ""
        if source_rel:
            source_path = workspace / source_rel
            if source_path.exists():
                source_text = source_path.read_text(encoding="utf-8", errors="ignore")

        if not source_text:
            fallback = next((workspace / "src").rglob("*.java"), None)
            if fallback is not None:
                source_rel = fallback.relative_to(workspace).as_posix()
                source_text = fallback.read_text(encoding="utf-8", errors="ignore")

        export_cmd = ["defects4j", "export", "-p", "tests.trigger"]
        export_proc = subprocess.run(export_cmd, cwd=str(workspace), capture_output=True, text=True, check=False)
        failing_tests = [line.strip() for line in export_proc.stdout.splitlines() if line.strip()]

        metadata_payload = dict(metadata)
        metadata_payload["format"] = "d4j_java"
        metadata_payload["workspace"] = str(workspace)

        return BugReport(
            bug_id=bug_id,
            language="java",
            buggy_source=source_text,
            file_path=source_rel,
            failing_tests=failing_tests,
            passing_tests=[],
            project_root=str(workspace),
            metadata=metadata_payload,
        )

    def _load_java_metadata(self, bug_id: str) -> dict:
        metadata_file = self.root / bug_id / "metadata.json"
        if not metadata_file.exists():
            return {}
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return payload


class Defects4JBenchmark:
    def __init__(self, d4j_root: str, config: VerifixConfig | None = None) -> None:
        self.config = VerifixConfig(**DEFAULT_CONFIG.model_dump()) if config is None else config
        self.loader = Defects4JLoader(d4j_root=d4j_root, format="d4j_python")
        self.agent = RepairAgent(self.config)

    def run(
        self,
        bug_ids: list[str] | None = None,
        output_dir: str = "./defects4j_results",
        compare_with_baseline: dict | None = None,
    ) -> dict:
        if bug_ids:
            reports = [self.loader.load_bug(bug_id) for bug_id in bug_ids]
        else:
            reports = self.loader.load_all()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_bug: dict[str, dict] = {}
        plausible_count = 0
        correct_count = 0
        terminated_by_budget = 0
        total_time = 0.0
        total_validations = 0

        for report in reports:
            result = self.agent.repair(report)

            success = bool(result.success)
            plausible_count += 1 if success else 0

            fixed_source = str(report.metadata.get("fixed_source_path", ""))
            correct_patch = self.compute_correct_rate(result, fixed_source) if fixed_source else False
            correct_count += 1 if correct_patch else 0

            if result.total_validations_run >= self.config.max_validations:
                terminated_by_budget += 1

            top_score = result.ranked_patches[0].score if result.ranked_patches else 0.0

            payload = {
                "bug_id": report.bug_id,
                "success": success,
                "correct_patch": correct_patch,
                "time": result.wall_time_seconds,
                "validations": result.total_validations_run,
                "top_patch_score": top_score,
                "error": result.error,
            }

            if compare_with_baseline and report.bug_id in compare_with_baseline:
                payload["baseline"] = compare_with_baseline[report.bug_id]

            per_bug[report.bug_id] = payload
            (out_dir / f"{report.bug_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            total_time += result.wall_time_seconds
            total_validations += result.total_validations_run

        total_bugs = len(reports)
        avg_time = total_time / total_bugs if total_bugs else 0.0
        avg_valids = total_validations / total_bugs if total_bugs else 0.0

        summary = {
            "total_bugs": total_bugs,
            "plausible_patches": plausible_count,
            "correct_patches": correct_count,
            "plausible_rate": (plausible_count / total_bugs) if total_bugs else 0.0,
            "correct_rate": (correct_count / total_bugs) if total_bugs else 0.0,
            "avg_time_seconds": avg_time,
            "avg_validations": avg_valids,
            "terminated_by_budget": terminated_by_budget,
            "per_bug": per_bug,
        }

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def compute_correct_rate(self, result: RepairResult, fixed_source_path: str) -> bool:
        if not result.ranked_patches:
            return False

        fixed_path = Path(fixed_source_path)
        if not fixed_path.exists():
            return False

        fixed_source = fixed_path.read_text(encoding="utf-8")
        patched_source = result.ranked_patches[0].patched_source

        return _normalize_source(fixed_source) == _normalize_source(patched_source)


def create_d4py_dataset_from_examples(output_dir: str) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    examples = {
        "Example-1": {
            "project": "Example",
            "version": "1",
            "buggy": """def find_max(arr):\n    max_val = arr[0]\n    for i in range(1, len(arr)):\n        if arr[i] < max_val:\n            max_val = arr[i]\n    return max_val\n""",
            "fixed": """def find_max(arr):\n    max_val = arr[0]\n    for i in range(1, len(arr)):\n        if arr[i] > max_val:\n            max_val = arr[i]\n    return max_val\n""",
            "failing": [
                "test_buggy.py::test_fail_one",
                "test_buggy.py::test_fail_two",
            ],
            "passing": [
                "test_buggy.py::test_pass_one",
                "test_buggy.py::test_pass_two",
            ],
        },
        "Example-2": {
            "project": "Example",
            "version": "2",
            "buggy": """def bubble_sort(arr):\n    arr = list(arr)\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n""",
            "fixed": """def bubble_sort(arr):\n    arr = list(arr)\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n""",
            "failing": ["test_buggy.py::test_descending"],
            "passing": ["test_buggy.py::test_already_sorted"],
        },
        "Example-3": {
            "project": "Example",
            "version": "3",
            "buggy": """def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            hi = mid - 1\n        else:\n            lo = mid + 1\n    return -1\n""",
            "fixed": """def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n""",
            "failing": ["test_buggy.py::test_found"],
            "passing": ["test_buggy.py::test_not_found"],
        },
    }

    for bug_id, payload in examples.items():
        bug_dir = root / bug_id
        bug_dir.mkdir(parents=True, exist_ok=True)

        (bug_dir / "buggy.py").write_text(payload["buggy"], encoding="utf-8")
        (bug_dir / "fixed.py").write_text(payload["fixed"], encoding="utf-8")
        (bug_dir / "failing_tests.txt").write_text("\n".join(payload["failing"]) + "\n", encoding="utf-8")
        (bug_dir / "passing_tests.txt").write_text("\n".join(payload["passing"]) + "\n", encoding="utf-8")

        metadata = {
            "bug_id": bug_id,
            "project": payload["project"],
            "version": payload["version"],
            "file_path": "buggy.py",
        }
        (bug_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_test_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_source(source: str) -> str:
    lines = [line.rstrip() for line in source.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


__all__ = [
    "Defects4JLoader",
    "Defects4JBenchmark",
    "create_d4py_dataset_from_examples",
]
