from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestCoverageRecord:
    __test__ = False

    test_id: str
    passed: bool
    covered_lines: frozenset[int]


@dataclass(frozen=True)
class SuspiciousnessScore:
    line: int
    score: float
    algorithm: str
    covered_by_failing: int
    covered_by_passing: int


def collect_coverage(
    project_root: str,
    source_file: str,
    test_ids: list[str],
    python_executable: str = sys.executable,
    timeout_seconds: float = 30.0,
) -> list[TestCoverageRecord]:
    root_path = Path(project_root)
    source_path = _resolve_source_path(root_path, source_file)
    source_arg = _coverage_source_arg(source_file, source_path)

    records: list[TestCoverageRecord] = []
    for test_id in test_ids:
        covered_lines: frozenset[int] = frozenset()
        passed = False

        with tempfile.TemporaryDirectory(prefix="verifix_cov_") as temp_dir:
            temp_path = Path(temp_dir)
            cov_data_file = temp_path / ".coverage"
            cov_json_file = temp_path / "coverage.json"

            run_cmd = [
                python_executable,
                "-m",
                "coverage",
                "run",
                f"--data-file={cov_data_file}",
                f"--source={source_arg}",
                "-m",
                "pytest",
                test_id,
                "--tb=no",
                "-q",
            ]

            try:
                run_proc = subprocess.run(
                    run_cmd,
                    cwd=str(root_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                passed = run_proc.returncode == 0
            except subprocess.TimeoutExpired:
                records.append(TestCoverageRecord(test_id=test_id, passed=False, covered_lines=frozenset()))
                continue

            json_cmd = [
                python_executable,
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_data_file}",
                "-o",
                str(cov_json_file),
            ]

            subprocess.run(
                json_cmd,
                cwd=str(root_path),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )

            if cov_json_file.exists():
                covered_lines = _extract_covered_lines(cov_json_file, root_path, source_path)

        records.append(TestCoverageRecord(test_id=test_id, passed=passed, covered_lines=covered_lines))

    return records


def ochiai_score(records: list[TestCoverageRecord]) -> list[SuspiciousnessScore]:
    all_lines = _all_covered_lines(records)
    total_failed = sum(1 for rec in records if not rec.passed)

    scores: list[SuspiciousnessScore] = []
    for line in sorted(all_lines):
        ef = sum(1 for rec in records if not rec.passed and line in rec.covered_lines)
        ep = sum(1 for rec in records if rec.passed and line in rec.covered_lines)
        nf = total_failed - ef

        if ef == 0:
            score = 0.0
        else:
            denominator = math.sqrt((ef + nf) * (ef + ep))
            score = ef / denominator if denominator > 0 else 0.0

        scores.append(
            SuspiciousnessScore(
                line=line,
                score=score,
                algorithm="ochiai",
                covered_by_failing=ef,
                covered_by_passing=ep,
            )
        )

    return sorted(scores, key=lambda item: (-item.score, item.line))


def tarantula_score(records: list[TestCoverageRecord]) -> list[SuspiciousnessScore]:
    total_failed = sum(1 for rec in records if not rec.passed)
    if total_failed == 0:
        return []

    total_passed = sum(1 for rec in records if rec.passed)
    all_lines = _all_covered_lines(records)

    scores: list[SuspiciousnessScore] = []
    for line in sorted(all_lines):
        ef = sum(1 for rec in records if not rec.passed and line in rec.covered_lines)
        ep = sum(1 for rec in records if rec.passed and line in rec.covered_lines)

        fail_ratio = ef / total_failed
        pass_ratio = (ep / total_passed) if total_passed > 0 else 0.0
        denominator = fail_ratio + pass_ratio
        score = (fail_ratio / denominator) if denominator > 0 else 0.0

        scores.append(
            SuspiciousnessScore(
                line=line,
                score=score,
                algorithm="tarantula",
                covered_by_failing=ef,
                covered_by_passing=ep,
            )
        )

    return sorted(scores, key=lambda item: (-item.score, item.line))


def localize_faults(
    project_root: str,
    source_file: str,
    failing_tests: list[str],
    passing_tests: list[str],
    algorithm: str = "ochiai",
    top_n: int = 10,
    python_executable: str = sys.executable,
    existing_coverage: list[TestCoverageRecord] | None = None,
    cache_dir: str | None = None,
) -> list[SuspiciousnessScore]:
    if top_n <= 0:
        return []

    # QuixBugs-like workloads often have failing tests but no passing tests,
    # which makes coverage-based Ochiai effectively uninformative while being expensive.
    if existing_coverage is None and failing_tests and not passing_tests:
        return _heuristic_lines(project_root, source_file, top_n, algorithm.lower())

    cache_path = _cache_path(cache_dir=cache_dir, project_root=project_root, source_file=source_file)

    if existing_coverage is not None:
        records = existing_coverage
    elif cache_path is not None and cache_path.exists():
        records = _read_cached_records(cache_path)
    else:
        all_tests = [*failing_tests, *passing_tests]
        records = collect_coverage(
            project_root=project_root,
            source_file=source_file,
            test_ids=all_tests,
            python_executable=python_executable,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_cached_records(cache_path, records)

    normalized_algorithm = algorithm.lower()
    if normalized_algorithm == "ochiai":
        scores = ochiai_score(records)
    elif normalized_algorithm == "tarantula":
        scores = tarantula_score(records)
    else:
        raise ValueError(f"Unsupported fault localization algorithm: {algorithm}")

    if not scores or all(item.score == 0.0 for item in scores):
        return _fallback_lines(project_root, source_file, top_n, normalized_algorithm)

    return scores[:top_n]


def _resolve_source_path(project_root: Path, source_file: str) -> Path:
    source_path = Path(source_file)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    return source_path.resolve()


def _coverage_source_arg(source_file: str, source_path: Path) -> str:
    source_input = Path(source_file)
    if source_input.suffix == ".py":
        return str(source_path.parent)
    return source_file


def _extract_covered_lines(cov_json_file: Path, project_root: Path, source_path: Path) -> frozenset[int]:
    try:
        payload = json.loads(cov_json_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return frozenset()

    files = payload.get("files", {})
    if not isinstance(files, dict):
        return frozenset()

    for file_name, file_payload in files.items():
        candidate = Path(file_name)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()

        if candidate == source_path or candidate.name == source_path.name:
            executed = file_payload.get("executed_lines", []) if isinstance(file_payload, dict) else []
            return frozenset(int(line) for line in executed)

    return frozenset()


def _all_covered_lines(records: list[TestCoverageRecord]) -> set[int]:
    covered: set[int] = set()
    for rec in records:
        covered.update(rec.covered_lines)
    return covered


def _fallback_lines(
    project_root: str,
    source_file: str,
    top_n: int,
    algorithm: str,
) -> list[SuspiciousnessScore]:
    source_path = _resolve_source_path(Path(project_root), source_file)
    content = source_path.read_text(encoding="utf-8")
    total_lines = len(content.splitlines())
    limit = min(top_n, total_lines)

    return [
        SuspiciousnessScore(
            line=line_no,
            score=0.0,
            algorithm=algorithm,
            covered_by_failing=0,
            covered_by_passing=0,
        )
        for line_no in range(1, limit + 1)
    ]


def _heuristic_lines(
    project_root: str,
    source_file: str,
    top_n: int,
    algorithm: str,
) -> list[SuspiciousnessScore]:
    source_path = _resolve_source_path(Path(project_root), source_file)
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    if not lines:
        return []

    structural_scores: dict[int, float] = {idx + 1: 0.0 for idx in range(len(lines))}
    token_counts: dict[int, int] = {}

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            token_counts[idx] = 0
            continue
        token_counts[idx] = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\+|\-|\*|/|%|\^|&|\|", stripped))

    try:
        import ast

        tree = ast.parse(source)
        weighted_nodes = (
            ast.Assign,
            ast.AugAssign,
            ast.Return,
            ast.If,
            ast.While,
            ast.For,
            ast.Compare,
            ast.BinOp,
            ast.BoolOp,
            ast.Call,
            ast.Subscript,
        )

        for node in ast.walk(tree):
            lineno = int(getattr(node, "lineno", 0) or 0)
            if lineno <= 0 or lineno not in structural_scores:
                continue
            if isinstance(node, weighted_nodes):
                structural_scores[lineno] += 1.0
    except Exception:
        pass

    non_zero_tokens = [count for count in token_counts.values() if count > 0]
    median_tokens = sorted(non_zero_tokens)[len(non_zero_tokens) // 2] if non_zero_tokens else 0

    ranked: list[tuple[float, int]] = []
    for line_no in range(1, len(lines) + 1):
        token_anomaly = abs(token_counts.get(line_no, 0) - median_tokens)
        score = structural_scores.get(line_no, 0.0) + (0.05 * token_anomaly)
        ranked.append((score, line_no))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[: min(top_n, len(ranked))]
    return [
        SuspiciousnessScore(
            line=line_no,
            score=score,
            algorithm=algorithm,
            covered_by_failing=0,
            covered_by_passing=0,
        )
        for score, line_no in selected
    ]


def _cache_path(cache_dir: str | None, project_root: str, source_file: str) -> Path | None:
    if cache_dir is None:
        return None

    source_path = _resolve_source_path(Path(project_root), source_file)
    cache_key = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{cache_key}.json"


def _write_cached_records(path: Path, records: list[TestCoverageRecord]) -> None:
    payload = [
        {
            "test_id": rec.test_id,
            "passed": rec.passed,
            "covered_lines": sorted(rec.covered_lines),
        }
        for rec in records
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_cached_records(path: Path) -> list[TestCoverageRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    records: list[TestCoverageRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        records.append(
            TestCoverageRecord(
                test_id=str(item.get("test_id", "")),
                passed=bool(item.get("passed", False)),
                covered_lines=frozenset(int(line) for line in item.get("covered_lines", [])),
            )
        )
    return records


__all__ = [
    "TestCoverageRecord",
    "SuspiciousnessScore",
    "collect_coverage",
    "ochiai_score",
    "tarantula_score",
    "localize_faults",
]
