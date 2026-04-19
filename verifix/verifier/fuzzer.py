from __future__ import annotations

import ast
import itertools
import json
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from verifix.core.config import VerifixConfig
from verifix.core.models import BugReport


@dataclass(frozen=False)
class FuzzTarget:
    function_name: str
    source: str
    signature: dict[str, str]
    test_cases: list[dict]


class FuzzStrategy(str, Enum):
    RANDOM = "random"
    BOUNDARY = "boundary"
    MUTATION = "mutation"
    PROPERTY = "property"


@dataclass(frozen=False)
class FuzzResult:
    survived: bool
    total_inputs_tested: int
    failing_inputs: list[dict]
    coverage_achieved: float
    fuzz_time_seconds: float
    strategy_used: str


def infer_signature(
    source: str,
    function_name: str,
    test_cases: list[dict] | None = None,
) -> dict[str, str]:
    tree = ast.parse(source)
    function_node = _find_function(tree, function_name)
    if function_node is None:
        raise ValueError(f"Function {function_name!r} not found in source")

    signature: dict[str, str] = {}
    for arg in function_node.args.args:
        signature[arg.arg] = _annotation_to_type(arg.annotation)

    if test_cases:
        inferred = _infer_param_types_from_cases(function_node, test_cases)
        for name, typ in inferred.items():
            if signature.get(name, "any") == "any" and typ != "any":
                signature[name] = typ

    for name, typ in list(signature.items()):
        if typ == "any" and name.lower() in {"arr", "array", "nums", "items", "lst", "values"}:
            signature[name] = "list"

    return signature


def generate_fuzz_inputs(
    fuzz_target: FuzzTarget,
    strategy: FuzzStrategy = FuzzStrategy.BOUNDARY,
    n_inputs: int = 50,
    seed: int = 42,
) -> list[list]:
    if n_inputs <= 0:
        return []

    param_types = list(fuzz_target.signature.values())
    if not param_types:
        return [[]]

    if strategy == FuzzStrategy.BOUNDARY:
        return _generate_boundary_inputs(param_types, n_inputs)
    if strategy == FuzzStrategy.RANDOM:
        return _generate_random_inputs(param_types, n_inputs, seed)
    if strategy == FuzzStrategy.MUTATION:
        return _generate_mutation_inputs(fuzz_target.test_cases, n_inputs, seed)
    if strategy == FuzzStrategy.PROPERTY:
        return _generate_random_inputs(param_types, n_inputs, seed + 17)

    return _generate_boundary_inputs(param_types, n_inputs)


def run_fuzz_test(
    fuzz_target: FuzzTarget,
    fuzz_inputs: list[list],
    reference_source: str | None = None,
    timeout_seconds: float = 5.0,
    working_dir: str = "/tmp/verifix_fuzz",
) -> FuzzResult:
    start = time.monotonic()

    if not fuzz_inputs:
        return FuzzResult(
            survived=True,
            total_inputs_tested=0,
            failing_inputs=[],
            coverage_achieved=0.0,
            fuzz_time_seconds=0.0,
            strategy_used="unknown",
        )

    work_root = Path(working_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="verifix_fuzz_", dir=str(work_root)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        patched_module = tmp_path / "patched_module.py"
        patched_module.write_text(fuzz_target.source, encoding="utf-8")

        use_reference = reference_source is not None
        if use_reference:
            reference_module = tmp_path / "reference_module.py"
            reference_module.write_text(reference_source or "", encoding="utf-8")

        inputs_path = tmp_path / "inputs.json"
        inputs_path.write_text(json.dumps(fuzz_inputs), encoding="utf-8")

        output_path = tmp_path / "result.json"
        runner_path = tmp_path / "runner.py"
        runner_path.write_text(
            _build_runner_script(
                function_name=fuzz_target.function_name,
                use_reference=use_reference,
            ),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(runner_path),
            str(inputs_path),
            str(output_path),
        ]

        try:
            proc = subprocess.run(
                command,
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return FuzzResult(
                survived=False,
                total_inputs_tested=len(fuzz_inputs),
                failing_inputs=[{"input": inp, "error": "timeout"} for inp in fuzz_inputs],
                coverage_achieved=0.0,
                fuzz_time_seconds=elapsed,
                strategy_used="unknown",
            )

        if proc.returncode != 0 or not output_path.exists():
            elapsed = time.monotonic() - start
            error_msg = (proc.stderr or proc.stdout or "runner_failed").strip()
            return FuzzResult(
                survived=False,
                total_inputs_tested=len(fuzz_inputs),
                failing_inputs=[{"input": inp, "error": error_msg} for inp in fuzz_inputs],
                coverage_achieved=0.0,
                fuzz_time_seconds=elapsed,
                strategy_used="unknown",
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        failing_inputs = payload.get("failing_inputs", []) if isinstance(payload, dict) else []
        signatures = payload.get("signatures", []) if isinstance(payload, dict) else []
        tested = int(payload.get("tested", len(fuzz_inputs))) if isinstance(payload, dict) else len(fuzz_inputs)

    coverage = (len(set(signatures)) / tested) if tested > 0 else 0.0
    elapsed = time.monotonic() - start
    return FuzzResult(
        survived=len(failing_inputs) == 0,
        total_inputs_tested=tested,
        failing_inputs=failing_inputs,
        coverage_achieved=max(0.0, min(1.0, coverage)),
        fuzz_time_seconds=elapsed,
        strategy_used="unknown",
    )


def fuzz_patch(
    patched_source: str,
    function_name: str,
    original_test_cases: list[dict],
    bug_report: BugReport,
    config: VerifixConfig,
    strategy: FuzzStrategy = FuzzStrategy.BOUNDARY,
) -> FuzzResult:
    signature = infer_signature(patched_source, function_name, test_cases=original_test_cases)
    fuzz_target = FuzzTarget(
        function_name=function_name,
        source=patched_source,
        signature=signature,
        test_cases=original_test_cases,
    )

    base_seed = int(config.fuzzer_seed)
    primary = generate_fuzz_inputs(
        fuzz_target,
        strategy=strategy,
        n_inputs=50,
        seed=base_seed,
    )
    boundary = generate_fuzz_inputs(
        fuzz_target,
        strategy=FuzzStrategy.BOUNDARY,
        n_inputs=50,
        seed=base_seed + 1,
    )

    combined = _dedupe_inputs([*primary, *boundary], limit=50)

    reference_source = None
    metadata_reference = bug_report.metadata.get("reference_source")
    if isinstance(metadata_reference, str) and metadata_reference.strip():
        reference_source = metadata_reference
    else:
        fixed_path = bug_report.metadata.get("fixed_source_path")
        if isinstance(fixed_path, str) and fixed_path.strip() and Path(fixed_path).exists():
            reference_source = Path(fixed_path).read_text(encoding="utf-8")

    timeout_seconds = max(1.0, min(5.0, float(config.test_timeout_seconds)))
    result = run_fuzz_test(
        fuzz_target=fuzz_target,
        fuzz_inputs=combined,
        reference_source=reference_source,
        timeout_seconds=timeout_seconds,
        working_dir=config.working_dir,
    )

    result.strategy_used = (
        f"{strategy.value}+{FuzzStrategy.BOUNDARY.value}"
        if strategy != FuzzStrategy.BOUNDARY
        else FuzzStrategy.BOUNDARY.value
    )
    return result


def _find_function(tree: ast.AST, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def _annotation_to_type(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "any"
    if isinstance(annotation, ast.Name):
        name = annotation.id.lower()
        if name in {"int", "list", "str", "bool"}:
            return name
    return "any"


def _infer_param_types_from_cases(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    test_cases: list[dict],
) -> dict[str, str]:
    arg_names = [arg.arg for arg in function_node.args.args]
    detected: dict[str, str] = {name: "any" for name in arg_names}

    per_arg_values: dict[str, list[Any]] = {name: [] for name in arg_names}
    for case in test_cases:
        inputs = case.get("input") if isinstance(case, dict) else None
        if not isinstance(inputs, (list, tuple)):
            continue
        for idx, name in enumerate(arg_names):
            if idx < len(inputs):
                per_arg_values[name].append(inputs[idx])

    for name, values in per_arg_values.items():
        if not values:
            continue
        inferred_types = {_value_type(value) for value in values}
        inferred_types.discard("any")
        if len(inferred_types) == 1:
            detected[name] = inferred_types.pop()

    return detected


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    return "any"


def _generate_boundary_inputs(param_types: list[str], n_inputs: int) -> list[list]:
    boundary_map = {
        "int": [-1000, -1, 0, 1, 1000, sys.maxsize],
        "list": [[], [0], [0, 0], [1, 0, -1], list(range(100)), list(range(100, 0, -1))],
        "str": ["", "a", "a" * 100, "hello world", "\n\t"],
        "bool": [False, True],
        "any": [0, 1, "", [], True],
    }

    pools = [boundary_map.get(param_type, boundary_map["any"]) for param_type in param_types]
    inputs = [list(values) for values in itertools.product(*pools)]
    return inputs[:n_inputs]


def _generate_random_inputs(param_types: list[str], n_inputs: int, seed: int) -> list[list]:
    rng = random.Random(seed)
    generated: list[list] = []

    for _ in range(n_inputs):
        args: list[Any] = []
        for param_type in param_types:
            if param_type == "int":
                args.append(rng.randint(-1000, 1000))
            elif param_type == "list":
                length = rng.randint(0, 8)
                args.append([rng.randint(-20, 20) for _ in range(length)])
            elif param_type == "str":
                length = rng.randint(0, 12)
                alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
                args.append("".join(rng.choice(alphabet) for _ in range(length)))
            elif param_type == "bool":
                args.append(bool(rng.randint(0, 1)))
            else:
                choice = rng.randint(0, 3)
                if choice == 0:
                    args.append(rng.randint(-10, 10))
                elif choice == 1:
                    args.append([rng.randint(-5, 5) for _ in range(rng.randint(0, 5))])
                elif choice == 2:
                    args.append("")
                else:
                    args.append(bool(rng.randint(0, 1)))

        generated.append(args)

    return generated


def _generate_mutation_inputs(test_cases: list[dict], n_inputs: int, seed: int) -> list[list]:
    rng = random.Random(seed)
    mutated: list[list] = []

    for case in test_cases:
        inputs = case.get("input") if isinstance(case, dict) else None
        if not isinstance(inputs, (list, tuple)):
            continue

        base = list(inputs)
        mutated.append(base)

        for idx, value in enumerate(base):
            for replacement in _mutate_value(value, rng):
                updated = list(base)
                updated[idx] = replacement
                mutated.append(updated)

    if not mutated:
        mutated = [[0], [1], [-1], [2], [3]]

    deduped = _dedupe_inputs(mutated, limit=n_inputs)
    return deduped


def _mutate_value(value: Any, rng: random.Random) -> list[Any]:
    mutations: list[Any] = []

    if isinstance(value, bool):
        mutations.extend([not value])
    elif isinstance(value, int):
        mutations.extend([value + 1, value - 1, value + 2, value - 2, value * 2])
        if value != 0:
            mutations.append(int(value / 2))
    elif isinstance(value, list):
        shuffled = list(value)
        rng.shuffle(shuffled)
        mutations.append(shuffled)
        if value:
            mutations.append(list(value) + [value[-1]])
        mutations.append([-1] + list(value))
        mutations.append(list(reversed(value)))
    elif isinstance(value, str):
        mutations.extend([value + "x", value[::-1], value.upper()])
    else:
        mutations.append(value)

    return mutations


def _dedupe_inputs(inputs: list[list], limit: int) -> list[list]:
    seen: set[str] = set()
    deduped: list[list] = []

    for item in inputs:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break

    return deduped


def _build_runner_script(function_name: str, use_reference: bool) -> str:
    return f"""import importlib.util
import json
import traceback
from pathlib import Path


def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import sys

    inputs_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    inputs = json.loads(inputs_path.read_text(encoding=\"utf-8\"))

    patched = _load_module(\"patched_module.py\", \"patched_module\")
    patched_fn = getattr(patched, {function_name!r})

    ref_fn = None
    if {use_reference!r}:
        reference = _load_module(\"reference_module.py\", \"reference_module\")
        ref_fn = getattr(reference, {function_name!r})

    failing_inputs = []
    signatures = []

    for fuzz_input in inputs:
        patched_ok = True
        patched_output = None
        patched_error = None
        try:
            patched_output = patched_fn(*fuzz_input)
        except Exception:
            patched_ok = False
            patched_error = traceback.format_exc(limit=1).strip()

        if ref_fn is None:
            if not patched_ok:
                failing_inputs.append({{\"input\": fuzz_input, \"error\": patched_error}})
                continue
            signatures.append(repr(patched_output))
            continue

        expected_ok = True
        expected_output = None
        expected_error = None
        try:
            expected_output = ref_fn(*fuzz_input)
        except Exception:
            expected_ok = False
            expected_error = traceback.format_exc(limit=1).strip()

        if patched_ok and expected_ok:
            if patched_output != expected_output:
                failing_inputs.append(
                    {{
                        \"input\": fuzz_input,
                        \"error\": f\"mismatch: patched={{patched_output!r}} expected={{expected_output!r}}\",
                    }}
                )
                continue
            signatures.append(repr(patched_output))
            continue

        if (not patched_ok) and (not expected_ok):
            # Matched exception behavior is acceptable when reference is provided.
            signatures.append(f\"exc:{{patched_error.split(':', 1)[0]}}\")
            continue

        if not patched_ok and expected_ok:
            failing_inputs.append(
                {{
                    \"input\": fuzz_input,
                    \"error\": f\"patched_exception: {{patched_error}}\",
                }}
            )
            continue

        failing_inputs.append(
            {{
                \"input\": fuzz_input,
                \"error\": f\"reference_exception: {{expected_error}}\",
            }}
        )

    payload = {{
        \"tested\": len(inputs),
        \"failing_inputs\": failing_inputs,
        \"signatures\": signatures,
    }}
    output_path.write_text(json.dumps(payload), encoding=\"utf-8\")
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
"""


__all__ = [
    "FuzzTarget",
    "infer_signature",
    "FuzzStrategy",
    "generate_fuzz_inputs",
    "run_fuzz_test",
    "FuzzResult",
    "fuzz_patch",
]
