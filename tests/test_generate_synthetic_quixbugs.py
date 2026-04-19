from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from verifix.core.action_space import action_id_to_name, operator_to_action_id
from verifix.core.models import Edit, EditOperator


def _load_generator_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_synthetic_quixbugs.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_quixbugs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load generate_synthetic_quixbugs module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hard_negative_uses_current_program_fixed_ast(monkeypatch, tmp_path: Path) -> None:
    generator = _load_generator_module()

    quixbugs_root = tmp_path / "quixbugs"
    quixbugs_root.mkdir(parents=True)
    output_path = tmp_path / "synthetic.jsonl"

    args = argparse.Namespace(
        quixbugs_root=str(quixbugs_root),
        output=str(output_path),
        max_synthetic_per_program=10,
        target_synthetic_count=1,
        num_mutations=1,
        hard_negative_ratio=1.0,
        split_strategy="stratified",
        split_artifact="",
        seed=2026,
    )
    monkeypatch.setattr(generator, "_parse_args", lambda: args)

    pairs = {
        "alpha": ("def alpha(value):\n    return value\n", "def alpha(value):\n    return value\n"),
        "beta": ("def beta(value):\n    return value\n", "def beta(value):\n    return value\n"),
    }
    monkeypatch.setattr(generator, "_load_program_pairs", lambda _root: pairs)
    monkeypatch.setattr(generator, "_build_split", lambda fixed_sources, strategy, seed: (["alpha", "beta"], []))

    def _fake_candidates(_ast_tree, suspicious_lines, max_edits_per_node=3, operator_tier="core"):
        del suspicious_lines, max_edits_per_node, operator_tier
        return [
            Edit(
                operator=EditOperator.REPLACE_EXPR,
                node_id="n1",
                node_type="Return",
                line_number=2,
                original_text="return value",
                replacement_text="return value + 1",
                metadata={"target": "variable"},
            )
        ]

    monkeypatch.setattr(generator, "get_candidate_edits", _fake_candidates)

    exit_code = generator.main()

    assert exit_code == 0
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hard_negative = [record for record in records if record.get("source") == "hard_negative"]

    assert len(hard_negative) == 1
    assert hard_negative[0]["program"] == "alpha"
    assert hard_negative[0]["fixed_ast"]["file_path"] == "alpha.py"
    assert hard_negative[0]["operator_tier"] == "synthetic_aux"


def test_synthetic_records_store_repair_direction_action(monkeypatch, tmp_path: Path) -> None:
    generator = _load_generator_module()

    quixbugs_root = tmp_path / "quixbugs"
    quixbugs_root.mkdir(parents=True)
    output_path = tmp_path / "synthetic_repair_direction.jsonl"

    args = argparse.Namespace(
        quixbugs_root=str(quixbugs_root),
        output=str(output_path),
        max_synthetic_per_program=10,
        target_synthetic_count=1,
        num_mutations=1,
        hard_negative_ratio=0.0,
        split_strategy="stratified",
        split_artifact="",
        seed=2026,
    )
    monkeypatch.setattr(generator, "_parse_args", lambda: args)

    pairs = {
        "alpha": (
            "def alpha(value):\n    return value\n",
            "def alpha(value):\n    return value\n",
        )
    }
    monkeypatch.setattr(generator, "_load_program_pairs", lambda _root: pairs)
    monkeypatch.setattr(generator, "_build_split", lambda fixed_sources, strategy, seed: (["alpha"], []))

    def _fake_candidates(ast_tree, suspicious_lines, max_edits_per_node=3, operator_tier="core"):
        del suspicious_lines, max_edits_per_node, operator_tier
        source = ast_tree.source.strip()
        if "return value + 1" in source:
            return [
                Edit(
                    operator=EditOperator.REPLACE_EXPR,
                    node_id="n2",
                    node_type="Return",
                    line_number=2,
                    original_text="return value + 1",
                    replacement_text="return value",
                    metadata={"target": "return_expr"},
                )
            ]

        return [
            Edit(
                operator=EditOperator.REPLACE_EXPR,
                node_id="n1",
                node_type="Return",
                line_number=2,
                original_text="return value",
                replacement_text="return value + 1",
                metadata={"target": "return_expr"},
            )
        ]

    monkeypatch.setattr(generator, "get_candidate_edits", _fake_candidates)

    exit_code = generator.main()

    assert exit_code == 0
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    synthetic_rows = [record for record in records if record.get("source") == "synthetic"]

    assert len(synthetic_rows) == 1
    row = synthetic_rows[0]

    expected_action_id = operator_to_action_id(
        EditOperator.REPLACE_EXPR,
        metadata={"target": "return_expr"},
    )
    assert row["action_id"] == expected_action_id
    assert row["action_operator"] == action_id_to_name(expected_action_id)
    assert row["action_resolution"] == "direct_candidate"
    assert row["buggy_ast"]["source"].strip().endswith("return value + 1")
    assert row["fixed_ast"]["source"].strip().endswith("return value")
