from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from verifix.core.action_space import ACTION_MAP, NUM_ACTIONS
from verifix.edit_dsl.operators import list_registered_operator_names


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def action_space_hash() -> str:
    payload = {
        "num_actions": NUM_ACTIONS,
        "action_map": ACTION_MAP,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def operator_space_hash() -> str:
    canonical = json.dumps(list_registered_operator_names(), sort_keys=True)
    return sha256_text(canonical)


def build_training_provenance(dataset_path: str | Path, run_id: str) -> dict[str, Any]:
    resolved_dataset = str(Path(dataset_path).resolve())
    return {
        "run_id": run_id,
        "dataset_path": resolved_dataset,
        "dataset_sha256": hash_file(resolved_dataset),
        "action_space_sha256": action_space_hash(),
        "operator_space_sha256": operator_space_hash(),
    }


def validate_checkpoint_provenance(
    metadata: dict[str, Any],
    required_run_prefix: str | None = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    run_id = str(metadata.get("run_id", "")).strip()
    if required_run_prefix is not None and not run_id.startswith(required_run_prefix):
        issues.append(
            f"run_id '{run_id}' does not start with required prefix '{required_run_prefix}'"
        )

    expected_action_hash = action_space_hash()
    actual_action_hash = str(metadata.get("action_space_sha256", "")).strip()
    if expected_action_hash != actual_action_hash:
        issues.append("action_space_sha256 mismatch")

    expected_operator_hash = operator_space_hash()
    actual_operator_hash = str(metadata.get("operator_space_sha256", "")).strip()
    if expected_operator_hash != actual_operator_hash:
        issues.append("operator_space_sha256 mismatch")

    return len(issues) == 0, issues


__all__ = [
    "sha256_text",
    "hash_file",
    "action_space_hash",
    "operator_space_hash",
    "build_training_provenance",
    "validate_checkpoint_provenance",
]
