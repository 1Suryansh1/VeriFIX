from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from verifix.core.provenance import build_training_provenance
from verifix.models.latent_jepa import (
    JEPATransitionPredictor,
    MultiTaskRepairGAT,
    save_checkpoint,
    train_step,
)
from verifix.models.pyg_converter import ASTtoPyG, annotated_ast_from_dict


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def train_from_jsonl(
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    alpha: float = 1.0,
    beta_critic: float = 10.0,
    beta_localization: float = 1.0,
    beta_policy: float = 1.0,
    max_records: int | None = None,
    run_id: str = "",
    required_run_prefix: str | None = None,
) -> dict[str, Any]:
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        normalized_run_id = "cycle_2026_04_04_default"

    if required_run_prefix is not None and not normalized_run_id.startswith(required_run_prefix):
        raise ValueError(
            f"run_id '{normalized_run_id}' must start with '{required_run_prefix}'"
        )

    records = load_jsonl_records(dataset_path)
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("No training records found")

    resolved_device = torch.device(device)
    converter = ASTtoPyG()

    model = MultiTaskRepairGAT().to(resolved_device)
    predictor = JEPATransitionPredictor().to(resolved_device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=learning_rate,
    )

    weights = {
        "alpha": alpha,
        "beta_c": beta_critic,
        "beta_l": beta_localization,
        "beta_p": beta_policy,
    }

    history: list[dict[str, float]] = []
    for epoch_idx in range(1, epochs + 1):
        metric_sum: dict[str, float] = defaultdict(float)
        steps = 0

        for record in records:
            try:
                buggy_graph, fixed_graph, action_tensor, fault_mask = _record_to_training_tensors(
                    record,
                    converter,
                )
            except Exception:
                continue

            metrics = train_step(
                model=model,
                predictor=predictor,
                optimizer=optimizer,
                buggy_graph=buggy_graph,
                fixed_graph=fixed_graph,
                action_ids=action_tensor,
                bug_node_mask=fault_mask,
                loss_weights=weights,
                device=resolved_device,
            )
            for key, value in metrics.items():
                metric_sum[key] += float(value)
            steps += 1

        if steps == 0:
            raise RuntimeError("No valid training steps were executed")

        epoch_metrics = {key: value / steps for key, value in metric_sum.items()}
        epoch_metrics["epoch"] = float(epoch_idx)
        history.append(epoch_metrics)
        print(
            "Epoch "
            f"{epoch_idx}/{epochs}: "
            f"total={epoch_metrics.get('total_loss', 0.0):.4f} "
            f"loc={epoch_metrics.get('loss_localization', 0.0):.4f} "
            f"pol={epoch_metrics.get('loss_policy', 0.0):.4f} "
            f"jepa={epoch_metrics.get('loss_jepa', 0.0):.4f}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "v3_multitask_gat.pt"
    provenance = build_training_provenance(dataset_path=dataset_path, run_id=normalized_run_id)

    metadata: dict[str, Any] = {
        "run_id": normalized_run_id,
        "dataset_path": str(Path(dataset_path).resolve()),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": str(resolved_device),
        "weights": weights,
        "records": len(records),
        "history": history,
        **provenance,
    }
    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        predictor=predictor,
        metadata=metadata,
    )

    result = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "metadata": metadata,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _record_to_training_tensors(
    record: dict[str, Any],
    converter: ASTtoPyG,
) -> tuple[Data, Data, Tensor, Tensor | None]:
    buggy_ast_dict = record.get("buggy_ast")
    fixed_ast_dict = record.get("fixed_ast")
    if not isinstance(buggy_ast_dict, dict) or not isinstance(fixed_ast_dict, dict):
        raise ValueError("Record missing buggy_ast/fixed_ast")

    buggy_ast = annotated_ast_from_dict(buggy_ast_dict)
    fixed_ast = annotated_ast_from_dict(fixed_ast_dict)

    bug_node_id = record.get("bug_node_id")
    if bug_node_id is not None:
        bug_node_id = str(bug_node_id)

    buggy_graph, _map_buggy, fault_mask = converter.convert(
        buggy_ast,
        bug_node_id=bug_node_id,
    )
    fixed_graph, _map_fixed, _fixed_labels = converter.convert(fixed_ast)

    action_raw = record.get("action_id", -1)
    action_id = int(action_raw) if isinstance(action_raw, int) else -1
    action_tensor = torch.tensor([action_id], dtype=torch.long)

    return buggy_graph, fixed_graph, action_tensor, fault_mask


__all__ = [
    "load_jsonl_records",
    "train_from_jsonl",
]
