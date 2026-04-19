from __future__ import annotations

import torch
from torch_geometric.data import Data

from verifix.models.latent_jepa import (
    JEPATransitionPredictor,
    MultiTaskRepairGAT,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def _toy_graph() -> Data:
    x = torch.tensor(
        [
            [0.1, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.2, 0.0],
            [0.3, 0.0, 1.0, 0.4, 0.0],
        ],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index)


def test_forward_shapes() -> None:
    model = MultiTaskRepairGAT()
    out = model(_toy_graph())

    assert out["fault_probs"].shape == (3, 1)
    assert out["policy_logits"].shape[0] == 3
    assert out["critic_scores"].shape == (1, 1)


def test_train_step_runs_and_returns_metrics() -> None:
    model = MultiTaskRepairGAT()
    predictor = JEPATransitionPredictor()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-3)

    buggy = _toy_graph()
    fixed = _toy_graph()
    action_ids = torch.tensor([1], dtype=torch.long)
    bug_mask = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float)

    metrics = train_step(
        model=model,
        predictor=predictor,
        optimizer=optimizer,
        buggy_graph=buggy,
        fixed_graph=fixed,
        action_ids=action_ids,
        bug_node_mask=bug_mask,
        device="cpu",
    )

    assert "total_loss" in metrics
    assert metrics["total_loss"] >= 0.0


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = MultiTaskRepairGAT()
    predictor = JEPATransitionPredictor()
    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        path=str(checkpoint),
        model=model,
        predictor=predictor,
        metadata={"epochs": 1},
    )

    loaded_model, loaded_predictor, metadata = load_checkpoint(str(checkpoint), device="cpu")
    assert isinstance(loaded_model, MultiTaskRepairGAT)
    assert isinstance(loaded_predictor, JEPATransitionPredictor)
    assert metadata["epochs"] == 1
