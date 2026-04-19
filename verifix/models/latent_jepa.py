from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from verifix.core.action_space import NUM_ACTIONS
from verifix.models.pyg_converter import FEATURE_DIM

LATENT_DIM = 128
ACTION_DIM = 64
HIDDEN_DIM = 256


def default_loss_weights() -> dict[str, float]:
    return {
        "alpha": 1.0,
        "beta_c": 10.0,
        "beta_l": 1.0,
        "beta_p": 1.0,
    }


class MultiTaskRepairGAT(nn.Module):
    """
    Shared GAT encoder -> 3 specialized heads.

    - Embedding head: graph latent z_graph
    - Fault localization head: node-level bug probability
    - Policy head: node-level action logits over NUM_ACTIONS
    - Critic head: graph-level repair closeness score
    """

    def __init__(self) -> None:
        super().__init__()
        self.gat1 = GATConv(FEATURE_DIM, 64, heads=4, concat=True)
        self.gat2 = GATConv(256, 64, heads=4, concat=True)
        self.gat3 = GATConv(256, LATENT_DIM, heads=1, concat=False)

        self.fault_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
          node_embeddings: [num_nodes, LATENT_DIM]
          z_graph:         [num_graphs, LATENT_DIM]
          batch_index:     [num_nodes]
        """
        batch_index = _resolve_batch_index(data)

        hidden = F.elu(self.gat1(data.x, data.edge_index))
        hidden = F.elu(self.gat2(hidden, data.edge_index))
        node_embeddings = self.gat3(hidden, data.edge_index)
        z_graph = global_mean_pool(node_embeddings, batch_index)
        return node_embeddings, z_graph, batch_index

    def forward(self, data: Data) -> dict[str, Tensor]:
        node_embeddings, z_graph, batch_index = self.encode(data)
        fault_probs = self.fault_head(node_embeddings)
        policy_logits = self.policy_head(node_embeddings)
        critic_scores = self.critic_head(z_graph)
        return {
            "node_embeddings": node_embeddings,
            "z_graph": z_graph,
            "fault_probs": fault_probs,
            "policy_logits": policy_logits,
            "critic_scores": critic_scores,
            "batch": batch_index,
        }


class JEPATransitionPredictor(nn.Module):
    """Predict next latent graph embedding from current embedding + action context."""

    def __init__(self) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(NUM_ACTIONS, ACTION_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(LATENT_DIM + ACTION_DIM + LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )

    def forward(
        self,
        z_graph: Tensor,
        action_ids: Tensor,
        node_context: Tensor | None = None,
    ) -> Tensor:
        action_emb = self.action_embedding(action_ids.long())
        if node_context is None:
            node_context = torch.zeros_like(z_graph)
        features = torch.cat([z_graph, action_emb, node_context], dim=-1)
        return self.mlp(features)


def train_step(
    model: MultiTaskRepairGAT,
    predictor: JEPATransitionPredictor,
    optimizer: torch.optim.Optimizer,
    buggy_graph: Data,
    fixed_graph: Data,
    action_ids: Tensor,
    bug_node_mask: Tensor | None,
    loss_weights: dict[str, float] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    model.train()
    predictor.train()

    resolved_weights = default_loss_weights()
    if loss_weights is not None:
        resolved_weights.update(loss_weights)

    resolved_device = torch.device(device) if device is not None else _infer_device(model)

    buggy_graph = buggy_graph.to(resolved_device)
    fixed_graph = fixed_graph.to(resolved_device)
    action_ids = action_ids.to(resolved_device).view(-1)
    if bug_node_mask is not None:
        bug_node_mask = bug_node_mask.to(resolved_device).view(-1, 1)

    optimizer.zero_grad(set_to_none=True)

    buggy_out = model(buggy_graph)
    fixed_out = model(fixed_graph)

    if bug_node_mask is None:
        loss_localization = torch.zeros((), device=resolved_device)
    else:
        loss_localization = F.binary_cross_entropy(buggy_out["fault_probs"], bug_node_mask.float())

    loss_critic_buggy = F.binary_cross_entropy(
        buggy_out["critic_scores"],
        torch.zeros_like(buggy_out["critic_scores"]),
    )
    loss_critic_fixed = F.binary_cross_entropy(
        fixed_out["critic_scores"],
        torch.ones_like(fixed_out["critic_scores"]),
    )

    num_graphs = int(buggy_out["z_graph"].shape[0])
    action_ids = _normalize_action_ids(action_ids, num_graphs=num_graphs, device=resolved_device)
    valid_mask = action_ids >= 0

    graph_policy_logits = _graph_policy_logits(
        node_policy_logits=buggy_out["policy_logits"],
        batch_index=buggy_out["batch"],
        bug_node_mask=bug_node_mask,
    )

    if bool(valid_mask.any()):
        policy_logits_valid = graph_policy_logits[valid_mask]
        action_valid = action_ids[valid_mask].long()
        loss_policy = F.cross_entropy(policy_logits_valid, action_valid)

        node_context = _graph_context_embeddings(
            node_embeddings=buggy_out["node_embeddings"],
            batch_index=buggy_out["batch"],
            bug_node_mask=bug_node_mask,
        )
        z_pred = predictor(
            buggy_out["z_graph"][valid_mask],
            action_valid,
            node_context=node_context[valid_mask],
        )
        z_target = fixed_out["z_graph"][valid_mask]
        loss_jepa = F.mse_loss(z_pred, z_target)
    else:
        loss_policy = torch.zeros((), device=resolved_device)
        loss_jepa = torch.zeros((), device=resolved_device)

    total_loss = (
        resolved_weights["alpha"] * loss_jepa
        + resolved_weights["beta_c"] * (loss_critic_buggy + loss_critic_fixed)
        + resolved_weights["beta_l"] * loss_localization
        + resolved_weights["beta_p"] * loss_policy
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), max_norm=5.0)
    optimizer.step()

    return {
        "total_loss": float(total_loss.detach().cpu().item()),
        "loss_localization": float(loss_localization.detach().cpu().item()),
        "loss_policy": float(loss_policy.detach().cpu().item()),
        "loss_critic_buggy": float(loss_critic_buggy.detach().cpu().item()),
        "loss_critic_fixed": float(loss_critic_fixed.detach().cpu().item()),
        "loss_jepa": float(loss_jepa.detach().cpu().item()),
    }


def _resolve_batch_index(data: Data) -> Tensor:
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)


def _infer_device(module: nn.Module) -> torch.device:
    first_param = next(module.parameters(), None)
    if first_param is None:
        return torch.device("cpu")
    return first_param.device


def _normalize_action_ids(action_ids: Tensor, num_graphs: int, device: torch.device) -> Tensor:
    if action_ids.numel() == num_graphs:
        return action_ids
    if action_ids.numel() == 1 and num_graphs > 1:
        return action_ids.repeat(num_graphs)
    if action_ids.numel() > num_graphs:
        return action_ids[:num_graphs]
    padding = torch.full((num_graphs - action_ids.numel(),), -1, dtype=action_ids.dtype, device=device)
    return torch.cat([action_ids, padding], dim=0)


def _graph_policy_logits(
    node_policy_logits: Tensor,
    batch_index: Tensor,
    bug_node_mask: Tensor | None,
) -> Tensor:
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
    logits: list[Tensor] = []

    for graph_idx in range(num_graphs):
        node_mask = batch_index == graph_idx
        graph_logits = node_policy_logits[node_mask]

        if graph_logits.numel() == 0:
            logits.append(torch.zeros((NUM_ACTIONS,), device=node_policy_logits.device))
            continue

        if bug_node_mask is not None:
            bug_mask_graph = bug_node_mask[node_mask].view(-1) > 0.5
            if bool(bug_mask_graph.any()):
                graph_logits = graph_logits[bug_mask_graph]

        logits.append(graph_logits.mean(dim=0))

    return torch.stack(logits, dim=0)


def _graph_context_embeddings(
    node_embeddings: Tensor,
    batch_index: Tensor,
    bug_node_mask: Tensor | None,
) -> Tensor:
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
    contexts: list[Tensor] = []

    for graph_idx in range(num_graphs):
        node_mask = batch_index == graph_idx
        graph_embeddings = node_embeddings[node_mask]
        if graph_embeddings.numel() == 0:
            contexts.append(torch.zeros((LATENT_DIM,), device=node_embeddings.device))
            continue

        if bug_node_mask is not None:
            bug_mask_graph = bug_node_mask[node_mask].view(-1) > 0.5
            if bool(bug_mask_graph.any()):
                graph_embeddings = graph_embeddings[bug_mask_graph]

        contexts.append(graph_embeddings.mean(dim=0))

    return torch.stack(contexts, dim=0)


def save_checkpoint(
    path: str,
    model: MultiTaskRepairGAT,
    predictor: JEPATransitionPredictor,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "predictor_state": predictor.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[MultiTaskRepairGAT, JEPATransitionPredictor, dict[str, Any]]:
    resolved_device = torch.device(device)
    payload = torch.load(path, map_location=resolved_device)

    model = MultiTaskRepairGAT().to(resolved_device)
    predictor = JEPATransitionPredictor().to(resolved_device)
    metadata = dict(payload.get("metadata", {}))

    model_state = dict(payload["model_state"])
    predictor_state = dict(payload["predictor_state"])

    try:
        model.load_state_dict(model_state)
        predictor.load_state_dict(predictor_state)
    except RuntimeError:
        checkpoint_action_dim = _infer_checkpoint_action_dim(model_state, predictor_state)
        if checkpoint_action_dim == NUM_ACTIONS:
            raise

        model_state = _align_action_dependent_state(
            checkpoint_state=model_state,
            template_state=model.state_dict(),
            checkpoint_action_dim=checkpoint_action_dim,
        )
        predictor_state = _align_action_dependent_state(
            checkpoint_state=predictor_state,
            template_state=predictor.state_dict(),
            checkpoint_action_dim=checkpoint_action_dim,
        )

        model.load_state_dict(model_state, strict=False)
        predictor.load_state_dict(predictor_state, strict=False)
        metadata.setdefault("checkpoint_action_dim", checkpoint_action_dim)
        metadata.setdefault("runtime_action_dim", NUM_ACTIONS)

    return model, predictor, metadata


def _infer_checkpoint_action_dim(
    model_state: dict[str, Tensor],
    predictor_state: dict[str, Tensor],
) -> int:
    policy_weight = model_state.get("policy_head.2.weight")
    if isinstance(policy_weight, Tensor) and policy_weight.ndim >= 1:
        return int(policy_weight.shape[0])

    action_embedding = predictor_state.get("action_embedding.weight")
    if isinstance(action_embedding, Tensor) and action_embedding.ndim >= 1:
        return int(action_embedding.shape[0])

    return NUM_ACTIONS


def _align_action_dependent_state(
    checkpoint_state: dict[str, Tensor],
    template_state: dict[str, Tensor],
    checkpoint_action_dim: int,
) -> dict[str, Tensor]:
    aligned = dict(checkpoint_state)
    for key in ("policy_head.2.weight", "policy_head.2.bias", "action_embedding.weight"):
        checkpoint_tensor = aligned.get(key)
        template_tensor = template_state.get(key)
        if not isinstance(checkpoint_tensor, Tensor) or not isinstance(template_tensor, Tensor):
            continue
        if checkpoint_tensor.shape == template_tensor.shape:
            continue
        rows_to_copy = min(checkpoint_action_dim, int(checkpoint_tensor.shape[0]), int(template_tensor.shape[0]))
        resized = template_tensor.detach().clone()
        resized[:rows_to_copy] = checkpoint_tensor[:rows_to_copy]
        aligned[key] = resized
    return aligned


__all__ = [
    "LATENT_DIM",
    "ACTION_DIM",
    "HIDDEN_DIM",
    "MultiTaskRepairGAT",
    "JEPATransitionPredictor",
    "default_loss_weights",
    "train_step",
    "save_checkpoint",
    "load_checkpoint",
]
