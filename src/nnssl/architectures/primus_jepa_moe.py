"""Sparse residual MoE extensions for teacher-free Primus MAE+JEPA.

The public Primus-M EVA encoder remains an intact ``base`` module. Sparse
experts are zero-initialized residual adapters after selected transformer
blocks. This preserves the public MAE function at initialization and lets the
downstream adaptation path load ``eva.base`` while discarding pretraining-only
experts.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from nnssl.architectures.primus_jepa import _gather_tokens, complement_indices
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA


class SparseResidualMoEAdapter(nn.Module):
    """Top-k token router with small residual experts and no balance loss.

    Expert output layers are initialized to zero, so inserting the adapter does
    not perturb a loaded MAE encoder. Routing imbalance is controlled with the
    auxiliary-loss-free bias update used by Neuro-JEPA rather than another
    gradient objective that could conflict with MAE or JEPA.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: int = 192,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError("top_k must satisfy 0 < top_k <= num_experts")
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.route_scale = float(route_scale)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("routing_bias", torch.zeros(num_experts))
        self.register_buffer("counts", torch.zeros(num_experts, dtype=torch.long))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, expert_hidden_dim),
                    nn.GELU(),
                    nn.Linear(expert_hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        nn.init.zeros_(self.gate.weight)
        for expert in self.experts:
            nn.init.trunc_normal_(expert[0].weight, std=0.02)
            nn.init.zeros_(expert[0].bias)
            nn.init.zeros_(expert[2].weight)
            nn.init.zeros_(expert[2].bias)
        self.last_probabilities: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = self.norm(x).reshape(-1, self.dim)
        probabilities = self.gate(flat).float().softmax(dim=-1)
        selection_scores = probabilities.detach() + self.routing_bias.float()
        indices = selection_scores.topk(self.top_k, dim=-1).indices
        weights = probabilities.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights.to(dtype=x.dtype) * self.route_scale

        with torch.no_grad():
            self.counts.add_(torch.bincount(indices.flatten(), minlength=self.num_experts))
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_index, topk_slot = torch.where(indices == expert_index)
            if token_index.numel() == 0:
                # Keep every expert in the graph for DDP static-graph execution.
                output = output + expert(flat[:1]).sum() * 0.0
                continue
            output[token_index] += expert(flat[token_index]) * weights[token_index, topk_slot, None]
        self.last_probabilities = probabilities.reshape(*shape[:-1], self.num_experts)
        return output.reshape(shape)

    @torch.no_grad()
    def update_routing_bias(self, rate: float = 1e-4, clip: float = 0.3) -> tuple[float, float]:
        counts = self.counts.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        mean = counts.float().mean()
        if mean <= 0:
            self.counts.zero_()
            return 1.0, 1.0
        relative = counts.float() / mean
        update = torch.sign(mean - counts.float())
        update.sub_(update.mean())
        self.routing_bias.add_(float(rate) * update)
        if clip > 0:
            self.routing_bias.clamp_(-float(clip), float(clip))
        self.routing_bias.sub_(self.routing_bias.mean())
        self.counts.zero_()
        return float(relative.min()), float(relative.max())

    @torch.no_grad()
    def reset_counts(self) -> None:
        self.counts.zero_()


class ResidualMoEEva(nn.Module):
    """An EVA encoder with sparse residual adapters after chosen blocks."""

    def __init__(
        self,
        base: nn.Module,
        layer_indices: Sequence[int],
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: int = 192,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        indices = tuple(sorted(set(int(i) for i in layer_indices)))
        if not indices or indices[0] < 0 or indices[-1] >= len(base.blocks):
            raise ValueError(f"Invalid MoE layer indices {indices} for depth {len(base.blocks)}")
        self.base = base
        self.layer_indices = indices
        self.adapters = nn.ModuleDict(
            {
                str(index): SparseResidualMoEAdapter(
                    base.embed_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    expert_hidden_dim=expert_hidden_dim,
                    route_scale=route_scale,
                )
                for index in indices
            }
        )
        self.last_router_probabilities: list[torch.Tensor] = []

    def forward_features(self, x: torch.Tensor):
        x, rope, keep_indices = self.base._pos_embed(x)
        probabilities = []
        for index, block in enumerate(self.base.blocks):
            if self.base.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, rope=rope, use_reentrant=False)
            else:
                x = block(x, rope=rope)
            key = str(index)
            if key in self.adapters:
                adapter = self.adapters[key]
                x = x + adapter(x)
                probabilities.append(adapter.last_probabilities)
        self.last_router_probabilities = probabilities
        return self.base.norm(x), keep_indices

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    @torch.no_grad()
    def update_routing_bias(self, rate: float = 1e-4, clip: float = 0.3) -> tuple[float, float]:
        violations = [adapter.update_routing_bias(rate, clip) for adapter in self.adapters.values()]
        return (
            sum(item[0] for item in violations) / len(violations),
            sum(item[1] for item in violations) / len(violations),
        )

    @torch.no_grad()
    def reset_routing_counts(self) -> None:
        for adapter in self.adapters.values():
            adapter.reset_counts()


def routing_consistency_loss(
    context_probabilities: Sequence[torch.Tensor],
    target_probabilities: Sequence[torch.Tensor],
    keep_indices: torch.Tensor,
) -> torch.Tensor:
    """Symmetric Jensen-Shannon loss for corresponding visible tokens."""
    if len(context_probabilities) != len(target_probabilities) or not context_probabilities:
        raise ValueError("Context and target routing lists must be non-empty and aligned")
    losses = []
    for context, target_full in zip(context_probabilities, target_probabilities):
        target = _gather_tokens(target_full, keep_indices)
        p = context.float().clamp_min(1e-7)
        q = target.float().clamp_min(1e-7)
        midpoint = 0.5 * (p + q)
        losses.append(
            0.5
            * (
                (p * (p.log() - midpoint.log())).sum(dim=-1).mean()
                + (q * (q.log() - midpoint.log())).sum(dim=-1).mean()
            )
        )
    return torch.stack(losses).mean()


class PrimusMoEOnlineTargetMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Teacher-free MAE+JEPA with pretraining-only sparse residual MoE."""

    def __init__(
        self,
        *args,
        moe_layer_indices: Sequence[int] = (8, 10, 12, 14),
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_hidden_dim: int = 192,
        moe_route_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eva = ResidualMoEEva(
            self.eva,
            layer_indices=moe_layer_indices,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            expert_hidden_dim=moe_hidden_dim,
            route_scale=moe_route_scale,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        context_routes = tuple(self.eva.last_router_probabilities)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        reconstruction = self.decode_mae(context, keep_indices, spatial_shape)
        target_raw, target_normalized = self.encode_online_target(x)
        target_routes = tuple(self.eva.last_router_probabilities)
        target_raw_masked = _gather_tokens(target_raw, masked_indices)
        target = _gather_tokens(target_normalized, masked_indices)
        prediction = self.predictor(context, keep_indices, masked_indices)
        consistency = routing_consistency_loss(context_routes, target_routes, keep_indices)
        with torch.no_grad():
            probabilities = torch.cat([item.reshape(-1, item.shape[-1]) for item in target_routes])
            entropy = -(probabilities.float().clamp_min(1e-7) * probabilities.float().clamp_min(1e-7).log()).sum(-1).mean()
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "regularizer_embedding": self.regularizer_projector(target_raw_masked),
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
            "routing_consistency": consistency,
            "routing_entropy": entropy,
        }

    @torch.no_grad()
    def update_routing_bias(self, rate: float = 1e-4, clip: float = 0.3) -> tuple[float, float]:
        return self.eva.update_routing_bias(rate, clip)

    @torch.no_grad()
    def reset_routing_counts(self) -> None:
        self.eva.reset_routing_counts()

    @torch.no_grad()
    def load_mae_state_dict(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        current = self.state_dict()
        loaded = []
        excluded = ("predictor.", "regularizer_projector.", "eva.adapters.")
        for raw_key, value in state_dict.items():
            key = raw_key.removeprefix("module.").removeprefix("_orig_mod.")
            if key.startswith("eva."):
                key = "eva.base." + key[len("eva.") :]
            if key in current and current[key].shape == value.shape and not key.startswith(excluded):
                current[key] = value
                loaded.append(key)
        self.load_state_dict(current)
        return loaded
