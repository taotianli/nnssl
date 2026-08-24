"""Structure-conditioned latent bridge for teacher-free Primus MAE+JEPA.

The module keeps the public Primus encoder and both SSL objectives, but makes
their roles complementary. JEPA predicts one latent per compact 3-D region;
those predictions condition the MAE decoder through an exactly zero-initialized
FiLM residual. The default stop-gradient prevents voxel reconstruction from
turning the semantic predictor into a second pixel decoder.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from einops import rearrange
from torch import nn

from nnssl.architectures.primus_jepa import _gather_tokens, complement_indices
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA


def compact_region_ids(
    spatial_shape: Sequence[int], region_grid: Sequence[int], device: torch.device
) -> torch.Tensor:
    """Return connected, approximately equal-size grid regions in token order.

    Primus stores projected volumes as ``(W, H, D)`` and flattens them as
    ``(H, W, D)``. Keeping that convention here is essential for conditioning
    the correct decoder positions.
    """
    if len(spatial_shape) != 3 or len(region_grid) != 3:
        raise ValueError("spatial_shape and region_grid must both be 3-D")
    flat_shape = (int(spatial_shape[1]), int(spatial_shape[0]), int(spatial_shape[2]))
    grid = tuple(int(value) for value in region_grid)
    if any(value <= 0 for value in grid):
        raise ValueError("region_grid entries must be positive")
    coordinates = torch.meshgrid(
        *(torch.arange(size, device=device) for size in flat_shape), indexing="ij"
    )
    bins = [
        torch.clamp(coordinate * count // size, max=count - 1)
        for coordinate, size, count in zip(coordinates, flat_shape, grid)
    ]
    return ((bins[0] * grid[1] + bins[1]) * grid[2] + bins[2]).reshape(-1).long()


def shuffled_region_control(region_ids: torch.Tensor) -> torch.Tensor:
    """Deterministically scatter regions while preserving their exact sizes."""
    count = region_ids.numel()
    # A private CPU generator makes the control reproducible without advancing
    # patch-drop, DropPath, augmentation, or regularizer RNG streams.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5C1B2026)
    permutation = torch.randperm(count, generator=generator, device="cpu")
    return region_ids[permutation.to(region_ids.device)]


def region_mean(
    tokens: torch.Tensor, region_ids: torch.Tensor, num_regions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool ``[B, K, C]`` tokens by per-token region labels."""
    batch, _, channels = tokens.shape
    if region_ids.shape[:2] != tokens.shape[:2]:
        raise ValueError("region_ids must have shape [B, K]")
    output = tokens.new_zeros(batch, num_regions, channels)
    counts = tokens.new_zeros(batch, num_regions, 1)
    output.scatter_add_(1, region_ids[..., None].expand(-1, -1, channels), tokens)
    counts.scatter_add_(1, region_ids[..., None], tokens.new_ones(batch, tokens.shape[1], 1))
    return output / counts.clamp_min(1), counts.squeeze(-1)


class RegionFiLMBridge(nn.Module):
    """Zero-initialized decoder-entry FiLM conditioned on predicted latents."""

    def __init__(self, dim: int, max_scale: float = 0.25):
        super().__init__()
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")
        self.max_scale = float(max_scale)
        self.norm = nn.LayerNorm(dim)
        self.to_gamma_beta = nn.Linear(dim, 2 * dim)
        # The projection is zero initialized, which makes the residual exactly
        # zero. A nonzero scalar is important: zeroing both factors would make
        # every first-step gradient zero and permanently deadlock the bridge.
        self.strength = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self, tokens: torch.Tensor, condition: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(condition).chunk(2, dim=-1)
        modulation = torch.tanh(gamma) * self.norm(tokens) + beta
        scale = self.max_scale * torch.tanh(self.strength)
        return tokens + active.to(tokens.dtype) * scale * modulation

    @property
    def effective_strength(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.strength)


class PrimusStructureConditionedMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Region-level JEPA whose predictions optionally condition MAE decoding."""

    def __init__(
        self,
        *args,
        region_grid: Sequence[int] = (4, 4, 4),
        bridge_enabled: bool = True,
        detach_bridge: bool = True,
        shuffled_regions: bool = False,
        bridge_max_scale: float = 0.25,
        context_gain: float = 0.0,
        context_noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.region_grid = tuple(int(value) for value in region_grid)
        self.num_regions = prod(self.region_grid)
        self.bridge_enabled = bool(bridge_enabled)
        self.detach_bridge = bool(detach_bridge)
        self.shuffled_regions = bool(shuffled_regions)
        self.context_gain = float(context_gain)
        self.context_noise_std = float(context_noise_std)
        self.region_bridge = RegionFiLMBridge(self.embed_dim, bridge_max_scale)

    def _target_view(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() or (self.context_gain == 0 and self.context_noise_std == 0):
            return x
        devices = [x.device.index] if x.device.type == "cuda" else []
        # Do not let the cross-view control change later decoder DropPath or
        # the next step's patch mask merely by consuming extra RNG samples.
        with torch.random.fork_rng(devices=devices):
            shape = (x.shape[0], 1, 1, 1, 1)
            gain = 1 + (
                2 * torch.rand(shape, device=x.device, dtype=x.dtype) - 1
            ) * self.context_gain
            return x * gain + torch.randn_like(x) * self.context_noise_std

    def _decode_conditioned(
        self,
        context: torch.Tensor,
        keep_indices: torch.Tensor,
        masked_indices: torch.Tensor,
        masked_region_ids: torch.Tensor,
        region_prediction: torch.Tensor,
        spatial_shape: Sequence[int],
    ) -> torch.Tensor:
        restored = self.restore_full_sequence(context, keep_indices, prod(spatial_shape))
        if self.bridge_enabled:
            condition = _gather_tokens(region_prediction, masked_region_ids)
            if self.detach_bridge:
                condition = condition.detach()
            full_condition = restored.new_zeros(restored.shape)
            active = restored.new_zeros((*restored.shape[:2], 1))
            full_condition.scatter_(
                1, masked_indices[..., None].expand_as(condition), condition
            )
            active.scatter_(1, masked_indices[..., None], 1)
            restored = self.region_bridge(restored, full_condition, active)
        decoded, _ = self.decoder(restored)
        decoded = rearrange(
            decoded,
            "b (h w d) c -> b c w h d",
            h=spatial_shape[0],
            w=spatial_shape[1],
            d=spatial_shape[2],
        )
        return self.up_projection(decoded)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # MAE always encodes and reconstructs the same clean view. Only the
        # target path changes in the cross-view ablation, isolating JEPA
        # invariance from denoising-MAE augmentation.
        context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        base_ids = compact_region_ids(spatial_shape, self.region_grid, x.device)
        if self.shuffled_regions:
            base_ids = shuffled_region_control(base_ids)
        all_region_ids = base_ids.expand(x.shape[0], -1)
        masked_region_ids = _gather_tokens(all_region_ids[..., None], masked_indices).squeeze(-1)

        target_raw, target_normalized = self.encode_online_target(self._target_view(x))
        patch_prediction = self.predictor(context, keep_indices, masked_indices)
        patch_target = _gather_tokens(target_normalized, masked_indices)
        region_prediction, counts = region_mean(
            patch_prediction, masked_region_ids, self.num_regions
        )
        region_target, _ = region_mean(patch_target, masked_region_ids, self.num_regions)
        valid = counts > 0
        # With the default 75% mask every 5^3 region is populated. The fallback
        # keeps unusual patch shapes numerically safe without changing shapes.
        region_prediction = region_prediction * valid[..., None]
        region_target = region_target * valid[..., None]

        reconstruction = self._decode_conditioned(
            context,
            keep_indices,
            masked_indices,
            masked_region_ids,
            region_prediction,
            spatial_shape,
        )
        regularizer_embedding = self.regularizer_projector(
            _gather_tokens(target_raw, masked_indices)
        )
        with torch.no_grad():
            centered = region_prediction.float() - region_prediction.float().mean(dim=1, keepdim=True)
            variance = centered.square().mean()
            condition_norm = region_prediction.float().norm(dim=-1).mean()
        return {
            "reconstruction": reconstruction,
            "prediction": region_prediction,
            "target": region_target,
            "regularizer_embedding": regularizer_embedding,
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
            "region_ids": all_region_ids,
            "region_prediction_variance": variance,
            "region_condition_norm": condition_norm,
            "bridge_strength": self.region_bridge.effective_strength.detach(),
        }
