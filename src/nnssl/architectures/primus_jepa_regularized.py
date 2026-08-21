"""Spatial teacher-free Primus variants used by regularization experiments."""

from __future__ import annotations

from itertools import product
from math import prod
import os
from typing import Sequence

import torch

from nnssl.architectures.primus_jepa import _gather_tokens, complement_indices
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA


class NoValidRegionPartitionError(RuntimeError):
    """Raised when a masked token set leaves an empty spatial region."""


def shifted_region_ids(
    token_indices: torch.Tensor,
    grid_size: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    """Map flattened ``(H, W, D)`` token indices to 64 non-wrapping regions."""
    grid = tuple(int(i) for i in grid_size)
    if len(grid) != 3 or any(i != 20 for i in grid):
        raise ValueError(f"ORR currently requires a 20x20x20 token grid, got {grid}")
    if len(offsets) != 3 or any(int(i) not in range(-2, 3) for i in offsets):
        raise ValueError(f"offsets must contain three integers in [-2, 2], got {offsets}")
    h = token_indices // (grid[1] * grid[2])
    remainder = token_indices % (grid[1] * grid[2])
    w = remainder // grid[2]
    d = remainder % grid[2]
    bins = []
    for coordinate, offset in zip((h, w, d), offsets):
        boundaries = coordinate.new_tensor([5 + int(offset), 10 + int(offset), 15 + int(offset)])
        # ``right=True`` makes a cut at 5 represent [0, 5) and [5, ...).
        bins.append(torch.bucketize(coordinate.contiguous(), boundaries, right=True))
    return bins[0] * 16 + bins[1] * 4 + bins[2]


def select_valid_region_partition(
    masked_indices: torch.Tensor,
    grid_size: Sequence[int],
    *,
    fixed: bool = False,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select reproducible valid offsets and return region ids for each volume."""
    if masked_indices.ndim != 2:
        raise ValueError("masked_indices must have shape [batch, masked_tokens]")
    candidates = [(0, 0, 0)] if fixed else list(product(range(-2, 3), repeat=3))
    assignment = []
    selected = []
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    for batch_index in range(masked_indices.shape[0]):
        order = torch.arange(len(candidates))
        if not fixed:
            order = torch.randperm(len(candidates), generator=generator)
        for candidate_index in order.tolist():
            offsets = candidates[candidate_index]
            region_ids = shifted_region_ids(masked_indices[batch_index], grid_size, offsets)
            if torch.bincount(region_ids, minlength=64).min().item() > 0:
                assignment.append(region_ids)
                selected.append(offsets)
                break
        else:
            raise NoValidRegionPartitionError(
                "No offset assigns at least one masked target token to all 64 regions"
            )
    return torch.stack(assignment), masked_indices.new_tensor(selected)


class PrimusSpatialOnlineTargetMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Online-target MAE+JEPA that exports a valid 64-region assignment."""

    def __init__(self, *args, region_fixed_offsets: bool = False, max_mask_attempts: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_fixed_offsets = bool(region_fixed_offsets)
        self.max_mask_attempts = int(max_mask_attempts)
        if self.max_mask_attempts < 1:
            raise ValueError("max_mask_attempts must be positive")
        self.region_seed = int(os.environ.get("NNSSL_EXPERIMENT_SEED", "20260821")) + 400_000
        self.register_buffer("region_rng_step", torch.zeros((), dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rejection_count = 0
        for rejection_count in range(self.max_mask_attempts):
            context, keep_indices, spatial_shape = self.encode_online(x)
            masked_indices = complement_indices(keep_indices, prod(spatial_shape))
            flat_grid_size = (spatial_shape[1], spatial_shape[0], spatial_shape[2])
            try:
                region_ids, region_offsets = select_valid_region_partition(
                    masked_indices,
                    flat_grid_size,
                    fixed=self.region_fixed_offsets,
                    seed=self.region_seed + int(self.region_rng_step.item()),
                )
                self.region_rng_step.add_(1)
                break
            except NoValidRegionPartitionError:
                continue
        else:
            raise NoValidRegionPartitionError(
                f"Failed to obtain a valid masked partition after {self.max_mask_attempts} attempts"
            )

        # All heads use the final accepted online mask. A rejected attempt never
        # contributes reconstruction, prediction, target, or regularization.
        reconstruction = self.decode_mae(context, keep_indices, spatial_shape)
        target_raw, target_normalized = self.encode_online_target(x)
        target_raw = _gather_tokens(target_raw, masked_indices)
        target = _gather_tokens(target_normalized, masked_indices)
        prediction = self.predictor(context, keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "regularizer_embedding": self.regularizer_projector(target_raw),
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
            "region_ids": region_ids,
            "region_offsets": region_offsets,
            "region_rejections": prediction.new_tensor(float(rejection_count)),
        }
