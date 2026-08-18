"""Experimental Primus MAE+JEPA architectures.

The variants in this module deliberately leave :mod:`primus_jepa` unchanged so
that old checkpoints and trainer names retain their original behaviour.
"""

from __future__ import annotations

from copy import deepcopy
from math import prod
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nnssl.architectures.primus_jepa import (
    PrimusMAEJEPA,
    _gather_tokens,
    complement_indices,
)


def select_contiguous_masked_tokens(
    masked_indices: torch.Tensor,
    grid_size: Sequence[int],
    num_targets: int,
) -> torch.Tensor:
    """Select a compact 3-D target block without exposing MAE-masked tokens.

    A random masked token is used as a centre and the nearest masked positions
    are retained.  Consequently every JEPA target remains hidden from the
    online encoder, while the MAE mask itself is unchanged.
    """
    if masked_indices.ndim != 2:
        raise ValueError("masked_indices must have shape [batch, tokens]")
    if num_targets <= 0:
        raise ValueError("num_targets must be positive")
    count = min(int(num_targets), masked_indices.shape[1])
    if count == masked_indices.shape[1]:
        return masked_indices

    device = masked_indices.device
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(int(n), device=device) for n in grid_size), indexing="ij"),
        dim=-1,
    ).reshape(-1, len(grid_size))
    masked_coordinates = coordinates[masked_indices.long()]
    centre_slot = torch.randint(masked_indices.shape[1], (masked_indices.shape[0],), device=device)
    centres = masked_coordinates[torch.arange(masked_indices.shape[0], device=device), centre_slot]
    distances = (masked_coordinates.float() - centres[:, None].float()).square().sum(dim=-1)
    nearest = distances.topk(count, dim=1, largest=False, sorted=False).indices
    return torch.gather(masked_indices, 1, nearest)


class PrimusBlockTargetMAEJEPA(PrimusMAEJEPA):
    """MAE random reconstruction mask plus an independent compact JEPA target."""

    def __init__(self, *args, jepa_num_targets: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.jepa_num_targets = int(jepa_num_targets)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        mae_masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        # encode_online flattens as (height, width, depth), see its einops pattern.
        flat_grid_size = (spatial_shape[1], spatial_shape[0], spatial_shape[2])
        target_indices = select_contiguous_masked_tokens(
            mae_masked_indices, flat_grid_size, self.jepa_num_targets
        )
        reconstruction = self.decode_mae(context, keep_indices, spatial_shape)
        with torch.no_grad():
            target = _gather_tokens(self.encode_target(x), target_indices)
        prediction = self.predictor(context, keep_indices, target_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "keep_indices": keep_indices,
            "masked_indices": mae_masked_indices,
            "jepa_target_indices": target_indices,
        }


class PrimusDualTeacherMAEJEPA(PrimusMAEJEPA):
    """Fuse a frozen public-MAE anchor with the adaptive EMA teacher.

    When the teachers disagree, more weight is assigned to the frozen anchor,
    reducing representation drift during continued pretraining.
    """

    def __init__(
        self,
        *args,
        anchor_weight_min: float = 0.25,
        anchor_weight_max: float = 0.75,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 <= anchor_weight_min <= anchor_weight_max <= 1:
            raise ValueError("anchor weights must satisfy 0 <= min <= max <= 1")
        self.anchor_weight_min = float(anchor_weight_min)
        self.anchor_weight_max = float(anchor_weight_max)
        self.anchor_down_projection = deepcopy(self.down_projection)
        self.anchor_eva = deepcopy(self.eva)
        self._freeze_anchor()

    def _freeze_anchor(self) -> None:
        self.anchor_down_projection.requires_grad_(False).eval()
        self.anchor_eva.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor_down_projection.eval()
        self.anchor_eva.eval()
        return self

    @torch.no_grad()
    def synchronize_anchor_encoder(self) -> None:
        self.anchor_down_projection.load_state_dict(self.down_projection.state_dict())
        self.anchor_eva.load_state_dict(self.eva.state_dict())
        self._freeze_anchor()

    @torch.no_grad()
    def encode_anchor(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.anchor_down_projection(x)
        tokens = rearrange(projected, "b c w h d -> b (h w d) c")
        target, _ = self.anchor_eva(tokens)
        return F.layer_norm(target, (target.shape[-1],))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        reconstruction = self.decode_mae(context, keep_indices, spatial_shape)
        with torch.no_grad():
            ema = _gather_tokens(self.encode_target(x), masked_indices)
            anchor = _gather_tokens(self.encode_anchor(x), masked_indices)
            agreement = F.cosine_similarity(anchor.float(), ema.float(), dim=-1).clamp(-1, 1)
            disagreement = 0.5 * (1.0 - agreement)
            anchor_weight = self.anchor_weight_min + (
                self.anchor_weight_max - self.anchor_weight_min
            ) * disagreement
            target = F.layer_norm(
                anchor_weight[..., None] * anchor + (1.0 - anchor_weight[..., None]) * ema,
                (anchor.shape[-1],),
            )
        prediction = self.predictor(context, keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
            "anchor_weight": anchor_weight.mean(),
        }

    @torch.no_grad()
    def load_mae_state_dict(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        loaded = super().load_mae_state_dict(state_dict)
        self.synchronize_anchor_encoder()
        return loaded


class PrimusOnlineTargetMAEJEPA(PrimusMAEJEPA):
    """Teacher-free MAE+JEPA with a second online, unmasked target view.

    The target branch shares weights with the context encoder and is not
    detached.  VICReg or SIGReg is therefore required by the corresponding
    trainer to discourage collapse.  This variant costs more activation memory
    than the EMA branch and is intentionally benchmarked separately.
    """

    def __init__(self, *args, regularizer_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        del self.target_down_projection
        del self.target_eva
        self.regularizer_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, int(regularizer_dim)),
        )

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        return self

    def encode_online_target(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.down_projection(x)
        tokens = rearrange(projected, "b c w h d -> b (h w d) c")
        was_training = self.eva.training
        self.eva.eval()  # disable patch dropping for the full target view
        target, keep_indices = self.eva(tokens)
        self.eva.train(was_training)
        if keep_indices is not None and target.shape[1] != tokens.shape[1]:
            raise RuntimeError("Online target encoder unexpectedly dropped patches")
        return target, F.layer_norm(target, (target.shape[-1],))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
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
        }

    @torch.no_grad()
    def load_mae_state_dict(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        current = self.state_dict()
        loaded = []
        for raw_key, value in state_dict.items():
            key = raw_key.removeprefix("module.").removeprefix("_orig_mod.")
            if key in current and current[key].shape == value.shape and not key.startswith(
                ("predictor.", "regularizer_projector.")
            ):
                current[key] = value
                loaded.append(key)
        self.load_state_dict(current)
        return loaded
