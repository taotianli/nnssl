"""Three independent modules for MAE + teacher-free JEPA + VICReg.

The variants isolate three different intervention axes:

* HOP changes where reconstruction supervision enters the encoder;
* ELC adds local 3-D equivariant correspondence;
* DOA adds objective-specific low-rank residual capacity.

All preserve the standard transferable ``down_projection`` + ``eva`` encoder.
"""

from __future__ import annotations

from math import prod

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nnssl.architectures.primus_jepa import _gather_tokens, complement_indices
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA


class ZeroResidualProjection(nn.Module):
    """A zero-output residual MLP that preserves the public MAE at init."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(self.norm(x))))


class PrimusHierarchicalObjectiveMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Place MAE reconstruction at an intermediate encoder depth.

    Blocks after ``reconstruction_block`` receive JEPA/VICReg gradients but no
    direct voxel-reconstruction gradient. The downstream representation remains
    the ordinary final EVA output.
    """

    def __init__(self, *args, reconstruction_block: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        depth = len(self.eva.blocks)
        if not 1 <= reconstruction_block <= depth:
            raise ValueError(
                f"reconstruction_block must be in [1, {depth}], got {reconstruction_block}"
            )
        self.reconstruction_block = int(reconstruction_block)

    def encode_online_hierarchical(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        projected = self.down_projection(x)
        spatial_shape = tuple(int(value) for value in projected.shape[2:])
        tokens = rearrange(projected, "b c w h d -> b (h w d) c")
        captured: list[torch.Tensor] = []

        def capture(_module, _inputs, output):
            captured.append(output)

        handle = self.eva.blocks[self.reconstruction_block - 1].register_forward_hook(capture)
        try:
            final, keep_indices = self.eva(tokens)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one HOP activation, captured {len(captured)} at block "
                f"{self.reconstruction_block}"
            )
        if keep_indices is None:
            keep_indices = torch.arange(tokens.shape[1], device=x.device).expand(x.shape[0], -1)
        # Parameter-free normalization keeps HOP a pure objective-placement
        # intervention rather than adding reconstruction-specific capacity.
        decoder_context = F.layer_norm(captured[0], (captured[0].shape[-1],))
        return final, decoder_context, keep_indices.long(), spatial_shape

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, decoder_context, keep_indices, spatial_shape = self.encode_online_hierarchical(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        reconstruction = self.decode_mae(decoder_context, keep_indices, spatial_shape)
        target_raw, target_normalized = self.encode_online_target(x)
        target_raw_masked = _gather_tokens(target_raw, masked_indices)
        prediction = self.predictor(context, keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": _gather_tokens(target_normalized, masked_indices),
            "regularizer_embedding": self.regularizer_projector(target_raw_masked),
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
        }


def align_flipped_tokens(
    tokens: torch.Tensor, spatial_shape: tuple[int, int, int], input_axis: int
) -> torch.Tensor:
    """Undo an input-volume flip on Primus' flattened ``(H,W,D)`` tokens."""
    if input_axis not in (2, 3, 4):
        raise ValueError("input_axis must be a spatial tensor axis: 2, 3, or 4")
    w, h, d = (int(value) for value in spatial_shape)
    grid = rearrange(tokens, "b (h w d) c -> b h w d c", h=h, w=w, d=d)
    grid_axis = {2: 2, 3: 1, 4: 3}[input_axis]
    return rearrange(torch.flip(grid, dims=(grid_axis,)), "b h w d c -> b (h w d) c")


class PrimusEquivariantLocalMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Add coordinate-aligned local correspondence to global VICReg."""

    def __init__(
        self,
        *args,
        local_projection_dim: int = 128,
        flip_axis: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if flip_axis not in (2, 3, 4):
            raise ValueError("flip_axis must be 2, 3, or 4")
        self.flip_axis = int(flip_axis)
        self.local_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, int(local_projection_dim)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        reconstruction = self.decode_mae(context, keep_indices, spatial_shape)

        target_raw, target_normalized = self.encode_online_target(x)
        _, flipped_normalized = self.encode_online_target(
            torch.flip(x, dims=(self.flip_axis,))
        )
        aligned_flipped = align_flipped_tokens(
            flipped_normalized, spatial_shape, self.flip_axis
        )
        prediction = self.predictor(context, keep_indices, masked_indices)
        target_raw_masked = _gather_tokens(target_raw, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": _gather_tokens(target_normalized, masked_indices),
            "regularizer_embedding": self.regularizer_projector(target_raw_masked),
            "local_embedding": self.local_projector(target_normalized),
            "local_embedding_aligned": self.local_projector(aligned_flipped),
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
        }


class PrimusDualObjectiveAdapterMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Give reconstruction and prediction separate low-rank residual capacity."""

    def __init__(self, *args, adapter_rank: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.mae_adapter = ZeroResidualProjection(self.embed_dim, int(adapter_rank))
        self.jepa_adapter = ZeroResidualProjection(self.embed_dim, int(adapter_rank))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        mae_context = shared_context + self.mae_adapter(shared_context)
        jepa_context = shared_context + self.jepa_adapter(shared_context)
        reconstruction = self.decode_mae(mae_context, keep_indices, spatial_shape)

        target_raw, target_normalized = self.encode_online_target(x)
        jepa_target = target_normalized + self.jepa_adapter(target_normalized)
        regularized_target = target_raw + self.jepa_adapter(target_raw)
        prediction = self.predictor(jepa_context, keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": _gather_tokens(jepa_target, masked_indices),
            # Regularize the adapter-modified target itself. Regularizing only
            # the base target would leave a low-rank collapse shortcut through
            # the JEPA-specific adapter. Raw-token input preserves exact
            # equivalence to the matched baseline at zero initialization.
            "regularizer_embedding": self.regularizer_projector(
                _gather_tokens(regularized_target, masked_indices)
            ),
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
        }
