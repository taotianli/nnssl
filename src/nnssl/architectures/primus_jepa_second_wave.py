"""Second-wave Primus MAE+JEPA architecture ablations.

These classes are intentionally additive: the original Primus MAE, JEPA and
first-wave experimental modules are not modified.  Each class changes one
coupling decision between reconstruction and latent prediction.
"""

from __future__ import annotations

from math import prod

import torch
from einops import rearrange
from torch import nn

from nnssl.architectures.primus_jepa import (
    PrimusMAEJEPA,
    _gather_tokens,
    complement_indices,
)


def _decode_complete_latent_grid(
    model: PrimusMAEJEPA,
    context: torch.Tensor,
    keep_indices: torch.Tensor,
    predicted: torch.Tensor,
    masked_indices: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Decode a complete grid whose missing tokens are predictor outputs."""
    batch, _, channels = context.shape
    full = context.new_zeros(batch, prod(spatial_shape), channels)
    full.scatter_(1, keep_indices[..., None].expand(-1, -1, channels), context)
    full.scatter_(1, masked_indices[..., None].expand(-1, -1, channels), predicted)
    decoded, _ = model.decoder(full)
    decoded = rearrange(
        decoded,
        "b (h w d) c -> b c w h d",
        h=spatial_shape[0],
        w=spatial_shape[1],
        d=spatial_shape[2],
    )
    return model.up_projection(decoded)


class PrimusSerialLatentMAEJEPA(PrimusMAEJEPA):
    """CAE-style serial coupling: predicted masked latents feed the decoder.

    In the baseline the MAE decoder receives learned mask tokens, so its pixel
    loss cannot improve the JEPA predictor. Here the predictor replaces those
    mask tokens. The same two losses remain, but both supervise the predictor.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        prediction = self.predictor(context, keep_indices, masked_indices)
        reconstruction = _decode_complete_latent_grid(
            self, context, keep_indices, prediction, masked_indices, spatial_shape
        )
        with torch.no_grad():
            target = _gather_tokens(self.encode_target(x), masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
        }


class PrimusGatedVisibleSkipMAEJEPA(PrimusMAEJEPA):
    """BootMAE-style low-level visible-token skip into the MAE decoder.

    The zero-initialized bounded gate makes initialization exactly equivalent
    to the public MAE path. Only the reconstruction decoder sees the skip; the
    JEPA predictor must still use semantic encoder context.
    """

    def __init__(self, *args, visible_skip_max: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        if visible_skip_max <= 0:
            raise ValueError("visible_skip_max must be positive")
        self.visible_skip_max = float(visible_skip_max)
        self.decoder_skip_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_skip_gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.down_projection(x)
        spatial_shape = tuple(int(i) for i in projected.shape[2:])
        raw_tokens = rearrange(projected, "b c w h d -> b (h w d) c")
        context, keep_indices = self.eva(raw_tokens)
        if keep_indices is None:
            keep_indices = torch.arange(raw_tokens.shape[1], device=x.device).expand(x.shape[0], -1)
        keep_indices = keep_indices.long()
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))

        visible_raw = _gather_tokens(raw_tokens, keep_indices)
        gate = self.visible_skip_max * torch.tanh(self.decoder_skip_gate)
        decoder_context = context + gate * self.decoder_skip_norm(visible_raw)
        reconstruction = self.decode_mae(decoder_context, keep_indices, spatial_shape)
        with torch.no_grad():
            target = _gather_tokens(self.encode_target(x), masked_indices)
        prediction = self.predictor(context, keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
            "decoder_skip_gate": gate,
        }


class PrimusIndependentCrossViewMAEJEPA(PrimusMAEJEPA):
    """Independent MAE/JEPA masks plus an aligned MRI intensity target.

    A first masked online pass drives reconstruction and a second independently
    masked pass supplies JEPA context. Spatial coordinates remain unchanged so
    the weak gain/bias/noise teacher view has exact token correspondence. The
    target intensity augmentation is disabled under validation ``no_grad``;
    nnSSL's validation patch mask remains stochastic by design.
    """

    def __init__(
        self,
        *args,
        intensity_gain: float = 0.10,
        intensity_bias_std: float = 0.05,
        intensity_noise_std: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intensity_gain = float(intensity_gain)
        self.intensity_bias_std = float(intensity_bias_std)
        self.intensity_noise_std = float(intensity_noise_std)
        if min(self.intensity_gain, self.intensity_bias_std, self.intensity_noise_std) < 0:
            raise ValueError("intensity-view magnitudes must be non-negative")

    @torch.no_grad()
    def make_target_view(self, x: torch.Tensor, augment: bool) -> torch.Tensor:
        if not augment:
            return x
        shape = (x.shape[0], 1, 1, 1, 1)
        gain = 1.0 + (2.0 * torch.rand(shape, device=x.device, dtype=x.dtype) - 1.0) * self.intensity_gain
        bias = torch.randn(shape, device=x.device, dtype=x.dtype) * self.intensity_bias_std
        noise = torch.randn_like(x) * self.intensity_noise_std
        return x * gain + bias + noise

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mae_context, mae_keep_indices, spatial_shape = self.encode_online(x)
        jepa_context, jepa_keep_indices, jepa_spatial_shape = self.encode_online(x)
        if jepa_spatial_shape != spatial_shape:
            raise RuntimeError("MAE and JEPA online views produced different patch grids")
        masked_indices = complement_indices(jepa_keep_indices, prod(spatial_shape))
        reconstruction = self.decode_mae(mae_context, mae_keep_indices, spatial_shape)
        # nnSSL deliberately leaves masked models in train mode for validation;
        # grad mode, rather than self.training, separates the two call sites.
        target_view = self.make_target_view(x, augment=torch.is_grad_enabled())
        with torch.no_grad():
            target = _gather_tokens(self.encode_target(target_view), masked_indices)
        prediction = self.predictor(jepa_context, jepa_keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target,
            "keep_indices": mae_keep_indices,
            "jepa_keep_indices": jepa_keep_indices,
            "masked_indices": masked_indices,
        }
