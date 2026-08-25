"""Pretraining-only prior heads for teacher-free Primus MAE+JEPA+VICReg."""

from __future__ import annotations

from math import prod

import torch
import torch.nn.functional as F
from torch import nn

from nnssl.architectures.primus_jepa import _gather_tokens, complement_indices
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA


class PrimusAnatomyPriorMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Predict anatomy occupancy and a signed boundary-distance proxy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anatomy_prior_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(x)
        output["anatomy_prior_prediction"] = self.anatomy_prior_head(output["prediction"])
        return output


class PrimusSoftRegionPriorMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Predict soft tissue or coarse-region distributions at masked patches."""

    def __init__(self, *args, prior_classes: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_classes = int(prior_classes)
        self.soft_region_prior_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.prior_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(x)
        output["soft_region_logits"] = self.soft_region_prior_head(output["prediction"])
        return output


class PrimusAcquisitionPriorMAEJEPA(PrimusOnlineTargetMAEJEPA):
    """Predict clean latents from a synthetically corrupted MRI context.

    Known corruption parameters condition only the MAE decoder. The JEPA
    predictor receives the unconditioned context and must recover clean-view
    targets, separating reconstruction style information from semantic latent
    prediction without changing the transferable encoder.
    """

    def __init__(
        self,
        *args,
        corruption_strength: float = 0.05,
        corruption_seed: int = 20260825,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.corruption_strength = float(corruption_strength)
        self.corruption_seed = int(corruption_seed)
        self.corruption_calls = 0
        self.acquisition_conditioner = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim),
        )
        nn.init.zeros_(self.acquisition_conditioner[-1].weight)
        nn.init.zeros_(self.acquisition_conditioner[-1].bias)

    def _corrupt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, strength = x.shape[0], self.corruption_strength
        devices = [x.device.index] if x.device.type == "cuda" else []
        # Keep acquisition randomness independent from patch masks and DropPath.
        with torch.random.fork_rng(devices=devices):
            seed = self.corruption_seed + self.corruption_calls
            torch.manual_seed(seed)
            if x.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            random = torch.rand(batch, 4, device=x.device, dtype=torch.float32)
            gain = 1.0 + (2.0 * random[:, 0] - 1.0) * strength
            bias = (2.0 * random[:, 1] - 1.0) * strength
            noise = random[:, 2] * strength
            blur_mix = random[:, 3] * min(0.5, 2.0 * strength)
            view_shape = (batch, 1, 1, 1, 1)
            corrupted = x * gain.view(view_shape).to(x.dtype)
            corrupted = corrupted + bias.view(view_shape).to(x.dtype)
            corrupted = corrupted + torch.randn_like(x) * noise.view(view_shape).to(x.dtype)
        self.corruption_calls += 1
        blurred = F.avg_pool3d(corrupted, kernel_size=3, stride=1, padding=1)
        corrupted = torch.lerp(corrupted, blurred, blur_mix.view(view_shape).to(x.dtype))
        parameters = torch.stack((gain - 1.0, bias, noise, blur_mix), dim=1)
        return corrupted, parameters

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context_view, corruption = self._corrupt(x)
        context, keep_indices, spatial_shape = self.encode_online(context_view)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))

        style = self.acquisition_conditioner(corruption.to(context.dtype)).unsqueeze(1)
        reconstruction = self.decode_mae(context + style, keep_indices, spatial_shape)

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
            "corruption_parameters": corruption,
        }
