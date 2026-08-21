"""Collapse-prevention losses for teacher-free latent prediction."""

from __future__ import annotations

import torch
from torch import nn


def _flatten_and_subsample(z: torch.Tensor, max_samples: int) -> torch.Tensor:
    if z.ndim < 2:
        raise ValueError("representations must have at least two dimensions")
    z = z.reshape(-1, z.shape[-1]).float()
    if z.shape[0] > max_samples:
        # Random sampling avoids systematically favouring one spatial region.
        indices = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[indices]
    return z


class VICRegVarianceCovarianceLoss(nn.Module):
    """VICReg variance/covariance terms; JEPA supplies the invariance term."""

    def __init__(
        self,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.04,
        target_std: float = 1.0,
        max_samples: int = 2048,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.target_std = float(target_std)
        self.max_samples = int(max_samples)
        self.eps = float(eps)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        z = _flatten_and_subsample(representations, self.max_samples)
        if z.shape[0] < 2:
            raise ValueError("VICReg requires at least two representation samples")
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        variance = torch.relu(self.target_std - std).mean()
        covariance = z.T @ z / (z.shape[0] - 1)
        covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance = covariance.square().sum() / covariance.shape[0]
        return self.variance_weight * variance + self.covariance_weight * covariance


class SIGRegLoss(nn.Module):
    """Stochastic sliced Epps--Pulley Gaussianity regularization.

    This is a compact single-GPU implementation of the SIGReg principle used by
    LeJEPA: random one-dimensional projections are matched to a standard normal
    through their empirical characteristic functions.  It has no teacher,
    stop-gradient, covariance matrix, or cross-device gather requirement.
    """

    def __init__(
        self,
        num_slices: int = 256,
        num_knots: int = 17,
        max_samples: int = 2048,
        max_frequency: float = 3.0,
    ):
        super().__init__()
        if num_slices <= 0 or num_knots < 2:
            raise ValueError("SIGReg requires positive slices and at least two knots")
        self.num_slices = int(num_slices)
        self.num_knots = int(num_knots)
        self.max_samples = int(max_samples)
        self.max_frequency = float(max_frequency)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        z = _flatten_and_subsample(representations, self.max_samples)
        if z.shape[0] < 2:
            raise ValueError("SIGReg requires at least two representation samples")
        directions = torch.randn(
            z.shape[1], self.num_slices, device=z.device, dtype=z.dtype
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-6)
        projections = z @ directions
        frequencies = torch.linspace(
            0.0, self.max_frequency, self.num_knots, device=z.device, dtype=z.dtype
        )
        step = self.max_frequency / (self.num_knots - 1)
        total = z.new_zeros(())
        for index, frequency in enumerate(frequencies):
            phase = projections * frequency
            empirical_real = phase.cos().mean(dim=0)
            empirical_imag = phase.sin().mean(dim=0)
            normal_real = torch.exp(-0.5 * frequency.square())
            discrepancy = (empirical_real - normal_real).square() + empirical_imag.square()
            trapezoid_weight = 0.5 if index in (0, self.num_knots - 1) else 1.0
            total = total + trapezoid_weight * discrepancy.mean()
        return total * step


class VISRegLoss(nn.Module):
    """FP32 variance/center/sliced-Wasserstein regularization.

    JEPA supplies the invariance/prediction term. Shape is computed after
    stop-gradient standardization so its gradients do not change feature scale.
    """

    def __init__(
        self,
        num_slices: int = 256,
        max_samples: int = 2048,
        scale_weight: float = 1.0,
        shape_weight: float = 1.0,
        center_weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        if num_slices <= 0 or max_samples < 2:
            raise ValueError("VISReg requires positive slices and at least two samples")
        self.num_slices = int(num_slices)
        self.max_samples = int(max_samples)
        self.scale_weight = float(scale_weight)
        self.shape_weight = float(shape_weight)
        self.center_weight = float(center_weight)
        self.target_std = float(target_std)
        self.eps = float(eps)

    def components(self, representations: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=representations.device.type, enabled=False):
            return self._components_fp32(representations.float())

    def _components_fp32(self, representations: torch.Tensor) -> dict[str, torch.Tensor]:
        z = _flatten_and_subsample(representations, self.max_samples)
        if z.shape[0] < 2:
            raise ValueError("VISReg requires at least two representation samples")
        center_vector = z.mean(dim=0, keepdim=True)
        centered = z - center_vector
        std = centered.norm(dim=0).div(z.shape[0] ** 0.5).clamp_min(self.eps)
        scale = (std - self.target_std).square().mean()
        center = center_vector.square().mean()

        standardized = centered / std.detach().clamp_min(self.eps)
        directions = torch.randn(
            standardized.shape[1], self.num_slices, device=z.device, dtype=torch.float32
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(self.eps)
        projected = standardized @ directions
        projected = projected.sort(dim=0).values
        quantiles = torch.linspace(
            1,
            projected.shape[0],
            projected.shape[0],
            device=z.device,
            dtype=torch.float32,
        ) / (projected.shape[0] + 1)
        gaussian = torch.sqrt(z.new_tensor(2.0)) * torch.erfinv(2.0 * quantiles - 1.0)
        shape = (projected - gaussian[:, None]).square().mean()
        total = (
            self.scale_weight * scale
            + self.shape_weight * shape
            + self.center_weight * center
        )
        return {"total": total, "scale": scale, "shape": shape, "center": center}

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.components(representations)["total"]


def _region_mean_and_residual(
    representations: torch.Tensor,
    region_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if representations.ndim != 3 or region_ids.shape != representations.shape[:2]:
        raise ValueError("expected representations [B, M, D] and region_ids [B, M]")
    means_per_volume = []
    residuals = []
    for z, ids in zip(representations.float(), region_ids.long()):
        counts = torch.bincount(ids, minlength=64)
        if counts.shape[0] != 64 or counts.min().item() < 1:
            raise ValueError("every volume must contain all 64 active regions")
        sums = z.new_zeros(64, z.shape[-1]).index_add(0, ids, z)
        means = sums / counts[:, None].to(dtype=z.dtype)
        means_per_volume.append(means)
        residuals.append(z - means[ids])
    return torch.cat(means_per_volume, dim=0), residuals


class RegionVISRegLoss(nn.Module):
    """VISReg over a balanced pool of 64 region means per volume."""

    def __init__(self, **visreg_kwargs):
        super().__init__()
        self.visreg = VISRegLoss(**visreg_kwargs)

    def components(
        self, representations: torch.Tensor, region_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=representations.device.type, enabled=False):
            region_means, _ = _region_mean_and_residual(representations.float(), region_ids)
            return self.visreg.components(region_means)

    def forward(self, representations: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        return self.components(representations, region_ids)["total"]


class ORRRegLoss(nn.Module):
    """Orthogonal Region--Residual regularization for masked 3-D tokens."""

    def __init__(
        self,
        residual_weight: float = 1.0,
        residual_max_samples: int = 512,
        target_std: float = 1.0,
        shuffled_regions: bool = False,
        **visreg_kwargs,
    ):
        super().__init__()
        self.coarse_visreg = VISRegLoss(**visreg_kwargs)
        self.residual_weight = float(residual_weight)
        self.residual_max_samples = int(residual_max_samples)
        self.target_std = float(target_std)
        self.shuffled_regions = bool(shuffled_regions)

    def components(
        self, representations: torch.Tensor, region_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=representations.device.type, enabled=False):
            ids = region_ids
            if self.shuffled_regions:
                ids = torch.stack(
                    [
                        volume_ids[
                            torch.randperm(volume_ids.numel(), device=volume_ids.device)
                        ]
                        for volume_ids in ids
                    ]
                )
            region_means, residuals = _region_mean_and_residual(representations.float(), ids)
            coarse_components = self.coarse_visreg.components(region_means)
            residual_losses = []
            for residual in residuals:
                if residual.shape[0] > self.residual_max_samples:
                    selection = torch.randperm(residual.shape[0], device=residual.device)[
                        : self.residual_max_samples
                    ]
                    residual = residual[selection]
                std = torch.sqrt(residual.var(dim=0, unbiased=False) + 1e-4)
                residual_losses.append(torch.relu(self.target_std - std).square().mean())
            residual_floor = torch.stack(residual_losses).mean()
            total = coarse_components["total"] + self.residual_weight * residual_floor
            return {
                "total": total,
                "coarse": coarse_components["total"],
                "residual": residual_floor,
                "scale": coarse_components["scale"],
                "shape": coarse_components["shape"],
                "center": coarse_components["center"],
            }

    def forward(self, representations: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        return self.components(representations, region_ids)["total"]
