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
