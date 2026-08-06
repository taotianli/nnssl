"""Latent prediction losses for JEPA trainers."""

import torch
from torch import nn


class JEPALatentLoss(nn.Module):
    """Mean per-token L1/Lp loss between predictions and detached targets."""

    def __init__(self, exponent: float = 1.0):
        super().__init__()
        if exponent <= 0:
            raise ValueError("exponent must be positive")
        self.exponent = float(exponent)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}")
        difference = torch.abs(prediction.float() - target.detach().float())
        return (difference.pow(self.exponent) / self.exponent).mean()
