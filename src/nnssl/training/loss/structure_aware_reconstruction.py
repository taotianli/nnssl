"""Lightweight structure-aware reconstruction losses for 3-D MRI."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_smoothed_gradient_l1(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    smoothing_kernel: int = 3,
) -> torch.Tensor:
    """Compare 3-D finite differences on edges touching masked voxels.

    Visible reconstruction values are replaced with the ground truth before
    differentiation. Consequently the auxiliary objective cannot waste effort
    on regions the encoder was allowed to observe.
    """
    if reconstruction.shape != target.shape or visible_mask.shape != target.shape:
        raise ValueError("reconstruction, target and visible_mask must have identical shapes")
    hidden = (1.0 - visible_mask).to(dtype=reconstruction.dtype)
    completed = reconstruction * hidden + target * (1.0 - hidden)
    if smoothing_kernel > 1:
        if smoothing_kernel % 2 == 0:
            raise ValueError("smoothing_kernel must be odd")
        padding = smoothing_kernel // 2
        completed = F.avg_pool3d(completed, smoothing_kernel, stride=1, padding=padding)
        target_for_gradient = F.avg_pool3d(target, smoothing_kernel, stride=1, padding=padding)
    else:
        target_for_gradient = target

    losses = []
    for axis in (2, 3, 4):
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        predicted_difference = completed[right] - completed[left]
        target_difference = target_for_gradient[right] - target_for_gradient[left]
        edge_mask = torch.maximum(hidden[right], hidden[left])
        denominator = edge_mask.sum().clamp_min(1.0)
        losses.append(((predicted_difference - target_difference).abs() * edge_mask).sum() / denominator)
    return torch.stack(losses).mean()
