"""Auxiliary MRI priors for MAE+JEPA+VICReg pretraining."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from nnssl.architectures.primus_jepa import _gather_tokens


def _pool_to_patches(volume: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    return F.avg_pool3d(volume.float(), kernel_size=tuple(patch_size), stride=tuple(patch_size))


def support_from_optional_mask(
    data: torch.Tensor,
    anatomy_mask: torch.Tensor | None,
    availability: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return foreground support and per-volume confidence.

    Anatomy-mask polarity differs across upstream sources. We choose the binary
    orientation with greater mean absolute image signal inside the valid image
    support. Missing masks fall back to non-zero image support at lower weight.
    """

    signal = data.detach().float().abs()
    valid_image = signal > 1e-6
    fallback = valid_image.float()
    batch = data.shape[0]
    if anatomy_mask is None:
        return fallback, data.new_full((batch,), 0.25, dtype=torch.float32)

    raw = anatomy_mask.detach().float() > 0.5
    candidate_one = raw & valid_image
    candidate_zero = (~raw) & valid_image

    def score(mask: torch.Tensor) -> torch.Tensor:
        axes = tuple(range(1, mask.ndim))
        return (signal * mask).sum(dim=axes) / mask.sum(dim=axes).clamp_min(1)

    choose_one = score(candidate_one) >= score(candidate_zero)
    support = torch.where(
        choose_one.view(batch, 1, 1, 1, 1), candidate_one, candidate_zero
    ).float()
    if availability is None:
        available = torch.ones(batch, device=data.device, dtype=torch.bool)
    else:
        available = availability.to(device=data.device).bool().reshape(batch)
    support = torch.where(
        available.view(batch, 1, 1, 1, 1), support, fallback
    )
    confidence = torch.where(
        available,
        torch.ones(batch, device=data.device),
        torch.full((batch,), 0.25, device=data.device),
    )
    return support, confidence


def _signed_grid_distance(binary: torch.Tensor, steps: int = 4) -> torch.Tensor:
    """Cheap signed distance proxy on the patch grid using iterative erosion."""

    binary = binary.float()

    def erosion(x: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.max_pool3d(1.0 - x, kernel_size=3, stride=1, padding=1)

    inside_state = binary
    outside_state = 1.0 - binary
    inside = torch.zeros_like(binary)
    outside = torch.zeros_like(binary)
    for _ in range(int(steps)):
        inside_state = erosion(inside_state)
        outside_state = erosion(outside_state)
        inside = inside + inside_state
        outside = outside + outside_state
    return (inside - outside) / max(1, int(steps))


def anatomy_prior_loss(
    prediction: torch.Tensor,
    masked_indices: torch.Tensor,
    data: torch.Tensor,
    anatomy_mask: torch.Tensor | None,
    availability: torch.Tensor | None,
    patch_size: Sequence[int],
    sdf_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support, confidence = support_from_optional_mask(data, anatomy_mask, availability)
    occupancy_grid = _pool_to_patches(support, patch_size).clamp(0, 1)
    signed_grid = _signed_grid_distance((occupancy_grid >= 0.5).float())
    occupancy = rearrange(occupancy_grid, "b c w h d -> b (h w d) c")
    signed = rearrange(signed_grid, "b c w h d -> b (h w d) c")
    occupancy = _gather_tokens(occupancy, masked_indices)
    signed = _gather_tokens(signed, masked_indices)

    sample_weight = confidence[:, None, None]
    occupancy_loss = F.binary_cross_entropy_with_logits(
        prediction[..., :1].float(), occupancy, reduction="none"
    )
    occupancy_loss = (occupancy_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1)
    occupancy_loss = occupancy_loss / occupancy.shape[1]
    sdf_loss = F.smooth_l1_loss(
        prediction[..., 1:2].float().tanh(), signed, reduction="none"
    )
    sdf_loss = (sdf_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1)
    sdf_loss = sdf_loss / signed.shape[1]
    total = occupancy_loss + float(sdf_weight) * sdf_loss
    return total, {
        "occupancy": occupancy_loss,
        "signed_distance": sdf_loss,
        "mask_coverage": (confidence >= 1.0).float().mean(),
    }


def soft_region_prior_loss(
    logits: torch.Tensor,
    masked_indices: torch.Tensor,
    data: torch.Tensor,
    anatomy_mask: torch.Tensor | None,
    availability: torch.Tensor | None,
    patch_size: Sequence[int],
    num_classes: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support, confidence = support_from_optional_mask(data, anatomy_mask, availability)
    occupancy = _pool_to_patches(support, patch_size).clamp(0, 1)
    patch_signal = _pool_to_patches(data.float() * support, patch_size)
    patch_signal = patch_signal / occupancy.clamp_min(1e-4)
    batch, _, grid_w, grid_h, grid_d = occupancy.shape

    if int(num_classes) == 3:
        flat_signal = patch_signal.reshape(batch, -1)
        flat_weight = occupancy.reshape(batch, -1)
        mean = (flat_signal * flat_weight).sum(1) / flat_weight.sum(1).clamp_min(1)
        variance = (
            (flat_signal - mean[:, None]).square() * flat_weight
        ).sum(1) / flat_weight.sum(1).clamp_min(1)
        normalized = (patch_signal - mean[:, None, None, None, None]) / torch.sqrt(
            variance[:, None, None, None, None] + 1e-4
        )
        centers = data.new_tensor((-1.0, 0.0, 1.0)).view(1, 3, 1, 1, 1)
        target = torch.softmax(-((normalized - centers) / 0.75).square(), dim=1)
    elif int(num_classes) == 8:
        w = torch.arange(grid_w, device=data.device) >= grid_w / 2
        h = torch.arange(grid_h, device=data.device) >= grid_h / 2
        d = torch.arange(grid_d, device=data.device) >= grid_d / 2
        region = (
            w[:, None, None].long() * 4
            + h[None, :, None].long() * 2
            + d[None, None, :].long()
        )
        target = F.one_hot(region, num_classes=8).permute(3, 0, 1, 2).float()
        target = target[None].expand(batch, -1, -1, -1, -1)
    else:
        raise ValueError(f"soft region prior supports 3 or 8 classes, got {num_classes}")

    target = rearrange(target, "b c w h d -> b (h w d) c")
    target = _gather_tokens(target, masked_indices)
    token_weight = _gather_tokens(
        rearrange(occupancy, "b c w h d -> b (h w d) c"), masked_indices
    ) * confidence[:, None, None]
    cross_entropy = -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1, keepdim=True)
    loss = (cross_entropy * token_weight).sum() / token_weight.sum().clamp_min(1)
    entropy = -(target.clamp_min(1e-8) * target.clamp_min(1e-8).log()).sum(-1)
    active = token_weight.squeeze(-1) > 0
    mean_entropy = entropy[active].mean() if active.any() else entropy.new_zeros(())
    return loss, {
        "region_entropy": mean_entropy,
        "mask_coverage": (confidence >= 1.0).float().mean(),
    }


def spectral_shell_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    num_shells: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.autocast(device_type=reconstruction.device.type, enabled=False):
        hidden = 1.0 - visible_mask.float()
        completed = reconstruction.float() * hidden + target.float() * visible_mask.float()
        predicted_fft = torch.fft.rfftn(completed, dim=(-3, -2, -1), norm="ortho").abs()
        target_fft = torch.fft.rfftn(target.float(), dim=(-3, -2, -1), norm="ortho").abs()
        predicted_fft = torch.log1p(predicted_fft)
        target_fft = torch.log1p(target_fft)
        axes = tuple(range(2, predicted_fft.ndim))
        predicted_fft = predicted_fft / predicted_fft.mean(dim=axes, keepdim=True).clamp_min(1e-6)
        target_fft = target_fft / target_fft.mean(dim=axes, keepdim=True).clamp_min(1e-6)

        fw = torch.fft.fftfreq(target.shape[-3], device=target.device).abs()
        fh = torch.fft.fftfreq(target.shape[-2], device=target.device).abs()
        fd = torch.fft.rfftfreq(target.shape[-1], device=target.device).abs()
        radius = torch.sqrt(
            fw[:, None, None].square() + fh[None, :, None].square() + fd[None, None, :].square()
        )
        radius = radius / radius.max().clamp_min(1e-6)
        shell_ids = torch.clamp((radius * int(num_shells)).long(), max=int(num_shells) - 1)
        predicted_flat = predicted_fft.flatten(start_dim=2)
        target_flat = target_fft.flatten(start_dim=2)
        shell_ids = shell_ids.flatten()
        shell_losses = []
        for shell in range(int(num_shells)):
            selected = shell_ids == shell
            if not selected.any():
                shell_losses.append(predicted_flat.new_zeros(()))
                continue
            predicted_energy = predicted_flat[..., selected].mean(dim=-1)
            target_energy = target_flat[..., selected].mean(dim=-1)
            shell_losses.append((predicted_energy - target_energy).abs().mean())
        shell_values = torch.stack(shell_losses)
        weights = torch.linspace(1.0, 2.0, int(num_shells), device=target.device)
        total = (shell_values * weights).sum() / weights.sum()
        return total, {
            "spectral_low": shell_values[: max(1, int(num_shells) // 3)].mean(),
            "spectral_high": shell_values[-max(1, int(num_shells) // 3) :].mean(),
        }


def _neighbor_offsets(neighborhood: int) -> list[tuple[int, int, int]]:
    if int(neighborhood) == 6:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if int(neighborhood) != 26:
        raise ValueError(f"neighborhood must be 6 or 26, got {neighborhood}")
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        first_nonzero = next(value for value in offset if value != 0)
        if first_nonzero > 0:
            offsets.append(offset)
    return offsets


def spatial_heat_kernel_loss(
    embeddings: torch.Tensor,
    masked_indices: torch.Tensor,
    data: torch.Tensor,
    patch_size: Sequence[int],
    neighborhood: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        batch, _, channels = embeddings.shape
        grid_w, grid_h, grid_d = [int(s // p) for s, p in zip(data.shape[-3:], patch_size)]
        num_tokens = grid_w * grid_h * grid_d
        full = embeddings.float().new_zeros(batch, num_tokens, channels)
        valid = torch.zeros(batch, num_tokens, device=data.device, dtype=torch.bool)
        full = full.scatter(
            1, masked_indices[..., None].expand(-1, -1, channels), embeddings.float()
        )
        valid.scatter_(1, masked_indices, True)
        full = full.reshape(batch, grid_h, grid_w, grid_d, channels)
        valid = valid.reshape(batch, grid_h, grid_w, grid_d)
        intensity = _pool_to_patches(data.float(), patch_size)
        intensity = rearrange(intensity, "b c w h d -> b h w d c")

        numerator = embeddings.float().new_zeros(())
        denominator = embeddings.float().new_zeros(())
        affinity_sum = embeddings.float().new_zeros(())
        pair_count = embeddings.float().new_zeros(())
        for dw, dh, dd in _neighbor_offsets(neighborhood):
            def slices(delta: int, size: int):
                return (slice(max(0, delta), min(size, size + delta)), slice(max(0, -delta), min(size, size - delta)))

            dst_w, src_w = slices(dw, grid_h)
            dst_h, src_h = slices(dh, grid_w)
            dst_d, src_d = slices(dd, grid_d)
            src = full[:, src_w, src_h, src_d]
            dst = full[:, dst_w, dst_h, dst_d]
            pair_valid = valid[:, src_w, src_h, src_d] & valid[:, dst_w, dst_h, dst_d]
            if not pair_valid.any():
                continue
            src_i = intensity[:, src_w, src_h, src_d]
            dst_i = intensity[:, dst_w, dst_h, dst_d]
            difference = (src_i - dst_i).abs().squeeze(-1)
            scale = difference[pair_valid].median().detach().clamp_min(1e-3)
            affinity = torch.exp(-difference.square() / (2.0 * scale.square())).detach()
            feature_distance = (src - dst).square().mean(dim=-1)
            weights = affinity * pair_valid.float()
            numerator = numerator + (weights * feature_distance).sum()
            denominator = denominator + weights.sum()
            affinity_sum = affinity_sum + affinity[pair_valid].sum()
            pair_count = pair_count + pair_valid.sum()
        total = numerator / denominator.clamp_min(1)
        return total, {
            "heat_affinity": affinity_sum / pair_count.clamp_min(1),
            "heat_pairs": pair_count,
        }
