"""Independent HOP, ELC and DOA experiments on MAE+JEPA+VICReg."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nnssl.architectures.primus_jepa_parallel_modules import (
    PrimusDualObjectiveAdapterMAEJEPA,
    PrimusEquivariantLocalMAEJEPA,
    PrimusHierarchicalObjectiveMAEJEPA,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPARegularizedTrainer import (
    PrimusJEPARegularized_200ep_BS4,
)


class _PrimusVICRegModuleTrainer(PrimusJEPARegularized_200ep_BS4):
    """Matched 200-epoch teacher-free MAE+JEPA+VICReg recipe."""

    regularizer_kind = "vicreg"
    extra_loss_weight = 0.005


class PrimusJEPAHOPB8_200ep_BS4(_PrimusVICRegModuleTrainer):
    """MAE taps block 8; final blocks are supervised by JEPA+VICReg."""

    architecture_class = PrimusHierarchicalObjectiveMAEJEPA
    architecture_kwargs = {"reconstruction_block": 8}


class PrimusJEPAELCVIC0p01_200ep_BS4(_PrimusVICRegModuleTrainer):
    """Coordinate-aligned local VICReg plus the matched global VICReg."""

    architecture_class = PrimusEquivariantLocalMAEJEPA
    architecture_kwargs = {"local_projection_dim": 128, "flip_axis": 4}
    auxiliary_prefixes_extra = ("local_projector.",)
    local_consistency_weight = 0.01
    local_max_samples = 2048

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_local_loss = torch.tensor(0.0)
        self._last_local_components = {
            "local_invariance": torch.tensor(0.0),
            "local_variance": torch.tensor(0.0),
            "local_covariance": torch.tensor(0.0),
        }
        for key in (
            "train_local_consistency_losses",
            "val_local_consistency_losses",
            "train_local_invariance_losses",
            "val_local_invariance_losses",
            "train_local_variance_losses",
            "val_local_variance_losses",
            "train_local_covariance_losses",
            "val_local_covariance_losses",
            "local_consistency_weights",
        ):
            self.logger.my_fantastic_logging.setdefault(key, [])

    def _local_loss(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.reshape(-1, first.shape[-1]).float()
            second = second.reshape(-1, second.shape[-1]).float()
            if first.shape != second.shape:
                raise RuntimeError("ELC feature pairs are not shape aligned")
            if first.shape[0] > self.local_max_samples:
                # Deterministic, spatially distributed sampling does not
                # perturb the matched mask/DropPath RNG stream.
                indices = torch.linspace(
                    0, first.shape[0] - 1, self.local_max_samples, device=first.device
                ).long()
                first = first[indices]
                second = second[indices]
            invariance = F.smooth_l1_loss(first, second)
            joined = torch.cat((first, second), dim=0)
            centered = joined - joined.mean(dim=0, keepdim=True)
            std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
            variance = torch.relu(1.0 - std).mean()
            covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
            covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
            covariance = covariance.square().sum() / covariance.shape[0]
            total = invariance + variance + 0.04 * covariance
            return total, {
                "local_invariance": invariance,
                "local_variance": variance,
                "local_covariance": covariance,
            }

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        loss, loss_mae, loss_jepa, loss_extra, weight = super()._losses(data, output)
        local, components = self._local_loss(
            output["local_embedding"], output["local_embedding_aligned"]
        )
        self._last_local_loss = local.detach()
        self._last_local_components = {
            key: value.detach() for key, value in components.items()
        }
        return (
            loss + self.local_consistency_weight * local,
            loss_mae,
            loss_jepa,
            loss_extra,
            weight,
        )

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        result["local_consistency"] = self._last_local_loss.cpu().numpy()
        result.update(
            {
                key: value.cpu().numpy()
                for key, value in self._last_local_components.items()
            }
        )
        return result

    def validation_step(self, batch: dict) -> dict:
        result = super().validation_step(batch)
        result["local_consistency"] = self._last_local_loss.cpu().numpy()
        result.update(
            {
                key: value.cpu().numpy()
                for key, value in self._last_local_components.items()
            }
        )
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log(
            "train_local_consistency_losses",
            self._mean_output(train_outputs, "local_consistency"),
            self.current_epoch,
        )
        self.logger.log(
            "local_consistency_weights",
            float(self.local_consistency_weight),
            self.current_epoch,
        )
        for component in ("local_invariance", "local_variance", "local_covariance"):
            self.logger.log(
                f"train_{component}_losses",
                self._mean_output(train_outputs, component),
                self.current_epoch,
            )

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log(
            "val_local_consistency_losses",
            self._mean_output(val_outputs, "local_consistency"),
            self.current_epoch,
        )
        for component in ("local_invariance", "local_variance", "local_covariance"):
            self.logger.log(
                f"val_{component}_losses",
                self._mean_output(val_outputs, component),
                self.current_epoch,
            )


class PrimusJEPADOAR32_200ep_BS4(_PrimusVICRegModuleTrainer):
    """Rank-32 objective-specific residual adapters."""

    architecture_class = PrimusDualObjectiveAdapterMAEJEPA
    architecture_kwargs = {"adapter_rank": 32}
    auxiliary_prefixes_extra = ("mae_adapter.", "jepa_adapter.")
