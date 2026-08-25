"""MRI-informed priors on top of the matched MAE+JEPA+VICReg baseline.

The public trainers in this file are isolated additions. They all retain the
teacher-free online target, VICReg weight 0.01, discriminative learning rates,
and JEPA warm-up/ramp of ``PrimusJEPANoEMA_200ep_BS4_VICReg0p01``.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

from nnssl.architectures.primus_jepa_priors import (
    PrimusAcquisitionPriorMAEJEPA,
    PrimusAnatomyPriorMAEJEPA,
    PrimusSoftRegionPriorMAEJEPA,
)
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA
from nnssl.ssl_data.dataloading.prior_data_loader_3d import nnsslPriorDataLoader3D
from nnssl.training.loss.mri_prior_losses import (
    anatomy_prior_loss,
    soft_region_prior_loss,
    spatial_heat_kernel_loss,
    spectral_shell_loss,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPARegularizedTrainer import (
    PrimusJEPARegularized_200ep_BS4,
)


_COMPONENT_NAMES = (
    "occupancy",
    "signed_distance",
    "mask_coverage",
    "heat_affinity",
    "heat_pairs",
    "spectral_low",
    "spectral_high",
    "region_entropy",
    "corruption_magnitude",
)


class _AnatomyPriorLoaderMixin:
    """Use the optional anatomy mask as the spatially transformed seg field."""

    def get_plain_dataloaders(self, initial_patch_size: Sequence[int]):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnsslPriorDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslPriorDataLoader3D(
            dataset_val,
            self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val


class _PrimusJEPAPriorTrainer(PrimusJEPARegularized_200ep_BS4):
    """Shared logging, scheduling, and resume safety for prior experiments."""

    architecture_class = PrimusOnlineTargetMAEJEPA
    architecture_kwargs: dict = {}
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.01
    prior_kind = "none"
    prior_loss_weight = 0.0
    auxiliary_prefixes_extra: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        override = os.environ.get("NNSSL_PRIOR_WEIGHT")
        if override is not None:
            self.prior_loss_weight = float(override)
        self._active_prior_batch: dict | None = None
        self._last_prior_loss = torch.tensor(0.0)
        self._last_prior_components: dict[str, torch.Tensor] = {}
        self._step_prior_components: dict[str, torch.Tensor] = {}
        history = self.logger.my_fantastic_logging
        for split in ("train", "val"):
            history.setdefault(f"{split}_prior_losses", [])
            for component in _COMPONENT_NAMES:
                history.setdefault(f"{split}_prior_{component}", [])
        history.setdefault("prior_loss_weights", [])

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ):
        network, adaptation_plan = super().build_architecture_and_adaptation_plan(
            config_plan, num_input_channels, num_output_channels
        )
        if hasattr(network, "corruption_seed"):
            network.corruption_seed = self.experiment_seed + 2_000_000
        return network, adaptation_plan

    def _current_prior_weight(self) -> float:
        if self.current_epoch < self.predictor_only_epochs:
            return 0.0
        ramp = min(
            1.0,
            (self.current_epoch - self.predictor_only_epochs + 1) / self.jepa_ramp_epochs,
        )
        return float(self.prior_loss_weight * ramp)

    def _anatomy_inputs(self, data: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._active_prior_batch is None:
            return None, None
        anatomy = self._active_prior_batch.get("seg")
        availability = self._active_prior_batch.get("prior_mask_available")
        if anatomy is not None:
            anatomy = anatomy.to(self.device, non_blocking=True)
        if availability is not None:
            availability = torch.as_tensor(availability, device=self.device)
        return anatomy, availability

    def _prior_loss(
        self, data: torch.Tensor, output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = output["prediction"].new_zeros(())
        return zero, {}

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        loss, loss_mae, loss_jepa, loss_extra, jepa_weight = super()._losses(data, output)
        prior_loss, components = self._prior_loss(data, output)
        prior_weight = self._current_prior_weight()
        loss = loss + prior_weight * prior_loss
        self._last_prior_loss = prior_loss.detach()
        self._last_prior_components = {key: value.detach() for key, value in components.items()}
        self._step_prior_components = self._last_prior_components
        return loss, loss_mae, loss_jepa, loss_extra, jepa_weight

    @staticmethod
    def _scalar(value: torch.Tensor | float | None) -> np.ndarray:
        if value is None:
            value = 0.0
        elif torch.is_tensor(value):
            value = float(value.detach().cpu())
        return np.asarray(value, dtype=np.float32)

    def train_step(self, batch: dict) -> dict:
        self._active_prior_batch = batch
        try:
            result = super().train_step(batch)
        finally:
            self._active_prior_batch = None
        result["loss_prior"] = self._scalar(self._last_prior_loss)
        result["prior_weight"] = self._scalar(self._current_prior_weight())
        for component in _COMPONENT_NAMES:
            result[f"prior_{component}"] = self._scalar(
                self._step_prior_components.get(component)
            )
        return result

    def validation_step(self, batch: dict) -> dict:
        self._active_prior_batch = batch
        try:
            result = super().validation_step(batch)
        finally:
            self._active_prior_batch = None
        result["loss_prior"] = self._scalar(self._last_prior_loss)
        result["prior_weight"] = self._scalar(self._current_prior_weight())
        for component in _COMPONENT_NAMES:
            result[f"prior_{component}"] = self._scalar(
                self._step_prior_components.get(component)
            )
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log(
            "train_prior_losses", self._mean_output(train_outputs, "loss_prior"), self.current_epoch
        )
        for component in _COMPONENT_NAMES:
            self.logger.log(
                f"train_prior_{component}",
                self._mean_output(train_outputs, f"prior_{component}"),
                self.current_epoch,
            )
        self.logger.log(
            "prior_loss_weights", self._mean_output(train_outputs, "prior_weight"), self.current_epoch
        )

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log(
            "val_prior_losses", self._mean_output(val_outputs, "loss_prior"), self.current_epoch
        )
        for component in _COMPONENT_NAMES:
            self.logger.log(
                f"val_prior_{component}",
                self._mean_output(val_outputs, f"prior_{component}"),
                self.current_epoch,
            )

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        requested = float(self.prior_loss_weight)
        super().load_checkpoint(filename_or_checkpoint)
        history = self.logger.my_fantastic_logging
        if history.get("prior_loss_weights"):
            # The logged value is ramped. A completed/ramped epoch stores the
            # actual terminal value; early checkpoints retain the class config.
            saved = float(history["prior_loss_weights"][-1])
            if self.current_epoch >= self.predictor_only_epochs + self.jepa_ramp_epochs:
                if "NNSSL_PRIOR_WEIGHT" in os.environ and not np.isclose(
                    requested, saved, rtol=1e-7, atol=1e-12
                ):
                    raise RuntimeError(
                        f"Checkpoint prior weight {saved} conflicts with "
                        f"NNSSL_PRIOR_WEIGHT={requested}"
                    )
                self.prior_loss_weight = saved
        network = self._actual_network()
        if hasattr(network, "corruption_seed"):
            network.corruption_seed = self.experiment_seed + 2_000_000
            network.corruption_calls = self.current_epoch * (
                self.num_iterations_per_epoch + self.num_val_iterations_per_epoch
            )
        # Older checkpoints may predate the prior logger keys. Re-registering
        # keeps a safe migration path without touching existing trainer logs.
        for split in ("train", "val"):
            history.setdefault(f"{split}_prior_losses", [0.0] * self.current_epoch)
            for component in _COMPONENT_NAMES:
                history.setdefault(
                    f"{split}_prior_{component}", [0.0] * self.current_epoch
                )
        history.setdefault("prior_loss_weights", [0.0] * self.current_epoch)


class _AnatomyPriorTrainer(_AnatomyPriorLoaderMixin, _PrimusJEPAPriorTrainer):
    architecture_class = PrimusAnatomyPriorMAEJEPA
    auxiliary_prefixes_extra = ("anatomy_prior_head.",)
    anatomy_sdf_weight = 0.0

    def _prior_loss(self, data, output):
        anatomy, availability = self._anatomy_inputs(data)
        return anatomy_prior_loss(
            output["anatomy_prior_prediction"],
            output["masked_indices"],
            data,
            anatomy,
            availability,
            self.vit_patch_size,
            self.anatomy_sdf_weight,
        )


class PrimusJEPAPriorAnatomyOcc_200ep_BS4_W0p02(_AnatomyPriorTrainer):
    """P1a: foreground-support occupancy only."""

    prior_kind = "anatomy_occupancy"
    prior_loss_weight = 0.02


class PrimusJEPAPriorAnatomySDF_200ep_BS4_W0p02(_AnatomyPriorTrainer):
    """P1b: occupancy plus patch-grid signed-distance proxy."""

    prior_kind = "anatomy_occupancy_sdf"
    prior_loss_weight = 0.02
    anatomy_sdf_weight = 0.5


class _HeatKernelPriorTrainer(_PrimusJEPAPriorTrainer):
    heat_neighborhood = 6
    prior_loss_weight = 0.001

    def _prior_loss(self, data, output):
        return spatial_heat_kernel_loss(
            output["regularizer_embedding"],
            output["masked_indices"],
            data,
            self.vit_patch_size,
            self.heat_neighborhood,
        )


class PrimusJEPAPriorHeat6_200ep_BS4_W0p001(_HeatKernelPriorTrainer):
    """P2a: sparse 6-neighbour intensity-gated manifold prior."""

    prior_kind = "spatial_heat_kernel_6"


class PrimusJEPAPriorHeat26_200ep_BS4_W0p001(_HeatKernelPriorTrainer):
    """P2b: 26-neighbour intensity-gated manifold prior."""

    prior_kind = "spatial_heat_kernel_26"
    heat_neighborhood = 26


class _SpectralPriorTrainer(_PrimusJEPAPriorTrainer):
    spectral_shells = 8
    prior_loss_weight = 0.01

    def _prior_loss(self, data, output):
        mask = self.create_mask(
            output["keep_indices"], self.config_plan.patch_size, self.vit_patch_size
        )
        return spectral_shell_loss(output["reconstruction"], data, mask, self.spectral_shells)


class PrimusJEPAPriorSpectral8_200ep_BS4_W0p01(_SpectralPriorTrainer):
    """P3a: eight radial frequency shells."""

    prior_kind = "spectral_8_shells"


class PrimusJEPAPriorSpectral16_200ep_BS4_W0p01(_SpectralPriorTrainer):
    """P3b: sixteen radial frequency shells."""

    prior_kind = "spectral_16_shells"
    spectral_shells = 16


class _AcquisitionPriorTrainer(_PrimusJEPAPriorTrainer):
    architecture_class = PrimusAcquisitionPriorMAEJEPA
    auxiliary_prefixes_extra = ("acquisition_conditioner.",)

    def _prior_loss(self, data, output):
        magnitude = output["corruption_parameters"].float().abs().mean()
        # This is a structural prior: gradients enter through clean-target JEPA
        # and acquisition-conditioned MAE, not through an additional scalar loss.
        return output["prediction"].new_zeros(()), {"corruption_magnitude": magnitude}


class PrimusJEPAPriorAcquisitionWeak_200ep_BS4(_AcquisitionPriorTrainer):
    """P4a: weak synthetic MRI acquisition/style corruption."""

    prior_kind = "acquisition_style_weak"
    architecture_kwargs = {"corruption_strength": 0.05}


class PrimusJEPAPriorAcquisitionMixed_200ep_BS4(_AcquisitionPriorTrainer):
    """P4b: stronger mixed acquisition/style corruption."""

    prior_kind = "acquisition_style_mixed"
    architecture_kwargs = {"corruption_strength": 0.10}


class _SoftRegionPriorTrainer(_AnatomyPriorLoaderMixin, _PrimusJEPAPriorTrainer):
    architecture_class = PrimusSoftRegionPriorMAEJEPA
    auxiliary_prefixes_extra = ("soft_region_prior_head.",)
    prior_classes = 3
    prior_loss_weight = 0.02

    def _prior_loss(self, data, output):
        anatomy, availability = self._anatomy_inputs(data)
        return soft_region_prior_loss(
            output["soft_region_logits"],
            output["masked_indices"],
            data,
            anatomy,
            availability,
            self.vit_patch_size,
            self.prior_classes,
        )


class PrimusJEPAPriorTissue3_200ep_BS4_W0p02(_SoftRegionPriorTrainer):
    """P5a: soft 3-bin within-support intensity pseudo-tissues."""

    prior_kind = "soft_tissue_3"
    prior_classes = 3
    architecture_kwargs = {"prior_classes": 3}


class PrimusJEPAPriorCoarseRegion8_200ep_BS4_W0p02(_SoftRegionPriorTrainer):
    """P5b: foreground-weighted 2x2x2 crop-relative coarse regions."""

    prior_kind = "coarse_region_8"
    prior_classes = 8
    architecture_kwargs = {"prior_classes": 8}
