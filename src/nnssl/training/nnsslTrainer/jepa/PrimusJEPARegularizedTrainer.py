"""Teacher-free Primus MAE+JEPA regularization controls and ORR experiments.

All trainer names are new so completed Wave1/Wave2 runs keep their behaviour.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch import nn

from nnssl.architectures.primus_jepa_regularized import PrimusSpatialOnlineTargetMAEJEPA
from nnssl.architectures.primus_jepa_variants import PrimusOnlineTargetMAEJEPA
from nnssl.training.loss.representation_regularization import (
    ORRRegLoss,
    RegionVISRegLoss,
    SIGRegLoss,
    VICRegVarianceCovarianceLoss,
    VISRegLoss,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPAExperimentalTrainer import (
    PrimusJEPANoEMA_200ep_BS4_VICReg0p01,
)


class PrimusJEPARegularized_200ep_BS4(PrimusJEPANoEMA_200ep_BS4_VICReg0p01):
    """Common teacher-free trainer with isolated, logged regularizer variants."""

    regularizer_kind: str | None = None
    extra_loss_weight = 0.0
    residual_weight = 1.0
    architecture_class = PrimusOnlineTargetMAEJEPA
    architecture_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        self.experiment_seed = int(os.environ.get("NNSSL_EXPERIMENT_SEED", "20260821"))
        random.seed(self.experiment_seed)
        np.random.seed(self.experiment_seed)
        torch.manual_seed(self.experiment_seed)
        torch.cuda.manual_seed_all(self.experiment_seed)
        self._regularizer_calls = 0
        super().__init__(*args, **kwargs)
        self.print_to_log_file(f"Regularization experiment seed: {self.experiment_seed}")
        weight_override = os.environ.get("NNSSL_REGULARIZER_WEIGHT")
        if weight_override is not None and self.regularizer_kind is not None:
            self.extra_loss_weight = float(weight_override)
            self.print_to_log_file(
                f"Using calibrated regularizer weight from NNSSL_REGULARIZER_WEIGHT="
                f"{self.extra_loss_weight:.10g}"
            )
        for key in (
            "train_extra_coarse_losses",
            "train_extra_residual_losses",
            "val_extra_coarse_losses",
            "val_extra_residual_losses",
            "regularizer_weights",
            "experiment_seeds",
        ):
            self.logger.my_fantastic_logging.setdefault(key, [])
        self._last_extra_components: dict[str, torch.Tensor] = {}

    def _seed_epoch(self) -> None:
        epoch_seed = self.experiment_seed + self.current_epoch * 100_003
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32 - 1))
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)

    def on_train_epoch_start(self):
        self._seed_epoch()
        super().on_train_epoch_start()

    def _build_extra_loss(self) -> nn.Module | None:
        if self.regularizer_kind is None:
            return None
        if self.regularizer_kind == "vicreg":
            return VICRegVarianceCovarianceLoss()
        if self.regularizer_kind == "sigreg":
            return SIGRegLoss()
        if self.regularizer_kind == "visreg":
            return VISRegLoss()
        if self.regularizer_kind == "region_visreg":
            return RegionVISRegLoss()
        if self.regularizer_kind in ("orr", "orr_shuffled"):
            return ORRRegLoss(
                residual_weight=self.residual_weight,
                shuffled_regions=self.regularizer_kind == "orr_shuffled",
            )
        raise ValueError(f"Unknown regularizer kind: {self.regularizer_kind}")

    def _extra_loss(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        zero = output["prediction"].new_zeros(())
        if self.extra_loss is None:
            self._last_extra_components = {"coarse": zero, "residual": zero}
            return zero
        embedding = output["regularizer_embedding"]
        devices = [embedding.device.index] if embedding.device.type == "cuda" else []
        # Regularizer projections/subsampling must not advance the RNG stream
        # used by patch dropping, DropPath, or the next online forward.
        with torch.random.fork_rng(devices=devices):
            regularizer_seed = self.experiment_seed + 1_000_000 + self._regularizer_calls
            torch.manual_seed(regularizer_seed)
            if embedding.device.type == "cuda":
                torch.cuda.manual_seed_all(regularizer_seed)
            if isinstance(self.extra_loss, (RegionVISRegLoss, ORRRegLoss)):
                components = self.extra_loss.components(embedding, output["region_ids"])
            elif isinstance(self.extra_loss, VISRegLoss):
                components = self.extra_loss.components(embedding)
            else:
                total = self.extra_loss(embedding)
                components = {"total": total}
        self._regularizer_calls += 1
        self._last_extra_components = {
            "coarse": components.get("coarse", components.get("total", zero)),
            "residual": components.get("residual", zero),
        }
        return components["total"]

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        requested_seed = self.experiment_seed
        requested_weight = self.extra_loss_weight
        super().load_checkpoint(filename_or_checkpoint)
        history = self.logger.my_fantastic_logging
        if history.get("experiment_seeds"):
            saved_seed = int(history["experiment_seeds"][-1])
            if "NNSSL_EXPERIMENT_SEED" in os.environ and requested_seed != saved_seed:
                raise RuntimeError(
                    f"Checkpoint seed {saved_seed} conflicts with NNSSL_EXPERIMENT_SEED="
                    f"{requested_seed}"
                )
            self.experiment_seed = saved_seed
        if history.get("regularizer_weights"):
            saved_weight = float(history["regularizer_weights"][-1])
            if (
                "NNSSL_REGULARIZER_WEIGHT" in os.environ
                and not np.isclose(requested_weight, saved_weight, rtol=1e-7, atol=1e-12)
            ):
                raise RuntimeError(
                    f"Checkpoint regularizer weight {saved_weight} conflicts with "
                    f"NNSSL_REGULARIZER_WEIGHT={requested_weight}"
                )
            self.extra_loss_weight = saved_weight
        calls_per_epoch = self.num_iterations_per_epoch + self.num_val_iterations_per_epoch
        self._regularizer_calls = self.current_epoch * calls_per_epoch
        network = self._actual_network()
        if hasattr(network, "region_seed"):
            network.region_seed = self.experiment_seed + 400_000
        self._seed_epoch()

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        losses = super()._losses(data, output)
        self._step_extra_components = {
            name: value.detach() for name, value in self._last_extra_components.items()
        }
        return losses

    @staticmethod
    def _component_result(components: dict[str, torch.Tensor], name: str) -> np.ndarray:
        value = components.get(name)
        return np.asarray(0.0 if value is None else float(value.cpu()), dtype=np.float32)

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        result["loss_extra_coarse"] = self._component_result(self._step_extra_components, "coarse")
        result["loss_extra_residual"] = self._component_result(
            self._step_extra_components, "residual"
        )
        return result

    def validation_step(self, batch: dict) -> dict:
        result = super().validation_step(batch)
        result["loss_extra_coarse"] = self._component_result(self._step_extra_components, "coarse")
        result["loss_extra_residual"] = self._component_result(
            self._step_extra_components, "residual"
        )
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log(
            "train_extra_coarse_losses",
            self._mean_output(train_outputs, "loss_extra_coarse"),
            self.current_epoch,
        )
        self.logger.log(
            "train_extra_residual_losses",
            self._mean_output(train_outputs, "loss_extra_residual"),
            self.current_epoch,
        )
        self.logger.log("regularizer_weights", float(self.extra_loss_weight), self.current_epoch)
        self.logger.log("experiment_seeds", float(self.experiment_seed), self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log(
            "val_extra_coarse_losses",
            self._mean_output(val_outputs, "loss_extra_coarse"),
            self.current_epoch,
        )
        self.logger.log(
            "val_extra_residual_losses",
            self._mean_output(val_outputs, "loss_extra_residual"),
            self.current_epoch,
        )


class PrimusJEPANoEMA_200ep_BS4_NoReg(PrimusJEPARegularized_200ep_BS4):
    """Causal control for the online target path and BS4."""


class PrimusJEPANoEMA_200ep_BS4_VICReg0p005(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.005


class PrimusJEPANoEMA_200ep_BS4_VICRegRho1(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.01


class PrimusJEPANoEMA_200ep_BS4_VICReg0p02(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.02


class PrimusJEPANoEMA_200ep_BS4_SIGReg0p01(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "sigreg"
    extra_loss_weight = 0.01


class PrimusJEPANoEMA_200ep_BS4_SIGRegRho1(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "sigreg"
    extra_loss_weight = 0.02


class PrimusJEPANoEMA_200ep_BS4_SIGReg0p05(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "sigreg"
    extra_loss_weight = 0.05


class PrimusJEPANoEMA_200ep_BS4_VISReg0p005(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "visreg"
    extra_loss_weight = 0.005


class PrimusJEPANoEMA_200ep_BS4_VISReg0p01(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "visreg"
    extra_loss_weight = 0.01


class PrimusJEPANoEMA_200ep_BS4_VISReg0p02(PrimusJEPARegularized_200ep_BS4):
    regularizer_kind = "visreg"
    extra_loss_weight = 0.02


class _SpatialRegularizedTrainer(PrimusJEPARegularized_200ep_BS4):
    architecture_class = PrimusSpatialOnlineTargetMAEJEPA


class PrimusJEPANoEMA_200ep_BS4_RegionVIS0p01(_SpatialRegularizedTrainer):
    regularizer_kind = "region_visreg"
    extra_loss_weight = 0.01


class PrimusJEPANoEMA_200ep_BS4_ORR0p005_B1(_SpatialRegularizedTrainer):
    regularizer_kind = "orr"
    extra_loss_weight = 0.005
    residual_weight = 1.0


class PrimusJEPANoEMA_200ep_BS4_ORR0p01_B1(_SpatialRegularizedTrainer):
    regularizer_kind = "orr"
    extra_loss_weight = 0.01
    residual_weight = 1.0


class PrimusJEPANoEMA_200ep_BS4_ORR0p01_B0p5(_SpatialRegularizedTrainer):
    regularizer_kind = "orr"
    extra_loss_weight = 0.01
    residual_weight = 0.5


class PrimusJEPANoEMA_200ep_BS4_ORRShuffled0p01_B1(_SpatialRegularizedTrainer):
    regularizer_kind = "orr_shuffled"
    extra_loss_weight = 0.01
    residual_weight = 1.0
