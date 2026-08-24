"""SCLB ablations for teacher-free Primus MAE+JEPA+VICReg."""

from __future__ import annotations

import numpy as np

from nnssl.architectures.primus_jepa_sclb import PrimusStructureConditionedMAEJEPA
from nnssl.training.nnsslTrainer.jepa.PrimusJEPARegularizedTrainer import (
    PrimusJEPARegularized_200ep_BS4,
)


class PrimusJEPASCLB_200ep_BS4(PrimusJEPARegularized_200ep_BS4):
    """Common 200-epoch SCLB recipe, initialized from public Primus-M MAE."""

    architecture_class = PrimusStructureConditionedMAEJEPA
    architecture_kwargs = {
        "region_grid": (4, 4, 4),
        "bridge_enabled": True,
        "detach_bridge": True,
        "shuffled_regions": False,
        "bridge_max_scale": 0.25,
    }
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.005
    auxiliary_prefixes_extra = ("region_bridge.",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in (
            "train_region_prediction_variances",
            "val_region_prediction_variances",
            "train_region_condition_norms",
            "val_region_condition_norms",
            "bridge_strengths",
        ):
            self.logger.my_fantastic_logging.setdefault(key, [])
        self._last_bridge_metrics: dict[str, np.ndarray] = {}

    def _losses(self, data, output):
        losses = super()._losses(data, output)
        self._last_bridge_metrics = {
            name: np.asarray(float(output[name].detach().cpu()), dtype=np.float32)
            for name in (
                "region_prediction_variance",
                "region_condition_norm",
                "bridge_strength",
            )
        }
        return losses

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        result.update(self._last_bridge_metrics)
        return result

    def validation_step(self, batch: dict) -> dict:
        result = super().validation_step(batch)
        result.update(self._last_bridge_metrics)
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log(
            "train_region_prediction_variances",
            self._mean_output(train_outputs, "region_prediction_variance"),
            self.current_epoch,
        )
        self.logger.log(
            "train_region_condition_norms",
            self._mean_output(train_outputs, "region_condition_norm"),
            self.current_epoch,
        )
        self.logger.log(
            "bridge_strengths",
            self._mean_output(train_outputs, "bridge_strength"),
            self.current_epoch,
        )

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log(
            "val_region_prediction_variances",
            self._mean_output(val_outputs, "region_prediction_variance"),
            self.current_epoch,
        )
        self.logger.log(
            "val_region_condition_norms",
            self._mean_output(val_outputs, "region_condition_norm"),
            self.current_epoch,
        )


class PrimusJEPASCLBParallel_200ep_BS4(PrimusJEPASCLB_200ep_BS4):
    """Region-level JEPA control with no predictor-to-decoder bridge."""

    architecture_kwargs = {**PrimusJEPASCLB_200ep_BS4.architecture_kwargs, "bridge_enabled": False}


class PrimusJEPASCLBDetached_200ep_BS4(PrimusJEPASCLB_200ep_BS4):
    """Main SCLB: connected region prediction conditions reconstruction."""


class PrimusJEPASCLBShuffled_200ep_BS4(PrimusJEPASCLB_200ep_BS4):
    """Size-matched spatial-control regions with the detached bridge."""

    architecture_kwargs = {**PrimusJEPASCLB_200ep_BS4.architecture_kwargs, "shuffled_regions": True}


class PrimusJEPASCLBBidirectional_200ep_BS4(PrimusJEPASCLB_200ep_BS4):
    """Ablation allowing MAE gradients to update the JEPA predictor."""

    architecture_kwargs = {**PrimusJEPASCLB_200ep_BS4.architecture_kwargs, "detach_bridge": False}


class PrimusJEPASCLBCrossView_200ep_BS4(PrimusJEPASCLB_200ep_BS4):
    """Main detached SCLB with an aligned weak intensity target view."""

    architecture_kwargs = {
        **PrimusJEPASCLB_200ep_BS4.architecture_kwargs,
        "context_gain": 0.10,
        "context_noise_std": 0.02,
    }
