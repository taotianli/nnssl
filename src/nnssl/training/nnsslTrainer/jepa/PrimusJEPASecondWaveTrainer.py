"""Paper-motivated second-wave Primus MAE+JEPA experiments.

All trainer names are new. Existing MAE, JEPA and first-wave checkpoints keep
their original implementation and output directories.
"""

from __future__ import annotations

import numpy as np
import torch

from nnssl.architectures.primus_jepa_second_wave import (
    PrimusGatedVisibleSkipMAEJEPA,
    PrimusIndependentCrossViewMAEJEPA,
    PrimusSerialLatentMAEJEPA,
)
from nnssl.training.loss.structure_aware_reconstruction import masked_smoothed_gradient_l1
from nnssl.training.nnsslTrainer.jepa.PrimusJEPAExperimentalTrainer import (
    PrimusJEPAStabilizedTrainer,
)


class PrimusJEPASerialCAE_200ep_BS8_L0p005(PrimusJEPAStabilizedTrainer):
    """CAE-style predictor-to-decoder serial latent bottleneck."""

    architecture_class = PrimusSerialLatentMAEJEPA
    jepa_loss_weight_default = 0.005


class PrimusJEPAGatedSkip_200ep_BS8_G0p25(PrimusJEPAStabilizedTrainer):
    """BootMAE-style gated low-level visible-token decoder skip."""

    architecture_class = PrimusGatedVisibleSkipMAEJEPA
    architecture_kwargs = {"visible_skip_max": 0.25}
    jepa_loss_weight_default = 0.005


class PrimusJEPAIndependentCrossView_200ep_BS4_Weak(PrimusJEPAStabilizedTrainer):
    """Independent task masks and an aligned weak EMA intensity view."""

    architecture_class = PrimusIndependentCrossViewMAEJEPA
    architecture_kwargs = {
        "intensity_gain": 0.10,
        "intensity_bias_std": 0.05,
        "intensity_noise_std": 0.02,
    }
    jepa_loss_weight_default = 0.005

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_ddp:
            raise RuntimeError(
                "PrimusJEPAGradGuard uses autograd gradient probes and currently supports single-GPU training only"
            )
        self.total_batch_size = 4


class PrimusJEPAStructure_200ep_BS8_E0p05(PrimusJEPAStabilizedTrainer):
    """Masked voxel reconstruction plus a weak smoothed 3-D edge target."""

    edge_loss_weight = 0.05
    edge_smoothing_kernel = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.my_fantastic_logging.setdefault("train_edge_losses", [])
        self.logger.my_fantastic_logging.setdefault("val_edge_losses", [])
        self._last_edge_loss = None

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        mask = self.create_mask(output["keep_indices"], self.config_plan.patch_size, self.vit_patch_size)
        loss_pixel = self.loss(output["reconstruction"], data, mask)
        loss_edge = masked_smoothed_gradient_l1(
            output["reconstruction"], data, mask, self.edge_smoothing_kernel
        )
        loss_mae = loss_pixel + self.edge_loss_weight * loss_edge
        loss_jepa = self._prediction_loss(output["prediction"], output["target"])
        loss_extra = self._extra_loss(output)
        weight = self._current_jepa_weight()
        loss = loss_mae + weight * loss_jepa + self.extra_loss_weight * loss_extra
        self._last_edge_loss = loss_edge.detach()
        return loss, loss_mae, loss_jepa, loss_extra, weight

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        result["loss_edge"] = self._last_edge_loss.cpu().numpy()
        return result

    def validation_step(self, batch: dict) -> dict:
        result = super().validation_step(batch)
        result["loss_edge"] = self._last_edge_loss.cpu().numpy()
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log("train_edge_losses", self._mean_output(train_outputs, "loss_edge"), self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log("val_edge_losses", self._mean_output(val_outputs, "loss_edge"), self.current_epoch)


class PrimusJEPAGradGuard_200ep_BS4_R0p25(PrimusJEPAStabilizedTrainer):
    """Adaptive fusion that protects the MAE encoder gradient.

    After predictor warm-up, the effective JEPA coefficient is capped so its
    gradient on the last online EVA block is at most 25% of the MAE gradient.
    A conflicting (negative-cosine) update is additionally attenuated by 0.1.
    Unlike manual gradient replacement, the final ordinary backward remains
    compatible with AMP and optimizer state handling.
    """

    max_jepa_to_mae_gradient_ratio = 0.25
    conflicting_gradient_scale = 0.10
    jepa_loss_weight_default = 0.005

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self._last_grad_cosine = 0.0
        self._last_grad_scale = 1.0
        self.logger.my_fantastic_logging.setdefault("encoder_grad_cosines", [])
        self.logger.my_fantastic_logging.setdefault("jepa_gradient_scales", [])

    def _gradient_probe_parameters(self) -> list[torch.nn.Parameter]:
        eva = self._actual_network().eva
        if hasattr(eva, "blocks") and len(eva.blocks):
            parameters = list(eva.blocks[-1].parameters())
        else:
            parameters = list(eva.parameters())[-16:]
        return [parameter for parameter in parameters if parameter.requires_grad]

    @staticmethod
    def _gradient_statistics(
        first: tuple[torch.Tensor | None, ...], second: tuple[torch.Tensor | None, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dot = None
        first_squared = None
        second_squared = None
        for first_gradient, second_gradient in zip(first, second):
            if first_gradient is None or second_gradient is None:
                continue
            local_dot = (first_gradient.float() * second_gradient.float()).sum()
            local_first = first_gradient.float().square().sum()
            local_second = second_gradient.float().square().sum()
            dot = local_dot if dot is None else dot + local_dot
            first_squared = local_first if first_squared is None else first_squared + local_first
            second_squared = local_second if second_squared is None else second_squared + local_second
        if dot is None:
            raise RuntimeError("MAE and JEPA losses have no shared gradient probe parameter")
        return dot, first_squared.sqrt(), second_squared.sqrt()

    def _protected_weight(self, loss_mae: torch.Tensor, loss_jepa: torch.Tensor) -> float:
        base_weight = self._current_jepa_weight()
        parameters = self._gradient_probe_parameters()
        if not torch.is_grad_enabled() or not parameters:
            self._last_grad_cosine = 0.0
            self._last_grad_scale = 1.0
            return base_weight
        mae_gradient = torch.autograd.grad(
            loss_mae, parameters, retain_graph=True, allow_unused=True
        )
        jepa_gradient = torch.autograd.grad(
            loss_jepa, parameters, retain_graph=True, allow_unused=True
        )
        dot, mae_norm, jepa_norm = self._gradient_statistics(mae_gradient, jepa_gradient)
        epsilon = torch.finfo(torch.float32).eps
        cosine = dot / (mae_norm * jepa_norm + epsilon)
        weighted_ratio = base_weight * jepa_norm / (mae_norm + epsilon)
        cap_scale = torch.clamp(
            self.max_jepa_to_mae_gradient_ratio / (weighted_ratio + epsilon), max=1.0
        )
        if float(cosine.detach()) < 0:
            cap_scale = cap_scale * self.conflicting_gradient_scale
        self._last_grad_cosine = float(cosine.detach())
        self._last_grad_scale = float(cap_scale.detach())
        return base_weight * self._last_grad_scale

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        mask = self.create_mask(output["keep_indices"], self.config_plan.patch_size, self.vit_patch_size)
        loss_mae = self.loss(output["reconstruction"], data, mask)
        loss_jepa = self._prediction_loss(output["prediction"], output["target"])
        loss_extra = self._extra_loss(output)
        weight = self._protected_weight(loss_mae, loss_jepa)
        loss = loss_mae + weight * loss_jepa + self.extra_loss_weight * loss_extra
        return loss, loss_mae, loss_jepa, loss_extra, weight

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        result["grad_cosine"] = np.asarray(self._last_grad_cosine, dtype=np.float32)
        result["gradient_scale"] = np.asarray(self._last_grad_scale, dtype=np.float32)
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log("encoder_grad_cosines", self._mean_output(train_outputs, "grad_cosine"), self.current_epoch)
        self.logger.log("jepa_gradient_scales", self._mean_output(train_outputs, "gradient_scale"), self.current_epoch)
