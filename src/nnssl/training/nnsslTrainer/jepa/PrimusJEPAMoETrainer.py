"""Sparse-MoE experiments for teacher-free Primus MAE+JEPA."""

from __future__ import annotations

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.architectures.primus_jepa_moe import PrimusMoEOnlineTargetMAEJEPA
from nnssl.training.nnsslTrainer.jepa.PrimusJEPARegularizedTrainer import (
    PrimusJEPARegularized_200ep_BS4,
)


class PrimusJEPAMoE_200ep_BS4(PrimusJEPARegularized_200ep_BS4):
    """Common sparse residual-MoE trainer.

    The original dense EVA path is the transferable encoder. New expert and
    router parameters use the predictor learning rate and are discarded after
    pretraining. Optional routing consistency aligns expert choice for the same
    visible token in masked and full online views.
    """

    architecture_class = PrimusMoEOnlineTargetMAEJEPA
    architecture_kwargs = {
        "moe_layer_indices": (8, 10, 12, 14),
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_hidden_dim": 192,
        "moe_route_scale": 0.25,
    }
    auxiliary_prefixes_extra = ("eva.adapters.",)
    routing_consistency_weight = 0.0
    routing_bias_rate = 1e-4
    routing_bias_clip = 0.3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in (
            "train_routing_consistency_losses",
            "val_routing_consistency_losses",
            "train_routing_entropies",
            "val_routing_entropies",
            "router_min_usage",
            "router_max_usage",
            "routing_consistency_weights",
        ):
            self.logger.my_fantastic_logging.setdefault(key, [])
        self._last_routing_consistency = torch.tensor(0.0)
        self._last_routing_entropy = torch.tensor(0.0)
        self._last_router_usage = (1.0, 1.0)

    def build_architecture_and_adaptation_plan(self, config_plan, num_input_channels, num_output_channels):
        network, adaptation_plan = super().build_architecture_and_adaptation_plan(
            config_plan, num_input_channels, num_output_channels
        )
        # Only the pretrained dense path is transferred to the standard PrimusM
        # segmentation encoder. Sparse experts remain a pretraining scaffold.
        adaptation_plan.key_to_encoder = "eva.base"
        adaptation_plan.key_to_lpe = "eva.base.pos_embed"
        save_json(adaptation_plan.serialize(), self.adaptation_json_plan)
        return network, adaptation_plan

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        loss, loss_mae, loss_jepa, loss_extra, weight = super()._losses(data, output)
        route_loss = output["routing_consistency"].float()
        route_weight = float(self.routing_consistency_weight)
        if self.current_epoch < self.predictor_only_epochs:
            route_weight = 0.0
        loss = loss + route_weight * route_loss
        self._last_routing_consistency = route_loss.detach()
        self._last_routing_entropy = output["routing_entropy"].detach()
        return loss, loss_mae, loss_jepa, loss_extra, weight

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        network = self._actual_network()
        self._last_router_usage = network.update_routing_bias(
            rate=self.routing_bias_rate,
            clip=self.routing_bias_clip,
        )
        result["routing_consistency"] = self._last_routing_consistency.cpu().numpy()
        result["routing_entropy"] = self._last_routing_entropy.cpu().numpy()
        result["router_min_usage"] = np.asarray(self._last_router_usage[0], dtype=np.float32)
        result["router_max_usage"] = np.asarray(self._last_router_usage[1], dtype=np.float32)
        return result

    def validation_step(self, batch: dict) -> dict:
        result = super().validation_step(batch)
        result["routing_consistency"] = self._last_routing_consistency.cpu().numpy()
        result["routing_entropy"] = self._last_routing_entropy.cpu().numpy()
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log(
            "train_routing_consistency_losses",
            self._mean_output(train_outputs, "routing_consistency"),
            self.current_epoch,
        )
        self.logger.log(
            "train_routing_entropies",
            self._mean_output(train_outputs, "routing_entropy"),
            self.current_epoch,
        )
        self.logger.log(
            "router_min_usage",
            self._mean_output(train_outputs, "router_min_usage"),
            self.current_epoch,
        )
        self.logger.log(
            "router_max_usage",
            self._mean_output(train_outputs, "router_max_usage"),
            self.current_epoch,
        )
        self.logger.log(
            "routing_consistency_weights",
            float(self.routing_consistency_weight),
            self.current_epoch,
        )

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log(
            "val_routing_consistency_losses",
            self._mean_output(val_outputs, "routing_consistency"),
            self.current_epoch,
        )
        self.logger.log(
            "val_routing_entropies",
            self._mean_output(val_outputs, "routing_entropy"),
            self.current_epoch,
        )
        # Validation routes are diagnostics and must not influence the next
        # training step's auxiliary-loss-free balancing update.
        self._actual_network().reset_routing_counts()


class PrimusJEPAMoE_200ep_BS4_NoReg(PrimusJEPAMoE_200ep_BS4):
    """Sparse-MoE causal control without a collapse regularizer."""


class PrimusJEPAMoE_200ep_BS4_VICReg0p01(PrimusJEPAMoE_200ep_BS4):
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.01


class PrimusJEPAMoE_200ep_BS4_SIGReg0p02(PrimusJEPAMoE_200ep_BS4):
    regularizer_kind = "sigreg"
    extra_loss_weight = 0.02


class PrimusJEPAMoE_200ep_BS4_VICReg0p01_RC0p01(PrimusJEPAMoE_200ep_BS4_VICReg0p01):
    routing_consistency_weight = 0.01


class PrimusJEPAMoE_200ep_BS4_VICReg0p01_RC0p05(PrimusJEPAMoE_200ep_BS4_VICReg0p01):
    routing_consistency_weight = 0.05


class PrimusJEPAMoE16_200ep_BS4_VICReg0p01_RC0p01(
    PrimusJEPAMoE_200ep_BS4_VICReg0p01_RC0p01
):
    """Capacity ablation: sixteen experts with four active per token."""

    architecture_kwargs = {
        **PrimusJEPAMoE_200ep_BS4.architecture_kwargs,
        "moe_num_experts": 16,
        "moe_top_k": 4,
    }
