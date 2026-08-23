"""Isolated MAE+JEPA ablations aimed at preserving dense-transfer quality.

Every public trainer name in this file is new.  The original
``PrimusJEPATrainer`` and its checkpoints are intentionally untouched.
"""

from __future__ import annotations

from time import perf_counter
from typing import override

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json
from torch import autocast, nn

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.primus_jepa import PrimusMAEJEPA
from nnssl.architectures.primus_jepa_variants import (
    PrimusBlockTargetMAEJEPA,
    PrimusDualTeacherMAEJEPA,
    PrimusOnlineTargetMAEJEPA,
)
from nnssl.training.loss.representation_regularization import (
    SIGRegLoss,
    VICRegVarianceCovarianceLoss,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPATrainer import PrimusJEPATrainer
from nnssl.utilities.helpers import dummy_context, empty_cache


class PerGroupWarmupPolyScheduler:
    """Warm-up/poly schedule that preserves discriminative group learning rates."""

    def __init__(self, optimizer, num_epochs: int, exponent: float = 0.9):
        self.optimizer = optimizer
        self.num_epochs = int(num_epochs)
        self.exponent = float(exponent)

    def step(self, epoch: int | None = None):
        epoch = 0 if epoch is None else int(epoch)
        for group in self.optimizer.param_groups:
            peak = float(group["peak_lr"])
            warmup = int(group["warmup_epochs"])
            if epoch < warmup:
                lr = peak * (epoch + 1) / max(1, warmup)
            else:
                progress = (epoch - warmup) / max(1, self.num_epochs - warmup)
                lr = peak * max(0.0, 1.0 - progress) ** self.exponent
            group["lr"] = lr

    def state_dict(self) -> dict:
        return {"num_epochs": self.num_epochs, "exponent": self.exponent}

    def load_state_dict(self, state_dict: dict) -> None:
        self.num_epochs = int(state_dict.get("num_epochs", self.num_epochs))
        self.exponent = float(state_dict.get("exponent", self.exponent))


class PrimusJEPAStabilizedTrainer(PrimusJEPATrainer):
    """Low encoder LR, predictor warm-up, and a gradual JEPA-loss ramp."""

    architecture_class = PrimusMAEJEPA
    architecture_kwargs: dict = {}
    encoder_lr = 3e-5
    auxiliary_lr = 3e-4
    predictor_only_epochs = 10
    jepa_ramp_epochs = 20
    jepa_loss_weight_default = 0.005
    uses_ema_teacher = True
    adaptive_contribution_ratio: float | None = None
    extra_loss_weight = 0.0
    regularizer_kind: str | None = None
    auxiliary_prefixes_extra: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200
        self.total_batch_size = 8
        self._adaptive_mae = None
        self._adaptive_jepa = None
        self._last_loss_weight = self.jepa_loss_weight
        self.extra_loss = self._build_extra_loss()
        for key in (
            "train_extra_losses",
            "val_extra_losses",
            "jepa_loss_weights",
        ):
            self.logger.my_fantastic_logging.setdefault(key, [])
        if issubclass(self.architecture_class, PrimusDualTeacherMAEJEPA):
            self.logger.my_fantastic_logging.setdefault("anchor_weights", [])

    def _build_extra_loss(self) -> nn.Module | None:
        if self.regularizer_kind == "vicreg":
            return VICRegVarianceCovarianceLoss()
        if self.regularizer_kind == "sigreg":
            return SIGRegLoss()
        return None

    @override
    def build_architecture_and_adaptation_plan(self, config_plan, num_input_channels, num_output_channels):
        network_kwargs = dict(self.architecture_kwargs)
        network = self.architecture_class(
            input_channels=1,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=1,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
            predictor_dim=self.predictor_dim,
            predictor_depth=self.predictor_depth,
            predictor_num_heads=self.predictor_num_heads,
            predictor_query_chunk_size=self.predictor_query_chunk_size,
            **network_kwargs,
        )
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="eva",
            key_to_stem="down_projection",
            keys_to_in_proj=("down_projection.proj",),
            key_to_lpe="eva.pos_embed",
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        return network, adapt_plan

    def configure_optimizers(self, stage: str = "warmup_all"):
        # BaseEvaMAETrainer calls this again at its historical epoch-50 boundary.
        if getattr(self, "optimizer", None) is not None and self.training_stage is not None:
            return self.optimizer, self.lr_scheduler
        network = self._actual_network()
        auxiliary_prefixes = (
            "predictor.",
            "regularizer_projector.",
            *self.auxiliary_prefixes_extra,
        )
        auxiliary = []
        backbone = []
        for name, parameter in network.named_parameters():
            if not parameter.requires_grad:
                continue
            (auxiliary if name.startswith(auxiliary_prefixes) else backbone).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone,
                    "lr": self.encoder_lr,
                    "peak_lr": self.encoder_lr,
                    "warmup_epochs": self.warmup_duration_whole_net,
                    "name": "mae_backbone",
                },
                {
                    "params": auxiliary,
                    "lr": self.auxiliary_lr,
                    "peak_lr": self.auxiliary_lr,
                    "warmup_epochs": self.predictor_only_epochs,
                    "name": "prediction_heads",
                },
            ],
            weight_decay=self.weight_decay,
            amsgrad=False,
            betas=(0.9, 0.98),
            fused=self.device.type == "cuda",
        )
        scheduler = PerGroupWarmupPolyScheduler(optimizer, self.num_epochs)
        self.training_stage = "discriminative"
        self.print_to_log_file(
            f"Discriminative LR: backbone={self.encoder_lr:g}, auxiliary={self.auxiliary_lr:g}; "
            f"predictor-only epochs={self.predictor_only_epochs}"
        )
        empty_cache(self.device)
        return optimizer, scheduler

    def _set_predictor_only(self, enabled: bool) -> None:
        network = self._actual_network()
        auxiliary_prefixes = (
            "predictor.",
            "regularizer_projector.",
            *self.auxiliary_prefixes_extra,
        )
        for name, parameter in network.named_parameters():
            if name.startswith(("target_", "anchor_")):
                parameter.requires_grad_(False)
            elif name.startswith(auxiliary_prefixes):
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(not enabled)

    def on_train_epoch_start(self):
        self._set_predictor_only(self.current_epoch < self.predictor_only_epochs)
        # Parent handles the logger/scheduler; configure_optimizers is idempotent.
        super().on_train_epoch_start()

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        self._jepa_steps = self.current_epoch * self.num_iterations_per_epoch
        history = self.logger.my_fantastic_logging
        if self.adaptive_contribution_ratio is not None:
            if history.get("train_mae_losses") and history.get("train_jepa_losses"):
                self._adaptive_mae = float(history["train_mae_losses"][-1])
                self._adaptive_jepa = float(history["train_jepa_losses"][-1])

    def _current_jepa_weight(self) -> float:
        if self.current_epoch < self.predictor_only_epochs:
            return 1.0
        if self.adaptive_contribution_ratio is not None and self._adaptive_mae is not None:
            raw = self.adaptive_contribution_ratio * self._adaptive_mae / max(self._adaptive_jepa, 1e-8)
            return float(np.clip(raw, 1e-4, 2e-2))
        ramp = min(1.0, (self.current_epoch - self.predictor_only_epochs + 1) / self.jepa_ramp_epochs)
        return float(self.jepa_loss_weight * ramp)

    def _update_adaptive_statistics(self, loss_mae: torch.Tensor, loss_jepa: torch.Tensor) -> None:
        decay = 0.99
        mae = float(loss_mae.detach())
        jepa = float(loss_jepa.detach())
        self._adaptive_mae = mae if self._adaptive_mae is None else decay * self._adaptive_mae + (1 - decay) * mae
        self._adaptive_jepa = jepa if self._adaptive_jepa is None else decay * self._adaptive_jepa + (1 - decay) * jepa

    def _prediction_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.jepa_loss(prediction, target)

    def _extra_loss(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.extra_loss is None:
            return output["prediction"].new_zeros(())
        return self.extra_loss(output["regularizer_embedding"])

    def _losses(self, data: torch.Tensor, output: dict[str, torch.Tensor]):
        mask = self.create_mask(output["keep_indices"], self.config_plan.patch_size, self.vit_patch_size)
        loss_mae = self.loss(output["reconstruction"], data, mask)
        loss_jepa = self._prediction_loss(output["prediction"], output["target"])
        loss_extra = self._extra_loss(output)
        if self.network.training and self.adaptive_contribution_ratio is not None:
            self._update_adaptive_statistics(loss_mae, loss_jepa)
        weight = self._current_jepa_weight()
        loss = loss_mae + weight * loss_jepa + self.extra_loss_weight * loss_extra
        return loss, loss_mae, loss_jepa, loss_extra, weight

    def train_step(self, batch: dict) -> dict:
        started = perf_counter()
        data = batch["data"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss, loss_mae, loss_jepa, loss_extra, weight = self._losses(data, output)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
        momentum = float("nan")
        if self.uses_ema_teacher:
            momentum = self._ema_momentum()
            self._actual_network().update_target_encoder(momentum)
        self._jepa_steps += 1
        self._last_loss_weight = weight
        result = {
            "loss": loss.detach().cpu().numpy(),
            "loss_mae": loss_mae.detach().cpu().numpy(),
            "loss_jepa": loss_jepa.detach().cpu().numpy(),
            "loss_extra": loss_extra.detach().cpu().numpy(),
            "loss_weight": np.asarray(weight, dtype=np.float32),
            "seconds": np.asarray(perf_counter() - started, dtype=np.float32),
            "ema_momentum": np.asarray(momentum, dtype=np.float32),
        }
        if "anchor_weight" in output:
            result["anchor_weight"] = output["anchor_weight"].detach().cpu().numpy()
        return result

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss, loss_mae, loss_jepa, loss_extra, weight = self._losses(data, output)
        result = {
            "loss": loss.detach().cpu().numpy(),
            "loss_mae": loss_mae.detach().cpu().numpy(),
            "loss_jepa": loss_jepa.detach().cpu().numpy(),
            "loss_extra": loss_extra.detach().cpu().numpy(),
            "loss_weight": np.asarray(weight, dtype=np.float32),
        }
        if "anchor_weight" in output:
            result["anchor_weight"] = output["anchor_weight"].detach().cpu().numpy()
        return result

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log("train_extra_losses", self._mean_output(train_outputs, "loss_extra"), self.current_epoch)
        self.logger.log("jepa_loss_weights", self._mean_output(train_outputs, "loss_weight"), self.current_epoch)
        if "anchor_weight" in train_outputs[0]:
            self.logger.log("anchor_weights", self._mean_output(train_outputs, "anchor_weight"), self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log("val_extra_losses", self._mean_output(val_outputs, "loss_extra"), self.current_epoch)

    def on_epoch_end(self):
        # PrimusJEPATrainer already serializes all newly registered logger keys.
        super().on_epoch_end()


class PrimusJEPAStaged_200ep_BS8_L0p005(PrimusJEPAStabilizedTrainer):
    jepa_loss_weight_default = 0.005


class PrimusJEPAStaged_200ep_BS8_L0p01(PrimusJEPAStabilizedTrainer):
    jepa_loss_weight_default = 0.01


class PrimusJEPAAdaptive_200ep_BS8_R0p05(PrimusJEPAStabilizedTrainer):
    adaptive_contribution_ratio = 0.05


class PrimusJEPAAdaptive_200ep_BS8_R0p10(PrimusJEPAStabilizedTrainer):
    adaptive_contribution_ratio = 0.10


class PrimusJEPABlock_200ep_BS8_K512(PrimusJEPAStabilizedTrainer):
    architecture_class = PrimusBlockTargetMAEJEPA
    architecture_kwargs = {"jepa_num_targets": 512}


class PrimusJEPABlock_200ep_BS8_K1024(PrimusJEPAStabilizedTrainer):
    architecture_class = PrimusBlockTargetMAEJEPA
    architecture_kwargs = {"jepa_num_targets": 1024}


class PrimusJEPADualTeacher_200ep_BS8_A0p25_0p75(PrimusJEPAStabilizedTrainer):
    architecture_class = PrimusDualTeacherMAEJEPA
    architecture_kwargs = {"anchor_weight_min": 0.25, "anchor_weight_max": 0.75}


class PrimusJEPADualTeacher_200ep_BS8_A0p50_0p90(PrimusJEPAStabilizedTrainer):
    architecture_class = PrimusDualTeacherMAEJEPA
    architecture_kwargs = {"anchor_weight_min": 0.50, "anchor_weight_max": 0.90}


class PrimusJEPANoEMA_200ep_BS4_VICReg0p01(PrimusJEPAStabilizedTrainer):
    architecture_class = PrimusOnlineTargetMAEJEPA
    uses_ema_teacher = False
    regularizer_kind = "vicreg"
    extra_loss_weight = 0.01

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4

    def _prediction_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction.float() - target.float()).abs().mean()


class PrimusJEPANoEMA_200ep_BS4_SIGReg0p02(PrimusJEPANoEMA_200ep_BS4_VICReg0p01):
    regularizer_kind = "sigreg"
    extra_loss_weight = 0.02
