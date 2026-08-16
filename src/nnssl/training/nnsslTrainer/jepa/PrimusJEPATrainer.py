from __future__ import annotations

import math
from copy import deepcopy
from time import perf_counter
from typing import override

import numpy as np
import torch
import torch.distributed as dist
from batchgenerators.utilities.file_and_folder_operations import save_json
from os.path import join
from torch import autocast, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.primus_jepa import PrimusMAEJEPA
from nnssl.training.loss.jepa_loss import JEPALatentLoss
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseEvaMAETrainer import BaseEvaMAETrainer
from nnssl.utilities.json_export import recursive_fix_for_json_export
from nnssl.utilities.helpers import dummy_context


class PrimusJEPATrainer(BaseEvaMAETrainer):
    """Joint Primus-M MAE + JEPA trainer using one shared online encoder."""

    jepa_loss_weight_default = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jepa_loss_weight = self.jepa_loss_weight_default
        self.mae_loss_weight = 1.0
        self.jepa_loss = JEPALatentLoss(exponent=1.0)
        self.predictor_dim = 384
        self.predictor_depth = 4
        self.predictor_num_heads = 12
        self.predictor_query_chunk_size = 512
        self.ema_start = 0.996
        self.ema_end = 1.0
        self._jepa_steps = 0
        for key in ("train_mae_losses", "train_jepa_losses", "val_mae_losses", "val_jepa_losses"):
            self.logger.my_fantastic_logging.setdefault(key, [])

    def _mean_output(self, outputs: list[dict], key: str) -> float:
        value = torch.as_tensor(
            np.mean([np.asarray(output[key], dtype=np.float64) for output in outputs]),
            dtype=torch.float64,
            device=self.device,
        )
        if self.is_ddp:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= dist.get_world_size()
        return float(value.cpu())

    @override
    def build_architecture_and_adaptation_plan(self, config_plan, num_input_channels, num_output_channels) -> nn.Module:
        network = PrimusMAEJEPA(
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

    def _actual_network(self) -> PrimusMAEJEPA:
        network = self.network.module if isinstance(self.network, DDP) else self.network
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod
        return network

    def _ema_momentum(self) -> float:
        total_steps = max(1, self.num_epochs * self.num_iterations_per_epoch)
        progress = min(1.0, self._jepa_steps / total_steps)
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.ema_start + (self.ema_end - self.ema_start) * cosine

    def train_step(self, batch: dict) -> dict:
        started = perf_counter()
        data = batch["data"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            mask = self.create_mask(output["keep_indices"], self.config_plan.patch_size, self.vit_patch_size)
            loss_mae = self.loss(output["reconstruction"], data, mask)
            loss_jepa = self.jepa_loss(output["prediction"], output["target"])
            loss = self.mae_loss_weight * loss_mae + self.jepa_loss_weight * loss_jepa

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

        momentum = self._ema_momentum()
        self._actual_network().update_target_encoder(momentum)
        self._jepa_steps += 1
        elapsed = perf_counter() - started
        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_mae": loss_mae.detach().cpu().numpy(),
            "loss_jepa": loss_jepa.detach().cpu().numpy(),
            "seconds": np.asarray(elapsed, dtype=np.float32),
            "ema_momentum": np.asarray(momentum, dtype=np.float32),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            mask = self.create_mask(output["keep_indices"], self.config_plan.patch_size, self.vit_patch_size)
            loss_mae = self.loss(output["reconstruction"], data, mask)
            loss_jepa = self.jepa_loss(output["prediction"], output["target"])
            loss = self.mae_loss_weight * loss_mae + self.jepa_loss_weight * loss_jepa
        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_mae": loss_mae.detach().cpu().numpy(),
            "loss_jepa": loss_jepa.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self.logger.log("train_mae_losses", self._mean_output(train_outputs, "loss_mae"), self.current_epoch)
        self.logger.log("train_jepa_losses", self._mean_output(train_outputs, "loss_jepa"), self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        self.logger.log("val_mae_losses", self._mean_output(val_outputs, "loss_mae"), self.current_epoch)
        self.logger.log("val_jepa_losses", self._mean_output(val_outputs, "loss_jepa"), self.current_epoch)

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank == 0:
            json_history = deepcopy(self.logger.my_fantastic_logging)
            recursive_fix_for_json_export(json_history)
            save_json(
                json_history,
                join(self.output_folder, "jepa_loss_history.json"),
                sort_keys=False,
            )


class PrimusJEPATrainer_BS1(PrimusJEPATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_batch_size = 1


class PrimusJEPATrainer_200ep_BS1(PrimusJEPATrainer_BS1):
    """Cluster-ready 200 epoch configuration with conservative memory use."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200


class PrimusJEPATrainer_BS8(PrimusJEPATrainer):
    """Eight samples per optimizer step for high-memory accelerators."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_batch_size = 8


class PrimusJEPATrainer_200ep_BS8(PrimusJEPATrainer_BS8):
    """H200-oriented 200 epoch configuration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200


class PrimusJEPATrainer_200ep_BS8_JEPA0p01(PrimusJEPATrainer_200ep_BS8):
    """H200 configuration with MAE + 0.01 * JEPA latent loss."""

    jepa_loss_weight_default = 0.01


class PrimusJEPATrainer_200ep_BS8_JEPA0p005(PrimusJEPATrainer_200ep_BS8):
    """H200 configuration with MAE + 0.005 * JEPA latent loss."""

    jepa_loss_weight_default = 0.005
