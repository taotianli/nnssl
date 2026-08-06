"""Short local Primus-M MAE+JEPA training benchmark on an OpenMind manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan
from nnssl.architectures.primus_jepa import PrimusMAEJEPA
from nnssl.preprocessing.preprocessors.default_preprocessor import preprocess_case
from nnssl.training.loss.jepa_loss import JEPALatentLoss


MODEL_SHAPE = (160, 160, 160)
PATCH_SHAPE = (8, 8, 8)


def center_crop_or_pad(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=np.float32)
    source_slices = []
    destination_slices = []
    for current, target in zip(array.shape, shape):
        if current >= target:
            start = (current - target) // 2
            source_slices.append(slice(start, start + target))
            destination_slices.append(slice(0, target))
        else:
            start = (target - current) // 2
            source_slices.append(slice(0, current))
            destination_slices.append(slice(start, start + current))
    output[tuple(destination_slices)] = array[tuple(source_slices)]
    return output


class OpenMindPreprocessedDataset(Dataset):
    def __init__(self, manifest: Path, adaptation_plan: Path, cache_dir: Path):
        table = pd.read_csv(manifest)
        if "local_path" not in table:
            raise KeyError(f"Manifest has no local_path column: {manifest}")
        table = table[table["local_path"].map(lambda value: Path(str(value)).is_file())].reset_index(drop=True)
        if table.empty:
            raise RuntimeError(f"No readable local images in {manifest}")
        self.table = table
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        adaptation = AdaptationPlan.from_dict(json.loads(adaptation_plan.read_text(encoding="utf-8")))
        self.plan = adaptation.pretrain_plan
        self.config = self.plan.configurations["onemmiso"]

    def __len__(self) -> int:
        return len(self.table)

    def _cache_path(self, index: int) -> Path:
        row = self.table.iloc[index]
        identity = str(row.get("unique_id", row["local_path"]))
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{index:04d}_{digest}.npy"

    def _preprocess(self, path: Path) -> np.ndarray:
        reader_writer = self.plan.image_reader_writer_class()()
        data, properties = reader_writer.read_images([str(path)])
        data, _ = preprocess_case(
            data,
            masks=None,
            properties=properties,
            plan=self.plan,
            config_plan=self.config,
            verbose=False,
        )
        return center_crop_or_pad(data[0], MODEL_SHAPE)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        cache_path = self._cache_path(index)
        if cache_path.is_file():
            array = np.load(cache_path, mmap_mode="r").astype(np.float32, copy=True)
        else:
            row = self.table.iloc[index]
            array = self._preprocess(Path(str(row["local_path"])))
            np.save(cache_path, array.astype(np.float32, copy=False))
        identity = str(self.table.iloc[index].get("unique_id", index))
        return torch.from_numpy(array[None]), identity


def visible_voxel_mask(keep_indices: torch.Tensor) -> torch.Tensor:
    batch_size = keep_indices.shape[0]
    token_grid = tuple(i // p for i, p in zip(MODEL_SHAPE, PATCH_SHAPE))
    flat = torch.zeros(batch_size, math.prod(token_grid), device=keep_indices.device)
    flat.scatter_(1, keep_indices, 1.0)
    mask = flat.reshape(batch_size, *token_grid)
    for axis, repeat in enumerate(PATCH_SHAPE, start=1):
        mask = mask.repeat_interleave(repeat, dim=axis)
    return mask[:, None]


def load_model(args: argparse.Namespace, device: torch.device) -> PrimusMAEJEPA:
    model = PrimusMAEJEPA(
        input_channels=1,
        embed_dim=864,
        patch_embed_size=PATCH_SHAPE,
        output_channels=1,
        input_shape=MODEL_SHAPE,
        encoder_eva_depth=16,
        encoder_eva_numheads=12,
        decoder_eva_depth=2,
        decoder_eva_numheads=12,
        patch_drop_rate=args.mask_ratio,
        drop_path_rate=0.2,
        attn_drop_rate=0.0,
        init_values=0.1,
        scale_attn_inner=True,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_heads,
        predictor_query_chunk_size=args.query_chunk_size,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    loaded = model.load_mae_state_dict(checkpoint["network_weights"])
    if not loaded:
        raise RuntimeError("The checkpoint did not contain compatible Primus-M MAE parameters")
    print(f"Loaded {len(loaded)} MAE tensors; predictor is newly initialized.", flush=True)
    del checkpoint
    return model.to(device)


def plot_metrics(table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(table["step"], table["loss_total"], alpha=0.25, color="tab:blue")
    axes[0].plot(table["step"], table["loss_jepa"], alpha=0.25, color="tab:orange")
    window = min(500, max(25, len(table) // 100))
    axes[0].plot(table["step"], table["loss_total"].rolling(window, min_periods=1).mean(), label="total")
    axes[0].plot(
        table["step"],
        table["loss_jepa"].rolling(window, min_periods=1).mean(),
        label="JEPA",
        color="tab:orange",
    )
    axes[0].plot(
        table["step"],
        table["loss_mae"].rolling(window, min_periods=1).mean(),
        label="MAE",
        color="tab:green",
    )
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(table["step"], table["samples_per_second"], color="tab:green")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("samples / second")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_epoch_metrics(table: pd.DataFrame, output: Path) -> None:
    epoch_table = table.assign(epoch_index=np.floor(table["epoch"]).astype(int)).groupby("epoch_index", as_index=False)[
        ["loss_total", "loss_jepa", "loss_mae"]
    ].mean()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epoch_table["epoch_index"] + 1, epoch_table["loss_total"], label="total")
    axis.plot(epoch_table["epoch_index"] + 1, epoch_table["loss_jepa"], label="JEPA")
    axis.plot(epoch_table["epoch_index"] + 1, epoch_table["loss_mae"], label="MAE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    epoch_table.to_csv(output.with_suffix(".csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adaptation-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--precache-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--predictor-dim", type=int, default=384)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--predictor-heads", type=int, default=12)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--ema", type=float, default=0.996)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = OpenMindPreprocessedDataset(args.manifest, args.adaptation_plan, args.cache_dir)
    print(f"Dataset contains {len(dataset)} readable OpenMind volumes.", flush=True)

    if args.precache_only:
        started = time.perf_counter()
        for index in range(len(dataset)):
            dataset[index]
            if (index + 1) % 20 == 0 or index + 1 == len(dataset):
                print(f"cached {index + 1}/{len(dataset)}", flush=True)
        print(f"Preprocessing completed in {time.perf_counter() - started:.1f}s", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    model = load_model(args, device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    jepa_loss_fn = JEPALatentLoss(1.0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        args.steps = args.epochs * len(loader)
        print(f"Training for {args.epochs} epochs = {args.steps} optimizer steps.", flush=True)
    iterator = iter(loader)
    records = []
    samples_seen = 0
    metrics_path = args.output_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = None
    torch.cuda.reset_peak_memory_stats()

    try:
        for step in range(args.steps):
            data_started = time.perf_counter()
            try:
                images, identities = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, identities = next(iterator)
            images = images.to(device, non_blocking=True)
            torch.cuda.synchronize()
            data_seconds = time.perf_counter() - data_started
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(images)
                mask = visible_voxel_mask(output["keep_indices"])
                masked_voxels = 1.0 - mask
                loss_mae = ((output["reconstruction"] - images).square() * masked_voxels).sum()
                loss_mae = loss_mae / masked_voxels.sum().clamp_min(1.0)
                loss_jepa = jepa_loss_fn(output["prediction"], output["target"])
                loss = args.mae_weight * loss_mae + args.jepa_weight * loss_jepa
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            model.update_target_encoder(args.ema)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            current_batch_size = int(images.shape[0])
            record = {
                "step": step,
                "epoch": samples_seen / len(dataset),
                "loss_total": float(loss.detach()),
                "loss_jepa": float(loss_jepa.detach()),
                "loss_mae": float(loss_mae.detach()),
                "seconds": seconds,
                "samples_per_second": current_batch_size / seconds,
                "data_seconds": data_seconds,
                "end_to_end_seconds": data_seconds + seconds,
                "samples_per_second_end_to_end": current_batch_size / (data_seconds + seconds),
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "gradient_norm": float(gradient_norm.detach()),
                "masked_tokens": int(output["masked_indices"].shape[1]),
                "sample_ids": "|".join(identities),
            }
            samples_seen += current_batch_size
            records.append(record)
            if metrics_writer is None:
                metrics_writer = csv.DictWriter(metrics_file, fieldnames=list(record))
                metrics_writer.writeheader()
            metrics_writer.writerow(record)
            metrics_file.flush()
            if step % args.log_every == 0 or step + 1 == args.steps:
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        metrics_file.close()

    metrics = pd.DataFrame(records)
    plot_metrics(metrics, args.output_dir / "loss_and_speed.png")
    plot_epoch_metrics(metrics, args.output_dir / "loss_by_epoch.png")
    if not args.no_save_checkpoint:
        torch.save(
            {
                "network_weights": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "steps": args.steps,
                "args": vars(args),
            },
            args.output_dir / "checkpoint_smoke.pth",
        )
    summary = {
        "samples": len(dataset),
        "steps": len(metrics),
        "epochs": args.epochs,
        "jepa_weight": args.jepa_weight,
        "mae_weight": args.mae_weight,
        "mean_seconds_excluding_first": float(metrics["seconds"].iloc[1:].mean()) if len(metrics) > 1 else None,
        "mean_samples_per_second_excluding_first": float(metrics["samples_per_second"].iloc[1:].mean())
        if len(metrics) > 1
        else None,
        "mean_end_to_end_seconds_excluding_first": float(metrics["end_to_end_seconds"].iloc[1:].mean())
        if len(metrics) > 1
        else None,
        "mean_end_to_end_samples_per_second_excluding_first": float(
            metrics["samples_per_second_end_to_end"].iloc[1:].mean()
        )
        if len(metrics) > 1
        else None,
        "peak_memory_gib": float(metrics["peak_memory_gib"].max()),
        "initial_jepa_loss": float(metrics["loss_jepa"].iloc[0]),
        "final_jepa_loss": float(metrics["loss_jepa"].iloc[-1]),
        "initial_total_loss": float(metrics["loss_total"].iloc[0]),
        "final_total_loss": float(metrics["loss_total"].iloc[-1]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
