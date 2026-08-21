"""Measure a regularizer-to-MAE encoder-gradient budget at MAE initialization."""

from __future__ import annotations

import argparse
import json
import math
import random
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from nnssl.run.load_pretrained_weights import load_pretrained_weights
from nnssl.run.run_training import get_trainer_from_args


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    squared = loss.new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.float().square().sum()
    return squared.sqrt()


def _measurement_parameters(network: torch.nn.Module) -> tuple[list[str], list[torch.nn.Parameter]]:
    if hasattr(network, "_orig_mod"):
        network = network._orig_mod
    depth = len(network.eva.blocks)
    final_blocks = tuple(f"eva.blocks.{index}." for index in range(max(0, depth - 4), depth))
    names = []
    parameters = []
    for name, parameter in network.named_parameters():
        if name.startswith("down_projection.") or name.startswith(final_blocks):
            names.append(name)
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("No encoder parameters matched down_projection/final four EVA blocks")
    return names, parameters


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        "all",
        args.trainer,
        args.plans,
        device,
    )
    trainer.initialize()
    load_pretrained_weights(trainer.network, args.pretrained_weights, verbose=False)
    trainer.on_train_start()
    network = trainer._actual_network()
    network.train()
    names, parameters = _measurement_parameters(network)
    ratios = []
    mae_norms = []
    regularizer_norms = []
    try:
        for step in range(args.batches):
            batch = next(trainer.dataloader_train)
            data = batch["data"].to(device, non_blocking=True)
            # Every independent calibration process gets the same model mask,
            # DropPath, and stochastic-forward stream for this batch.
            torch.manual_seed(args.seed + step)
            torch.cuda.manual_seed_all(args.seed + step)
            output = network(data)
            mask = trainer.create_mask(
                output["keep_indices"], trainer.config_plan.patch_size, trainer.vit_patch_size
            )
            loss_mae = trainer.loss(output["reconstruction"], data, mask)
            loss_regularizer = trainer._extra_loss(output)
            if not loss_regularizer.requires_grad:
                raise RuntimeError(f"Trainer {args.trainer} has no differentiable regularizer")
            mae_norm = _gradient_norm(loss_mae, parameters, retain_graph=True)
            regularizer_norm = _gradient_norm(loss_regularizer, parameters, retain_graph=False)
            ratio = regularizer_norm / (mae_norm + args.eps)
            ratios.append(float(ratio.detach().cpu()))
            mae_norms.append(float(mae_norm.detach().cpu()))
            regularizer_norms.append(float(regularizer_norm.detach().cpu()))
            if (step + 1) % 10 == 0:
                print(
                    f"[{step + 1}/{args.batches}] median q="
                    f"{float(np.median(ratios)):.8g}",
                    flush=True,
                )
    finally:
        for loader in (trainer.dataloader_train, trainer.dataloader_val):
            if loader is not None and hasattr(loader, "_finish"):
                loader._finish()

    checkpoint = Path(args.pretrained_weights).resolve()
    result = {
        "schema_version": 1,
        "trainer": args.trainer,
        "dataset": args.dataset,
        "configuration": args.configuration,
        "plans": args.plans,
        "batches": args.batches,
        "batch_size": trainer.total_batch_size,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "calibration_script_sha256": _file_sha256(Path(__file__).resolve()),
        "manifest_sha256": _file_sha256(Path(args.manifest).resolve()) if args.manifest else None,
        "measurement_parameters": names,
        "q_median": float(np.median(ratios)),
        "q_mean": float(np.mean(ratios)),
        "mae_grad_norm_median": float(np.median(mae_norms)),
        "regularizer_grad_norm_median": float(np.median(regularizer_norms)),
        "ratios": ratios,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "ratios"}, indent=2))


def derive(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    compatibility_fields = (
        "dataset",
        "configuration",
        "plans",
        "batches",
        "batch_size",
        "seed",
        "checkpoint_sha256",
        "calibration_script_sha256",
        "manifest_sha256",
        "measurement_parameters",
    )
    mismatched = [
        field for field in compatibility_fields if reference.get(field) != candidate.get(field)
    ]
    if mismatched:
        raise ValueError(f"Incompatible calibration JSON fields: {mismatched}")
    q_reference = float(reference["q_median"])
    q_candidate = float(candidate["q_median"])
    if not math.isfinite(q_reference) or not math.isfinite(q_candidate):
        raise ValueError("Calibration q values must be finite")
    if q_reference <= args.eps or q_candidate <= args.eps:
        raise ValueError("Calibration q values must be positive")
    rho_reference = args.reference_weight * q_reference
    target = args.rho_multiplier * rho_reference
    recommended = target / q_candidate
    if not math.isfinite(recommended) or recommended <= 0:
        raise ValueError("Derived regularizer weight must be finite and positive")
    if args.json:
        print(
            json.dumps(
                {
                    "rho_reference": rho_reference,
                    "rho_multiplier": args.rho_multiplier,
                    "rho_target": target,
                    "recommended_lambda": recommended,
                    "reference_json_sha256": _file_sha256(Path(args.reference_json)),
                    "candidate_json_sha256": _file_sha256(Path(args.candidate_json)),
                }
            )
        )
    else:
        print(f"{recommended:.12g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--dataset", default="746")
    measure.add_argument("--configuration", default="onemmiso")
    measure.add_argument("--plans", default="nnsslPlans")
    measure.add_argument("--trainer", required=True)
    measure.add_argument("--pretrained-weights", required=True)
    measure.add_argument("--batches", type=int, default=200)
    measure.add_argument("--seed", type=int, default=20260821)
    measure.add_argument("--device", default="cuda")
    measure.add_argument("--output", required=True)
    measure.add_argument("--manifest")
    measure.add_argument("--eps", type=float, default=1e-12)
    measure.set_defaults(func=calibrate)

    derive_parser = subparsers.add_parser("derive")
    derive_parser.add_argument("--reference-json", required=True)
    derive_parser.add_argument("--candidate-json", required=True)
    derive_parser.add_argument("--reference-weight", type=float, default=0.01)
    derive_parser.add_argument("--rho-multiplier", type=float, default=1.0)
    derive_parser.add_argument("--eps", type=float, default=1e-12)
    derive_parser.add_argument("--json", action="store_true")
    derive_parser.set_defaults(func=derive)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
