#!/usr/bin/env python3
"""Run one OpenMind-style Primus-M downstream segmentation benchmark.

This is a thin, auditable wrapper around TaWald/nnUNet's
``run_training_from_pretrained`` interface. It deliberately keeps the uploaded
preprocessed data untouched apart from writing a model-specific copy of the
plans JSON next to the source plan.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_NNUNET_REPO = Path("/home/u6mn/taotl.u6mn/u6mn/nnUNet-openmind")
DEFAULT_PREPROCESSED = Path(
    "/lus/lfs1aip2/projects/u6mn/datasets/OpenMind_downstream/segmentation_preprocessed"
)
DEFAULT_RESULTS = Path("/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark")
DEFAULT_WEIGHTS = Path("/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights")

DATASET_LABELS = {
    201: "AMOS",
    202: "KiTS19",
    203: "ISLES22",
    204: "HNTSMRG",
    205: "HaNSeg",
    206: "MSFLAIR",
    207: "TopCoW_MRA",
    208: "YaleBrainMets",
    209: "ACDC",
}

PAPER_PROTOCOL = {
    "architecture": "PrimusM",
    "trainer": "PretrainedTrainer_Primus_150ep",
    "configuration": "3d_fullres",
    "epochs": 150,
    "train_iterations_per_epoch": 250,
    "validation_iterations_per_epoch": 50,
    "batch_size": 2,
    "initial_lr": 3e-5,
    "warmup_epochs": 15,
    "patch_size": [160, 160, 160],
    "optimizer": "AdamW",
    "weight_decay": 5e-2,
    "schedule": "linear warm-up then polynomial decay",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not name:
        raise ValueError(f"Cannot derive a run name from {value!r}")
    return name


def resolve_dataset(root: Path, selector: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Preprocessed root does not exist: {root}")
    candidates = sorted(path for path in root.glob("Dataset???_*") if path.is_dir())
    if selector.isdigit():
        prefix = f"Dataset{int(selector):03d}_"
        matches = [path for path in candidates if path.name.startswith(prefix)]
    elif selector.startswith("Dataset"):
        matches = [path for path in candidates if path.name == selector]
    else:
        key = selector.casefold()
        matches = [path for path in candidates if path.name[11:].casefold() == key]
    if len(matches) != 1:
        available = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Dataset selector {selector!r} matched {len(matches)} folders. Available: {available}"
        )
    return matches[0]


def resolve_checkpoint(
    model: str | None, checkpoint: Path | None, weights_root: Path
) -> tuple[Path, str]:
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        model_name = model or resolved.parent.name
    else:
        if not model:
            raise ValueError("Provide --model or --checkpoint")
        model_path = Path(model).expanduser()
        if model_path.is_file():
            resolved = model_path.resolve()
            model_name = resolved.parent.name
        elif model_path.is_dir():
            resolved = (model_path / "checkpoint_final.pth").resolve()
            model_name = model_path.name
        else:
            resolved = (weights_root / model / "checkpoint_final.pth").resolve()
            model_name = model
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved, model_name


def adaptation_architecture(checkpoint: Path) -> tuple[str | None, Path | None]:
    adaptation_path = checkpoint.parent / "adaptation_plan.json"
    if not adaptation_path.is_file():
        return None, None
    adaptation = read_json(adaptation_path)
    architecture = adaptation.get("architecture_plans", {}).get("arch_class_name")
    return architecture, adaptation_path


def prepare_plan(
    dataset_dir: Path,
    source_identifier: str,
    target_identifier: str,
    checkpoint: Path,
    model_name: str,
    configuration: str,
    batch_size: int,
) -> tuple[Path, dict[str, Any]]:
    source = dataset_dir / f"{source_identifier}.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Source plan not found: {source}. These uploaded datasets are expected to contain PMPrep.json."
        )
    plan = read_json(source)
    if configuration not in plan.get("configurations", {}):
        raise KeyError(f"Configuration {configuration!r} is missing from {source}")
    pretrain_info = plan.get("pretrain_info")
    if not isinstance(pretrain_info, dict):
        raise KeyError(f"pretrain_info is missing from {source}")

    plan["plans_name"] = target_identifier
    pretrain_info["checkpoint_path"] = str(checkpoint)
    pretrain_info["checkpoint_name"] = model_name
    plan["configurations"][configuration]["batch_size"] = batch_size
    target = dataset_dir / f"{target_identifier}.json"
    write_json(target, plan)
    return target, plan


def validate_inputs(
    dataset_dir: Path,
    plan: dict[str, Any],
    configuration: str,
    fold: int,
) -> dict[str, Any]:
    required = [
        dataset_dir / "dataset.json",
        dataset_dir / "dataset_fingerprint.json",
        dataset_dir / "splits_final.json",
        dataset_dir / "gt_segmentations",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required downstream artifacts are missing: {missing}")
    splits = read_json(dataset_dir / "splits_final.json")
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Fold {fold} is unavailable; splits_final.json contains {len(splits)} split(s)")
    config = plan["configurations"][configuration]
    data_dir = dataset_dir / config["data_identifier"]
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed configuration folder not found: {data_dir}")
    split = splits[fold]
    return {
        "train_cases": len(split["train"]),
        "validation_cases": len(split["val"]),
        "data_identifier": config["data_identifier"],
        "patch_size": list(config["patch_size"]),
        "spacing": list(config["spacing"]),
        "normalization_schemes": list(config["normalization_schemes"]),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Train and full-volume validate one public Primus-M OpenMind checkpoint on one downstream dataset."
    )
    result.add_argument("--model", help="Model directory name under --weights-root, e.g. PrimusM-OpenMind-MAE")
    result.add_argument("--dataset", required=True, help="Dataset ID/name, e.g. 203, ISLES22, or Dataset203_ISLES22")
    result.add_argument("--checkpoint", type=Path, help="Direct checkpoint path (overrides model lookup)")
    result.add_argument("--run-name", help="Optional stable result/plan name")
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--nnunet-repo", type=Path, default=Path(os.environ.get("NNUNET_ADAPTATION_REPO", DEFAULT_NNUNET_REPO)))
    result.add_argument("--preprocessed-root", type=Path, default=Path(os.environ.get("OPENMIND_DOWNSTREAM_PREPROCESSED", DEFAULT_PREPROCESSED)))
    result.add_argument("--results-root", type=Path, default=Path(os.environ.get("OPENMIND_DOWNSTREAM_RESULTS", DEFAULT_RESULTS)))
    result.add_argument("--weights-root", type=Path, default=Path(os.environ.get("OPENMIND_WEIGHTS", DEFAULT_WEIGHTS)))
    result.add_argument("--raw-root", type=Path, default=None, help="Optional nnUNet_raw root; not needed for validation of preprocessed cases")
    result.add_argument("--source-plan", default="PMPrep")
    result.add_argument("--configuration", default=PAPER_PROTOCOL["configuration"])
    result.add_argument("--trainer", default=PAPER_PROTOCOL["trainer"])
    result.add_argument("--epochs", type=int, default=PAPER_PROTOCOL["epochs"])
    result.add_argument("--steps-per-epoch", type=int, default=PAPER_PROTOCOL["train_iterations_per_epoch"])
    result.add_argument("--val-steps", type=int, default=PAPER_PROTOCOL["validation_iterations_per_epoch"])
    result.add_argument("--batch-size", type=int, default=PAPER_PROTOCOL["batch_size"])
    result.add_argument("--learning-rate", type=float, default=PAPER_PROTOCOL["initial_lr"])
    result.add_argument("--warmup-epochs", type=int, default=PAPER_PROTOCOL["warmup_epochs"])
    result.add_argument("--continue", dest="continue_training", action="store_true")
    result.add_argument("--validation-only", action="store_true")
    result.add_argument("--val-best", action="store_true")
    result.add_argument("--npz", action="store_true", help="Also export validation probabilities")
    result.add_argument("--dry-run", action="store_true", help="Prepare/check inputs and print the resolved run without using a GPU")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.continue_training and args.validation_only:
        raise ValueError("--continue and --validation-only are mutually exclusive")
    if min(args.epochs, args.steps_per_epoch, args.val_steps, args.batch_size) <= 0:
        raise ValueError("Epoch, step, validation-step, and batch-size values must all be positive")
    if args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be smaller than --epochs")

    nnunet_repo = args.nnunet_repo.expanduser().resolve()
    if not (nnunet_repo / "nnunetv2").is_dir():
        raise FileNotFoundError(f"TaWald/nnUNet adaptation checkout not found: {nnunet_repo}")
    dataset_dir = resolve_dataset(args.preprocessed_root.expanduser().resolve(), args.dataset)
    checkpoint, model_name = resolve_checkpoint(args.model, args.checkpoint, args.weights_root)
    architecture, adaptation_path = adaptation_architecture(checkpoint)
    if architecture is not None and architecture != "PrimusM":
        raise ValueError(
            f"This runner currently targets Primus-M, but {adaptation_path} declares {architecture!r}"
        )

    run_name = args.run_name or f"OMBench_{safe_name(model_name)}"
    plan_identifier = safe_name(run_name)
    plan_path, plan = prepare_plan(
        dataset_dir,
        args.source_plan,
        plan_identifier,
        checkpoint,
        model_name,
        args.configuration,
        args.batch_size,
    )
    inventory = validate_inputs(dataset_dir, plan, args.configuration, args.fold)
    protocol = dict(PAPER_PROTOCOL)
    protocol.update(
        trainer=args.trainer,
        configuration=args.configuration,
        epochs=args.epochs,
        train_iterations_per_epoch=args.steps_per_epoch,
        validation_iterations_per_epoch=args.val_steps,
        batch_size=args.batch_size,
        initial_lr=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
    )
    resolved = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "adaptation_plan": str(adaptation_path) if adaptation_path else None,
        "declared_architecture": architecture,
        "dataset": dataset_dir.name,
        "fold": args.fold,
        "source_plan": args.source_plan,
        "generated_plan": str(plan_path),
        "plan_identifier": plan_identifier,
        "nnunet_repo": str(nnunet_repo),
        "preprocessed_root": str(args.preprocessed_root.expanduser().resolve()),
        "results_root": str(args.results_root.expanduser().resolve()),
        "protocol": protocol,
        "inventory": inventory,
    }
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0

    results_root = args.results_root.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_preprocessed"] = str(args.preprocessed_root.expanduser().resolve())
    os.environ["nnUNet_results"] = str(results_root)
    if args.raw_root is not None:
        os.environ["nnUNet_raw"] = str(args.raw_root.expanduser().resolve())
    os.environ.setdefault("nnUNet_compile", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    sys.path.insert(0, str(nnunet_repo))

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from nnunetv2.run.run_training_from_pretrained import get_trainer_from_args
    from torch.backends import cudnn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (use --dry-run for a CPU-only preflight)")
    trainer = get_trainer_from_args(
        dataset_dir.name,
        args.configuration,
        args.fold,
        args.trainer,
        plan_identifier,
        device=torch.device("cuda"),
        pretrained_from_scratch=False,
        overwrite_ckpt_path=None,
    )
    trainer.num_epochs = args.epochs
    trainer.num_iterations_per_epoch = args.steps_per_epoch
    trainer.num_val_iterations_per_epoch = args.val_steps
    trainer.batch_size = args.batch_size
    trainer.configuration_manager.configuration["batch_size"] = args.batch_size
    trainer.initial_lr = args.learning_rate
    trainer.warmup_duration_whole_net = args.warmup_epochs
    trainer.use_pretrained_weights = not (args.continue_training or args.validation_only)

    output_folder = Path(trainer.output_folder)
    final_checkpoint = output_folder / "checkpoint_final.pth"
    if final_checkpoint.is_file() and not (args.continue_training or args.validation_only):
        raise FileExistsError(
            f"A completed run already exists at {final_checkpoint}. Use --validation-only, --continue, or a new --run-name."
        )
    if args.continue_training:
        resumable = [
            output_folder / "checkpoint_final.pth",
            output_folder / "checkpoint_latest.pth",
            output_folder / "checkpoint_best.pth",
        ]
        if not any(path.is_file() for path in resumable):
            raise FileNotFoundError(
                f"--continue was requested, but no training checkpoint exists in {output_folder}"
            )

    run_record = dict(resolved)
    run_record.update(
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        output_folder=str(output_folder),
        host=socket.gethostname(),
        platform=platform.platform(),
        python=sys.version,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        command=[sys.executable, *sys.argv],
    )
    record_path = output_folder / "benchmark_run.json"
    write_json(record_path, run_record)
    started = time.perf_counter()
    try:
        maybe_load_checkpoint(trainer, args.continue_training, args.validation_only, None)
        cudnn.deterministic = False
        cudnn.benchmark = True
        if not args.validation_only:
            trainer.run_training()
        if args.val_best:
            trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_best.pth"))
        trainer.perform_actual_validation(args.npz)

        summary_path = output_folder / "validation" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Full-volume validation did not create {summary_path}")
        summary = read_json(summary_path)
        run_record.update(
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            validation_summary=str(summary_path),
            foreground_mean=summary.get("foreground_mean"),
            per_class_mean=summary.get("mean"),
        )
        write_json(record_path, run_record)
        print(json.dumps(run_record["foreground_mean"], indent=2, ensure_ascii=False))
        print(f"Benchmark record: {record_path}")
        return 0
    except BaseException as error:
        run_record.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        write_json(record_path, run_record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
