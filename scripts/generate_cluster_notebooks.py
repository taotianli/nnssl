"""Generate the two cluster notebooks tracked in notebooks/."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write(name: str, cells: list):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3 (nnssl)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(notebook, OUT / name)


write(
    "01_openmind2000_preprocess_and_primus_jepa.ipynb",
    [
        md(
            """
            # OpenMind 2000-volume preprocessing and Primus MAE+JEPA training

            This notebook is designed for the Isambard cluster. It performs four explicit stages:

            1. deterministically select 2,000 real MRI acquisitions from `openneuro_metadata.csv`, including BIDS paths with and without `ses-*`;
            2. build an nnSSL `pretrain_data.json` and run the official fingerprint → plan → `onemmiso` preprocessing pipeline;
            3. visually and numerically verify raw versus preprocessed volumes;
            4. load the released PrimusM OpenMind MAE checkpoint and train the joint MAE+JEPA trainer for 200 epochs, then plot total/MAE/JEPA losses.

            The raw OpenMind download is never modified. All generated files go to project storage under
            `/lus/lfs1aip2/projects/u6mn/openmind_jepa` by default.
            """
        ),
        code(
            """
            from __future__ import annotations

            import hashlib
            import json
            import os
            import pickle
            import shutil
            import subprocess
            import sys
            from collections import defaultdict
            from pathlib import Path

            import blosc2
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            # Cluster paths. Override any of these with environment variables before starting Jupyter.
            default_repo = Path("/home/u6mn/taotl.u6mn/u6mn/nnssl")
            REPO_ROOT = Path(os.environ.get("NNSSL_REPO", Path.cwd() if (Path.cwd() / "src/nnssl").is_dir() else default_repo))
            OPENMIND_DOWNLOAD_ROOT = Path(os.environ.get("OPENMIND_DOWNLOAD_ROOT", "/home/u6mn/taotl.u6mn/u6mn/OpenMind"))
            OPENMIND_IMAGE_ROOT = OPENMIND_DOWNLOAD_ROOT / "OpenMind"
            METADATA_CSV = OPENMIND_DOWNLOAD_ROOT / "openneuro_metadata.csv"
            WORK_ROOT = Path(os.environ.get("OPENMIND_JEPA_WORK", "/lus/lfs1aip2/projects/u6mn/openmind_jepa"))
            NNSSL_RAW = WORK_ROOT / "nnssl_raw"
            NNSSL_PREPROCESSED = WORK_ROOT / "nnssl_preprocessed"
            NNSSL_RESULTS = WORK_ROOT / "nnssl_results"
            CHECKPOINT = Path(os.environ.get("PRIMUS_MAE_CHECKPOINT", REPO_ROOT / "weights/PrimusM-OpenMind-MAE/checkpoint_final.pth"))

            DATASET_ID = 745
            DATASET_NAME = "Dataset745_OpenMind2000"
            CONFIGURATION = "onemmiso"
            PLANS = "nnsslPlans"
            N_IMAGES = 2000
            SEED = 2026
            NUM_PROCESSES_FINGERPRINT = 12
            NUM_PROCESSES_PREPROCESS = 12

            for directory in (NNSSL_RAW, NNSSL_PREPROCESSED, NNSSL_RESULTS):
                directory.mkdir(parents=True, exist_ok=True)
            os.environ.update(
                nnssl_raw=str(NNSSL_RAW),
                nnssl_preprocessed=str(NNSSL_PREPROCESSED),
                nnssl_results=str(NNSSL_RESULTS),
                OMP_NUM_THREADS="8",
                MKL_NUM_THREADS="8",
            )
            if str(REPO_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(REPO_ROOT / "src"))

            required = [REPO_ROOT / "src/nnssl", OPENMIND_IMAGE_ROOT, METADATA_CSV, CHECKPOINT]
            missing = [str(path) for path in required if not path.exists()]
            assert not missing, f"Missing required cluster paths: {missing}"
            print(json.dumps({
                "repo": str(REPO_ROOT), "metadata": str(METADATA_CSV), "raw_definition": str(NNSSL_RAW),
                "preprocessed": str(NNSSL_PREPROCESSED), "results": str(NNSSL_RESULTS),
                "checkpoint": str(CHECKPOINT),
            }, indent=2))
            print(shutil.disk_usage(WORK_ROOT))
            """
        ),
        md(
            """
            ## 1. Build a deterministic, dataset-balanced manifest

            Selection is based on the official metadata table rather than assumptions about directory depth. Therefore both
            `ds000001/sub-01/anat/...` and `ds000017/sub-6/ses-timepoint1/anat/...` are handled correctly. Mask files are never
            selected as images because selection uses the metadata `image_path` column. Original acquisitions are preferred
            over derived volumes, and datasets are sampled round-robin so one large OpenNeuro dataset cannot dominate the 2,000 images.
            """
        ),
        code(
            """
            metadata = pd.read_csv(METADATA_CSV, low_memory=False)
            required_columns = {"unique_id", "image_path", "modality"}
            assert required_columns.issubset(metadata.columns), required_columns - set(metadata.columns)
            metadata = metadata.dropna(subset=["unique_id", "image_path", "modality"]).copy()
            metadata["dataset_id"] = metadata["image_path"].astype(str).str.split("/").str[0]
            metadata["local_path"] = metadata["image_path"].map(lambda value: str(OPENMIND_IMAGE_ROOT / str(value)))

            derived = metadata.get("derived_from", pd.Series(index=metadata.index, dtype=object))
            originals = metadata[derived.isna() | derived.astype(str).str.strip().isin(["", "nan", "None"])].copy()
            candidates = originals if len(originals) >= N_IMAGES else metadata
            candidates["selection_key"] = candidates["unique_id"].map(
                lambda value: hashlib.sha1(f"{SEED}:{value}".encode()).hexdigest()
            )
            groups = {
                dataset_id: group.sort_values("selection_key").reset_index(drop=True)
                for dataset_id, group in candidates.groupby("dataset_id", sort=True)
            }
            selected_rows = []
            rank = 0
            while len(selected_rows) < N_IMAGES:
                made_progress = False
                for dataset_id in sorted(groups):
                    group = groups[dataset_id]
                    if rank >= len(group):
                        continue
                    made_progress = True
                    row = group.iloc[rank]
                    if Path(row["local_path"]).is_file():
                        selected_rows.append(row)
                        if len(selected_rows) == N_IMAGES:
                            break
                if not made_progress:
                    break
                rank += 1

            manifest = pd.DataFrame(selected_rows).drop(columns=["selection_key"], errors="ignore").reset_index(drop=True)
            assert len(manifest) == N_IMAGES, f"Only found {len(manifest)} readable images"
            assert manifest["unique_id"].is_unique
            MANIFEST_CSV = WORK_ROOT / "openmind2000_manifest.csv"
            manifest.to_csv(MANIFEST_CSV, index=False)
            print(f"Saved {len(manifest)} rows to {MANIFEST_CSV}")
            display(manifest[["unique_id", "modality", "dataset_id", "local_path"]].head())
            display(manifest["modality"].value_counts().head(20).rename("images").to_frame())
            print("OpenNeuro datasets represented:", manifest["dataset_id"].nunique())
            """
        ),
        md("## 2. Convert the selected rows to the official nnSSL collection format"),
        code(
            """
            from nnssl.data.raw_dataset import AssociatedMasks, Collection, Dataset, Image, Session, Subject

            subject_info_keys = ["age", "sex", "handedness", "race", "weight", "bmi", "health_status"]
            image_info_keys = [
                "derived_from", "is_brain_extract", "manufacturer", "model_name", "phase_encoding_direction",
                "magnetic_field_strength", "repetition_time", "echo_time", "image_quality_score",
            ]
            collection = Collection(collection_name=DATASET_NAME, collection_index=DATASET_ID)

            def optional_absolute_path(value):
                if value is None or pd.isna(value) or not str(value).strip():
                    return None
                path = OPENMIND_IMAGE_ROOT / str(value)
                return str(path) if path.is_file() else None

            for row in manifest.to_dict(orient="records"):
                parts = Path(row["image_path"]).parts
                dataset_id = parts[0]
                subject_id = next(part for part in parts if part.startswith("sub-"))
                session_id = next((part for part in parts if part.startswith("ses-")), "ses-DEFAULT")
                subject_info = {key: row[key] for key in subject_info_keys if key in row and not pd.isna(row[key])}
                image_info = {key: row[key] for key in image_info_keys if key in row and not pd.isna(row[key])}
                dataset = collection.datasets.setdefault(
                    dataset_id, Dataset(dataset_index=dataset_id, name=None, dataset_info={})
                )
                subject = dataset.subjects.setdefault(
                    subject_id, Subject(subject_id=subject_id, subject_info=subject_info)
                )
                session = subject.sessions.setdefault(session_id, Session(session_id=session_id, images=[]))
                session.images.append(Image(
                    name=Path(row["image_path"]).name,
                    image_path=row["local_path"],
                    modality=row["modality"],
                    image_info=image_info,
                    associated_masks=AssociatedMasks(
                        anonymization_mask=optional_absolute_path(row.get("anon_mask_path")),
                        anatomy_mask=optional_absolute_path(row.get("anat_mask_path")),
                    ),
                ))

            PRETRAIN_JSON = NNSSL_RAW / DATASET_NAME / "pretrain_data.json"
            PRETRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
            PRETRAIN_JSON.write_text(json.dumps(collection.to_dict(relative_paths=False), indent=2), encoding="utf-8")
            assert len(collection.to_independent_images()) == N_IMAGES
            print(f"Saved nnSSL collection: {PRETRAIN_JSON}")
            """
        ),
        md(
            """
            ## 3. Fingerprint, plan and preprocess all 2,000 volumes

            The output is compressed `.b2nd` data plus properties, stored separately from the raw BIDS tree. Set the flags to
            `False` after a successful run. The `valid_imgs.json` assertion is the completion marker; do not start 200-epoch
            training unless it contains exactly 2,000 successful images.
            """
        ),
        code(
            """
            from nnssl.experiment_planning.plan_and_preprocess_api import (
                extract_fingerprint_dataset, plan_experiment_dataset, preprocess_dataset,
            )

            RUN_FINGERPRINT = True
            RUN_PLANNING = True
            RUN_PREPROCESSING = True

            fingerprint_path = NNSSL_PREPROCESSED / DATASET_NAME / "dataset_fingerprint.json"
            plans_path = NNSSL_PREPROCESSED / DATASET_NAME / f"{PLANS}.json"
            valid_path = NNSSL_PREPROCESSED / DATASET_NAME / "valid_imgs.json"

            if RUN_FINGERPRINT or not fingerprint_path.is_file():
                extract_fingerprint_dataset(DATASET_ID, num_processes=NUM_PROCESSES_FINGERPRINT, clean=True, verbose=False)
            if RUN_PLANNING or not plans_path.is_file():
                plan_experiment_dataset(DATASET_ID)
            if RUN_PREPROCESSING or not valid_path.is_file():
                preprocess_dataset(
                    DATASET_ID, plans_identifier=PLANS, configurations=(CONFIGURATION,),
                    num_processes=[NUM_PROCESSES_PREPROCESS], verbose=False,
                )

            valid_images = json.loads(valid_path.read_text())
            failures = N_IMAGES - len(valid_images)
            print({"requested": N_IMAGES, "preprocessed": len(valid_images), "failures": failures})
            assert failures == 0, "Inspect preprocessing stdout before training; one or more volumes failed."
            """
        ),
        md("## 4. Raw-versus-preprocessed QC"),
        code(
            """
            from nnssl.data.raw_dataset import Collection

            pp_json = NNSSL_PREPROCESSED / DATASET_NAME / f"pretrain_data__{CONFIGURATION}.json"
            pp_collection = Collection.from_dict(json.loads(pp_json.read_text()))
            raw_collection = Collection.from_dict(json.loads(PRETRAIN_JSON.read_text()))
            pp_images = {image.get_unique_id(): image for image in pp_collection.to_independent_images()}
            raw_images = {image.get_unique_id(): image for image in raw_collection.to_independent_images()}
            common_ids = sorted(set(pp_images) & set(raw_images))
            assert len(common_ids) == N_IMAGES

            rng = np.random.default_rng(SEED)
            qc_ids = rng.choice(common_ids, size=4, replace=False)

            def load_b2nd(path):
                return np.asarray(blosc2.open(urlpath=str(path), mode="r", mmap_mode="r"))

            def middle_planes(volume):
                volume = np.asarray(volume).squeeze()
                return [volume[volume.shape[0] // 2], volume[:, volume.shape[1] // 2], volume[:, :, volume.shape[2] // 2]]

            def robust_limits(volume):
                finite = np.asarray(volume)[np.isfinite(volume)]
                return tuple(np.percentile(finite, [1, 99]))

            qc_records = []
            figure, axes = plt.subplots(len(qc_ids), 6, figsize=(18, 4 * len(qc_ids)))
            for row_index, identity in enumerate(qc_ids):
                raw_image = raw_images[identity]
                pp_image = pp_images[identity]
                import SimpleITK as sitk
                raw = sitk.GetArrayFromImage(sitk.ReadImage(raw_image.image_path)).astype(np.float32)
                processed = load_b2nd(pp_image.image_path)[0]
                raw_limits, pp_limits = robust_limits(raw), robust_limits(processed)
                for column, plane in enumerate(middle_planes(raw)):
                    axes[row_index, column].imshow(plane, cmap="gray", vmin=raw_limits[0], vmax=raw_limits[1])
                    axes[row_index, column].axis("off")
                for column, plane in enumerate(middle_planes(processed), start=3):
                    axes[row_index, column].imshow(plane, cmap="gray", vmin=pp_limits[0], vmax=pp_limits[1])
                    axes[row_index, column].axis("off")
                axes[row_index, 0].set_title(f"RAW {raw.shape}\\n{identity[-55:]}")
                axes[row_index, 3].set_title(f"PREPROCESSED {processed.shape}")
                qc_records.append({
                    "id": identity, "raw_shape": raw.shape, "processed_shape": processed.shape,
                    "finite": bool(np.isfinite(processed).all()), "mean": float(processed.mean()),
                    "std": float(processed.std()), "p01": float(pp_limits[0]), "p99": float(pp_limits[1]),
                })
            plt.tight_layout()
            QC_PNG = WORK_ROOT / "openmind2000_preprocessing_qc.png"
            plt.savefig(QC_PNG, dpi=160, bbox_inches="tight")
            plt.show()
            qc_table = pd.DataFrame(qc_records)
            display(qc_table)
            assert qc_table["finite"].all() and (qc_table["std"] > 0).all()
            print("QC figure:", QC_PNG)
            """
        ),
        md(
            """
            ## 5. GPU witness and 200-epoch MAE+JEPA training

            The trainer loads `checkpoint_final.pth` into the shared online encoder/MAE decoder, initializes the predictor,
            synchronizes the EMA target encoder, and optimizes `L = L_MAE + 0.1 * L_JEPA`. It saves latest/best/final
            checkpoints and `jepa_loss_history.json`. Set `CONTINUE_TRAINING=True` to resume after a scheduler or wall-time stop.

            nnSSL defines an epoch as 250 training iterations plus 50 validation iterations. Thus this 200-epoch run performs
            50,000 optimizer steps while sampling from the full 2,000-volume pool; it is not 200 exhaustive passes over every file.
            """
        ),
        code(
            """
            import torch

            assert torch.cuda.is_available(), "Run this section in a GPU Jupyter/Slurm allocation."
            torch.manual_seed(SEED)
            witness = torch.randn(8, 8, device="cuda")
            print("WITNESS", (witness @ witness).shape, torch.cuda.get_device_name(), torch.__version__)

            TRAINER = "PrimusJEPATrainer_200ep_BS1"
            CONTINUE_TRAINING = False
            RUN_TRAINING = False  # Inspect paths and QC first, then change to True.
            command = [
                sys.executable, "-m", "nnssl.run.run_training", str(DATASET_ID), CONFIGURATION,
                "-tr", TRAINER, "-p", PLANS,
            ]
            if CONTINUE_TRAINING:
                command.append("--c")
            else:
                command.extend(["-pretrained_weights", str(CHECKPOINT)])
            print(" ".join(command))
            if RUN_TRAINING:
                subprocess.run(command, cwd=REPO_ROOT, check=True, env=os.environ.copy())
            """
        ),
        md("## 6. Plot total, MAE and JEPA losses during or after training"),
        code(
            """
            history_files = sorted(NNSSL_RESULTS.rglob("jepa_loss_history.json"), key=lambda path: path.stat().st_mtime)
            assert history_files, "No history yet. Start training first, then rerun this cell at any epoch."
            history_path = history_files[-1]
            history = json.loads(history_path.read_text())
            epochs = np.arange(1, len(history["train_losses"]) + 1)

            figure, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(epochs, history["train_losses"], label="train total")
            axes[0].plot(epochs, history["val_losses"], label="validation total")
            axes[1].plot(epochs, history["train_mae_losses"], label="train MAE")
            axes[1].plot(epochs, history["train_jepa_losses"], label="train JEPA")
            axes[1].plot(epochs, history["val_mae_losses"], "--", label="validation MAE")
            axes[1].plot(epochs, history["val_jepa_losses"], "--", label="validation JEPA")
            for axis in axes:
                axis.set_xlabel("epoch")
                axis.set_ylabel("loss")
                axis.grid(alpha=0.25)
                axis.legend()
            figure.suptitle(f"Primus MAE+JEPA — {len(epochs)}/200 epochs")
            figure.tight_layout()
            loss_png = history_path.parent / "mae_jepa_loss_curves.png"
            figure.savefig(loss_png, dpi=180)
            plt.show()
            print("History:", history_path)
            print("Figure:", loss_png)
            """
        ),
    ],
)


write(
    "02_primus_downstream_9datasets_one_epoch.ipynb",
    [
        md(
            """
            # PrimusM-MAE: nine downstream segmentation datasets, one epoch each

            This cluster notebook reproduces the local compatibility test against every uploaded OpenMind-preprocessed
            downstream dataset. Each task loads the released PrimusM MAE checkpoint, runs 250 training and 50 validation
            patch iterations, records loss/pseudo Dice/memory/runtime, and continues after per-dataset failures.

            It creates cluster-specific plan copies and does not edit the uploaded `PMPrep.json` files or `.b2nd` volumes.
            """
        ),
        code(
            """
            from __future__ import annotations

            import gc
            import json
            import os
            import sys
            import time
            import traceback
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch

            NNSSL_REPO = Path(os.environ.get("NNSSL_REPO", "/home/u6mn/taotl.u6mn/u6mn/nnssl"))
            NNUNET_REPO = Path(os.environ.get("NNUNET_ADAPTATION_REPO", "/home/u6mn/taotl.u6mn/u6mn/nnUNet-openmind"))
            PREPROCESSED_ROOT = Path(os.environ.get(
                "OPENMIND_DOWNSTREAM_PREPROCESSED",
                "/lus/lfs1aip2/projects/u6mn/datasets/OpenMind_downstream/segmentation_preprocessed",
            ))
            RESULTS_ROOT = Path(os.environ.get(
                "OPENMIND_DOWNSTREAM_RESULTS",
                "/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_primus_smoke",
            ))
            CHECKPOINT = Path(os.environ.get(
                "PRIMUS_MAE_CHECKPOINT",
                NNSSL_REPO / "weights/PrimusM-OpenMind-MAE/checkpoint_final.pth",
            ))
            TRAINER_NAME = os.environ.get("NNUNET_PRIMUS_TRAINER", "PMMAE1")
            SOURCE_PLAN = "PMPrep"
            CLUSTER_PLAN = "PMPrepCluster"
            BATCH_SIZE = 2
            DATASET_IDS = list(range(201, 210))

            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            os.environ.update(
                nnUNet_preprocessed=str(PREPROCESSED_ROOT), nnUNet_results=str(RESULTS_ROOT),
                nnUNet_compile="false", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
            )
            if NNUNET_REPO.is_dir() and str(NNUNET_REPO) not in sys.path:
                sys.path.insert(0, str(NNUNET_REPO))
            assert PREPROCESSED_ROOT.is_dir(), PREPROCESSED_ROOT
            assert CHECKPOINT.is_file(), CHECKPOINT
            print({"data": str(PREPROCESSED_ROOT), "results": str(RESULTS_ROOT), "checkpoint": str(CHECKPOINT)})
            """
        ),
        md("## 1. Discover and audit all uploaded datasets"),
        code(
            """
            dataset_dirs = {int(path.name[7:10]): path for path in PREPROCESSED_ROOT.glob("Dataset???_*")}
            missing = [dataset_id for dataset_id in DATASET_IDS if dataset_id not in dataset_dirs]
            assert not missing, f"Datasets not uploaded yet: {missing}"

            inventory = []
            for dataset_id in DATASET_IDS:
                folder = dataset_dirs[dataset_id]
                dataset_json = json.loads((folder / "dataset.json").read_text())
                splits = json.loads((folder / "splits_final.json").read_text())
                data_folders = [path for path in folder.iterdir() if path.is_dir() and path.name.endswith("3d_fullres")]
                assert len(data_folders) == 1, (folder, data_folders)
                b2nd = list(data_folders[0].glob("*.b2nd"))
                inventory.append({
                    "id": dataset_id, "dataset": folder.name, "channels": len(dataset_json["channel_names"]),
                    "labels": len(dataset_json["labels"]), "train": len(splits[0]["train"]),
                    "validation": len(splits[0]["val"]), "b2nd_files": len(b2nd),
                })
            inventory_table = pd.DataFrame(inventory)
            display(inventory_table)
            """
        ),
        md(
            """
            ## 2. Install runtime compatibility hooks

            Released OpenMind property files store foreground coordinates as `(z,y,x)`, while some nnU-Net branches expect
            `(class,z,y,x)`. The first hook adds a temporary dummy coordinate column before bounding-box selection. The second
            maps sparse source label IDs (TopCoW uses `0..12,15`) to contiguous training channels. These hooks live only in the
            current Python process and do not rewrite uploaded data.
            """
        ),
        code(
            """
            from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

            if not getattr(nnUNetDataLoader, "_openmind_compat_installed", False):
                original_get_bbox = nnUNetDataLoader.get_bbox
                original_generate = nnUNetDataLoader.generate_train_batch

                def openmind_get_bbox(self, data_shape, force_fg, class_locations, *args, **kwargs):
                    if class_locations is not None:
                        normalized = {}
                        spatial_dims = len(data_shape)
                        for key, locations in class_locations.items():
                            array = np.asarray(locations)
                            if array.ndim == 2 and array.shape[1] == spatial_dims:
                                array = np.concatenate([np.zeros((len(array), 1), dtype=array.dtype), array], axis=1)
                            normalized[key] = array
                        class_locations = normalized
                    return original_get_bbox(self, data_shape, force_fg, class_locations, *args, **kwargs)

                def remap_target(target, mapping):
                    if isinstance(target, list):
                        return [remap_target(item, mapping) for item in target]
                    source = target.clone() if torch.is_tensor(target) else target.copy()
                    for label_value, train_id in mapping.items():
                        target[source == label_value] = train_id
                    return target

                def openmind_generate(self):
                    batch = original_generate(self)
                    labels = list(self.annotated_classes_key[1:])
                    mapping = {int(label): train_id for train_id, label in enumerate(labels) if int(label) != train_id}
                    if mapping:
                        batch["target"] = remap_target(batch["target"], mapping)
                    return batch

                nnUNetDataLoader.get_bbox = openmind_get_bbox
                nnUNetDataLoader.generate_train_batch = openmind_generate
                nnUNetDataLoader._openmind_compat_installed = True
            print("OpenMind coordinate and sparse-label compatibility hooks installed")
            """
        ),
        md("## 3. Create cluster plan copies pointing to the cluster checkpoint"),
        code(
            """
            for dataset_id, folder in dataset_dirs.items():
                if dataset_id not in DATASET_IDS:
                    continue
                source = folder / f"{SOURCE_PLAN}.json"
                assert source.is_file(), source
                plan = json.loads(source.read_text())
                plan["plans_name"] = CLUSTER_PLAN
                plan.setdefault("pretrain_info", {})["checkpoint_path"] = str(CHECKPOINT)
                target = folder / f"{CLUSTER_PLAN}.json"
                target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
                print(target)
            """
        ),
        md(
            """
            ## 4. Run one epoch per dataset

            `RUN_TRAINING` defaults to `False` so the path audit can be executed safely on a login/Jupyter node. Change it to
            `True` inside a GPU allocation. Checkpoint writing is disabled because this is a numerical compatibility test.
            """
        ),
        code(
            """
            from nnunetv2.run.run_training import get_trainer_from_args

            RUN_TRAINING = False
            if RUN_TRAINING:
                assert torch.cuda.is_available(), "Run this cell in a GPU allocation"
                torch.manual_seed(12345)
                print("GPU:", torch.cuda.get_device_name())
                summaries = []
                for dataset_id in DATASET_IDS:
                    started = time.perf_counter()
                    summary = {"id": dataset_id, "dataset": dataset_dirs[dataset_id].name, "status": "failed"}
                    trainer = None
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        trainer = get_trainer_from_args(
                            str(dataset_id), "3d_fullres", 0, TRAINER_NAME, CLUSTER_PLAN,
                            device=torch.device("cuda"), pretrained_from_scratch=False,
                            overwrite_ckpt_path=None,
                        )
                        trainer.num_epochs = 1
                        trainer.num_iterations_per_epoch = 250
                        trainer.num_val_iterations_per_epoch = 50
                        trainer.batch_size = BATCH_SIZE
                        trainer.configuration_manager.configuration["batch_size"] = BATCH_SIZE
                        trainer.disable_checkpointing = True
                        trainer.run_training()
                        logs = trainer.logger.my_fantastic_logging
                        summary.update(
                            status="completed", train_loss=float(logs["train_losses"][-1]),
                            val_loss=float(logs["val_losses"][-1]),
                            pseudo_dice=float(logs.get("mean_fg_dice", [np.nan])[-1]),
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                        )
                    except Exception as error:
                        summary.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                    finally:
                        summary["elapsed_seconds"] = time.perf_counter() - started
                        summaries.append(summary)
                        (RESULTS_ROOT / "one_epoch_summary.json").write_text(json.dumps(summaries, indent=2))
                        print(json.dumps(summary, indent=2))
                        del trainer
                        gc.collect()
                        torch.cuda.empty_cache()
                results_table = pd.DataFrame(summaries)
                display(results_table)
            else:
                print("Audit complete. Set RUN_TRAINING=True in this cell on a GPU node.")
            """
        ),
        md("## 5. Load existing results and visualize loss / pseudo Dice"),
        code(
            """
            summary_path = RESULTS_ROOT / "one_epoch_summary.json"
            assert summary_path.is_file(), f"Run the training cell first: {summary_path}"
            results_table = pd.DataFrame(json.loads(summary_path.read_text()))
            display(results_table)
            completed = results_table[results_table.status == "completed"].copy()
            assert len(completed) == len(DATASET_IDS), results_table[["dataset", "status", "error"]]
            assert np.isfinite(completed[["train_loss", "val_loss"]].to_numpy()).all()

            figure, axes = plt.subplots(1, 2, figsize=(15, 5))
            x = np.arange(len(completed))
            axes[0].bar(x - 0.2, completed.train_loss, width=0.4, label="train")
            axes[0].bar(x + 0.2, completed.val_loss, width=0.4, label="validation")
            axes[0].set_xticks(x, completed.dataset, rotation=65, ha="right")
            axes[0].set_ylabel("loss")
            axes[0].legend()
            axes[1].bar(x, completed.pseudo_dice)
            axes[1].set_xticks(x, completed.dataset, rotation=65, ha="right")
            axes[1].set_ylabel("patch-level pseudo Dice")
            for axis in axes:
                axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            output = RESULTS_ROOT / "one_epoch_summary.png"
            figure.savefig(output, dpi=180, bbox_inches="tight")
            plt.show()
            print(output)
            """
        ),
    ],
)

print(f"Generated notebooks in {OUT}")
