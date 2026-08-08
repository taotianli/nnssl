import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "openmind_downstream_benchmark.py"
SPEC = importlib.util.spec_from_file_location("openmind_downstream_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class BenchmarkHelpersTest(unittest.TestCase):
    def test_resolve_dataset_accepts_id_name_and_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset203_ISLES22"
            dataset.mkdir()
            self.assertEqual(MODULE.resolve_dataset(root, "203"), dataset)
            self.assertEqual(MODULE.resolve_dataset(root, "Dataset203_ISLES22"), dataset)
            self.assertEqual(MODULE.resolve_dataset(root, "isles22"), dataset)

    def test_resolve_checkpoint_from_weights_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "PrimusM-OpenMind-MAE" / "checkpoint_final.pth"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            resolved, model = MODULE.resolve_checkpoint("PrimusM-OpenMind-MAE", None, root)
            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(model, "PrimusM-OpenMind-MAE")

    def test_prepare_plan_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset203_ISLES22"
            source = {
                "plans_name": "PMPrep",
                "pretrain_info": {"checkpoint_path": "old", "key_to_encoder": "eva"},
                "configurations": {"3d_fullres": {"batch_size": 1}},
            }
            write_json(dataset / "PMPrep.json", source)
            checkpoint = root / "checkpoint_final.pth"
            checkpoint.touch()
            target_path, target = MODULE.prepare_plan(
                dataset, "PMPrep", "OMBench_MAE", checkpoint, "PrimusM-OpenMind-MAE", "3d_fullres", 2
            )
            self.assertEqual(json.loads((dataset / "PMPrep.json").read_text()), source)
            self.assertEqual(target_path.name, "OMBench_MAE.json")
            self.assertEqual(target["plans_name"], "OMBench_MAE")
            self.assertEqual(target["pretrain_info"]["checkpoint_path"], str(checkpoint))
            self.assertEqual(target["configurations"]["3d_fullres"]["batch_size"], 2)

    def test_validate_inputs_rejects_unavailable_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "Dataset203_ISLES22"
            (dataset / "gt_segmentations").mkdir(parents=True)
            (dataset / "data").mkdir()
            write_json(dataset / "dataset.json", {})
            write_json(dataset / "dataset_fingerprint.json", {})
            write_json(dataset / "splits_final.json", [{"train": ["a"], "val": ["b"]}])
            plan = {
                "configurations": {
                    "3d_fullres": {
                        "data_identifier": "data",
                        "patch_size": [160, 160, 160],
                        "spacing": [1, 1, 1],
                        "normalization_schemes": ["ZScoreNormalization"],
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "contains 1 split"):
                MODULE.validate_inputs(dataset, plan, "3d_fullres", 1)

    def test_dry_run_prepares_a_model_specific_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adaptation_repo = root / "nnUNet-openmind"
            (adaptation_repo / "nnunetv2").mkdir(parents=True)
            weights = root / "weights"
            checkpoint = weights / "PrimusM-OpenMind-MAE" / "checkpoint_final.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            write_json(
                checkpoint.parent / "adaptation_plan.json",
                {"architecture_plans": {"arch_class_name": "PrimusM"}},
            )
            preprocessed = root / "preprocessed"
            dataset = preprocessed / "Dataset203_ISLES22"
            (dataset / "gt_segmentations").mkdir(parents=True)
            (dataset / "data").mkdir()
            write_json(dataset / "dataset.json", {})
            write_json(dataset / "dataset_fingerprint.json", {})
            write_json(dataset / "splits_final.json", [{"train": ["a"], "val": ["b"]}])
            write_json(
                dataset / "PMPrep.json",
                {
                    "plans_name": "PMPrep",
                    "pretrain_info": {"checkpoint_path": "old"},
                    "configurations": {
                        "3d_fullres": {
                            "batch_size": 1,
                            "data_identifier": "data",
                            "patch_size": [160, 160, 160],
                            "spacing": [1, 1, 1],
                            "normalization_schemes": ["ZScoreNormalization"],
                        }
                    },
                },
            )
            status = MODULE.main(
                [
                    "--model",
                    "PrimusM-OpenMind-MAE",
                    "--dataset",
                    "203",
                    "--nnunet-repo",
                    str(adaptation_repo),
                    "--preprocessed-root",
                    str(preprocessed),
                    "--weights-root",
                    str(weights),
                    "--results-root",
                    str(root / "results"),
                    "--dry-run",
                ]
            )
            self.assertEqual(status, 0)
            generated = dataset / "OMBench_PrimusM_OpenMind_MAE.json"
            self.assertTrue(generated.is_file())
            self.assertEqual(json.loads(generated.read_text())["plans_name"], generated.stem)

    def test_compatibility_accepts_three_column_class_locations(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("numpy and torch are required for the compatibility hook test")

        class FakeLoader:
            annotated_classes_key = (-1, 0, 2, 5)

            def get_bbox(self, data_shape, force_fg, class_locations, *args, **kwargs):
                return class_locations

            def generate_train_batch(self):
                return {"target": torch.tensor([[[[0, 2, 5]]]])}

        class FakeLabelManager:
            has_regions = False
            all_labels = [0, 2, 5]

            def convert_probabilities_to_segmentation(self, predicted_probabilities):
                return predicted_probabilities.argmax(0)

        loader_module = types.ModuleType("nnunetv2.training.dataloading.data_loader")
        loader_module.nnUNetDataLoader = FakeLoader
        label_module = types.ModuleType("nnunetv2.utilities.label_handling.label_handling")
        label_module.LabelManager = FakeLabelManager
        names = {
            "nnunetv2": types.ModuleType("nnunetv2"),
            "nnunetv2.training": types.ModuleType("nnunetv2.training"),
            "nnunetv2.training.dataloading": types.ModuleType("nnunetv2.training.dataloading"),
            "nnunetv2.training.dataloading.data_loader": loader_module,
            "nnunetv2.utilities": types.ModuleType("nnunetv2.utilities"),
            "nnunetv2.utilities.label_handling": types.ModuleType("nnunetv2.utilities.label_handling"),
            "nnunetv2.utilities.label_handling.label_handling": label_module,
        }
        previous = {name: sys.modules.get(name) for name in names}
        try:
            sys.modules.update(names)
            installed = MODULE.install_uploaded_downstream_compatibility()
            locations = FakeLoader().get_bbox(
                (32, 32, 32), True, {1: np.asarray([[4, 5, 6]])}
            )
            self.assertEqual(locations[1].shape, (1, 4))
            self.assertEqual(locations[1].tolist(), [[0, 4, 5, 6]])
            target = FakeLoader().generate_train_batch()["target"]
            self.assertEqual(target.flatten().tolist(), [0, 1, 2])
            probabilities = np.asarray(
                [[[[1, 0, 0]]], [[[0, 1, 0]]], [[[0, 0, 1]]]], dtype=np.float32
            )
            segmentation = FakeLabelManager().convert_probabilities_to_segmentation(probabilities)
            self.assertEqual(segmentation.flatten().tolist(), [0, 2, 5])
            self.assertIn("foreground coordinates (z,y,x) -> (class,z,y,x)", installed)
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
