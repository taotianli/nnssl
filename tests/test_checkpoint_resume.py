from pathlib import Path

from nnssl.run.run_training import maybe_load_checkpoint


class _FakeTrainer:
    def __init__(self, output_folder: Path):
        self.output_folder = str(output_folder)
        self.loaded = []

    def load_checkpoint(self, filename: str):
        self.loaded.append(Path(filename).name)
        if filename.endswith("checkpoint_latest.pth"):
            raise RuntimeError(
                "PytorchStreamReader failed reading zip archive: failed finding central directory"
            )


def test_continue_skips_truncated_latest_and_uses_valid_best(tmp_path):
    (tmp_path / "checkpoint_latest.pth").write_bytes(b"truncated")
    (tmp_path / "checkpoint_best.pth").write_bytes(b"valid")
    trainer = _FakeTrainer(tmp_path)
    maybe_load_checkpoint(trainer, True, False)
    assert trainer.loaded == ["checkpoint_latest.pth", "checkpoint_best.pth"]


def test_continue_does_not_hide_non_deserialization_runtime_errors(tmp_path):
    (tmp_path / "checkpoint_latest.pth").write_bytes(b"present")

    class _StateMismatch(_FakeTrainer):
        def load_checkpoint(self, filename: str):
            raise RuntimeError("checkpoint seed conflicts with requested experiment seed")

    trainer = _StateMismatch(tmp_path)
    try:
        maybe_load_checkpoint(trainer, True, False)
    except RuntimeError as error:
        assert "seed conflicts" in str(error)
    else:
        raise AssertionError("non-deserialization RuntimeError must not be swallowed")
