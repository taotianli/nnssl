"""Primus-M MAE-only continued-pretraining controls.

These trainers intentionally differ only in their peak learning rate.  They
use the same Primus-M architecture, MAE reconstruction objective, 160^3 patch,
batch size, epoch budget, optimizer, schedule, and augmentations as the
corresponding MAE + JEPA experiments.
"""

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseEvaMAETrainer import (
    BaseEvaMAETrainer_BS8,
)


class PrimusMAEOnlyTrainer_200ep_BS8_LR3e4(BaseEvaMAETrainer_BS8):
    """Matched MAE-only control using the JEPA trainer's 3e-4 learning rate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200
        self.initial_lr = 3e-4


class PrimusMAEOnlyTrainer_200ep_BS8_LR3e5(BaseEvaMAETrainer_BS8):
    """Low-learning-rate MAE-only control for continued pretraining."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200
        self.initial_lr = 3e-5
