import torch

from nnssl.architectures.primus_jepa_regularized import (
    select_valid_region_partition,
    shifted_region_ids,
)
from nnssl.training.loss.representation_regularization import (
    ORRRegLoss,
    RegionVISRegLoss,
    VISRegLoss,
    _region_mean_and_residual,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPARegularizedTrainer import (
    PrimusJEPANoEMA_200ep_BS4_NoReg,
    PrimusJEPANoEMA_200ep_BS4_ORR0p01_B1,
    PrimusJEPANoEMA_200ep_BS4_ORRShuffled0p01_B1,
    PrimusJEPANoEMA_200ep_BS4_RegionVIS0p01,
    PrimusJEPANoEMA_200ep_BS4_SIGReg0p01,
    PrimusJEPANoEMA_200ep_BS4_VICReg0p005,
    PrimusJEPANoEMA_200ep_BS4_VISReg0p01,
)


def _all_indices():
    return torch.arange(20**3).unsqueeze(0)


def test_shifted_regions_have_constant_cardinality_and_no_wrap():
    indices = _all_indices()[0]
    for offset in (-2, -1, 0, 1, 2):
        ids = shifted_region_ids(indices, (20, 20, 20), (offset, offset, offset))
        counts = torch.bincount(ids, minlength=64)
        assert counts.shape[0] == 64
        assert int(counts.min()) >= 27
        assert int(counts.sum()) == 20**3
    first = shifted_region_ids(torch.tensor([0]), (20, 20, 20), (2, 2, 2))
    last = shifted_region_ids(torch.tensor([20**3 - 1]), (20, 20, 20), (2, 2, 2))
    assert int(first) != int(last)


def test_partition_selects_all_active_regions():
    region_ids, offsets = select_valid_region_partition(_all_indices(), (20, 20, 20))
    assert region_ids.shape == (1, 20**3)
    assert offsets.shape == (1, 3)
    assert torch.bincount(region_ids[0], minlength=64).min() > 0


def test_region_projection_is_orthogonal_and_zero_mean():
    ids = shifted_region_ids(_all_indices()[0], (20, 20, 20), (0, 0, 0))
    z = torch.randn(1, 20**3, 8, requires_grad=True)
    means, residuals = _region_mean_and_residual(z, ids.unsqueeze(0))
    residual = residuals[0]
    coarse = means[ids]
    assert torch.allclose(coarse + residual, z[0], atol=1e-6)
    assert torch.allclose((coarse * residual).sum(), torch.zeros(()), atol=2e-4)
    sums = torch.zeros(64, residual.shape[-1]).index_add(0, ids, residual)
    assert torch.allclose(sums, torch.zeros_like(sums), atol=2e-4)


def test_vis_region_vis_and_orr_are_finite_and_differentiable():
    torch.manual_seed(3)
    ids = shifted_region_ids(_all_indices()[0], (20, 20, 20), (0, 0, 0))
    # Keep the test inexpensive while preserving all 64 active regions.
    chosen = torch.cat([(ids == region).nonzero()[:2, 0] for region in range(64)])
    active_ids = ids[chosen].unsqueeze(0).repeat(2, 1)
    z = torch.randn(2, 128, 16, requires_grad=True)
    loss = VISRegLoss(num_slices=8, max_samples=128)(z)
    loss = loss + RegionVISRegLoss(num_slices=8, max_samples=128)(z, active_ids)
    loss = loss + ORRRegLoss(num_slices=8, max_samples=128)(z, active_ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_regularization_sweep_exposes_distinct_trainer_names():
    trainers = {
        PrimusJEPANoEMA_200ep_BS4_NoReg,
        PrimusJEPANoEMA_200ep_BS4_VICReg0p005,
        PrimusJEPANoEMA_200ep_BS4_SIGReg0p01,
        PrimusJEPANoEMA_200ep_BS4_VISReg0p01,
        PrimusJEPANoEMA_200ep_BS4_RegionVIS0p01,
        PrimusJEPANoEMA_200ep_BS4_ORR0p01_B1,
        PrimusJEPANoEMA_200ep_BS4_ORRShuffled0p01_B1,
    }
    assert len(trainers) == 7
    assert PrimusJEPANoEMA_200ep_BS4_NoReg.extra_loss_weight == 0.0
    assert PrimusJEPANoEMA_200ep_BS4_ORR0p01_B1.regularizer_kind == "orr"
