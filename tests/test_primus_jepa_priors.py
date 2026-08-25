import torch

from nnssl.architectures.primus_jepa_priors import (
    PrimusAcquisitionPriorMAEJEPA,
    PrimusAnatomyPriorMAEJEPA,
    PrimusSoftRegionPriorMAEJEPA,
)
from nnssl.training.loss.mri_prior_losses import (
    anatomy_prior_loss,
    soft_region_prior_loss,
    spatial_heat_kernel_loss,
    spectral_shell_loss,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPAPriorTrainer import (
    PrimusJEPAPriorAcquisitionMixed_200ep_BS4,
    PrimusJEPAPriorAcquisitionWeak_200ep_BS4,
    PrimusJEPAPriorAnatomyOcc_200ep_BS4_W0p02,
    PrimusJEPAPriorAnatomySDF_200ep_BS4_W0p02,
    PrimusJEPAPriorCoarseRegion8_200ep_BS4_W0p02,
    PrimusJEPAPriorHeat26_200ep_BS4_W0p001,
    PrimusJEPAPriorHeat6_200ep_BS4_W0p001,
    PrimusJEPAPriorSpectral16_200ep_BS4_W0p01,
    PrimusJEPAPriorSpectral8_200ep_BS4_W0p01,
    PrimusJEPAPriorTissue3_200ep_BS4_W0p02,
)


def _kwargs():
    return dict(
        input_channels=1,
        embed_dim=48,
        patch_embed_size=(4, 4, 4),
        output_channels=1,
        input_shape=(16, 16, 16),
        encoder_eva_depth=2,
        encoder_eva_numheads=4,
        decoder_eva_depth=1,
        decoder_eva_numheads=4,
        patch_drop_rate=0.5,
        use_rot_pos_emb=True,
        predictor_dim=24,
        predictor_depth=1,
        predictor_num_heads=4,
        predictor_query_chunk_size=8,
    )


def test_prior_architectures_keep_transferable_output_contract():
    x = torch.randn(1, 1, 16, 16, 16)
    anatomy = PrimusAnatomyPriorMAEJEPA(**_kwargs()).train()(x)
    tissue = PrimusSoftRegionPriorMAEJEPA(**_kwargs(), prior_classes=3).train()(x)
    acquisition = PrimusAcquisitionPriorMAEJEPA(
        **_kwargs(), corruption_strength=0.05
    ).train()(x)
    for output in (anatomy, tissue, acquisition):
        assert output["reconstruction"].shape == x.shape
        assert output["prediction"].shape == output["target"].shape
        assert output["masked_indices"].shape[:2] == output["prediction"].shape[:2]
    assert anatomy["anatomy_prior_prediction"].shape[-1] == 2
    assert tissue["soft_region_logits"].shape[-1] == 3
    assert acquisition["corruption_parameters"].shape == (1, 4)


def test_acquisition_randomness_does_not_advance_patch_mask_rng():
    plain = PrimusAnatomyPriorMAEJEPA(**_kwargs()).train()
    acquisition = PrimusAcquisitionPriorMAEJEPA(**_kwargs()).train()
    acquisition.load_mae_state_dict(plain.state_dict())
    x = torch.randn(1, 1, 16, 16, 16)
    torch.manual_seed(17)
    plain_indices = plain(x)["keep_indices"]
    torch.manual_seed(17)
    acquisition_indices = acquisition(x)["keep_indices"]
    torch.testing.assert_close(acquisition_indices, plain_indices)


def test_anatomy_and_soft_region_losses_are_finite_and_backpropagate():
    data = torch.randn(2, 1, 8, 8, 8)
    anatomy = torch.zeros_like(data)
    anatomy[:, :, 2:6, 2:6, 2:6] = 1
    indices = torch.tensor([[0, 1, 6, 7], [0, 1, 6, 7]])
    anatomy_prediction = torch.randn(2, 4, 2, requires_grad=True)
    anatomy_loss, anatomy_parts = anatomy_prior_loss(
        anatomy_prediction,
        indices,
        data,
        anatomy,
        torch.ones(2),
        (4, 4, 4),
        sdf_weight=0.5,
    )
    logits = torch.randn(2, 4, 3, requires_grad=True)
    region_loss, region_parts = soft_region_prior_loss(
        logits,
        indices,
        data,
        anatomy,
        torch.ones(2),
        (4, 4, 4),
        num_classes=3,
    )
    (anatomy_loss + region_loss).backward()
    assert torch.isfinite(anatomy_loss) and torch.isfinite(region_loss)
    assert anatomy_prediction.grad is not None and logits.grad is not None
    assert set(anatomy_parts) == {"occupancy", "signed_distance", "mask_coverage"}
    assert "region_entropy" in region_parts


def test_heat_kernel_and_spectral_losses_are_finite():
    data = torch.randn(1, 1, 8, 8, 8)
    indices = torch.tensor([[0, 1, 2, 3]])
    embedding = torch.randn(1, 4, 8, requires_grad=True)
    heat, parts = spatial_heat_kernel_loss(
        embedding, indices, data, (4, 4, 4), neighborhood=26
    )
    reconstruction = torch.randn_like(data, requires_grad=True)
    visible = torch.zeros_like(data)
    visible[..., :4] = 1
    spectral, spectral_parts = spectral_shell_loss(
        reconstruction, data, visible, num_shells=4
    )
    (heat + spectral).backward()
    assert torch.isfinite(heat) and torch.isfinite(spectral)
    assert embedding.grad is not None and reconstruction.grad is not None
    assert parts["heat_pairs"] > 0
    assert set(spectral_parts) == {"spectral_low", "spectral_high"}


def test_ten_trainers_are_matched_vicreg_baseline_extensions():
    trainers = (
        PrimusJEPAPriorAnatomyOcc_200ep_BS4_W0p02,
        PrimusJEPAPriorAnatomySDF_200ep_BS4_W0p02,
        PrimusJEPAPriorHeat6_200ep_BS4_W0p001,
        PrimusJEPAPriorHeat26_200ep_BS4_W0p001,
        PrimusJEPAPriorSpectral8_200ep_BS4_W0p01,
        PrimusJEPAPriorSpectral16_200ep_BS4_W0p01,
        PrimusJEPAPriorAcquisitionWeak_200ep_BS4,
        PrimusJEPAPriorAcquisitionMixed_200ep_BS4,
        PrimusJEPAPriorTissue3_200ep_BS4_W0p02,
        PrimusJEPAPriorCoarseRegion8_200ep_BS4_W0p02,
    )
    assert len({trainer.__name__ for trainer in trainers}) == 10
    for trainer in trainers:
        assert trainer.regularizer_kind == "vicreg"
        assert trainer.extra_loss_weight == 0.01
    assert PrimusJEPAPriorHeat6_200ep_BS4_W0p001.heat_neighborhood == 6
    assert PrimusJEPAPriorHeat26_200ep_BS4_W0p001.heat_neighborhood == 26
    assert PrimusJEPAPriorSpectral8_200ep_BS4_W0p01.spectral_shells == 8
    assert PrimusJEPAPriorSpectral16_200ep_BS4_W0p01.spectral_shells == 16
