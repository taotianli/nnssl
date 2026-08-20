import torch

from nnssl.architectures.primus_jepa_second_wave import (
    PrimusGatedVisibleSkipMAEJEPA,
    PrimusIndependentCrossViewMAEJEPA,
    PrimusSerialLatentMAEJEPA,
)
from nnssl.training.loss.structure_aware_reconstruction import masked_smoothed_gradient_l1
from nnssl.training.nnsslTrainer.jepa.PrimusJEPASecondWaveTrainer import (
    PrimusJEPAIndependentCrossView_200ep_BS4_Weak,
    PrimusJEPAGatedSkip_200ep_BS8_G0p25,
    PrimusJEPAGradGuard_200ep_BS4_R0p25,
    PrimusJEPASerialCAE_200ep_BS8_L0p005,
    PrimusJEPAStructure_200ep_BS8_E0p05,
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


def test_serial_reconstruction_backpropagates_to_predictor():
    model = PrimusSerialLatentMAEJEPA(**_kwargs()).train()
    output = model(torch.randn(1, 1, 16, 16, 16))
    output["reconstruction"].square().mean().backward()
    assert next(model.predictor.parameters()).grad is not None
    assert output["prediction"].shape == output["target"].shape == (1, 32, 48)


def test_gated_visible_skip_starts_as_exactly_zero_and_learns():
    model = PrimusGatedVisibleSkipMAEJEPA(**_kwargs(), visible_skip_max=0.25).train()
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert float(output["decoder_skip_gate"]) == 0.0
    output["reconstruction"].square().mean().backward()
    assert model.decoder_skip_gate.grad is not None


def test_public_mae_loading_keeps_extension_initialized_and_syncs_teacher():
    source = PrimusGatedVisibleSkipMAEJEPA(**_kwargs(), visible_skip_max=0.25)
    target = PrimusGatedVisibleSkipMAEJEPA(**_kwargs(), visible_skip_max=0.25)
    source_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith(("predictor.", "target_", "decoder_skip_"))
    }
    target.load_mae_state_dict(source_state)
    assert float(target.decoder_skip_gate) == 0.0
    for online, teacher in zip(target.eva.parameters(), target.target_eva.parameters()):
        assert torch.equal(online, teacher)
        assert not teacher.requires_grad


def test_cross_view_intensity_is_aligned_and_can_be_disabled():
    torch.manual_seed(7)
    model = PrimusIndependentCrossViewMAEJEPA(**_kwargs()).train()
    image = torch.randn(2, 1, 16, 16, 16)
    assert torch.equal(model.make_target_view(image, augment=False), image)
    changed = model.make_target_view(image, augment=True)
    assert changed.shape == image.shape
    assert not torch.equal(changed, image)
    output = model(image)
    assert output["keep_indices"].shape == output["jepa_keep_indices"].shape
    for kept, masked in zip(output["jepa_keep_indices"], output["masked_indices"]):
        assert set(kept.tolist()).isdisjoint(masked.tolist())
        assert len(set(kept.tolist()) | set(masked.tolist())) == 64


def test_masked_gradient_loss_ignores_visible_only_errors_and_is_finite():
    target = torch.zeros(1, 1, 8, 8, 8)
    reconstruction = target.clone()
    reconstruction[:, :, :2] = 4.0
    visible = torch.ones_like(target)
    visible[:, :, 4:] = 0.0
    ignored = masked_smoothed_gradient_l1(reconstruction, target, visible, smoothing_kernel=1)
    assert float(ignored) == 0.0
    reconstruction[:, :, 6:] = 1.0
    penalized = masked_smoothed_gradient_l1(reconstruction, target, visible, smoothing_kernel=3)
    assert torch.isfinite(penalized) and float(penalized) > 0


def test_gradient_statistics_report_conflict_and_norms():
    first = (torch.tensor([1.0, 0.0]),)
    conflicting = (torch.tensor([-1.0, 1.0]),)
    dot, first_norm, second_norm = PrimusJEPAGradGuard_200ep_BS4_R0p25._gradient_statistics(
        first, conflicting
    )
    assert float(dot) < 0
    assert float(first_norm) == 1.0
    assert torch.isclose(second_norm, torch.sqrt(torch.tensor(2.0)))


def test_second_wave_exposes_five_distinct_trainer_names():
    trainers = {
        PrimusJEPASerialCAE_200ep_BS8_L0p005,
        PrimusJEPAGatedSkip_200ep_BS8_G0p25,
        PrimusJEPAIndependentCrossView_200ep_BS4_Weak,
        PrimusJEPAStructure_200ep_BS8_E0p05,
        PrimusJEPAGradGuard_200ep_BS4_R0p25,
    }
    assert len(trainers) == 5
