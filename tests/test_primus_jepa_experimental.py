import torch

from nnssl.architectures.primus_jepa_variants import (
    PrimusBlockTargetMAEJEPA,
    PrimusDualTeacherMAEJEPA,
    PrimusOnlineTargetMAEJEPA,
    select_contiguous_masked_tokens,
)
from nnssl.training.loss.representation_regularization import (
    SIGRegLoss,
    VICRegVarianceCovarianceLoss,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPAExperimentalTrainer import (
    PerGroupWarmupPolyScheduler,
    PrimusJEPAAdaptive_200ep_BS8_R0p05,
    PrimusJEPABlock_200ep_BS8_K512,
    PrimusJEPADualTeacher_200ep_BS8_A0p25_0p75,
    PrimusJEPANoEMA_200ep_BS4_SIGReg0p02,
    PrimusJEPANoEMA_200ep_BS4_VICReg0p01,
    PrimusJEPAStaged_200ep_BS8_L0p005,
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


def test_block_selector_only_returns_masked_tokens_and_is_compact():
    torch.manual_seed(4)
    masked = torch.tensor([[0, 1, 2, 4, 5, 6, 16, 17, 18, 21, 22, 26]])
    selected = select_contiguous_masked_tokens(masked, (3, 3, 3), 5)
    assert selected.shape == (1, 5)
    assert set(selected[0].tolist()).issubset(set(masked[0].tolist()))


def test_block_target_keeps_mae_mask_but_predicts_fewer_tokens():
    model = PrimusBlockTargetMAEJEPA(**_kwargs(), jepa_num_targets=7).train()
    output = model(torch.randn(2, 1, 16, 16, 16))
    assert output["masked_indices"].shape == (2, 32)
    assert output["jepa_target_indices"].shape == (2, 7)
    assert output["prediction"].shape == output["target"].shape == (2, 7, 48)


def test_dual_teacher_anchor_is_frozen_and_initialized_from_mae():
    source = PrimusDualTeacherMAEJEPA(**_kwargs())
    target = PrimusDualTeacherMAEJEPA(**_kwargs())
    state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith(("predictor.", "target_", "anchor_"))
    }
    target.load_mae_state_dict(state)
    for online, anchor in zip(target.eva.parameters(), target.anchor_eva.parameters()):
        assert torch.equal(online, anchor)
        assert not anchor.requires_grad
    output = target(torch.randn(1, 1, 16, 16, 16))
    assert 0.25 <= float(output["anchor_weight"]) <= 0.75


def test_online_target_has_no_ema_modules_and_backpropagates_into_encoder():
    model = PrimusOnlineTargetMAEJEPA(**_kwargs()).train()
    assert not hasattr(model, "target_eva")
    output = model(torch.randn(1, 1, 16, 16, 16))
    loss = (output["prediction"] - output["target"]).abs().mean()
    loss.backward()
    assert next(model.eva.parameters()).grad is not None
    assert output["regularizer_embedding"].shape[-1] == 128


def test_collapse_regularizers_are_finite_and_differentiable():
    z = torch.randn(4, 32, 16, requires_grad=True)
    loss = VICRegVarianceCovarianceLoss(max_samples=64)(z) + SIGRegLoss(
        num_slices=8, num_knots=5, max_samples=64
    )(z)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_discriminative_scheduler_preserves_lr_ratio():
    first = torch.nn.Parameter(torch.ones(()))
    second = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "peak_lr": 3e-5, "warmup_epochs": 50},
            {"params": [second], "peak_lr": 3e-4, "warmup_epochs": 10},
        ],
        lr=1.0,
    )
    scheduler = PerGroupWarmupPolyScheduler(optimizer, 200)
    scheduler.step(9)
    assert optimizer.param_groups[0]["lr"] == 6e-6
    assert optimizer.param_groups[1]["lr"] == 3e-4


def test_public_ablation_classes_are_independent_trainer_names():
    assert PrimusJEPAStaged_200ep_BS8_L0p005.jepa_loss_weight_default == 0.005
    assert PrimusJEPAAdaptive_200ep_BS8_R0p05.adaptive_contribution_ratio == 0.05
    assert PrimusJEPABlock_200ep_BS8_K512.architecture_class is PrimusBlockTargetMAEJEPA
    assert PrimusJEPADualTeacher_200ep_BS8_A0p25_0p75.architecture_class is PrimusDualTeacherMAEJEPA
    assert PrimusJEPANoEMA_200ep_BS4_VICReg0p01.regularizer_kind == "vicreg"
    assert PrimusJEPANoEMA_200ep_BS4_SIGReg0p02.regularizer_kind == "sigreg"
