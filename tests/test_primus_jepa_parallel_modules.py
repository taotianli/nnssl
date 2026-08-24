import torch

from nnssl.architectures.primus_jepa_parallel_modules import (
    PrimusDualObjectiveAdapterMAEJEPA,
    PrimusEquivariantLocalMAEJEPA,
    PrimusHierarchicalObjectiveMAEJEPA,
    ZeroResidualProjection,
    align_flipped_tokens,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPAParallelModulesTrainer import (
    PrimusJEPADOAR32_200ep_BS4,
    PrimusJEPAELCVIC0p01_200ep_BS4,
    PrimusJEPAHOPB8_200ep_BS4,
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


def test_zero_residual_projection_preserves_input_and_receives_gradient():
    module = ZeroResidualProjection(12, 4)
    x = torch.randn(2, 5, 12, requires_grad=True)
    residual = module(x)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    residual.sum().backward()
    assert module.up.weight.grad is not None and module.up.weight.grad.abs().sum() > 0


def test_hop_uses_intermediate_tokens_but_keeps_final_jepa_shape():
    model = PrimusHierarchicalObjectiveMAEJEPA(
        **_kwargs(), reconstruction_block=1
    ).train()
    output = model(torch.randn(2, 1, 16, 16, 16))
    assert output["reconstruction"].shape == (2, 1, 16, 16, 16)
    assert output["prediction"].shape == output["target"].shape == (2, 32, 48)


def test_hop_reconstruction_gradient_stops_before_late_encoder_blocks():
    model = PrimusHierarchicalObjectiveMAEJEPA(
        **_kwargs(), reconstruction_block=1
    ).train()
    output = model(torch.randn(1, 1, 16, 16, 16))
    output["reconstruction"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.eva.blocks[0].parameters())
    assert all(parameter.grad is None for parameter in model.eva.blocks[1].parameters())


def test_flipped_token_alignment_is_exact():
    tokens = torch.arange(2 * 3 * 4).reshape(1, 24, 1).float()
    flipped = align_flipped_tokens(tokens, (3, 2, 4), input_axis=4)
    restored = align_flipped_tokens(flipped, (3, 2, 4), input_axis=4)
    torch.testing.assert_close(restored, tokens)


def test_elc_outputs_aligned_local_pairs():
    model = PrimusEquivariantLocalMAEJEPA(
        **_kwargs(), local_projection_dim=12, flip_axis=4
    ).train()
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert output["local_embedding"].shape == (1, 64, 12)
    assert output["local_embedding_aligned"].shape == (1, 64, 12)


def test_doa_is_public_mae_equivalent_at_initialization_and_backpropagates():
    model = PrimusDualObjectiveAdapterMAEJEPA(**_kwargs(), adapter_rank=8).train()
    output = model(torch.randn(1, 1, 16, 16, 16))
    loss = output["reconstruction"].square().mean() + output["prediction"].abs().mean()
    loss.backward()
    assert model.mae_adapter.up.weight.grad is not None
    assert model.jepa_adapter.up.weight.grad is not None


def test_public_trainer_names_are_isolated_and_matched():
    assert PrimusJEPAHOPB8_200ep_BS4.architecture_kwargs["reconstruction_block"] == 8
    assert PrimusJEPAELCVIC0p01_200ep_BS4.local_consistency_weight == 0.01
    assert PrimusJEPADOAR32_200ep_BS4.architecture_kwargs["adapter_rank"] == 32
    for trainer in (
        PrimusJEPAHOPB8_200ep_BS4,
        PrimusJEPAELCVIC0p01_200ep_BS4,
        PrimusJEPADOAR32_200ep_BS4,
    ):
        assert trainer.regularizer_kind == "vicreg"
        assert trainer.extra_loss_weight == 0.005
