import torch

from nnssl.architectures.primus_jepa_sclb import (
    PrimusStructureConditionedMAEJEPA,
    RegionFiLMBridge,
    compact_region_ids,
    region_mean,
    shuffled_region_control,
)
from nnssl.training.nnsslTrainer.jepa.PrimusJEPASCLBTrainer import (
    PrimusJEPASCLBBidirectional_200ep_BS4,
    PrimusJEPASCLBDetached_200ep_BS4,
    PrimusJEPASCLBParallel_200ep_BS4,
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
        region_grid=(2, 2, 2),
    )


def test_compact_regions_follow_primus_flattening_and_are_balanced():
    labels = compact_region_ids((20, 20, 20), (4, 4, 4), torch.device("cpu"))
    assert labels.shape == (8000,)
    assert labels.unique().numel() == 64
    torch.testing.assert_close(torch.bincount(labels), torch.full((64,), 125))


def test_shuffled_control_preserves_region_sizes_but_changes_layout():
    labels = compact_region_ids((20, 20, 20), (4, 4, 4), torch.device("cpu"))
    shuffled = shuffled_region_control(labels)
    assert not torch.equal(labels, shuffled)
    torch.testing.assert_close(torch.bincount(labels), torch.bincount(shuffled))
    assert torch.equal(shuffled, shuffled_region_control(labels))
    compact_neighbours = (labels[1:] == labels[:-1]).float().mean()
    shuffled_neighbours = (shuffled[1:] == shuffled[:-1]).float().mean()
    assert shuffled_neighbours < compact_neighbours * 0.25


def test_region_mean_matches_manual_pooling():
    tokens = torch.tensor([[[1.0], [3.0], [2.0], [6.0]]])
    labels = torch.tensor([[0, 0, 1, 1]])
    pooled, counts = region_mean(tokens, labels, 2)
    torch.testing.assert_close(pooled, torch.tensor([[[2.0], [4.0]]]))
    torch.testing.assert_close(counts, torch.tensor([[2.0, 2.0]]))


def test_film_bridge_is_exact_identity_at_initialization_and_learns():
    bridge = RegionFiLMBridge(8, max_scale=0.25)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    condition = torch.randn_like(tokens)
    active = torch.ones(2, 5, 1)
    output = bridge(tokens, condition, active)
    torch.testing.assert_close(output, tokens)
    output.sum().backward()
    assert bridge.to_gamma_beta.weight.grad is not None
    assert bridge.to_gamma_beta.weight.grad.abs().sum() > 0


def test_detached_condition_blocks_condition_gradient():
    bridge = RegionFiLMBridge(4)
    with torch.no_grad():
        bridge.strength.fill_(1.0)
        bridge.to_gamma_beta.weight.normal_(std=0.1)
    tokens = torch.randn(1, 3, 4, requires_grad=True)
    condition = torch.randn(1, 3, 4, requires_grad=True)
    bridge(tokens, condition.detach(), torch.ones(1, 3, 1)).sum().backward()
    assert condition.grad is None


def test_sclb_tiny_forward_has_region_targets_and_standard_reconstruction():
    model = PrimusStructureConditionedMAEJEPA(**_kwargs()).train()
    output = model(torch.randn(2, 1, 16, 16, 16))
    assert output["reconstruction"].shape == (2, 1, 16, 16, 16)
    assert output["prediction"].shape == output["target"].shape == (2, 8, 48)
    assert output["regularizer_embedding"].shape == (2, 32, 128)


def test_detached_and_bidirectional_bridge_have_different_predictor_gradients():
    gradients = []
    for detach in (True, False):
        model = PrimusStructureConditionedMAEJEPA(**_kwargs(), detach_bridge=detach).train()
        with torch.no_grad():
            model.region_bridge.to_gamma_beta.weight.normal_(std=0.02)
        output = model(torch.randn(1, 1, 16, 16, 16))
        output["reconstruction"].square().mean().backward()
        gradient = model.predictor.output_projection.weight.grad
        gradients.append(0.0 if gradient is None else float(gradient.abs().sum()))
    assert gradients[0] == 0.0
    assert gradients[1] > 0.0


def test_sclb_trainer_controls_share_architecture_but_change_bridge_contract():
    assert PrimusJEPASCLBParallel_200ep_BS4.architecture_kwargs["bridge_enabled"] is False
    assert PrimusJEPASCLBDetached_200ep_BS4.architecture_kwargs["detach_bridge"] is True
    assert PrimusJEPASCLBBidirectional_200ep_BS4.architecture_kwargs["detach_bridge"] is False
    assert PrimusJEPASCLBDetached_200ep_BS4.regularizer_kind == "vicreg"
