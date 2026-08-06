import torch

from nnssl.architectures.primus_jepa import CrossAttention, PrimusMAEJEPA, complement_indices
from nnssl.training.loss.jepa_loss import JEPALatentLoss


def _small_model() -> PrimusMAEJEPA:
    return PrimusMAEJEPA(
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
        predictor_depth=2,
        predictor_num_heads=4,
        predictor_query_chunk_size=7,
    )


def test_complement_indices_preserves_patch_identity():
    keep = torch.tensor([[3, 0, 6], [1, 4, 2]])
    masked = complement_indices(keep, 7)
    assert torch.equal(masked, torch.tensor([[1, 2, 4, 5], [0, 3, 5, 6]]))


def test_shared_mask_forward_and_backward():
    torch.manual_seed(0)
    model = _small_model().train()
    output = model(torch.randn(2, 1, 16, 16, 16))
    assert output["reconstruction"].shape == (2, 1, 16, 16, 16)
    assert output["keep_indices"].shape == (2, 32)
    assert output["masked_indices"].shape == (2, 32)
    assert output["prediction"].shape == output["target"].shape == (2, 32, 48)
    loss = JEPALatentLoss()(output["prediction"], output["target"])
    loss.backward()
    assert model.predictor.output_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_eva.parameters())


def test_mae_loading_synchronizes_target_encoder():
    torch.manual_seed(1)
    source = _small_model()
    target = _small_model()
    mae_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith(("predictor.", "target_"))
    }
    loaded = target.load_mae_state_dict(mae_state)
    assert "eva.pos_embed" in loaded
    for online, ema in zip(target.eva.parameters(), target.target_eva.parameters()):
        assert torch.equal(online, ema)


def test_cross_attention_query_chunking_is_exact():
    torch.manual_seed(2)
    chunked = CrossAttention(dim=24, num_heads=4, query_chunk_size=3).eval()
    unchunked = CrossAttention(dim=24, num_heads=4, query_chunk_size=0).eval()
    unchunked.load_state_dict(chunked.state_dict())
    query = torch.randn(2, 11, 24)
    context = torch.randn(2, 7, 24)
    assert torch.allclose(chunked(query, context), unchunked(query, context), atol=1e-6, rtol=1e-5)
