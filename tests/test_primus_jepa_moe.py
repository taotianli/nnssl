import torch
from torch import nn

from nnssl.architectures.primus_jepa_moe import (
    ResidualMoEEva,
    SparseResidualMoEAdapter,
    routing_consistency_loss,
)


class _Block(nn.Module):
    def forward(self, x, rope=None):
        return x + 1.0


class _Eva(nn.Module):
    def __init__(self, dim=12, depth=3):
        super().__init__()
        self.embed_dim = dim
        self.blocks = nn.ModuleList([_Block() for _ in range(depth)])
        self.norm = nn.Identity()
        self.grad_checkpointing = False

    def _pos_embed(self, x):
        return x, None, None


def test_sparse_residual_moe_is_identity_at_initialization():
    adapter = SparseResidualMoEAdapter(12, num_experts=4, top_k=2, expert_hidden_dim=8)
    x = torch.randn(2, 7, 12)
    output = adapter(x)
    torch.testing.assert_close(output, torch.zeros_like(output))
    assert adapter.last_probabilities.shape == (2, 7, 4)
    assert int(adapter.counts.sum()) == 2 * 7 * 2


def test_residual_moe_eva_preserves_base_function_at_initialization():
    wrapped = ResidualMoEEva(
        _Eva(), layer_indices=(1,), num_experts=4, top_k=2, expert_hidden_dim=8
    )
    x = torch.randn(2, 5, 12)
    output, keep = wrapped(x)
    torch.testing.assert_close(output, x + 3.0)
    assert keep is None
    assert len(wrapped.last_router_probabilities) == 1


def test_routing_consistency_matches_tokens_by_keep_index():
    target = torch.tensor(
        [[[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]]], dtype=torch.float32
    )
    keep = torch.tensor([[3, 1]])
    context = torch.gather(target, 1, keep[..., None].expand(-1, -1, 2))
    same = routing_consistency_loss([context], [target], keep)
    different = routing_consistency_loss([context.flip(-1)], [target], keep)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-7, rtol=0)
    assert different > same


def test_auxiliary_free_bias_update_favours_underused_experts():
    adapter = SparseResidualMoEAdapter(12, num_experts=4, top_k=2, expert_hidden_dim=8)
    adapter.counts.copy_(torch.tensor([100, 50, 10, 0]))
    minimum, maximum = adapter.update_routing_bias(rate=1e-2)
    assert minimum == 0.0
    assert maximum > 1.0
    assert adapter.routing_bias[3] > adapter.routing_bias[0]
    assert int(adapter.counts.sum()) == 0
