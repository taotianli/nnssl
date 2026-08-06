"""Primus-M MAE with an EMA-target JEPA prediction branch.

The online EVA encoder is shared by the original MAE reconstruction branch and
the JEPA branch.  Both branches use exactly the same patch mask.  The predictor
uses masked positions as queries and visible tokens as keys/values, avoiding an
8,000-token self-attention operation for the default 160^3 Primus-M input.
"""

from __future__ import annotations

from copy import deepcopy
from math import prod
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nnssl.architectures.evaMAE_module import EvaMAE


def _gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, K]`` token indices from a ``[B, N, C]`` tensor."""
    return torch.gather(x, 1, indices[..., None].expand(-1, -1, x.shape[-1]))


def complement_indices(keep_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Return sorted token indices not present in ``keep_indices``."""
    batch_size = keep_indices.shape[0]
    is_masked = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=keep_indices.device)
    is_masked.scatter_(1, keep_indices.long(), False)
    all_indices = torch.arange(num_tokens, device=keep_indices.device).expand(batch_size, -1)
    return all_indices[is_masked].reshape(batch_size, num_tokens - keep_indices.shape[1])


def _sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError(f"A sin/cos axis dimension must be even, got {dim}")
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (dim / 2.0)))
    angles = positions.reshape(-1, 1).float() * omega.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def make_3d_sincos_position_embedding(grid_size: Sequence[int], embed_dim: int) -> torch.Tensor:
    """Create a fixed ``[1, D*H*W, C]`` 3-D sine/cosine embedding."""
    if len(grid_size) != 3:
        raise ValueError(f"Expected a 3-D grid, got {tuple(grid_size)}")
    axis_dims = [embed_dim // 3] * 3
    axis_dims[-1] = embed_dim - sum(axis_dims[:-1])
    for axis in range(3):
        if axis_dims[axis] % 2:
            donor = next((i for i in range(3) if i != axis and axis_dims[i] >= 2), None)
            if donor is None:
                raise ValueError(f"Cannot split embed_dim={embed_dim} into even 3-D axis dimensions")
            axis_dims[axis] += 1
            axis_dims[donor] -= 1

    coordinates = torch.meshgrid(
        *(torch.arange(int(size), dtype=torch.float32) for size in grid_size), indexing="ij"
    )
    embedding = torch.cat(
        [_sincos_1d(axis_dim, coordinate.reshape(-1)) for axis_dim, coordinate in zip(axis_dims, coordinates)],
        dim=1,
    )
    if embedding.shape != (prod(grid_size), embed_dim):
        raise RuntimeError(f"Unexpected positional embedding shape {tuple(embedding.shape)}")
    return embedding.unsqueeze(0)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with exact, memory-bounded query chunking."""

    def __init__(self, dim: int, num_heads: int, attention_dropout: float = 0.0, query_chunk_size: int = 512):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_dropout = float(attention_dropout)
        self.query_chunk_size = int(query_chunk_size)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query_heads = self._to_heads(self.q_proj(query))
        key, value = self.kv_proj(context).chunk(2, dim=-1)
        key_heads = self._to_heads(key)
        value_heads = self._to_heads(value)
        dropout = self.attention_dropout if self.training else 0.0
        chunk_size = self.query_chunk_size if self.query_chunk_size > 0 else query.shape[1]
        chunks = []
        for query_chunk in query_heads.split(chunk_size, dim=2):
            chunks.append(
                F.scaled_dot_product_attention(
                    query_chunk,
                    key_heads,
                    value_heads,
                    dropout_p=dropout,
                )
            )
        attended = torch.cat(chunks, dim=2).transpose(1, 2).contiguous()
        attended = attended.reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(attended)


class CrossAttentionPredictorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        query_chunk_size: int = 512,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = CrossAttention(dim, num_heads, attention_dropout, query_chunk_size)
        self.attention_drop = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = query + self.attention_drop(
            self.cross_attention(self.query_norm(query), self.context_norm(context))
        )
        return query + self.mlp(self.mlp_norm(query))


class PrimusJEPAPredictor(nn.Module):
    """Predict all masked Primus latent tokens from visible context tokens."""

    def __init__(
        self,
        grid_size: Sequence[int] = (20, 20, 20),
        encoder_dim: int = 864,
        predictor_dim: int = 384,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        query_chunk_size: int = 512,
    ):
        super().__init__()
        self.grid_size = tuple(int(i) for i in grid_size)
        self.num_tokens = prod(self.grid_size)
        self.input_projection = nn.Linear(encoder_dim, predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.register_buffer(
            "position_embedding",
            make_3d_sincos_position_embedding(self.grid_size, predictor_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                CrossAttentionPredictorBlock(
                    predictor_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    attention_dropout,
                    query_chunk_size,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        keep_indices: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        position = self.position_embedding.to(device=context_tokens.device, dtype=context_tokens.dtype)
        position = position.expand(context_tokens.shape[0], -1, -1)
        context = self.input_projection(context_tokens) + _gather_tokens(position, keep_indices)
        query = self.mask_token.to(dtype=context.dtype).expand(context.shape[0], masked_indices.shape[1], -1)
        query = query + _gather_tokens(position, masked_indices)
        for block in self.blocks:
            query = block(query, context)
        return self.output_projection(self.norm(query))


class PrimusMAEJEPA(EvaMAE):
    """State-dict-compatible Primus MAE extended with a JEPA branch."""

    def __init__(
        self,
        *args,
        predictor_dim: int = 384,
        predictor_depth: int = 4,
        predictor_num_heads: int = 12,
        predictor_mlp_ratio: float = 4.0,
        predictor_query_chunk_size: int = 512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.use_decoder:
            raise ValueError("PrimusMAEJEPA requires the MAE decoder")
        input_shape = tuple(int(i) for i in kwargs["input_shape"])
        grid_size = tuple(i // p for i, p in zip(input_shape, self.patch_embed_size))
        self.predictor = PrimusJEPAPredictor(
            grid_size=grid_size,
            encoder_dim=self.embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=predictor_mlp_ratio,
            query_chunk_size=predictor_query_chunk_size,
        )
        self.target_down_projection = deepcopy(self.down_projection)
        self.target_eva = deepcopy(self.eva)
        self._freeze_target_encoder()

    def _freeze_target_encoder(self) -> None:
        self.target_down_projection.requires_grad_(False)
        self.target_eva.requires_grad_(False)
        self.target_down_projection.eval()
        self.target_eva.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_down_projection.eval()
        self.target_eva.eval()
        return self

    def encode_online(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        x = self.down_projection(x)
        spatial_shape = tuple(int(i) for i in x.shape[2:])
        tokens = rearrange(x, "b c w h d -> b (h w d) c")
        encoded, keep_indices = self.eva(tokens)
        if keep_indices is None:
            keep_indices = torch.arange(tokens.shape[1], device=tokens.device).expand(tokens.shape[0], -1)
        return encoded, keep_indices.long(), spatial_shape

    def decode_mae(
        self,
        encoded: torch.Tensor,
        keep_indices: torch.Tensor,
        spatial_shape: Sequence[int],
    ) -> torch.Tensor:
        restored = self.restore_full_sequence(encoded, keep_indices, prod(spatial_shape))
        decoded, _ = self.decoder(restored)
        decoded = rearrange(
            decoded,
            "b (h w d) c -> b c w h d",
            h=spatial_shape[0],
            w=spatial_shape[1],
            d=spatial_shape[2],
        )
        return self.up_projection(decoded)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.target_down_projection(x)
        tokens = rearrange(projected, "b c w h d -> b (h w d) c")
        target, _ = self.target_eva(tokens)
        return F.layer_norm(target, (target.shape[-1],))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context, keep_indices, spatial_shape = self.encode_online(x)
        masked_indices = complement_indices(keep_indices, prod(spatial_shape))
        reconstruction = self.decode_mae(context, keep_indices, spatial_shape)
        with torch.no_grad():
            target_full = self.encode_target(x)
            target_masked = _gather_tokens(target_full, masked_indices)
        prediction = self.predictor(context, keep_indices, masked_indices)
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "target": target_masked,
            "keep_indices": keep_indices,
            "masked_indices": masked_indices,
        }

    @torch.no_grad()
    def synchronize_target_encoder(self) -> None:
        self.target_down_projection.load_state_dict(self.down_projection.state_dict())
        self.target_eva.load_state_dict(self.eva.state_dict())
        self._freeze_target_encoder()

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for online, target in (
            (self.down_projection, self.target_down_projection),
            (self.eva, self.target_eva),
        ):
            for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)

    @torch.no_grad()
    def load_mae_state_dict(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        """Load matching original MAE keys and initialize the target encoder."""
        current = self.state_dict()
        loaded = []
        for raw_key, value in state_dict.items():
            key = raw_key.removeprefix("module.").removeprefix("_orig_mod.")
            if key in current and current[key].shape == value.shape and not key.startswith(("predictor.", "target_")):
                current[key] = value
                loaded.append(key)
        self.load_state_dict(current)
        self.synchronize_target_encoder()
        return loaded
