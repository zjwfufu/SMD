# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 model building blocks.

Consolidates all transformer building blocks needed by ``SLatFlowModel``:
embedders, normalization, modulated sparse transformer blocks, and utility
functions.  Sparse tensor primitives are imported from
``smd.networks.Trellis2.sparse``.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smd.networks.Trellis2.sparse import (
    SparseGELU,
    SparseLinear,
    SparseMultiHeadAttention,
    SparseTensor,
    VarLenTensor,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def convert_module_to(module, dtype):
    """Convert primitive modules (Linear, Conv, SparseLinear) to the given dtype."""
    _MIX_PRECISION = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
                       nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear, SparseLinear)
    if isinstance(module, _MIX_PRECISION):
        for p in module.parameters():
            p.data = p.data.to(dtype)


def manual_cast(tensor, dtype):
    """Cast if autocast is not enabled."""
    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor


def str_to_dtype(dtype_str: str):
    return {
        "f16": torch.float16, "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "f32": torch.float32, "fp32": torch.float32, "float32": torch.float32,
    }[dtype_str]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class LayerNorm32(nn.LayerNorm):
    """LayerNorm that casts to float32 for numerical stability, then casts back."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → 2-layer MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class AbsolutePositionEmbedder(nn.Module):
    """Sinusoidal 3D position embedding for sparse voxel coordinates."""

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(x.device)
        out = torch.cat([
            torch.sin(x[..., None] * self.freqs),
            torch.cos(x[..., None] * self.freqs),
        ], dim=-1)
        return out.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self._sin_cos_embedding(x.float())
        if embed.shape[-1] < self.channels:
            embed = torch.cat([
                embed,
                torch.zeros(*embed.shape[:-1], self.channels - embed.shape[-1], device=embed.device),
            ], dim=-1)
        return embed


# ---------------------------------------------------------------------------
# Sparse Transformer Blocks
# ---------------------------------------------------------------------------


class SparseFeedForwardNet(nn.Module):
    """SparseLinear → GELU → SparseLinear."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x):
        return self.mlp(x)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Modulated sparse transformer block with self-attention + cross-attention + FFN.

    This is the core building block of ``SLatFlowModel`` — a DiT-style block
    with adaLN modulation from the timestep embedding, sparse self-attention
    over voxel tokens, and cross-attention to the condition (DINOv3 features).
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "full",
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qkv_bias: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, type="self", attn_mode=attn_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope, rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            type="cross", attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
