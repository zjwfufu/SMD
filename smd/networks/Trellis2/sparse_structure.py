# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

"""TRELLIS.2 sparse-structure flow model and decoder.

This is the minimal dense-voxel model stack needed by the first stage of the
official TRELLIS.2 image-to-3D pipeline. Module and parameter names intentionally
match the released checkpoints.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _str_to_dtype(dtype: str) -> torch.dtype:
    return {
        "f16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype]


def _convert_module_to(module: nn.Module, dtype: torch.dtype) -> None:
    if isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        ),
    ):
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype)


def _manual_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor if torch.is_autocast_enabled() else tensor.to(dtype)


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.group_norm(x.float(), self.num_groups, weight, bias, self.eps).to(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.dim()
        x = x.permute(0, *range(2, ndim), 1).contiguous()
        x = super().forward(x)
        return x.permute(0, ndim - 1, *range(1, ndim - 1)).contiguous()


class RotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2")
        self.head_dim = head_dim
        self.dim = dim
        freq_dim = head_dim // 2 // dim
        freqs = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
        self.register_buffer("freqs", rope_freq[0] / (rope_freq[1] ** freqs), persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(f"Expected coordinate dim {self.dim}, got {indices.shape[-1]}")
        phases = torch.outer(indices.reshape(-1), self.freqs.to(indices.device))
        phases = torch.polar(torch.ones_like(phases), phases).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            padding = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(*phases.shape[:-1], padding, device=phases.device),
                        torch.zeros(*phases.shape[:-1], padding, device=phases.device),
                    ),
                ],
                dim=-1,
            )
        return phases

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        rotated = x_complex * phases.unsqueeze(-2)
        return torch.view_as_real(rotated).reshape(*rotated.shape[:-1], -1).to(x.dtype)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    try:
        if q.is_cuda:
            import flash_attn

            return flash_attn.flash_attn_func(q, k, v)
    except ImportError:
        pass
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        del window_size, shift_window, rope_freq
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if attn_mode != "full":
            raise NotImplementedError("Sparse-structure inference only supports full attention")
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        elif type == "cross":
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        else:
            raise ValueError(f"Unsupported attention type: {type}")

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        if self._type == "self":
            q, k, v = self.to_qkv(x).reshape(batch, length, 3, self.num_heads, -1).unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if self.use_rope:
                if phases is None:
                    raise ValueError("RoPE phases are required")
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
        else:
            if context is None:
                raise ValueError("Cross attention requires context")
            kv_length = context.shape[1]
            q = self.to_q(x).reshape(batch, length, self.num_heads, -1)
            k, v = self.to_kv(context).reshape(batch, kv_length, 2, self.num_heads, -1).unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
        return self.to_out(_attention(q, k, v).reshape(batch, length, -1))


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "full",
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
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
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if share_mod:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )

    def _forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.share_mod:
            modulation = (self.modulation + mod).to(mod.dtype)
        else:
            modulation = self.adaLN_modulation(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.self_attn(h, phases=phases) * gate_msa.unsqueeze(1)
        x = x + self.cross_attn(self.norm2(x), context)
        h = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.mlp(h) * gate_mlp.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                mod,
                context,
                phases,
                use_reentrant=False,
            )
        return self._forward(x, mod, context, phases)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = "vanilla",
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.dtype = _str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        coords = torch.stack(
            torch.meshgrid(*[torch.arange(resolution)] * 3, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        if pe_mode == "rope":
            rope = RotaryPositionEmbedder(model_channels // self.num_heads, 3, rope_freq)
            self.register_buffer("rope_phases", rope(coords), persistent=True)
        else:
            raise NotImplementedError("Released TRELLIS.2 sparse-structure model uses RoPE")

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=use_checkpoint,
                    use_rope=True,
                    rope_freq=rope_freq,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)
        self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(_convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)

        def scaled_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in self.blocks:
            block.self_attn.to_out.apply(scaled_init)
            block.cross_attn.to_out.apply(scaled_init)
            block.mlp.mlp[2].apply(scaled_init)
        nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
        nn.init.zeros_(self.input_layer.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        else:
            for block in self.blocks:
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        expected = [x.shape[0], self.in_channels, self.resolution, self.resolution, self.resolution]
        if list(x.shape) != expected:
            raise ValueError(f"Input shape mismatch: got {list(x.shape)}, expected {expected}")
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = _manual_cast(t_emb, self.dtype)
        h = _manual_cast(h, self.dtype)
        cond = _manual_cast(cond, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases)
        h = self.out_layer(F.layer_norm(_manual_cast(h, x.dtype), h.shape[-1:]))
        return h.permute(0, 2, 1).view(h.shape[0], h.shape[2], self.resolution, self.resolution, self.resolution)


def _pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    batch, channels, height, width, depth = x.shape
    out_channels = channels // scale_factor**3
    x = x.reshape(
        batch,
        out_channels,
        scale_factor,
        scale_factor,
        scale_factor,
        height,
        width,
        depth,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(
        batch,
        out_channels,
        height * scale_factor,
        width * scale_factor,
        depth * scale_factor,
    )


def _norm_layer(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, channels)
    if norm_type == "layer":
        return ChannelLayerNorm32(channels)
    raise ValueError(f"Invalid norm type: {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = _norm_layer(norm_type, channels)
        self.norm2 = _norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_connection(x)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        elif mode == "nearest":
            if in_channels != out_channels:
                raise ValueError("Nearest upsampling requires matching channels")
        else:
            raise ValueError(f"Invalid upsample mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv":
            return _pixel_shuffle_3d(self.conv(x), 2)
        return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: list[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[0], channels[0], norm_type) for _ in range(num_res_blocks_middle)]
        )
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.blocks.extend([ResBlock3d(channel, channel, norm_type) for _ in range(num_res_blocks)])
            if index < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(channel, channels[index + 1]))
        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )
        if use_fp16:
            self.blocks.apply(partial(_convert_module_to, dtype=torch.float16))
            self.middle_block.apply(partial(_convert_module_to, dtype=torch.float16))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x).to(self.dtype)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        return self.out_layer(h.to(x.dtype))


__all__ = ["SparseStructureDecoder", "SparseStructureFlowModel"]
