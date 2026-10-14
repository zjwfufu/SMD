# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored & adapted from Tencent Hunyuan3D-2.1
#   hy3dshape/hy3dshape/models/denoisers/hunyuandit.py
# Original: Copyright (C) 2024 THL A29 Limited (Tencent), licensed under the
# TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT. Architecture preserved
# verbatim; loading classmethods and original-repo utility imports stripped so
# this file is self-contained inside SMD.
#
# ============================================================================
# TIME / SIGN CONVENTION (read before touching forward!)
# ----------------------------------------------------------------------------
# There are TWO distinct "times" and they must never be conflated:
#
#   * ODE-time `t` (SMD convention): parametrizes the probability-flow ODE.
#       t=0 -> data, t=1 -> noise. The SMD noise_scheduler / sigma grid /
#       Euler solver / basis / imitation collocation all live in THIS time.
#       It is the ONLY time the rest of SMD ever sees.
#
#   * embedding-time `t_emb` (Hunyuan native): the scalar fed into the pretrained
#       `TimestepEmbedder`. It is purely a *conditioning feature* ("which noise
#       level am I at"). Hunyuan trained with t_emb=0 -> noise, t_emb=1 -> data,
#       normalized to [0, 1] (the original pipeline feeds timestep/1000). It does
#       NOT drive ODE integration.
#
# Algebra forced by the two conventions:
#       t_emb  = 1 - t                      (time-embedding-time from ODE-time)
#       u      = x_data - x_noise           (Hunyuan native velocity)
#       flow   = x_noise - x_data = -u      (SMD "flow" prediction)
#
# `_forward_native(x, t_emb, contexts)` is the faithful Hunyuan computation:
# it consumes native embedding-time and returns native velocity u. The public
# `forward(x, t, contexts)` is a thin adapter that applies the two conversions
# in ONE place each, so from SMD's side this network natively speaks our
# convention. (The pivot, which replicates the tail blocks, applies the
# matching output negation in its own forward — see pivot/backends/hunyuan3d.py.)
# ============================================================================

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .moe import MoEBlock


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, downscale_freq_shift: float = 0.0, scale: int = 1, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)
        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):
        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)
        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight

    def forward(self, x, y):
        b, s1, c = x.shape

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim :, :]
            kv_dca = self.kv_proj_dca(context_dca).view(b, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, : (token_len - self.dca_dim), :]

        _, s2, c = y.shape
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.view(b, s2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads), (q, k, v))
        context = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s1, -1)

        if self.with_dca:
            k_dca, v_dca = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads), (k_dca, v_dca))
            context_dca = F.scaled_dot_product_attention(q, k_dca, v_dca).transpose(1, 2).reshape(b, s1, -1)
            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)
        return out


class Attention(nn.Module):
    """Multi-head self-attention (names aligned with flash-attention)."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x


class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        init_scale=1.0,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        use_ele_affine = True

        # ---- Self-Attention ----
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        self.attn1 = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=qk_norm_layer
        )

        # ---- FFN norm ----
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)

        # ---- Optional timestep additive modulation ----
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_emb_size, hidden_size, bias=True))

        # ---- Cross-Attention ----
        self.attn2 = CrossAttention(
            hidden_size,
            text_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
            with_decoupled_ca=with_decoupled_ca,
            decoupled_ca_dim=decoupled_ca_dim,
            decoupled_ca_weight=decoupled_ca_weight,
            init_scale=init_scale,
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        # ---- U-Net skip connection ----
        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        # ---- FFN: dense MLP or MoE ----
        self.use_moe = use_moe
        if self.use_moe:
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None):
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-Attention
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa
        attn_out = self.attn1(self.norm1(x))
        x = x + attn_out

        # Cross-Attention
        x = x + self.attn2(self.norm2(x), text_states)

        # FFN
        mlp_inputs = self.norm3(x)
        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)
        return x


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)
        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class FinalLayer(nn.Module):
    """Final layer of HunYuanDiT: strips the prepended conditioning token + projects.

    `out_channels_mult` > 1 expands the output projection for the pivot
    (n_grid velocity slots). The base model uses mult=1.
    """

    def __init__(self, final_hidden_size, out_channels, out_channels_mult: int = 1):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.out_channels = out_channels
        self.out_channels_mult = out_channels_mult
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, out_channels * out_channels_mult, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]  # strip prepended conditioning token
        x = self.linear(x)
        return x


class HunYuanDiTPlain(nn.Module):
    """Hunyuan3D 2.1 Shape denoiser (U-Net-style DiT with skip connections + MoE).

    Architecture preserved verbatim from the Tencent release. The forward path is
    split into `_forward_native` (faithful Hunyuan semantics) and the public
    `forward` (SMD convention adapter). See the module-level convention note.
    """

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type="layer",
        qk_norm_type="rms",
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == "layer" else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == "rms" else nn.LayerNorm
        self.context_dim = context_dim
        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim
        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if use_attention_pooling:
            self.pooler = AttentionPool(self.text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size, bias=True),
            )

        if with_decoupled_ca:
            self.additional_cond_hidden_state = additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(additional_cond_hidden_state, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, 1024, bias=True),
            )

        self.blocks = nn.ModuleList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=context_dim,
                    qk_norm=qk_norm,
                    norm_layer=self.norm,
                    qk_norm_layer=self.qk_norm,
                    skip_connection=layer > depth // 2,
                    with_decoupled_ca=with_decoupled_ca,
                    decoupled_ca_dim=decoupled_ca_dim,
                    decoupled_ca_weight=decoupled_ca_weight,
                    qkv_bias=qkv_bias,
                    use_moe=True if depth - layer <= num_moe_layers else False,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                )
                for layer in range(depth)
            ]
        )
        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        # Pivot-distillation capture hook (set externally by Hunyuan3DPivotBackend).
        # When `_pivot_capture` is a dict and `_pivot_split_k` is set, the forward
        # loop records the post-split-block hidden_states + everything the pivot
        # decoder needs to continue (timestep token `c`, conditioning, remaining
        # skip stack). It is a plain attribute, NOT a torch forward-hook, because
        # the U-Net skip stack is forward-local and invisible to module hooks.
        self._pivot_capture: dict | None = None
        self._pivot_split_k: int | None = None

    # ------------------------------------------------------------------
    # Faithful Hunyuan computation (native embedding-time, native velocity u)
    # ------------------------------------------------------------------
    def _forward_native(self, x, t_emb, contexts, **kwargs):
        """Native Hunyuan forward.

        x        : [B, N, in_channels] latent tokens
        t_emb    : [B] embedding-time in [0, 1] (Hunyuan: 0=noise, 1=data)
        contexts : {"main": [B, text_len, context_dim], optional "additional": ...}
        returns  : [B, N, in_channels] native velocity u = x_data - x_noise
        """
        cond = contexts["main"]

        t = self.t_embedder(t_emb, condition=kwargs.get("guidance_cond"))
        x = self.x_embedder(x)

        if self.use_pos_emb:
            x = x + self.pos_embed.to(x.dtype)

        if self.use_attention_pooling:
            extra_vec = self.pooler(cond, None)
            c = t + self.extra_embedder(extra_vec)
        else:
            c = t

        if self.with_decoupled_ca:
            additional_cond = self.additional_cond_proj(contexts["additional"])
            cond = torch.cat([cond, additional_cond], dim=1)

        x = torch.cat([c, x], dim=1)  # prepend conditioning token

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)
            # Pivot capture: record state just after the split block so the pivot
            # decoder can continue from here. skip_stack is the *remaining* encoder
            # activations the tail decoder blocks will pop (LIFO), copied so later
            # pops in this loop don't mutate it.
            if self._pivot_capture is not None and self._pivot_split_k is not None and layer == self._pivot_split_k:
                self._pivot_capture.clear()
                self._pivot_capture.update(
                    hidden_states=x,
                    c=c,
                    time_condition_offset=c - t,
                    guidance_cond=kwargs.get("guidance_cond"),
                    cond=cond,
                    skip_stack=list(skip_value_list),
                )

        x = self.final_layer(x)
        return x

    # ------------------------------------------------------------------
    # Public SMD-convention forward (thin adapter; see module note)
    # ------------------------------------------------------------------
    def forward(self, x, t, contexts, **kwargs):
        """SMD-convention forward.

        t : [B] ODE-time in [0, 1] (SMD: 0=data, 1=noise). Used ONLY to derive
            the embedding-time; it does not drive ODE integration here.
        returns : [B, N, in_channels] SMD "flow" = x_noise - x_data.
        """
        t_emb = 1.0 - t  # ODE-time -> embedding-time (the only place this happens)
        u = self._forward_native(x, t_emb, contexts, **kwargs)
        return -u  # native velocity u -> SMD flow (the only place this happens)
