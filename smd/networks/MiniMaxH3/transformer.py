# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin, get_parameter_dtype
from diffusers.utils import BaseOutput, apply_lora_scale, logging
from torch import _dynamo, nn

from .context_parallel import (
    MiniMaxH3ContextParallelState,
    gather_sequence,
    heads_to_sequence,
    sequence_to_heads,
)

logger = logging.get_logger(__name__)

MINIMAX_H3_MODALITY_NUM = 3

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    _dynamo.config.optimize_ddp = False
    if os.environ.get("SMD_FLEX_COMPILE", "1") != "0":
        _flex_attention = torch.compile(
            _flex_attention,
            dynamic=False,
            mode=os.environ.get("TORCH_COMPILE_MODE", "default"),
        )
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    BlockMask = Any
    create_block_mask = None
    _flex_attention = None
    _FLEX_ATTENTION_AVAILABLE = False


@dataclass
class MiniMaxH3TransformerOutput(BaseOutput):
    sample: torch.Tensor
    audio_sample: torch.Tensor


@dataclass(frozen=True)
class MiniMaxH3SparseAttentionMask:
    block_mask: BlockMask | None
    score_mod: Callable[..., torch.Tensor] | None
    sequence_length: int
    padded_length: int
    prefix_length: int
    video_rows_per_frame: int
    temporal_radius: int
    padding_start: int
    padding_end: int
    device: torch.device


def _default_gradient_checkpointing_func(module: Callable[..., torch.Tensor], *args: torch.Tensor) -> torch.Tensor:
    try:
        return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=False)
    except TypeError:
        return torch.utils.checkpoint.checkpoint(module.__call__, *args)


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    if rotary_dim > hidden_states.shape[-1]:
        raise ValueError(f"Rotary dim {rotary_dim} exceeds attention head dim {hidden_states.shape[-1]}.")
    half = rotary_dim // 2
    if rotary_dim % 2:
        raise ValueError(f"Rotary dim must be even, got {rotary_dim}.")
    rotary = hidden_states[..., :rotary_dim]
    cos = cos.to(hidden_states.dtype)[None, :, None, :half]
    sin = sin.to(hidden_states.dtype)[None, :, None, :half]
    # H3's fp64 layout can produce ~38k rows. Rotating the full 96-channel
    # prefix with cat/multiply creates several 388MiB temporaries, so update
    # the newly-created Q/K tensors in small channel chunks instead.
    for start in range(0, half, 4):
        end = min(start + 4, half)
        x1 = rotary[..., start:end].clone()
        x2 = rotary[..., half + start : half + end].clone()
        cos_chunk = cos[..., start:end]
        sin_chunk = sin[..., start:end]
        rotary[..., start:end] = x1 * cos_chunk - x2 * sin_chunk
        rotary[..., half + start : half + end] = x2 * cos_chunk + x1 * sin_chunk
    return hidden_states.contiguous()


def _chunked_feed_forward(
    feed_forward: nn.Module,
    hidden_states: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0 or hidden_states.shape[1] <= chunk_size:
        return feed_forward(hidden_states)

    output = None
    for start in range(0, hidden_states.shape[1], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[1])
        chunk_input = hidden_states[:, start:end]
        # Keep GEMM shapes fixed: a short CP tail can select a different BF16
        # reduction kernel and amplify rounding differences through the stack.
        # Feed-forward layers are row-local, so padded rows are discarded.
        if end - start < chunk_size:
            chunk_input = F.pad(chunk_input, (0, 0, 0, chunk_size - (end - start)))
        chunk = feed_forward(chunk_input)[:, : end - start]
        if output is None:
            output = chunk.new_empty((*hidden_states.shape[:-1], chunk.shape[-1]))
        output[:, start:end].copy_(chunk)
    if output is None:
        raise RuntimeError("MiniMax H3 feed-forward input must contain at least one sequence row.")
    return output


def _chunked_rms_norm(norm: nn.Module, hidden_states: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or hidden_states.shape[1] <= chunk_size:
        return norm(hidden_states)

    output = None
    for start in range(0, hidden_states.shape[1], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[1])
        chunk = norm(hidden_states[:, start:end])
        if output is None:
            output = chunk.new_empty(hidden_states.shape)
        output[:, start:end].copy_(chunk)
    if output is None:
        raise RuntimeError("MiniMax H3 RMSNorm input must contain at least one sequence row.")
    return output


def _h3_temporal_sparse_visibility(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    sequence_length: int,
    prefix_length: int,
    video_rows_per_frame: int,
    temporal_radius: int,
) -> torch.Tensor:
    q_valid = q_idx < sequence_length
    kv_valid = kv_idx < sequence_length
    q_prefix = q_idx < prefix_length
    kv_prefix = kv_idx < prefix_length
    q_time = (q_idx - prefix_length) // video_rows_per_frame
    kv_time = (kv_idx - prefix_length) // video_rows_per_frame
    visible = q_prefix | kv_prefix | ((q_time - kv_time).abs() <= temporal_radius)
    return (q_valid & kv_valid & visible) | ((~q_valid) & (q_idx == kv_idx))


def _h3_padded_sequence_length(sequence_length: int, context_parallel_degree: int = 1) -> int:
    alignment = math.lcm(128, int(context_parallel_degree))
    return math.ceil(sequence_length / alignment) * alignment


def _h3_dense_key_padding_mask(
    sequence_length: int,
    padded_length: int,
    device: torch.device,
) -> torch.Tensor:
    if sequence_length <= 0 or padded_length < sequence_length:
        raise ValueError(
            f"MiniMax H3 dense padding requires 0 < sequence_length <= padded_length, "
            f"got {sequence_length} and {padded_length}."
        )
    return (torch.arange(padded_length, device=device) < sequence_length).reshape(1, 1, 1, padded_length)


def _h3_sparse_padding_score_mod(
    sequence_length: int,
    padded_length: int,
    padding_start: int,
    padding_end: int,
    device: torch.device,
    query_start: int = 0,
) -> Callable[..., torch.Tensor]:
    row_indices = torch.arange(padded_length, device=device)
    valid_rows = (row_indices < sequence_length) & ~((row_indices >= padding_start) & (row_indices < padding_end))

    def score_mod(score, _batch, _head, q_idx, kv_idx):
        q_global = q_idx + query_start
        q_valid = valid_rows[q_global]
        kv_valid = valid_rows[kv_idx]
        keep = (q_valid & kv_valid) | ((~q_valid) & (q_global == kv_idx))
        return torch.where(keep, score, -torch.inf)

    return score_mod


@functools.lru_cache(maxsize=128)
def _h3_temporal_sparse_query_mask(
    sequence_length: int,
    padded_length: int,
    prefix_length: int,
    video_rows_per_frame: int,
    temporal_radius: int,
    padding_start: int,
    padding_end: int,
    query_start: int,
    query_length: int,
    device: torch.device,
) -> MiniMaxH3SparseAttentionMask:
    if not _FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise RuntimeError("MiniMax H3 sparse attention requires PyTorch FlexAttention.")

    def mask_mod(_batch, _head, q_idx, kv_idx):
        return _h3_temporal_sparse_visibility(
            q_idx + query_start,
            kv_idx,
            sequence_length=sequence_length,
            prefix_length=prefix_length,
            video_rows_per_frame=video_rows_per_frame,
            temporal_radius=temporal_radius,
        )

    return MiniMaxH3SparseAttentionMask(
        block_mask=create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=padded_length,
            device=device,
            BLOCK_SIZE=128,
            _compile=True,
        ),
        score_mod=_h3_sparse_padding_score_mod(
            sequence_length,
            padded_length,
            padding_start,
            padding_end,
            device,
            query_start,
        ),
        sequence_length=sequence_length,
        padded_length=padded_length,
        prefix_length=prefix_length,
        video_rows_per_frame=video_rows_per_frame,
        temporal_radius=temporal_radius,
        padding_start=padding_start,
        padding_end=padding_end,
        device=device,
    )


def _slice_sparse_attention_mask(
    attention_mask: MiniMaxH3SparseAttentionMask,
    query_start: int,
    query_length: int,
) -> MiniMaxH3SparseAttentionMask:
    return _h3_temporal_sparse_query_mask(
        attention_mask.sequence_length,
        attention_mask.padded_length,
        attention_mask.prefix_length,
        attention_mask.video_rows_per_frame,
        attention_mask.temporal_radius,
        attention_mask.padding_start,
        attention_mask.padding_end,
        query_start,
        query_length,
        attention_mask.device,
    )


@functools.lru_cache(maxsize=32)
def _h3_temporal_sparse_block_mask(
    sequence_length: int,
    prefix_length: int,
    video_rows_per_frame: int,
    temporal_radius: int,
    device: torch.device,
    minimum_padded_length: int = 0,
) -> tuple[BlockMask, int]:
    if not _FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise RuntimeError("MiniMax H3 sparse attention requires PyTorch FlexAttention.")
    minimum_padded_length = max(minimum_padded_length, sequence_length)
    padded_length = math.ceil(minimum_padded_length / 128) * 128

    def mask_mod(_batch, _head, q_idx, kv_idx):
        return _h3_temporal_sparse_visibility(
            q_idx,
            kv_idx,
            sequence_length=sequence_length,
            prefix_length=prefix_length,
            video_rows_per_frame=video_rows_per_frame,
            temporal_radius=temporal_radius,
        )

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=device,
        BLOCK_SIZE=128,
        _compile=True,
    )
    logger.info(
        "MiniMax H3 temporal sparse mask: sequence=%d padded=%d prefix=%d rows_per_frame=%d radius=%d sparsity=%.2f%%",
        sequence_length,
        padded_length,
        prefix_length,
        video_rows_per_frame,
        temporal_radius,
        block_mask.sparsity(),
    )
    return block_mask, padded_length


class MiniMaxH3RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids.to(torch.float32)
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(F.silu(temb).to(get_parameter_dtype(self.linear)))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(get_parameter_dtype(self.linear))).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


class MiniMaxH3AttnProcessor:
    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: MiniMaxH3SparseAttentionMask | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn.fused_projections and attn.to_qkv is not None:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        batch_size, seq_len, _ = query.shape
        query = query.view(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.view(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.view(batch_size, seq_len, attn.heads, attn.head_dim)

        query = _chunked_rms_norm(attn.norm_q, query, attn.qk_norm_chunk_size)
        key = _chunked_rms_norm(attn.norm_k, key, attn.qk_norm_chunk_size)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        context_parallel = attn.context_parallel_state
        if context_parallel is not None and context_parallel.enabled:
            query = sequence_to_heads(query, context_parallel)
            key = sequence_to_heads(key, context_parallel)
            value = sequence_to_heads(value, context_parallel)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        if isinstance(attention_mask, MiniMaxH3SparseAttentionMask):
            if _flex_attention is None:
                raise RuntimeError("MiniMax H3 sparse attention requires PyTorch FlexAttention.")
            head_chunk_size = attn.context_parallel_head_chunk_size

            def attend(query_chunk, key_chunk, value_chunk, sparse_mask):
                if sparse_mask.block_mask is None or sparse_mask.score_mod is None:
                    raise RuntimeError("MiniMax H3 sparse attention mask was not materialized.")
                if head_chunk_size > 0 and query_chunk.shape[1] > head_chunk_size:
                    return torch.cat(
                        [
                            _flex_attention(
                                query_chunk[:, start : start + head_chunk_size],
                                key_chunk[:, start : start + head_chunk_size],
                                value_chunk[:, start : start + head_chunk_size],
                                score_mod=sparse_mask.score_mod,
                                block_mask=sparse_mask.block_mask,
                            )
                            for start in range(0, query_chunk.shape[1], head_chunk_size)
                        ],
                        dim=1,
                    )
                return _flex_attention(
                    query_chunk,
                    key_chunk,
                    value_chunk,
                    score_mod=sparse_mask.score_mod,
                    block_mask=sparse_mask.block_mask,
                )

            query_chunk_size = attn.context_parallel_query_chunk_size
            if query_chunk_size > 0:
                hidden_states = torch.cat(
                    [
                        attend(
                            query[:, :, start : start + query_chunk_size],
                            key,
                            value,
                            _slice_sparse_attention_mask(
                                attention_mask,
                                start,
                                min(query_chunk_size, query.shape[2] - start),
                            ),
                        )
                        for start in range(0, query.shape[2], query_chunk_size)
                    ],
                    dim=2,
                )
            else:
                hidden_states = attend(query, key, value, attention_mask)
        elif attention_mask is None:
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
        elif isinstance(attention_mask, torch.Tensor):
            if attention_mask.dtype != torch.bool or attention_mask.ndim != 4:
                raise TypeError(
                    "MiniMax H3 dense attention mask must be a broadcastable rank-4 bool key mask, got "
                    f"dtype={attention_mask.dtype}, shape={list(attention_mask.shape)}."
                )
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            raise TypeError(f"Unsupported MiniMax H3 attention mask type: {type(attention_mask).__name__}.")
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        if context_parallel is not None and context_parallel.enabled:
            hidden_states = heads_to_sequence(hidden_states, context_parallel)
        hidden_states = hidden_states.reshape(batch_size, seq_len, attn.inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float = 1e-5,
        qk_norm_chunk_size: int = 0,
        processor: MiniMaxH3AttnProcessor | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.use_bias = False
        self.fused_projections = False
        self.to_qkv: nn.Linear | None = None
        self.qk_norm_chunk_size = int(qk_norm_chunk_size)
        self.context_parallel_state: MiniMaxH3ContextParallelState | None = None
        self.context_parallel_head_chunk_size = 0
        self.context_parallel_query_chunk_size = 0
        if self.qk_norm_chunk_size < 0:
            raise ValueError(f"`qk_norm_chunk_size` must be non-negative, got {qk_norm_chunk_size}.")

        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Dropout(0.0)])
        self.set_processor(processor or MiniMaxH3AttnProcessor())

    def set_processor(self, processor: MiniMaxH3AttnProcessor) -> None:
        self.processor = processor

    def get_processor(self) -> MiniMaxH3AttnProcessor:
        return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb, attention_mask)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func: Callable[..., torch.Tensor] | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gradient_checkpointing_func = self._gradient_checkpointing_func or _default_gradient_checkpointing_func
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(time_embed_dim=time_embed_dim, hidden_size=hidden_size)
        self.ff_chunk_size = 0
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func: Callable[..., torch.Tensor] | None = None

    def _attention_residual(
        self,
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale.index_select(0, adaln_indices)) + shift.index_select(
            0, adaln_indices
        )
        attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        gate = gate.index_select(0, adaln_indices)
        # Preserve the reference's BF16 product rounding before residual addition.
        # Fusing with addcmul changes pretrained outputs under context parallelism.
        return residual + gate * attn_output

    def _feed_forward_residual(
        self,
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        adaln_indices: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale.index_select(0, adaln_indices)) + shift.index_select(
            0, adaln_indices
        )
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self.ff_chunk_size)
        gate = gate.index_select(0, adaln_indices)
        # Keep the same two rounding steps as the unsharded reference path.
        return residual + gate * ff_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        context_parallel = self.attn.context_parallel_state
        use_split_checkpoint = (
            context_parallel is not None
            and context_parallel.enabled
            and self.gradient_checkpointing
            and torch.is_grad_enabled()
        )
        if use_split_checkpoint:
            # Reentrant checkpointing needs a differentiable input, including
            # when a custom LoRA recipe leaves the input embeddings frozen.
            if not any(value.requires_grad for value in (hidden_states, shift_msa, scale_msa, gate_msa)):
                hidden_states.requires_grad_(True)
            hidden_states = torch.utils.checkpoint.checkpoint(
                self._attention_residual,
                hidden_states,
                shift_msa,
                scale_msa,
                gate_msa,
                adaln_indices,
                rotary_emb,
                attention_mask,
                use_reentrant=True,
            )
            return torch.utils.checkpoint.checkpoint(
                self._feed_forward_residual,
                hidden_states,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                adaln_indices,
                use_reentrant=True,
            )
        hidden_states = self._attention_residual(
            hidden_states,
            shift_msa,
            scale_msa,
            gate_msa,
            adaln_indices,
            rotary_emb,
            attention_mask,
        )
        return self._feed_forward_residual(
            hidden_states,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            adaln_indices,
        )


class MiniMaxH3Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = [
        "MiniMaxH3TransformerBlock",
        "MiniMaxH3TokenRefinerBlock",
        "MiniMaxH3AdaLayerNormOut",
    ]
    _repeated_blocks: ClassVar[list[str]] = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"]
    _skip_layerwise_casting_patterns: ClassVar[list[str]] = ["norm"]
    _keep_in_fp32_modules: ClassVar[list[str]] = [
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        patch_size = tuple(int(value) for value in patch_size)
        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        rotary_dim = 2 * 3 * rope_freq_dim
        if rotary_dim > attention_head_dim:
            raise ValueError(
                f"`attention_head_dim` must be >= {rotary_dim} for rope_freq_dim={rope_freq_dim}, got {attention_head_dim}."
            )

        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        self.time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=time_embed_hidden_dim,
            out_dim=time_embed_dim,
        )

        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=hidden_size, time_embed_dim=time_embed_dim, eps=final_norm_eps
        )
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

        self.gradient_checkpointing = False
        self.external_gradient_checkpointing = False
        self.sparse_attention_temporal_radius = 0
        self.sparse_attention_max_text_tokens = 0
        self.context_parallel_state: MiniMaxH3ContextParallelState | None = None
        self.active_block_indices: frozenset[int] | None = None

    @property
    def attn_processors(self) -> dict[str, MiniMaxH3AttnProcessor]:
        return {
            f"{name}.processor": module.get_processor()
            for name, module in self.named_modules()
            if isinstance(module, MiniMaxH3Attention)
        }

    def set_attn_processor(self, processor: MiniMaxH3AttnProcessor | dict[str, MiniMaxH3AttnProcessor]) -> None:
        if isinstance(processor, dict):
            processors = processor
            expected = set(self.attn_processors)
            if set(processors) != expected:
                missing = sorted(expected - set(processors))
                extra = sorted(set(processors) - expected)
                raise ValueError(f"Processor dict mismatch. Missing={missing}, extra={extra}.")
            for name, module in self.named_modules():
                if isinstance(module, MiniMaxH3Attention):
                    module.set_processor(processors[f"{name}.processor"])
            return
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.set_processor(processor)

    def set_default_attn_processor(self) -> None:
        self.set_attn_processor(MiniMaxH3AttnProcessor())

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        if enable and self.external_gradient_checkpointing:
            raise RuntimeError("MiniMax H3 external FSDP checkpoint wrappers are already active.")
        if enable:
            self.enable_gradient_checkpointing()
        else:
            self.disable_gradient_checkpointing()

    def use_external_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.token_refiner.gradient_checkpointing = False
        for block in [*self.token_refiner.refiner_blocks, *self.transformer_blocks]:
            block.gradient_checkpointing = False
        self.external_gradient_checkpointing = True

    def set_ff_chunk_size(self, chunk_size: int = 0) -> None:
        if chunk_size < 0:
            raise ValueError(f"`chunk_size` must be non-negative, got {chunk_size}.")
        for block in self.transformer_blocks:
            block.ff_chunk_size = int(chunk_size)

    def set_qk_norm_chunk_size(self, chunk_size: int = 0) -> None:
        if chunk_size < 0:
            raise ValueError(f"`chunk_size` must be non-negative, got {chunk_size}.")
        for block in self.transformer_blocks:
            block.attn.qk_norm_chunk_size = int(chunk_size)

    def set_sparse_attention(self, temporal_radius: int = 0, max_text_tokens: int = 0) -> None:
        if temporal_radius < 0:
            raise ValueError(f"`temporal_radius` must be non-negative, got {temporal_radius}.")
        if max_text_tokens < 0:
            raise ValueError(f"`max_text_tokens` must be non-negative, got {max_text_tokens}.")
        if temporal_radius > 0 and not _FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("MiniMax H3 sparse attention requires PyTorch FlexAttention.")
        if temporal_radius > 0 and max_text_tokens == 0:
            raise ValueError("MiniMax H3 sparse attention requires a positive fixed max_text_tokens.")
        self.sparse_attention_temporal_radius = int(temporal_radius)
        self.sparse_attention_max_text_tokens = int(max_text_tokens)

    def set_context_parallel(self, state: MiniMaxH3ContextParallelState) -> None:
        if state.enabled and int(self.config.num_attention_heads) % state.degree:
            raise ValueError(
                f"MiniMax H3 attention heads {self.config.num_attention_heads} must be divisible "
                f"by context_parallel_size={state.degree}."
            )
        self.context_parallel_state = state
        for block in self.transformer_blocks:
            block.attn.context_parallel_state = state

    def set_context_parallel_head_chunk_size(self, chunk_size: int) -> None:
        if chunk_size < 0:
            raise ValueError(f"MiniMax H3 context-parallel head chunk size must be non-negative, got {chunk_size}.")
        for block in self.transformer_blocks:
            block.attn.context_parallel_head_chunk_size = int(chunk_size)

    def set_context_parallel_query_chunk_size(self, chunk_size: int) -> None:
        if chunk_size < 0 or (chunk_size > 0 and chunk_size % 128):
            raise ValueError(
                "MiniMax H3 context-parallel query chunk size must be zero or a positive multiple of 128, "
                f"got {chunk_size}."
            )
        for block in self.transformer_blocks:
            block.attn.context_parallel_query_chunk_size = int(chunk_size)
        if chunk_size > 0:
            _dynamo.config.recompile_limit = max(int(_dynamo.config.recompile_limit), 128)

    def set_active_block_indices(self, indices: list[int] | tuple[int, ...] | None) -> None:
        if indices is None:
            self.active_block_indices = None
            return
        resolved = []
        num_blocks = len(self.transformer_blocks)
        for index in indices:
            position = int(index) if int(index) >= 0 else num_blocks + int(index)
            if not 0 <= position < num_blocks:
                raise IndexError(f"MiniMax H3 block index {index} is outside {num_blocks} blocks.")
            resolved.append(position)
        if not resolved:
            raise ValueError("MiniMax H3 active block list must not be empty.")
        self.active_block_indices = frozenset(resolved)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        skip_layers: list[int] | tuple[int, ...] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must match `position_ids` in shape `(seq_len,)`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)}."
            )

        token_tags = token_tags.to(dtype=torch.long, device=position_ids.device)
        timestep_indices = timestep_indices.to(dtype=torch.long, device=position_ids.device)
        video_indices = video_indices.to(dtype=torch.long, device=position_ids.device)
        audio_indices = audio_indices.to(dtype=torch.long, device=position_ids.device)
        text_indices = text_indices.to(dtype=torch.long, device=position_ids.device)

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
        attention_mask = None
        padded_length = sequence_length
        context_parallel = self.context_parallel_state
        if self.sparse_attention_temporal_radius > 0:
            if attention_kwargs is None or "video_rows_per_frame" not in attention_kwargs:
                raise ValueError("MiniMax H3 sparse attention requires `attention_kwargs.video_rows_per_frame`.")
            video_rows_per_frame = int(attention_kwargs["video_rows_per_frame"])
            if video_rows_per_frame <= 0:
                raise ValueError(f"`video_rows_per_frame` must be positive, got {video_rows_per_frame}.")
            prefix_length = int(video_indices[0])
            num_text_tokens = int(text_indices.numel())
            if num_text_tokens > self.sparse_attention_max_text_tokens:
                raise ValueError(
                    f"MiniMax H3 sparse attention received {num_text_tokens} text tokens, exceeding "
                    f"max_text_tokens={self.sparse_attention_max_text_tokens}."
                )
            padding_start = int(attention_kwargs.get("text_padding_start", num_text_tokens))
            padding_end = int(attention_kwargs.get("text_padding_end", num_text_tokens))
            if not 0 <= padding_start <= padding_end <= prefix_length:
                raise ValueError(
                    "MiniMax H3 sparse text padding must lie inside the packed prefix, got "
                    f"[{padding_start},{padding_end}) with prefix_length={prefix_length}."
                )
            query_chunk_size = max(
                (block.attn.context_parallel_query_chunk_size for block in self.transformer_blocks),
                default=0,
            )
            use_query_chunks = context_parallel is not None and context_parallel.enabled and query_chunk_size > 0
            context_parallel_degree = (
                context_parallel.degree if context_parallel is not None and context_parallel.enabled else 1
            )
            minimum_padded_length = _h3_padded_sequence_length(sequence_length, context_parallel_degree)
            if use_query_chunks:
                padded_length = minimum_padded_length
                block_mask = None
                score_mod = None
            else:
                block_mask, padded_length = _h3_temporal_sparse_block_mask(
                    sequence_length,
                    prefix_length,
                    video_rows_per_frame,
                    self.sparse_attention_temporal_radius,
                    position_ids.device,
                    minimum_padded_length,
                )
                score_mod = _h3_sparse_padding_score_mod(
                    sequence_length,
                    padded_length,
                    padding_start,
                    padding_end,
                    position_ids.device,
                )
            attention_mask = MiniMaxH3SparseAttentionMask(
                block_mask=block_mask,
                score_mod=score_mod,
                sequence_length=sequence_length,
                padded_length=padded_length,
                prefix_length=prefix_length,
                video_rows_per_frame=video_rows_per_frame,
                temporal_radius=self.sparse_attention_temporal_radius,
                padding_start=padding_start,
                padding_end=padding_end,
                device=position_ids.device,
            )
        elif context_parallel is not None and context_parallel.enabled:
            padded_length = math.ceil(sequence_length / context_parallel.degree) * context_parallel.degree
            if padded_length != sequence_length:
                attention_mask = _h3_dense_key_padding_mask(
                    sequence_length,
                    padded_length,
                    position_ids.device,
                )
        pad_rows = padded_length - sequence_length
        if pad_rows:
            position_ids = F.pad(position_ids, (0, 0, 0, pad_rows))
            token_tags = F.pad(token_tags, (0, pad_rows))
            timestep_indices = F.pad(timestep_indices, (0, pad_rows))

        if context_parallel is not None and context_parallel.enabled:
            if padded_length % context_parallel.degree:
                raise ValueError(
                    f"MiniMax H3 padded sequence {padded_length} must be divisible by "
                    f"context_parallel_size={context_parallel.degree}."
                )
            local_sequence = padded_length // context_parallel.degree
            local_start = context_parallel.rank * local_sequence
            local_end = local_start + local_sequence

            def local_projection(
                values: torch.Tensor,
                global_indices: torch.Tensor,
                projection: nn.Module,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                owned = (global_indices >= local_start) & (global_indices < local_end)
                source_indices = torch.nonzero(owned, as_tuple=False).flatten()
                local_indices = global_indices.index_select(0, source_indices) - local_start
                selected = values.index_select(1, source_indices)
                return projection(selected.to(get_parameter_dtype(projection))), local_indices

            text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
            text_embeds = self.token_refiner(text_embeds)
            owned_text = (text_indices >= local_start) & (text_indices < local_end)
            text_source_indices = torch.nonzero(owned_text, as_tuple=False).flatten()
            local_text_indices = text_indices.index_select(0, text_source_indices) - local_start
            local_text_embeds = text_embeds.index_select(1, text_source_indices)
            video_embeds, local_video_indices = local_projection(hidden_states, video_indices, self.proj_in)
            audio_embeds, local_audio_indices = local_projection(
                audio_hidden_states,
                audio_indices,
                self.audio_proj_in,
            )
            packed_hidden_states = text_embeds.new_zeros((text_embeds.shape[0], local_sequence, text_embeds.shape[-1]))
            packed_hidden_states = packed_hidden_states.index_copy(1, local_text_indices, local_text_embeds)
            packed_hidden_states = packed_hidden_states.index_copy(
                1,
                local_video_indices,
                video_embeds.to(text_embeds.dtype),
            )
            packed_hidden_states = packed_hidden_states.index_copy(
                1,
                local_audio_indices,
                audio_embeds.to(text_embeds.dtype),
            )
            graph_anchor = text_embeds.sum() + video_embeds.sum() + audio_embeds.sum()
            packed_hidden_states = packed_hidden_states + graph_anchor.to(packed_hidden_states.dtype) * 0
            position_ids = position_ids[local_start:local_end]
            token_tags = token_tags[local_start:local_end]
            timestep_indices = timestep_indices[local_start:local_end]
        else:
            text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
            text_embeds = self.token_refiner(text_embeds)
            video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
            audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
            packed_hidden_states = text_embeds.new_zeros((text_embeds.shape[0], padded_length, text_embeds.shape[-1]))
            packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
            packed_hidden_states = packed_hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
            packed_hidden_states = packed_hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        rotary_emb = self.rope(position_ids)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        gradient_checkpointing_func = self._gradient_checkpointing_func or _default_gradient_checkpointing_func
        skipped = frozenset() if skip_layers is None else frozenset(int(index) for index in skip_layers)
        if any(index < 0 or index >= len(self.transformer_blocks) for index in skipped):
            raise IndexError(
                f"MiniMax H3 skip_layers must be zero-based indices in [0,{len(self.transformer_blocks) - 1}], "
                f"got {sorted(skipped)}."
            )
        if len(skipped) == len(self.transformer_blocks):
            raise ValueError("MiniMax H3 skip_layers must not skip all Transformer blocks.")

        hidden_states = packed_hidden_states
        for index, block in enumerate(self.transformer_blocks):
            if index in skipped or (self.active_block_indices is not None and index not in self.active_block_indices):
                continue
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if context_parallel is not None and context_parallel.enabled:
                    hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
                else:
                    hidden_states = gradient_checkpointing_func(
                        block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
                    )
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)

        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(hidden_states)
        audio_output = self.audio_proj_out(hidden_states)
        if context_parallel is not None and context_parallel.enabled:
            video_output = gather_sequence(
                video_output,
                context_parallel,
                autograd=torch.is_grad_enabled() and video_output.requires_grad,
            )
            audio_output = gather_sequence(
                audio_output,
                context_parallel,
                autograd=torch.is_grad_enabled() and audio_output.requires_grad,
            )
        video_output = video_output.index_select(1, video_indices)
        audio_output = audio_output.index_select(1, audio_indices)

        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
