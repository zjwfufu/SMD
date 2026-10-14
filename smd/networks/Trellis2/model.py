# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLatFlowModel — TRELLIS.2 Structured Latent Flow Model.

A sparse-voxel DiT for generating 3D shape and texture latents via flow
matching.  Operates on ``SparseTensor`` inputs (variable-length voxel
features with integer coordinates).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smd.networks.Trellis2.sparse import (
    SparseLinear,
    SparseTensor,
    VarLenTensor,
    sparse_cat,
)

from .modules import (
    AbsolutePositionEmbedder,
    LayerNorm32,
    ModulatedSparseTransformerCrossBlock,
    TimestepEmbedder,
    convert_module_to,
    manual_cast,
    str_to_dtype,
)


class SLatFlowModel(nn.Module):
    """Structured Latent Flow Model (TRELLIS.2).

    A sparse DiT that processes voxel features with cross-attention to
    image condition embeddings and adaLN modulation from timestep.
    """

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
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = "vanilla",
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
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
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = SparseLinear(in_channels, model_channels)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_mode="full", use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"), rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

        # Elastic memory controller (set via register_memory_controller)
        self._memory_controller = None

    # ------------------------------------------------------------------
    # Elastic memory management
    # ------------------------------------------------------------------

    def register_memory_controller(self, controller) -> None:
        """Register a :class:`LinearMemoryController` for dynamic checkpointing."""
        self._memory_controller = controller

    @contextmanager
    def with_mem_ratio(self, mem_ratio: float = 1.0):
        """Dynamically toggle per-block activation checkpointing.

        ``mem_ratio=1.0`` → no checkpointing (fastest, most memory).
        ``mem_ratio=0.0`` → all blocks checkpointed (slowest, least memory).
        """
        if mem_ratio >= 1.0:
            yield 1.0
            return
        num_blocks = len(self.blocks)
        num_ckpt = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        exact = 1.0 - (num_ckpt - 1) / num_blocks
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = i < num_ckpt
        try:
            yield exact
        finally:
            for i in range(num_blocks):
                self.blocks[i].use_checkpoint = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == "vanilla":
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        elif self.initialization == "scaled":
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[SparseTensor] = None,
        **kwargs,
    ) -> SparseTensor:
        if concat_cond is not None:
            x = sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h
