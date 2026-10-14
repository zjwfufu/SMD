# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..common import Pivot
from .flux import FluxPivotBackend

# ---------------------------------------------------------------------------
# LensPivot
# ---------------------------------------------------------------------------


class LensPivot(Pivot):
    """Deep pivot for Lens transformers.

    Lens blocks use ``attention_mask`` (not ``encoder_hidden_states_mask``)
    and do not take ``joint_attention_kwargs``.
    """

    @staticmethod
    def _collect_all_blocks(transformer: nn.Module) -> list[nn.Module]:
        return list(transformer.transformer_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


class LensPivotBackend(FluxPivotBackend):
    """Lens-specific backend for SimplifiedDistillation.

    Like QwenImage: single flat ``transformer_blocks`` list, ``norm_out`` +
    ``proj_out``. But block forward uses ``attention_mask`` instead of
    ``encoder_hidden_states_mask``.
    """

    def enumerate_blocks(self, net: nn.Module) -> list[nn.Module]:
        transformer = self.get_transformer(net)
        return list(transformer.transformer_blocks)

    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> LensPivot:
        transformer = self.get_transformer(net)
        return LensPivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        def hook_fn(module: nn.Module, input: Any, kwargs: dict, output: Any) -> Any:
            if isinstance(output, tuple) and len(output) == 2:
                enc_hs, hs = output
            else:
                enc_hs = None
                hs = output if not isinstance(output, tuple) else output[-1]

            capture_store.clear()
            capture_store["hidden_states"] = hs
            capture_store["encoder_hidden_states"] = enc_hs if enc_hs is not None else hs
            capture_store["temb"] = kwargs.get("temb", None)
            capture_store["image_rotary_emb"] = kwargs.get("image_rotary_emb", None)
            capture_store["attention_mask"] = kwargs.get("attention_mask", None)
            return None

        return block.register_forward_hook(hook_fn, with_kwargs=True)

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            temb=hook_data["temb"],
            image_rotary_emb=hook_data["image_rotary_emb"],
            attention_mask=hook_data.get("attention_mask"),
        )

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach().requires_grad_(True)
        enc_hs = hook_data["encoder_hidden_states"]
        if enc_hs is not None:
            enc_hs = enc_hs.detach().requires_grad_(True)
        temb = hook_data["temb"]
        rope = hook_data["image_rotary_emb"]
        attn_mask = hook_data.get("attention_mask")

        all_blocks = list(transformer.transformer_blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        for blk in all_blocks[pos_k + 1 :]:
            enc_hs, hs = blk(
                hidden_states=hs,
                encoder_hidden_states=enc_hs,
                temb=temb,
                image_rotary_emb=rope,
                attention_mask=attn_mask,
            )
        h = transformer.norm_out(hs, temb)
        return transformer.proj_out(h)
