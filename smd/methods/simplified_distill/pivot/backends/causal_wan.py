# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal Wan Pivot backend with prefix-KV-aware copied tail blocks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .wan import WanPivot, WanPivotBackend


def _causal_block_inputs(
    causal_context: dict[str, Any],
    block_index: int,
) -> tuple[dict[str, Any], int, Any]:
    attention_kwargs = dict(causal_context["attention_kwargs"])
    cache_tag = str(attention_kwargs.get("cache_tag", "pos"))
    cur_start_frame = int(attention_kwargs.get("cur_start_frame", 0))
    frame_seqlen = int(attention_kwargs["frame_seqlen"])
    proper_cache_len = cur_start_frame * frame_seqlen

    self_kv_list = causal_context.get("external_self_kv_list")
    if self_kv_list is not None:
        entry = self_kv_list[block_index].get(cache_tag)
        if entry is not None:
            attention_kwargs["external_self_kv"] = {
                cache_tag: {
                    "k": entry["k"],
                    "v": entry["v"],
                    "len": proper_cache_len,
                }
            }

    cross_kv_list = causal_context.get("external_cross_kv_list")
    if cross_kv_list is not None:
        attention_kwargs["external_cross_kv"] = cross_kv_list[block_index]

    block_mask = None if bool(attention_kwargs.get("is_ar", False)) else causal_context.get("block_mask")
    return attention_kwargs, proper_cache_len, block_mask


def _causal_output_modulation(
    *,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    scale_shift_table: torch.Tensor,
    norm_out: nn.Module,
) -> torch.Tensor:
    if temb.ndim != 3:
        raise ValueError(f"Causal Wan requires per-frame temb [B,F,D], got {tuple(temb.shape)}")
    shift, scale = (scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
    shift = shift.squeeze(2)
    scale = scale.squeeze(2)
    frame_seqlen = hidden_states.shape[1] // temb.shape[1]
    normalized = norm_out(hidden_states.float()).unflatten(1, (temb.shape[1], frame_seqlen))
    return (normalized * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)


class CausalWanPivot(WanPivot):
    """Wan Pivot whose copied blocks read the causal student's prefix caches."""

    def __init__(
        self,
        transformer: nn.Module,
        split_k: int,
        n_grid: int = 10,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ):
        super().__init__(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )
        total = len(transformer.blocks)
        self.split_pos = split_k if split_k >= 0 else total + split_k

    def forward_hidden(
        self,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        norm_temb: bool,
        temb: torch.Tensor,
        causal_context: dict[str, Any],
    ) -> torch.Tensor:
        """Run the copied causal tail and return its final hidden tokens."""
        for offset, block in enumerate(self.blocks):
            block_index = self.split_pos + 1 + offset
            attention_kwargs, cache_start_idx, block_mask = _causal_block_inputs(
                causal_context,
                block_index,
            )
            # The copied tail has independent K/V projections. Student text
            # caches would make no-grad collocation differ from live replay.
            attention_kwargs.pop("external_cross_kv", None)
            hidden_states = block(
                hidden_states,
                timestep_proj,
                encoder_hidden_states,
                rotary_emb,
                norm_temb,
                block_mask,
                attention_kwargs,
                cache_start_idx=cache_start_idx,
            )

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        norm_temb: bool,
        temb: torch.Tensor,
        causal_context: dict[str, Any],
        return_hidden: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.forward_hidden(
            hidden_states=hidden_states,
            timestep_proj=timestep_proj,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
            norm_temb=norm_temb,
            temb=temb,
            causal_context=causal_context,
        )
        if return_hidden:
            return hidden_states
        hidden_states = _causal_output_modulation(
            hidden_states=hidden_states,
            temb=temb,
            scale_shift_table=self.scale_shift_table,
            norm_out=self.norm_out,
        )
        return self.proj_out(hidden_states)


class CausalWanPivotBackend(WanPivotBackend):
    """Pivot backend isolated to :class:`CausalWan` networks."""

    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> CausalWanPivot:
        return CausalWanPivot(
            self.get_transformer(net),
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> Any:
        transformer = self.get_transformer(net)
        original_prepare = transformer.classify_forward_prepare

        def wrapped_prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            capture_store["temb"] = result[5]
            attention_kwargs = kwargs.get("attention_kwargs")
            if attention_kwargs is None and len(args) >= 6:
                attention_kwargs = args[5]
            if not isinstance(attention_kwargs, dict) or "frame_seqlen" not in attention_kwargs:
                raise RuntimeError("Causal Wan Pivot capture did not receive causal attention metadata.")
            capture_store["causal_context"] = {
                "attention_kwargs": dict(attention_kwargs),
                "external_self_kv_list": getattr(transformer, "_external_self_kv_list", None),
                "external_cross_kv_list": getattr(transformer, "_external_cross_kv_list", None),
                "block_mask": getattr(transformer, "block_mask", None),
            }
            return result

        transformer.classify_forward_prepare = wrapped_prepare

        def hook_fn(module: nn.Module, args: tuple, output: Any) -> None:
            del module, output
            capture_store["hidden_states"] = args[0]
            capture_store["timestep_proj"] = args[1]
            capture_store["encoder_hidden_states"] = args[2]
            capture_store["rotary_emb"] = args[3]
            capture_store["norm_temb"] = args[4]

        forward_handle = block.register_forward_hook(hook_fn)

        class _CausalWanCaptureHandle:
            def remove(self_inner) -> None:
                forward_handle.remove()
                transformer.classify_forward_prepare = original_prepare

        return _CausalWanCaptureHandle()

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            timestep_proj=hook_data["timestep_proj"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            rotary_emb=hook_data["rotary_emb"],
            norm_temb=hook_data["norm_temb"],
            temb=hook_data["temb"],
            causal_context=hook_data["causal_context"],
        )

    def decode_hidden(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            timestep_proj=hook_data["timestep_proj"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            rotary_emb=hook_data["rotary_emb"],
            norm_temb=hook_data["norm_temb"],
            temb=hook_data["temb"],
            causal_context=hook_data["causal_context"],
            return_hidden=True,
        )

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        transformer = self.get_transformer(net)
        hidden_states = hook_data["hidden_states"].detach().requires_grad_(True)
        timestep_proj = hook_data["timestep_proj"]
        encoder_hidden_states = hook_data["encoder_hidden_states"]
        rotary_emb = hook_data["rotary_emb"]
        norm_temb = hook_data["norm_temb"]
        causal_context = hook_data["causal_context"]

        blocks = list(transformer.blocks)
        split_pos = split_k if split_k >= 0 else len(blocks) + split_k
        for block_index, block in enumerate(blocks[split_pos + 1 :], start=split_pos + 1):
            attention_kwargs, cache_start_idx, block_mask = _causal_block_inputs(
                causal_context,
                block_index,
            )
            hidden_states = block(
                hidden_states,
                timestep_proj,
                encoder_hidden_states,
                rotary_emb,
                norm_temb,
                block_mask,
                attention_kwargs,
                cache_start_idx=cache_start_idx,
            )

        hidden_states = _causal_output_modulation(
            hidden_states=hidden_states,
            temb=hook_data["temb"],
            scale_shift_table=transformer.scale_shift_table,
            norm_out=transformer.norm_out,
        )
        return transformer.proj_out(hidden_states)
