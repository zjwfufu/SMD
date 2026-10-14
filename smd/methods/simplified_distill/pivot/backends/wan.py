# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import types
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from ..common import PivotBackend

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WanPivot
# ---------------------------------------------------------------------------


class WanPivot(nn.Module):
    """Deep pivot for Wan-style transformers (3D video).

    From the student transformer, deepcopy the last few blocks (after the
    split point *split_k*), ``norm_out``, ``scale_shift_table``, and
    ``proj_out``.  ``proj_out`` is expanded to output
    ``n_grid × base_out_features`` so it predicts ``n_grid`` velocity
    anchors in a single forward pass.

    Wan blocks use **positional** args::

        block(hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, norm_temb)

    and return a single tensor (not a tuple like Flux).
    """

    def __init__(
        self,
        transformer: nn.Module,
        split_k: int,
        n_grid: int = 10,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ):
        super().__init__()
        self.n_grid = max(n_grid, 1)
        self.split_k = split_k

        # Wan: single flat list of blocks
        all_blocks = list(transformer.blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        if not (0 <= pos_k < total):
            raise IndexError(
                f"split_k={split_k} resolves to position {pos_k}, "
                f"which is out of range for {total} blocks (valid: -{total}..{total - 1})"
            )

        # Deepcopy blocks pos_k+1 .. total-1
        self.blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(pos_k + 1, total)])

        # Deepcopy norm_out (plain LayerNorm for Wan)
        self.norm_out = copy.deepcopy(transformer.norm_out)

        # Deepcopy scale_shift_table (nn.Parameter, shape [2, D] for global modulation)
        self.scale_shift_table = nn.Parameter(transformer.scale_shift_table.data.clone())

        # Deepcopy & expand proj_out for n_grid
        orig = transformer.proj_out
        self._base_out_features = orig.out_features
        self.proj_out = nn.Linear(
            orig.in_features,
            orig.out_features * self.n_grid,
            bias=orig.bias is not None,
            device=orig.weight.device,
            dtype=orig.weight.dtype,
        )
        self._proj_out_init_mode = proj_out_init_mode
        with torch.no_grad():
            if proj_out_init_mode == "copy":
                self.proj_out.weight.copy_(orig.weight.repeat(self.n_grid, 1))
                if orig.bias is not None:
                    self.proj_out.bias.copy_(orig.bias.repeat(self.n_grid))
            elif proj_out_init_mode == "zero":
                self.proj_out.weight.zero_()
                if self.proj_out.bias is not None:
                    self.proj_out.bias.zero_()
            elif proj_out_init_mode == "hybrid":
                if proj_out_init_plan is None or len(proj_out_init_plan) != self.n_grid:
                    raise ValueError(
                        f"hybrid mode requires proj_out_init_plan with {self.n_grid} entries, got {proj_out_init_plan}"
                    )
                out_f = orig.out_features
                for slot, mode in enumerate(proj_out_init_plan):
                    row_start = slot * out_f
                    row_end = (slot + 1) * out_f
                    if mode == "copy":
                        self.proj_out.weight[row_start:row_end].copy_(orig.weight)
                        if orig.bias is not None:
                            self.proj_out.bias[row_start:row_end].copy_(orig.bias)
                    elif mode == "zero":
                        self.proj_out.weight[row_start:row_end].zero_()
                        if orig.bias is not None:
                            self.proj_out.bias[row_start:row_end].zero_()
                    else:
                        raise ValueError(f"Unknown slot init mode {mode!r} at slot {slot}")
            else:
                raise ValueError(f"Unknown proj_out_init_mode={proj_out_init_mode!r}. Use 'copy', 'zero', or 'hybrid'.")

        log.info(
            "WanPivot: split_k=%d (pos=%d), copied %d blocks, proj_out expanded %d → %d (n_grid=%d, init=%s)",
            split_k,
            pos_k,
            len(self.blocks),
            self._base_out_features,
            orig.out_features * self.n_grid,
            self.n_grid,
            proj_out_init_mode,
        )

    @property
    def base_out_features(self) -> int:
        """Original (un-expanded) ``proj_out.out_features``."""
        return self._base_out_features

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        norm_temb: bool,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """Run copied blocks → modulated norm_out → expanded proj_out.

        All arguments follow the Wan block interface except ``temb`` which
        is used for the final norm_out modulation.

        Returns
        -------
        torch.Tensor
            ``[B, seq, base_out_features * n_grid]``
        """
        for block in self.blocks:
            hidden_states = block(hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, norm_temb)

        # Global modulation (same logic as Wan classify_forward else branch)
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# WanPivotBackend
# ---------------------------------------------------------------------------


class WanPivotBackend(PivotBackend):
    """Wan-specific backend for SimplifiedDistillation (3D video models).

    Handles the single flat block list (``transformer.blocks``),
    positional-arg block forward, 3D unpatchify, and the modulated
    ``norm_out`` + ``proj_out`` output-layer logic.
    """

    def get_transformer(self, net: nn.Module) -> nn.Module:
        return getattr(net, "transformer", net)

    def enumerate_blocks(self, net: nn.Module) -> list[nn.Module]:
        transformer = self.get_transformer(net)
        return list(transformer.blocks)

    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> WanPivot:
        transformer = self.get_transformer(net)
        return WanPivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def configure_pivot_blocks(
        self,
        pivot: WanPivot,
        net: nn.Module,
        split_k: int,
        num_blocks: int | None,
        init_mode: str,
        init_seed: int,
    ) -> None:
        transformer = self.get_transformer(net)
        student_blocks = list(transformer.blocks)
        pos = split_k if split_k >= 0 else len(student_blocks) + split_k
        natural_count = len(student_blocks) - pos - 1
        target_count = natural_count if num_blocks is None else num_blocks

        if target_count < 0:
            raise ValueError(f"pivot_num_blocks must be >= 0, got {target_count}")
        if init_mode == "pretrained":
            if target_count != natural_count:
                raise ValueError(
                    "pivot_block_init='pretrained' requires pivot_num_blocks to match the natural "
                    f"student tail ({natural_count} blocks for pivot_hook_layer={split_k}), got {target_count}."
                )
            return
        if init_mode != "random":
            raise ValueError(f"pivot_block_init must be 'pretrained' or 'random', got {init_mode!r}")
        if init_seed < 0:
            raise ValueError(f"pivot_block_init_seed must be >= 0, got {init_seed}")

        reference = student_blocks[-1]
        reference_parameter = next(reference.parameters())
        devices = [reference_parameter.device.index] if reference_parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(init_seed)
            random_blocks = [self._random_block_from_transformer(transformer) for _ in range(target_count)]
        pivot.blocks = nn.ModuleList(random_blocks)
        log.info(
            "Wan Pivot: replaced natural %d-block tail with %d random blocks (seed=%d)",
            natural_count,
            target_count,
            init_seed,
        )

    @staticmethod
    def _random_block_from_transformer(transformer: nn.Module) -> WanTransformerBlock:
        config = transformer.config
        reference = transformer.blocks[-1]
        reference_parameter = next(reference.parameters())
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(reference_parameter.dtype)
            with torch.device(reference_parameter.device):
                block = WanTransformerBlock(
                    dim=config.num_attention_heads * config.attention_head_dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    qk_norm=config.qk_norm,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    added_kv_proj_dim=config.added_kv_proj_dim,
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        if not isinstance(reference.forward, types.MethodType):
            raise TypeError(f"Expected Wan block forward to be a bound method, got {type(reference.forward).__name__}.")
        block.forward = types.MethodType(reference.forward.__func__, block)
        block.train(reference.training)
        return block

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> Any:
        """Install the split-block capture for Wan blocks.

        Wan blocks take ``(hidden_states, timestep_proj, encoder_hidden_states,
        rotary_emb, norm_temb)`` as positional args and return a single tensor.

        ``temb`` (used for norm_out modulation) is captured separately by wrapping
        ``classify_forward_prepare`` on the transformer. The returned handle removes
        the block forward hook **and** restores the original
        ``classify_forward_prepare`` so no monkeypatch leaks past the capture scope.
        """
        transformer = self.get_transformer(net)

        # Wrap classify_forward_prepare to capture temb (restored on handle.remove()).
        _orig_prepare = transformer.classify_forward_prepare

        def wrapped_prepare(*args, **kwargs):
            result = _orig_prepare(*args, **kwargs)
            # result is (hidden_states, timestep_proj, r_timestep_proj,
            #            encoder_hidden_states, encoder_hidden_states_image, temb, rotary_emb)
            capture_store["temb"] = result[5]
            return result

        transformer.classify_forward_prepare = wrapped_prepare

        def hook_fn(module: nn.Module, args: tuple, output: Any) -> None:
            # Wan block positional args:
            # (hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, norm_temb)
            capture_store["hidden_states"] = args[0]  # input hs to this block
            capture_store["timestep_proj"] = args[1]
            capture_store["encoder_hidden_states"] = args[2]
            capture_store["rotary_emb"] = args[3]
            capture_store["norm_temb"] = args[4]

        forward_handle = block.register_forward_hook(hook_fn)

        class _WanCaptureHandle:
            def remove(self_inner) -> None:
                forward_handle.remove()
                transformer.classify_forward_prepare = _orig_prepare

        return _WanCaptureHandle()

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            timestep_proj=hook_data["timestep_proj"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            rotary_emb=hook_data["rotary_emb"],
            norm_temb=hook_data["norm_temb"],
            temb=hook_data["temb"],
        )

    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        """3D unpatchify: tokens → [B, C, T, H, W].

        ``input_shape`` is the latent shape ``(C, T, H, W)``.
        """
        transformer = self.get_transformer(net)
        p_t, p_h, p_w = transformer.config.patch_size
        _C, T, H, W = int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
        post_T = T // p_t
        post_H = H // p_h
        post_W = W // p_w
        B = tokens.shape[0]
        # tokens: [B, post_T*post_H*post_W, C_out*p_t*p_h*p_w] (or expanded by n_grid)
        hs = tokens.reshape(B, post_T, post_H, post_W, p_t, p_h, p_w, -1)
        hs = hs.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C_out, post_T, p_t, post_H, p_h, post_W, p_w]
        return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)  # [B, C_out, T, H, W]

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        """Re-run student blocks after the hook + modulated norm_out + proj_out.

        Returns raw token output ``[B, seq, out_dim]`` before spatial unpacking.
        """
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach().requires_grad_(True)
        timestep_proj = hook_data["timestep_proj"]
        enc_hs = hook_data["encoder_hidden_states"]
        rotary_emb = hook_data["rotary_emb"]
        norm_temb = hook_data["norm_temb"]
        temb = hook_data["temb"]

        # Run blocks after the split point — with grad.
        all_blocks = list(transformer.blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        for blk in all_blocks[pos_k + 1 :]:
            hs = blk(hs, timestep_proj, enc_hs, rotary_emb, norm_temb)

        # Global modulation + proj_out (matching classify_forward else branch)
        shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hs.device)
        scale = scale.to(hs.device)
        h = (transformer.norm_out(hs.float()) * (1 + scale) + shift).type_as(hs)
        return transformer.proj_out(h)
