# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn

from ..common import PivotBackend

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SD35Pivot
# ---------------------------------------------------------------------------


class SD35Pivot(nn.Module):
    """Deep pivot for SD3.5-Medium (MMDiT-X JointTransformerBlocks).

    SD3's JointTransformerBlock forward signature::

        block(hidden_states, encoder_hidden_states, temb, joint_attention_kwargs)
        → (encoder_hidden_states, hidden_states)

    Key differences from Flux/QwenImage:
    - **No RoPE**: SD3 uses 2D sin-cos positional embedding baked into pos_embed.
    - **No encoder_hidden_states_mask**: No mask argument needed.
    - **context_pre_only block**: The last block (block 23) has ``context_pre_only=True``
      and returns ``(None, hidden_states)`` — subsequent blocks only have image tokens.
    - **norm_out**: ``AdaLayerNormContinuous(hs, temb)`` — same calling convention as Flux.
    - **proj_out → unpatchify**: Output must be unpatchified ``nhwpqc → nchpwq``.
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

        # SD3: single flat list of transformer_blocks
        all_blocks = list(transformer.transformer_blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        if not (0 <= pos_k < total):
            raise IndexError(
                f"split_k={split_k} resolves to position {pos_k}, "
                f"which is out of range for {total} blocks (valid: -{total}..{total - 1})"
            )

        # Deepcopy blocks pos_k+1 .. total-1
        self.blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(pos_k + 1, total)])

        # Track which blocks are context_pre_only (e.g. block 23)
        self._context_pre_only = [blk.context_pre_only for blk in self.blocks]

        # Deepcopy norm_out (AdaLayerNormContinuous)
        self.norm_out = copy.deepcopy(transformer.norm_out)

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
            "SD35Pivot: split_k=%d (pos=%d), copied %d blocks, proj_out expanded %d → %d (n_grid=%d, init=%s)",
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
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """Run copied blocks → norm_out → expanded proj_out.

        Args:
            hidden_states: Image token features ``[B, seq, 1536]``.
            encoder_hidden_states: Text context features (may be None after
                a context_pre_only block).
            temb: Timestep + pooled text embedding ``[B, 1536]``.

        Returns:
            ``[B, seq, base_out_features * n_grid]``
        """
        for i, block in enumerate(self.blocks):
            enc_hs_out, hs_out = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
            hidden_states = hs_out
            # context_pre_only blocks return (None, hs) — stop passing context
            if not self._context_pre_only[i]:
                encoder_hidden_states = enc_hs_out

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# SD35PivotBackend
# ---------------------------------------------------------------------------


class SD35PivotBackend(PivotBackend):
    """SD3.5-Medium backend for SimplifiedDistillation.

    SD3.5-M is the simplest architecture to hook into:
    - Single flat list of ``transformer_blocks`` (24 JointTransformerBlocks)
    - No RoPE — positional embedding is baked into ``pos_embed`` module
    - Hook only needs to capture ``(hidden_states, encoder_hidden_states, temb)``
    - Unpatchify: ``nhwpqc → nchpwq`` with ``patch_size=2``
    """

    def get_transformer(self, net: nn.Module) -> nn.Module:
        return getattr(net, "transformer", net)

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
    ) -> SD35Pivot:
        transformer = self.get_transformer(net)
        return SD35Pivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        """Register a hook that captures ``(hidden_states, encoder_hidden_states, temb)``.

        SD3 blocks are called with keyword args. When gradient checkpointing is
        enabled, ``_gradient_checkpointing_func`` passes args positionally:
        ``block(hidden_states, encoder_hidden_states, temb, joint_attention_kwargs)``.
        """

        def hook_fn(module: nn.Module, input: Any, kwargs: dict, output: Any) -> Any:
            if isinstance(output, tuple) and len(output) == 2:
                enc_hs, hs = output
            else:
                enc_hs = None
                hs = output if not isinstance(output, tuple) else output[-1]

            # Extract temb from kwargs or positional args
            temb = kwargs.get("temb", None)
            if temb is None and len(input) > 2:
                temb = input[2]

            capture_store.clear()
            capture_store["hidden_states"] = hs
            capture_store["encoder_hidden_states"] = enc_hs
            capture_store["temb"] = temb

            return None

        return block.register_forward_hook(hook_fn, with_kwargs=True)

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            temb=hook_data["temb"],
        )

    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        """Unpatchify SD3 token output to spatial tensor.

        SD3 unpatchify: ``[B, seq, patch*patch*C] → nhwpqc → nchpwq → [B, C, H, W]``
        where ``patch_size=2``, ``C = out_channels (16 for base, 16*n_grid for pivot)``.
        """
        transformer = self.get_transformer(net)
        patch_size = transformer.config.patch_size  # 2
        # input_shape = (C_latent, H_latent, W_latent) e.g. (16, 128, 128)
        _C_latent, H, W = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
        patched_h = H // patch_size
        patched_w = W // patch_size

        B = tokens.shape[0]
        # tokens: [B, patched_h * patched_w, patch*patch*C_expanded]
        C_expanded = tokens.shape[-1] // (patch_size * patch_size)

        tokens = tokens.reshape(B, patched_h, patched_w, patch_size, patch_size, C_expanded)
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(B, C_expanded, H, W)

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        """Re-run student blocks after the hook + norm_out + proj_out.

        Returns raw token output ``[B, seq, out_dim]`` before spatial unpacking.
        """
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach().requires_grad_(True)
        enc_hs = hook_data["encoder_hidden_states"]
        if enc_hs is not None:
            enc_hs = enc_hs.detach().requires_grad_(True)
        temb = hook_data["temb"]

        all_blocks = list(transformer.transformer_blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        for blk in all_blocks[pos_k + 1 :]:
            out = blk(
                hidden_states=hs,
                encoder_hidden_states=enc_hs,
                temb=temb,
            )
            if isinstance(out, tuple) and len(out) == 2:
                new_enc_hs, hs = out
                if new_enc_hs is not None:
                    enc_hs = new_enc_hs
            else:
                hs = out

        h = transformer.norm_out(hs, temb)
        return transformer.proj_out(h)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
