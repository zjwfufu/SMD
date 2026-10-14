# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..common import PivotBackend, Pivot, _enumerate_flux_blocks

# ---------------------------------------------------------------------------
# FluxPivotBackend
# ---------------------------------------------------------------------------


class FluxPivotBackend(PivotBackend):
    """Flux-specific backend for SimplifiedDistillation.

    Handles the dual block-list structure (``transformer_blocks`` +
    ``single_transformer_blocks``), keyword-arg hook, ``_unpack_latents``,
    and ``norm_out(hs, temb)`` + ``proj_out(h)`` output-layer logic.
    """

    def get_transformer(self, net: nn.Module) -> nn.Module:
        return getattr(net, "transformer", net)

    def enumerate_blocks(self, net: nn.Module) -> list[nn.Module]:
        transformer = self.get_transformer(net)
        return _enumerate_flux_blocks(transformer)

    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> Pivot:
        transformer = self.get_transformer(net)
        return Pivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        """Register a ``with_kwargs=True`` hook for Flux keyword-arg blocks.

        The hook writes ``hidden_states``, ``encoder_hidden_states``,
        ``temb``, and ``image_rotary_emb`` into *capture_store*.  It never
        modifies the output — gradient flow through the shared prefix is
        controlled externally by the caller (conditional ``torch.no_grad()``
        around the student forward).
        """

        def hook_fn(module: nn.Module, input: Any, kwargs: dict, output: Any) -> Any:
            # Flux block output: (encoder_hidden_states, hidden_states)
            if isinstance(output, tuple) and len(output) == 2:
                enc_hs, hs = output
            else:
                # Fallback for non-standard blocks
                enc_hs = None
                hs = output if not isinstance(output, tuple) else output[-1]

            # Extract temb and image_rotary_emb from kwargs OR positional args.
            # When gradient checkpointing is enabled, _gradient_checkpointing_func
            # calls the block with positional args, so temb/image_rotary_emb end up
            # in `input` (index 2 and 3) rather than `kwargs`.
            # Signature: forward(self, hidden_states, encoder_hidden_states, temb,
            #                    image_rotary_emb=None, joint_attention_kwargs=None)
            temb = kwargs.get("temb", None)
            if temb is None and len(input) > 2:
                temb = input[2]
            image_rotary_emb = kwargs.get("image_rotary_emb", None)
            if image_rotary_emb is None and len(input) > 3:
                image_rotary_emb = input[3]

            capture_store.clear()
            capture_store["hidden_states"] = hs
            capture_store["encoder_hidden_states"] = enc_hs if enc_hs is not None else hs
            capture_store["temb"] = temb
            capture_store["image_rotary_emb"] = image_rotary_emb

            return None

        return block.register_forward_hook(hook_fn, with_kwargs=True)

    def decode(self, pivot: Pivot, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            temb=hook_data["temb"],
            image_rotary_emb=hook_data["image_rotary_emb"],
        )

    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        if hasattr(net, "_unpack_latents"):
            return net._unpack_latents(tokens, *input_shape[1:])
        # Fallback: simple reshape
        B = tokens.shape[0]
        _C, H, W = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
        return tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        """Re-run student blocks after the hook + norm_out + proj_out.

        Returns raw token output ``[B, seq, out_dim]`` before spatial
        unpacking.  Gradients flow through blocks ``split_k+1 .. end``,
        ``norm_out``, and ``proj_out``.
        """
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach().requires_grad_(True)
        enc_hs = hook_data["encoder_hidden_states"]
        if enc_hs is not None:
            enc_hs = enc_hs.detach().requires_grad_(True)
        temb = hook_data["temb"]
        rope = hook_data["image_rotary_emb"]

        # Run blocks after the split point — with grad.
        all_blocks = _enumerate_flux_blocks(transformer)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        for blk in all_blocks[pos_k + 1 :]:
            enc_hs, hs = blk(
                hidden_states=hs,
                encoder_hidden_states=enc_hs,
                temb=temb,
                image_rotary_emb=rope,
            )

        h = transformer.norm_out(hs, temb)
        return transformer.proj_out(h)
