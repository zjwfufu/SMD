# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm

from ..common import Pivot, _enumerate_flux_blocks
from .flux import FluxPivotBackend

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QwenImagePivot
# ---------------------------------------------------------------------------


class QwenImagePivot(Pivot):
    """Deep pivot for QwenImage-style transformers.

    Identical to ``Pivot`` (Flux) except that block forward calls pass
    extra ``encoder_hidden_states_mask`` and ``joint_attention_kwargs``.
    QwenImage has a single list of ``transformer_blocks`` (no
    ``single_transformer_blocks``).
    """

    @staticmethod
    def _collect_all_blocks(transformer: nn.Module) -> list[nn.Module]:
        """QwenImage: single flat list of transformer_blocks."""
        return list(transformer.transformer_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        joint_attention_kwargs: dict | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run copied blocks → norm_out → expanded proj_out.

        Same as Flux Pivot but passes ``encoder_hidden_states_mask``
        and ``joint_attention_kwargs`` to each block.
        """
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                modulate_index=modulate_index,
            )
        if self.blocks and getattr(self.blocks[0], "zero_cond_t", False):
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# QwenImagePivotBackend
# ---------------------------------------------------------------------------


class QwenImagePivotBackend(FluxPivotBackend):
    """QwenImage-specific backend for SimplifiedDistillation.

    QwenImage is structurally very similar to Flux (dual-stream MMDiT,
    ``transformer_blocks`` list, ``norm_out(hs, temb)`` + ``proj_out``),
    but differs in two ways:
    1. No ``single_transformer_blocks`` — only a single flat block list.
    2. Blocks require ``encoder_hidden_states_mask`` and
       ``joint_attention_kwargs`` as additional keyword arguments.
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
    ) -> QwenImagePivot:
        transformer = self.get_transformer(net)
        return QwenImagePivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def configure_pivot_blocks(
        self,
        pivot: QwenImagePivot,
        net: nn.Module,
        split_k: int,
        num_blocks: int | None,
        init_mode: str,
        init_seed: int,
    ) -> None:
        transformer = self.get_transformer(net)
        student_blocks = list(transformer.transformer_blocks)
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

        template = student_blocks[-1]
        devices = [template.img_mod[1].weight.device.index] if template.img_mod[1].weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(init_seed)
            random_blocks = [self._random_block_from_template(template) for _ in range(target_count)]
        pivot.blocks = nn.ModuleList(random_blocks)
        log.info(
            "QwenImage Pivot: replaced natural %d-block tail with %d random blocks (seed=%d)",
            natural_count,
            target_count,
            init_seed,
        )

    @staticmethod
    def _random_block_from_template(template: nn.Module) -> nn.Module:
        block = copy.deepcopy(template)
        with torch.no_grad():
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
                elif isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
        return block

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        """Register a hook that also captures ``encoder_hidden_states_mask``."""

        def hook_fn(module: nn.Module, input: Any, kwargs: dict, output: Any) -> Any:
            if isinstance(output, tuple) and len(output) == 2:
                enc_hs, hs = output
            else:
                enc_hs = None
                hs = output if not isinstance(output, tuple) else output[-1]

            capture_store.clear()
            capture_store["hidden_states"] = hs
            capture_store["encoder_hidden_states"] = enc_hs if enc_hs is not None else hs
            capture_store["encoder_hidden_states_mask"] = kwargs.get(
                "encoder_hidden_states_mask",
                input[2] if len(input) > 2 else None,
            )
            capture_store["temb"] = kwargs.get("temb", input[3] if len(input) > 3 else None)
            capture_store["image_rotary_emb"] = kwargs.get("image_rotary_emb", input[4] if len(input) > 4 else None)
            capture_store["joint_attention_kwargs"] = kwargs.get(
                "joint_attention_kwargs",
                input[5] if len(input) > 5 else None,
            )
            capture_store["modulate_index"] = kwargs.get("modulate_index", input[6] if len(input) > 6 else None)

            return None

        return block.register_forward_hook(hook_fn, with_kwargs=True)

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        return pivot(
            hidden_states=hook_data["hidden_states"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            temb=hook_data["temb"],
            image_rotary_emb=hook_data["image_rotary_emb"],
            encoder_hidden_states_mask=hook_data.get("encoder_hidden_states_mask"),
            joint_attention_kwargs=hook_data.get("joint_attention_kwargs"),
            modulate_index=hook_data.get("modulate_index"),
        )


# ---------------------------------------------------------------------------
# QwenImageEditPivotBackend
# ---------------------------------------------------------------------------


class QwenImageEditPivotBackend(QwenImagePivotBackend):
    """QwenImageEdit-specific backend for SimplifiedDistillation.

    Extends QwenImagePivotBackend to handle the source image latent tokens
    that are concatenated to noise tokens in the sequence dimension.

    The edit model's forward pass concatenates noise + source tokens, so the
    hook captures activations for ALL tokens. The pivot also runs on all
    tokens. But we only want noise-token predictions for the loss, so we
    slice the pivot output to ``[:, :noise_seq_len, :]`` before unpacking.
    """

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        """Register a hook that also captures the noise sequence length."""

        def hook_fn(module: nn.Module, input: Any, kwargs: dict, output: Any) -> Any:
            if isinstance(output, tuple) and len(output) == 2:
                enc_hs, hs = output
            else:
                enc_hs = None
                hs = output if not isinstance(output, tuple) else output[-1]

            capture_store.clear()
            capture_store["hidden_states"] = hs
            capture_store["encoder_hidden_states"] = enc_hs if enc_hs is not None else hs
            capture_store["encoder_hidden_states_mask"] = kwargs.get(
                "encoder_hidden_states_mask",
                input[2] if len(input) > 2 else None,
            )
            capture_store["temb"] = kwargs.get("temb", input[3] if len(input) > 3 else None)
            capture_store["image_rotary_emb"] = kwargs.get("image_rotary_emb", input[4] if len(input) > 4 else None)
            capture_store["joint_attention_kwargs"] = kwargs.get(
                "joint_attention_kwargs",
                input[5] if len(input) > 5 else None,
            )
            capture_store["modulate_index"] = kwargs.get("modulate_index", input[6] if len(input) > 6 else None)
            # Capture noise_seq_len from the network for output slicing
            capture_store["noise_seq_len"] = getattr(net, "_noise_seq_len", None)

            return None

        return block.register_forward_hook(hook_fn, with_kwargs=True)

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        """Run pivot and slice output to noise tokens only."""
        raw_output = pivot(
            hidden_states=hook_data["hidden_states"],
            encoder_hidden_states=hook_data["encoder_hidden_states"],
            temb=hook_data["temb"],
            image_rotary_emb=hook_data["image_rotary_emb"],
            encoder_hidden_states_mask=hook_data.get("encoder_hidden_states_mask"),
            joint_attention_kwargs=hook_data.get("joint_attention_kwargs"),
            modulate_index=hook_data.get("modulate_index"),
        )
        # Slice to noise tokens only (discard source image token predictions)
        noise_seq_len = hook_data.get("noise_seq_len")
        if noise_seq_len is not None:
            raw_output = raw_output[:, :noise_seq_len, :]
        return raw_output

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        """Re-run student blocks after the hook + norm_out + proj_out.

        Like ``decode()``, the output is sliced to ``noise_seq_len`` so the
        source-image tokens (concatenated in the sequence dim by the edit
        model's forward) are discarded before the caller unpacks to spatial.
        """
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach().requires_grad_(True)
        enc_hs = hook_data["encoder_hidden_states"]
        if enc_hs is not None:
            enc_hs = enc_hs.detach().requires_grad_(True)
        temb = hook_data["temb"]
        rope = hook_data["image_rotary_emb"]
        enc_mask = hook_data.get("encoder_hidden_states_mask")
        attn_kwargs = hook_data.get("joint_attention_kwargs")
        modulate_index = hook_data.get("modulate_index")

        all_blocks = _enumerate_flux_blocks(transformer)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        for blk in all_blocks[pos_k + 1 :]:
            enc_hs, hs = blk(
                hidden_states=hs,
                encoder_hidden_states=enc_hs,
                encoder_hidden_states_mask=enc_mask,
                temb=temb,
                image_rotary_emb=rope,
                joint_attention_kwargs=attn_kwargs,
                modulate_index=modulate_index,
            )

        # diffusers: norm_out(hs, temb) + proj_out(h)
        # BFL: final_layer(hs, vec) — combines norm + proj
        if all_blocks[pos_k + 1 :] and getattr(all_blocks[pos_k + 1], "zero_cond_t", False):
            temb = temb.chunk(2, dim=0)[0]
        if hasattr(transformer, "norm_out"):
            h = transformer.norm_out(hs, temb)
            raw_output = transformer.proj_out(h)
        else:
            raw_output = transformer.final_layer(hs, temb)

        # Slice to noise tokens only (mirror decode()) — discard source tokens.
        noise_seq_len = hook_data.get("noise_seq_len")
        if noise_seq_len is not None:
            raw_output = raw_output[:, :noise_seq_len, :]
        return raw_output
