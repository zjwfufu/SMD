# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hunyuan3D 2.1 Shape backend for SimplifiedDistillation.

Implements :class:`Hunyuan3DPivot` and :class:`Hunyuan3DPivotBackend`
for the vendored ``HunYuanDiTPlain`` (a U-Net-style DiT with skip connections +
MoE). The latent is a dense ``[B, 4096, 64]`` vec2set token tensor.

Two architecture details drive this backend:

1. **U-Net skip connections.** The decoder blocks after the split point each
   consume an encoder activation popped (LIFO) from a forward-local stack. A
   plain ``register_forward_hook`` on the split block cannot see that stack, so
   instead of a torch hook we use the ``_pivot_capture`` mechanism baked into
   ``HunYuanDiTPlain._forward_native``: it records ``hidden_states`` (post-split
   block), the conditioning token ``c``, the cross-attn ``cond``, and the
   *remaining* ``skip_stack`` exactly when the split block runs. See ``dit.py``.

2. **Time/sign convention.** ``HunYuanDiTPlain.forward`` returns SMD ``flow``
   ( = ``-u`` ). The capture happens inside ``_forward_native`` (native velocity
   ``u`` space), so the pivot — which continues from captured native
   activations and reuses ``final_layer`` — produces native ``u`` and must apply
   the same ``-u`` negation at its output to stay in SMD ``flow`` space. This
   is the ONLY place the pivot touches the convention, mirroring the network.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from einops import rearrange

from ..common import PivotBackend

log = logging.getLogger(__name__)


def build_expanded_final_layer(
    orig_final_layer: nn.Module,
    n_grid: int,
    init_mode: str = "copy",
    init_plan: list[str] | None = None,
) -> nn.Module:
    """Deepcopy ``FinalLayer`` and expand its ``linear`` to ``out_channels*n_grid``.

    The ``norm_final`` is copied verbatim; the output ``linear`` is replaced by an
    expanded one initialized per *init_mode* (``copy`` tiles the teacher weights
    n_grid times so every slot starts as the teacher prediction; ``zero`` starts
    all slots at zero; ``hybrid`` chooses per-slot via *init_plan*).
    """
    new_fl = copy.deepcopy(orig_final_layer)
    orig_linear = orig_final_layer.linear
    out_f = orig_final_layer.out_channels
    in_f = orig_linear.in_features

    expanded = nn.Linear(
        in_f,
        out_f * n_grid,
        bias=orig_linear.bias is not None,
        device=orig_linear.weight.device,
        dtype=orig_linear.weight.dtype,
    )
    with torch.no_grad():
        if init_mode == "copy":
            expanded.weight.copy_(orig_linear.weight.repeat(n_grid, 1))
            if orig_linear.bias is not None:
                expanded.bias.copy_(orig_linear.bias.repeat(n_grid))
        elif init_mode == "zero":
            expanded.weight.zero_()
            if expanded.bias is not None:
                expanded.bias.zero_()
        elif init_mode == "hybrid":
            if init_plan is None or len(init_plan) != n_grid:
                raise ValueError(f"hybrid mode requires init_plan with {n_grid} entries, got {init_plan}")
            for slot, mode in enumerate(init_plan):
                rs, re = slot * out_f, (slot + 1) * out_f
                if mode == "copy":
                    expanded.weight[rs:re].copy_(orig_linear.weight)
                    if orig_linear.bias is not None:
                        expanded.bias[rs:re].copy_(orig_linear.bias)
                elif mode == "zero":
                    expanded.weight[rs:re].zero_()
                    if orig_linear.bias is not None:
                        expanded.bias[rs:re].zero_()
                else:
                    raise ValueError(f"Unknown slot init mode {mode!r} at slot {slot}")
        else:
            raise ValueError(f"Unknown proj_out_init_mode={init_mode!r}.")

    new_fl.linear = expanded
    new_fl.out_channels_mult = n_grid
    return new_fl


class Hunyuan3DPivot(nn.Module):
    """Deep pivot for the Hunyuan3D Shape DiT.

    Deepcopies the decoder blocks after *split_k* plus ``final_layer`` from the
    student ``HunYuanDiTPlain``, expanding ``final_layer.linear`` to output
    ``in_channels × n_grid`` velocity slots. Reuses the captured U-Net skip
    stack so the copied decoder blocks see the same skip connections the student
    would have.
    """

    def __init__(
        self,
        transformer: nn.Module,
        split_k: int,
        n_grid: int = 4,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ):
        super().__init__()
        self.n_grid = max(n_grid, 1)
        self.split_k = split_k

        all_blocks = list(transformer.blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        if not (0 <= pos_k < total):
            raise IndexError(
                f"split_k={split_k} resolves to position {pos_k}, "
                f"out of range for {total} blocks (valid: -{total}..{total - 1})"
            )
        self.pos_k = pos_k
        self.depth = total

        # Deepcopy decoder blocks pos_k+1 .. total-1 (these are decoder-half blocks,
        # each with skip_connection and possibly MoE).
        self.blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(pos_k + 1, total)])

        # How many skip values the copied blocks will pop, in LIFO order.
        # In HunYuanDiTPlain, block `layer` (> depth//2) pops one skip; the split
        # capture hands us the remaining stack so each copied decoder block pops one.
        self._n_skips_consumed = sum(1 for i in range(pos_k + 1, total) if i > total // 2)

        # Expanded final layer: in_channels -> in_channels * n_grid.
        self.final_layer = build_expanded_final_layer(
            transformer.final_layer, self.n_grid, proj_out_init_mode, proj_out_init_plan
        )
        self._base_out_features = transformer.final_layer.out_channels

        log.info(
            "Hunyuan3DPivot: split_k=%d (pos=%d), copied %d decoder blocks "
            "(%d consume skips), final_layer expanded %d -> %d (n_grid=%d, init=%s)",
            split_k,
            pos_k,
            len(self.blocks),
            self._n_skips_consumed,
            self._base_out_features,
            self._base_out_features * self.n_grid,
            self.n_grid,
            proj_out_init_mode,
        )

    @property
    def base_out_features(self) -> int:
        return self._base_out_features

    def forward(self, hidden_states, c, cond, skip_stack) -> torch.Tensor:
        """Continue the student tail from captured state.

        hidden_states : ``[B, N+1, hidden]`` post-split-block activations (token 0
                        is the prepended conditioning token).
        c             : ``[B, 1, hidden]`` timestep+pooled conditioning token.
        cond          : ``[B, L, context_dim]`` cross-attention conditioning.
        skip_stack    : list of remaining encoder activations (LIFO) for the copied
                        decoder blocks to pop, mirroring the student forward.

        Returns ``[B, N, in_channels * n_grid]`` SMD-flow velocity slots
        (token 0 stripped by ``final_layer``; ``-u`` negation applied).
        """
        x = hidden_states
        skip_stack = list(skip_stack)  # local copy; we pop from it
        # autocast(bf16) mirrors the network-level autocast in HunYuanDiTPlain's
        # student forward (network.py:180) and matches TrellisPivot.forward
        # (trellis.py). The captured ``hidden_states`` arrives as fp32 —
        # Hunyuan uses plain nn.LayerNorm, whose skip_norm output autocast leaves
        # in fp32, and that propagates to the split point. The pivot is
        # called OUTSIDE any autocast (model.py:_build_coeffs), so without this
        # wrapper the deepcopy blocks would run their bf16 FSDP params down an
        # fp32 path: dtype-mismatch errors when bare, and ~70GB of fp32 activations
        # under an outer autocast. Wrapping here keeps the tail in bf16 (~6GB).
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # The student forward pops a skip for every block with layer > depth//2.
            # Our copied blocks correspond to positions pos_k+1 .. depth-1; each that
            # satisfies (global_layer > depth//2) pops one skip.
            for offset, block in enumerate(self.blocks):
                global_layer = self.pos_k + 1 + offset
                skip = skip_stack.pop() if global_layer > self.depth // 2 else None
                x = block(x, c, cond, skip_value=skip)

            u = self.final_layer(x)  # [B, N, in_channels * n_grid], native velocity
        return -u  # native u -> SMD flow (mirror of HunYuanDiTPlain.forward)


class Hunyuan3DPivotBackend(PivotBackend):
    """Hunyuan3D-specific backend for SimplifiedDistillation (dense vec2set tokens)."""

    def get_transformer(self, net: nn.Module) -> nn.Module:
        return getattr(net, "model", net)

    def enumerate_blocks(self, net: nn.Module) -> list[nn.Module]:
        return list(self.get_transformer(net).blocks)

    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> Hunyuan3DPivot:
        return Hunyuan3DPivot(
            self.get_transformer(net),
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict):
        """Arm the DiT's ``_pivot_capture`` mechanism (NOT a torch hook).

        The U-Net skip stack is forward-local, so a module forward-hook cannot
        capture it. Instead we point the transformer's ``_pivot_capture`` dict at
        ``capture_store`` and set ``_pivot_split_k`` to the resolved split index;
        ``_forward_native`` fills it during the student forward. We return a small
        handle object exposing ``.remove()`` so the caller can disarm it like a
        real hook.
        """
        transformer = self.get_transformer(net)
        all_blocks = list(transformer.blocks)
        # Resolve the split index from the block identity.
        split_pos = next(i for i, b in enumerate(all_blocks) if b is block)
        transformer._pivot_capture = capture_store
        transformer._pivot_split_k = split_pos

        class _CaptureHandle:
            def remove(self_inner):
                transformer._pivot_capture = None
                transformer._pivot_split_k = None

        return _CaptureHandle()

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        """Run the pivot from captured state -> ``[B, N, C*n_grid]``."""
        return pivot(
            hook_data["hidden_states"],
            hook_data["c"],
            hook_data.get("cond", None),
            hook_data.get("skip_stack", []),
        )

    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        """Rearrange decoder output ``[B, N, n_grid*C]`` -> ``[B, n_grid, N, C]``.

        The downstream ``_build_coeffs`` does ``flow.reshape(B, n_grid, *x_t.shape[1:])``
        with ``x_t`` shape ``[B, N, C]``; producing a contiguous ``(B, n_grid, N, C)``
        tensor here makes that reshape a no-op view with the correct slot ordering
        (slot is the OUTER channel group, matching the expanded final layer).
        """
        n_grid = tokens.shape[-1] // self._infer_c(net, tokens)
        return rearrange(tokens, "b n (g c) -> b g n c", g=n_grid).contiguous()

    @staticmethod
    def _infer_c(net: nn.Module, tokens: torch.Tensor) -> int:
        return getattr(net, "latent_channels", getattr(net.model, "in_channels"))

    def rerun_output_layer(self, net: nn.Module, hook_data: dict, split_k: int) -> torch.Tensor:
        """Re-run the student tail (no n_grid expansion) for v_base alignment.

        Returns ``[B, N, in_channels]`` SMD flow, matching what the student's
        own forward would produce from the captured split state.
        """
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach()
        c = hook_data["c"].detach()
        cond = hook_data.get("cond", None)
        skip_stack = [s.detach() for s in hook_data.get("skip_stack", [])]

        all_blocks = list(transformer.blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k

        # autocast(bf16) for the same reason as Hunyuan3DPivot.forward:
        # the captured fp32 hidden_states must run through the bf16 tail blocks in
        # a bf16 path, or activation memory explodes (~70GB).
        x = hs
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for offset, blk in enumerate(all_blocks[pos_k + 1 :]):
                global_layer = pos_k + 1 + offset
                skip = skip_stack.pop() if global_layer > total // 2 else None
                x = blk(x, c, cond, skip_value=skip)
            u = transformer.final_layer(x)  # [B, N, in_channels], native velocity
        return -u  # -> SMD flow
