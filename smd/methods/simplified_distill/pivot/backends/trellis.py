# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 backend for SimplifiedDistillation.

Implements :class:`TrellisPivot` and :class:`TrellisPivotBackend` for
the TRELLIS.2 Shape DiT (``SLatFlowModel``), which operates on SparseTensor
inputs (variable-length voxel features).

Architecture of SLatFlowModel::

    input_layer:  SparseLinear(32 → 1536)
    blocks:       [ModulatedSparseTransformerCrossBlock × 30]
    out_layer:    LayerNorm → SparseLinear(1536 → 32)

The pivot deepcopy's tail blocks and replaces ``out_layer`` with an
expanded version that outputs ``32 × n_grid`` channels.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import PivotBackend

log = logging.getLogger(__name__)


class TrellisPivot(nn.Module):
    """Deep pivot for TRELLIS.2 sparse-voxel transformers.

    From the student transformer (``SLatFlowModel``), deepcopy the last few
    blocks after *split_k* and the output layer, expanding
    ``out_layer`` to output ``out_channels × n_grid``.

    Parameters
    ----------
    transformer : nn.Module
        The student's ``SLatFlowModel``.
    split_k : int
        Block index (supports negative indexing) at which the student is
        split.  Blocks ``split_k + 1 .. num_blocks - 1`` are deepcopy'd.
    n_grid : int
        Number of velocity predictions.
    proj_out_init_mode : str
        Initialization mode for the expanded output layer.
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
                f"which is out of range for {total} blocks (valid: -{total}..{total - 1})"
            )

        # Deepcopy blocks pos_k+1 .. total-1
        self.blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(pos_k + 1, total)])

        # TRELLIS.2 output: LayerNorm on feats + SparseLinear (out_layer)
        # We deepcopy out_layer and expand it for n_grid outputs
        orig_out = transformer.out_layer
        self._base_out_features = orig_out.out_features

        # Create expanded output layer (standard nn.Linear, applied on feats)
        self.out_layer = nn.Linear(
            orig_out.in_features,
            orig_out.out_features * self.n_grid,
            bias=orig_out.bias is not None,
            device=orig_out.weight.device,
            dtype=orig_out.weight.dtype,
        )

        # Initialize
        with torch.no_grad():
            if proj_out_init_mode == "copy":
                self.out_layer.weight.copy_(orig_out.weight.repeat(self.n_grid, 1))
                if orig_out.bias is not None:
                    self.out_layer.bias.copy_(orig_out.bias.repeat(self.n_grid))
            elif proj_out_init_mode == "zero":
                self.out_layer.weight.zero_()
                if self.out_layer.bias is not None:
                    self.out_layer.bias.zero_()
            elif proj_out_init_mode == "hybrid":
                if proj_out_init_plan is None or len(proj_out_init_plan) != self.n_grid:
                    raise ValueError(
                        f"hybrid mode requires proj_out_init_plan with {self.n_grid} entries, got {proj_out_init_plan}"
                    )
                out_f = orig_out.out_features
                for slot, mode in enumerate(proj_out_init_plan):
                    row_start = slot * out_f
                    row_end = (slot + 1) * out_f
                    if mode == "copy":
                        self.out_layer.weight[row_start:row_end].copy_(orig_out.weight)
                        if orig_out.bias is not None:
                            self.out_layer.bias[row_start:row_end].copy_(orig_out.bias)
                    elif mode == "zero":
                        self.out_layer.weight[row_start:row_end].zero_()
                        if orig_out.bias is not None:
                            self.out_layer.bias[row_start:row_end].zero_()
                    else:
                        raise ValueError(f"Unknown slot init mode {mode!r} at slot {slot}")
            else:
                raise ValueError(f"Unknown proj_out_init_mode={proj_out_init_mode!r}.")

        log.info(
            "TrellisPivot: split_k=%d (pos=%d), copied %d blocks, out_layer expanded %d → %d (n_grid=%d, init=%s)",
            split_k,
            pos_k,
            len(self.blocks),
            self._base_out_features,
            orig_out.out_features * self.n_grid,
            self.n_grid,
            proj_out_init_mode,
        )

    @property
    def base_out_features(self) -> int:
        return self._base_out_features

    def forward(self, h, mod: torch.Tensor, cond) -> Any:
        """Run copied blocks → LayerNorm → expanded out_layer.

        Parameters
        ----------
        h : SparseTensor
            Hidden states from the split point.
        mod : Tensor
            Time embedding modulation (``t_emb``, already through ``adaLN_modulation``
            if ``share_mod=True``).
        cond : Tensor or VarLenTensor
            Cross-attention condition features.

        Returns
        -------
        SparseTensor
            ``feats: [N_total, base_out * n_grid]``
        """
        # Autocast to match the student's precision — prevents LayerNorm32
        # internal float32 cast from producing mixed-dtype gradients during
        # FSDP reduce-scatter.
        with torch.amp.autocast("cuda", dtype=h.feats.dtype):
            for block in self.blocks:
                h = block(h, mod, cond)

            # LayerNorm on feats (matching SLatFlowModel.forward)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))

            # Expanded output layer (applied on feats, returning SparseTensor)
            out_feats = self.out_layer(h.feats)  # [N_total, C * n_grid]
        return h.replace(out_feats)


class TrellisPivotBackend(PivotBackend):
    """TRELLIS.2-specific backend for SimplifiedDistillation.

    Handles the sparse transformer block structure, ``SparseLinear`` layers,
    SparseTensor inputs/outputs, and the ``share_mod`` adaLN pattern.
    """

    def get_transformer(self, net: nn.Module) -> nn.Module:
        return getattr(net, "model", net)

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
    ) -> TrellisPivot:
        transformer = self.get_transformer(net)
        return TrellisPivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        """Register a hook on a TRELLIS.2 ``ModulatedSparseTransformerCrossBlock``.

        The block signature is ``forward(h: SparseTensor, mod: Tensor, cond)``
        and returns a SparseTensor.

        We capture:
        - ``hidden_states``: SparseTensor output of the hook block (h)
        - ``mod``: time embedding modulation
        - ``cond``: cross-attention condition
        """
        # We also need to capture the mod (t_emb) and cond from the transformer forward.
        # Since TRELLIS.2 blocks are called with positional args, we use args[0..2].
        transformer = self.get_transformer(net)

        # Capture mod/cond via wrapper on the first block call or on the transformer forward
        _orig_forward = transformer.forward

        def wrapped_forward(*args, **kwargs):
            # SLatFlowModel.forward(x, t, cond, ...) computes t_emb internally
            # We need to recapture the intermediate t_emb and cond.
            result = _orig_forward(*args, **kwargs)
            return result

        def hook_fn(module: nn.Module, args: tuple, output: Any) -> None:
            # Block called as: block(h, mod, cond) — positional args
            capture_store.clear()
            capture_store["hidden_states"] = output  # SparseTensor
            if len(args) >= 3:
                capture_store["mod"] = args[1]  # t_emb (possibly through adaLN_modulation)
                capture_store["cond"] = args[2]  # condition features
            elif len(args) >= 2:
                capture_store["mod"] = args[1]
                capture_store["cond"] = None

        return block.register_forward_hook(hook_fn)

    def decode(self, pivot: nn.Module, hook_data: dict) -> Any:
        """Run the TrellisPivot using captured hook data.

        Returns a SparseTensor with ``feats: [N_total, C * n_grid]``.
        """
        return pivot(hook_data["hidden_states"], hook_data["mod"], hook_data.get("cond", None))

    def unpack_to_spatial(self, net: nn.Module, tokens: Any, input_shape: tuple) -> Any:
        """For SparseTensor: no spatial unpacking needed.

        The tokens are already a SparseTensor with
        ``feats: [N_total, C * n_grid]``.  We return it as-is.
        """
        return tokens

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> Any:
        """Re-run student blocks after the hook + LayerNorm + out_layer.

        Returns SparseTensor with ``feats: [N_total, out_channels]``.
        """
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach()
        mod = hook_data["mod"]
        cond = hook_data.get("cond", None)

        all_blocks = list(transformer.blocks)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k

        for blk in all_blocks[pos_k + 1 :]:
            hs = blk(hs, mod, cond)

        # LayerNorm + out_layer (matching SLatFlowModel.forward)
        hs = hs.replace(F.layer_norm(hs.feats, hs.feats.shape[-1:]))
        out_feats = transformer.out_layer(hs.feats)
        return hs.replace(out_feats)
