# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep pivot and architecture-specific backends.

``Pivot`` deepcopy's the last *N* blocks plus ``norm_out`` and
``proj_out`` from a student transformer, expanding ``proj_out`` to produce
``n_grid`` velocity predictions.  The pivot is small (typically 3–4 blocks)
and deliberately excluded from FSDP wrapping.

``PivotBackend`` is an abstract adapter that encapsulates all architecture-
specific operations.  ``distillation.py`` is architecture-agnostic and delegates
to the backend via duck-typing.
"""

from __future__ import annotations

import abc
import contextlib
import copy
import logging
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _enumerate_flux_blocks(transformer: nn.Module) -> list[nn.Module]:
    """Collect all blocks in flat order for Flux: joint/double + single.

    Supports both diffusers naming (``transformer_blocks`` / ``single_transformer_blocks``)
    and BFL naming (``double_blocks`` / ``single_blocks``).
    """
    # Try diffusers names first, fall back to BFL names
    joint = list(getattr(transformer, "transformer_blocks", None) or getattr(transformer, "double_blocks", []))
    single = list(getattr(transformer, "single_transformer_blocks", None) or getattr(transformer, "single_blocks", []))
    return joint + single


# ---------------------------------------------------------------------------
# Pivot
# ---------------------------------------------------------------------------


class Pivot(nn.Module):
    """Deep pivot for Flux-style transformers.

    From the student transformer, deepcopy the last few blocks (after the
    split point *split_k*), ``norm_out``, and ``proj_out``.  ``proj_out`` is
    expanded to output ``n_grid × base_out_features`` so it predicts
    ``n_grid`` velocity anchors in a single forward pass.

    The pivot is small (typically only the last 3–4 blocks) and does **not**
    need FSDP.

    Parameters
    ----------
    transformer : nn.Module
        The student's transformer (e.g. ``FluxTransformer2DModel``).
        Must have ``transformer_blocks``, ``single_transformer_blocks``,
        ``norm_out``, and ``proj_out``.
    split_k : int
        Flat block index (supports negative indexing) at which the student
        is *split*.  Blocks ``split_k + 1 .. total - 1`` are deepcopy'd
        into the pivot.
    n_grid : int
        Number of velocity predictions.  ``proj_out`` is expanded to
        ``base_out_features * n_grid``.
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

        # Collect all blocks (Flux: joint + single) in flat order.
        all_blocks = self._collect_all_blocks(transformer)
        total = len(all_blocks)
        pos_k = split_k if split_k >= 0 else total + split_k
        if not (0 <= pos_k < total):
            raise IndexError(
                f"split_k={split_k} resolves to position {pos_k}, "
                f"which is out of range for {total} blocks (valid: -{total}..{total - 1})"
            )

        # Deepcopy blocks pos_k+1 .. total-1
        self.blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(pos_k + 1, total)])

        # Deepcopy norm_out (diffusers: AdaLayerNormContinuous, BFL: LastLayer)
        if hasattr(transformer, "norm_out"):
            self.norm_out = copy.deepcopy(transformer.norm_out)
        elif hasattr(transformer, "final_layer"):
            self.norm_out = copy.deepcopy(transformer.final_layer)
        else:
            raise AttributeError("Transformer has neither 'norm_out' nor 'final_layer'")

        # Deepcopy & expand proj_out for n_grid
        # diffusers: transformer.proj_out; BFL: transformer.final_layer.linear
        if hasattr(transformer, "proj_out"):
            orig = transformer.proj_out
        elif hasattr(transformer, "final_layer") and hasattr(transformer.final_layer, "linear"):
            orig = transformer.final_layer.linear
        else:
            raise AttributeError("Cannot find proj_out or final_layer.linear on transformer")
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
            "Pivot: split_k=%d (pos=%d), copied %d blocks, proj_out expanded %d → %d (n_grid=%d, init=%s)",
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

    @staticmethod
    def _collect_all_blocks(transformer: nn.Module) -> list[nn.Module]:
        """Collect all blocks in flat order.  Flux: joint + single."""
        if hasattr(transformer, "transformer_blocks") or hasattr(transformer, "double_blocks"):
            return _enumerate_flux_blocks(transformer)
        raise ValueError(
            "Pivot requires a Flux-style transformer with "
            "transformer_blocks + single_transformer_blocks.  "
            f"Got {type(transformer).__name__}."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run copied blocks → norm_out → expanded proj_out.

        All arguments follow the Flux block interface.

        Returns
        -------
        torch.Tensor
            ``[B, seq, base_out_features * n_grid]``
        """
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# Local pivot-feature capture
# ---------------------------------------------------------------------------


class PivotCapture:
    """Local, single-use container for one pivot-feature capture.

    Created per :meth:`PivotBackend.capture_pivot_features` scope and populated by
    the split-block forward hook during **exactly one** student forward.

    ``store`` is the backend-specific payload dict — its keys differ per
    architecture (e.g. Flux writes ``hidden_states`` / ``encoder_hidden_states`` /
    ``temb`` / ``image_rotary_emb``; Wan adds ``timestep_proj`` / ``rotary_emb`` /
    ``norm_temb``; the TRELLIS sparse backend stores a ``SparseTensor``
    ``hidden_states`` plus ``mod`` / ``cond``; Hunyuan3D stores ``hidden_states`` /
    ``c`` / ``cond`` / ``skip_stack``). A dedicated counting hook records how many
    times the split block fired so :attr:`result` can prove it fired once and only
    once.

    The container never detaches captured tensors: for a live trainable forward the
    payload stays attached (gradients reach the student prefix + pivot); for an
    EMA / ``torch.no_grad`` forward no autograd graph is built in the first place.
    Detach policy is therefore uniform across backends — there is no per-call
    ``detach_input`` flag.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._fired: int = 0

    def _mark_fired(self) -> None:
        self._fired += 1

    @property
    def store(self) -> dict[str, Any]:
        """Mutable backend payload dict the split-block hook writes into."""
        return self._store

    @property
    def fired(self) -> int:
        """Number of times the split block fired inside the capture scope."""
        return self._fired

    @property
    def result(self) -> dict[str, Any]:
        """Validated backend payload dict.

        Raises unless the split block fired exactly once and wrote a payload.
        """
        if self._fired == 0:
            raise RuntimeError(
                "pivot capture hook did not fire during the student forward. Ensure pivot_hook_layer "
                "resolves to a transformer block that runs exactly once in the forward pass."
            )
        if self._fired > 1:
            raise RuntimeError(
                f"pivot capture hook fired {self._fired} times during a single student forward "
                "(expected exactly 1). The split block ran more than once — a duplicated forward or a "
                "gradient-checkpointing recompute leaked into the capture scope."
            )
        if not self._store:
            raise RuntimeError("pivot capture hook fired but wrote no payload into the capture store.")
        return self._store


# ---------------------------------------------------------------------------
# PivotBackend ABC
# ---------------------------------------------------------------------------


class PivotBackend(abc.ABC):
    """Architecture adapter for SimplifiedDistillation.

    Encapsulates all architecture-specific operations that ``model.py``
    needs so that the training loop remains architecture-agnostic.

    Backends are **stateless** duck-typed adapters: one instance serves any net of
    the same architecture (the live student, an EMA copy, a teacher). They hold no
    per-forward capture state — a single forward's features are captured locally
    via :meth:`capture_pivot_features`.
    """

    @abc.abstractmethod
    def get_transformer(self, net: nn.Module) -> nn.Module:
        """Extract the transformer sub-module from the network."""

    @abc.abstractmethod
    def enumerate_blocks(self, net: nn.Module) -> list[nn.Module]:
        """Flat list of all transformer blocks."""

    @abc.abstractmethod
    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> Pivot:
        """Construct a Pivot for this architecture."""

    def get_lora_backbone(self, pivot: nn.Module) -> nn.Module:
        """Return the copied block container that may receive Pivot LoRA."""
        blocks = getattr(pivot, "blocks", None)
        if not isinstance(blocks, nn.Module):
            raise TypeError(f"Pivot {type(pivot).__name__} does not expose a module-valued 'blocks' LoRA backbone.")
        return blocks

    def configure_pivot_blocks(
        self,
        pivot: Pivot,
        net: nn.Module,
        split_k: int,
        num_blocks: int | None,
        init_mode: str,
        init_seed: int,
    ) -> None:
        """Apply an optional architecture-specific Pivot block override."""
        del net, split_k, init_seed
        natural_count = len(pivot.blocks)
        target_count = natural_count if num_blocks is None else num_blocks
        if init_mode != "pretrained" or target_count != natural_count:
            raise NotImplementedError(
                f"{type(self).__name__} does not support pivot_num_blocks={num_blocks!r} "
                f"with pivot_block_init={init_mode!r}."
            )

    def set_lora_backbone(self, pivot: nn.Module, backbone: nn.Module) -> None:
        """Replace the copied block container after adapter injection."""
        if not hasattr(pivot, "blocks"):
            raise TypeError(f"Pivot {type(pivot).__name__} has no 'blocks' attribute.")
        pivot.blocks = backbone

    @abc.abstractmethod
    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> Any:
        """Install the split-block capture on *block*, writing into *capture_store*.

        Returns a handle exposing ``.remove()`` that fully tears down whatever the
        backend installed (a torch forward hook, and — for backends that also wrap
        a transformer method such as Wan's ``classify_forward_prepare`` — the
        restoration of that method). Called only by :meth:`capture_pivot_features`.
        """

    def _resolve_split_block(self, net: nn.Module, split_k: int) -> tuple[nn.Module, int]:
        """Resolve the split block and its flat position (supports negative indexing)."""
        blocks = self.enumerate_blocks(net)
        n = len(blocks)
        pos = split_k if split_k >= 0 else n + split_k
        if not (0 <= pos < n):
            raise IndexError(f"split_k={split_k} out of range for {n} blocks (valid: -{n}..{n - 1})")
        return blocks[pos], pos

    @contextlib.contextmanager
    def capture_pivot_features(self, net: nn.Module, split_k: int):
        """Scoped, single-forward capture of the pivot split-point features.

        Usage::

            with backend.capture_pivot_features(net, split_k) as capture:
                output = net(...)          # exactly ONE forward
                payload = capture.result   # validated backend payload dict

        A backend payload hook (:meth:`register_hook`) and a dedicated counting hook
        are installed on the resolved split block and **both removed in the
        ``finally`` block** — no persistent instance hook state survives the scope.
        The counting hook is measured during the wrapped forward only, so an
        activation-checkpointing recompute (which happens later, during backward,
        after the hooks are already removed) can never double-count.

        Works identically for a live trainable student (payload stays attached, so
        the Pivot loss trains the student prefix) and for an EMA / ``no_grad``
        student (no graph is built). The caller runs the forward inside the ``with``
        block and reads :attr:`PivotCapture.result` before leaving it.
        """
        block, _pos = self._resolve_split_block(net, split_k)
        capture = PivotCapture()
        payload_handle = self.register_hook(net, block, capture.store)
        count_handle = block.register_forward_hook(lambda _module, *_args: capture._mark_fired())
        try:
            yield capture
        finally:
            try:
                count_handle.remove()
            finally:
                try:
                    payload_handle.remove()
                finally:
                    capture.store.clear()

    @abc.abstractmethod
    def decode(self, pivot: Pivot, hook_data: dict) -> torch.Tensor:
        """Run the pivot using captured hook data → raw tokens.

        ``hook_data`` is a :attr:`PivotCapture.result` payload. Captured tensors are
        consumed as-is (never detached here): the live path stays attached; the
        EMA / no-grad path builds no graph.
        """

    @abc.abstractmethod
    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        """Unpack raw tokens to spatial tensor ``[B, C_expanded, H, W, ...]``."""

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        """Re-run blocks after the hook + norm_out + proj_out → raw tokens (for alignment).

        Returns raw token output ``[B, seq, out_dim]`` **before** spatial
        unpacking / flow conversion.  The caller is responsible for those steps.

        This is an **optional** alignment/diagnostic helper, not part of the
        production training path — it is exercised only by standalone tests that
        prove the pivot decoder tail matches the student's own output layer. New
        architectures need not implement it; the default raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rerun_output_layer (optional alignment helper)."
        )
