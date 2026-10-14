# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn

from ..common import PivotBackend, _enumerate_flux_blocks

log = logging.getLogger(__name__)


class _Flux2CaptureHandle:
    """Teardown handle for Flux2 scoped capture.

    Flux2 capture installs both a block forward hook and a temporary wrapper on
    ``transformer.forward``.  The generic capture context only knows it can call
    ``remove()``, so this handle owns the full teardown.
    """

    def __init__(
        self,
        *,
        block_handle: torch.utils.hooks.RemovableHandle,
        transformer: nn.Module,
        original_forward: Any,
        capture_store: dict,
    ) -> None:
        self._block_handle = block_handle
        self._transformer = transformer
        self._original_forward = original_forward
        self._capture_store = capture_store
        self._removed = False

    def remove(self) -> None:
        if self._removed:
            return
        try:
            self._block_handle.remove()
        finally:
            self._transformer.forward = self._original_forward
            self._capture_store.clear()
            self._removed = True


# ---------------------------------------------------------------------------
# Flux2Pivot
# ---------------------------------------------------------------------------


class Flux2Pivot(nn.Module):
    """Deep pivot for Flux2-style transformers.

    Flux2 differs from Flux1 in block forward signatures:
    - Double blocks take ``temb_mod_img`` + ``temb_mod_txt`` (shared modulation)
    - Single blocks take ``temb_mod`` and concatenated text+image hidden states

    The pivot receives pre-computed modulation tensors and runs copied tail
    blocks → norm_out → expanded proj_out.
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

        # Collect all blocks in flat order (joint + single)
        all_blocks = _enumerate_flux_blocks(transformer)
        total = len(all_blocks)
        self._num_double = len(
            list(getattr(transformer, "transformer_blocks", None) or getattr(transformer, "double_blocks", []))
        )
        pos_k = split_k if split_k >= 0 else total + split_k
        if not (0 <= pos_k < total):
            raise IndexError(
                f"split_k={split_k} resolves to position {pos_k}, "
                f"which is out of range for {total} blocks (valid: -{total}..{total - 1})"
            )

        # Deepcopy blocks pos_k+1 .. total-1
        self.blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(pos_k + 1, total)])
        # Track which copied blocks are double vs single
        self._block_is_double = [i < self._num_double for i in range(pos_k + 1, total)]

        # Deepcopy norm_out (diffusers) or final_layer (BFL)
        if hasattr(transformer, "norm_out"):
            self.norm_out = copy.deepcopy(transformer.norm_out)
        elif hasattr(transformer, "final_layer"):
            self.norm_out = copy.deepcopy(transformer.final_layer)
        else:
            raise AttributeError("Transformer has neither 'norm_out' nor 'final_layer'")

        # Deepcopy & expand proj_out
        # diffusers: transformer.proj_out (nn.Linear)
        # BFL: transformer.final_layer.linear (nn.Linear)
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
            "Flux2Pivot: split_k=%d (pos=%d), copied %d blocks (%d double, %d single), "
            "proj_out expanded %d → %d (n_grid=%d, init=%s)",
            split_k,
            pos_k,
            len(self.blocks),
            sum(self._block_is_double),
            sum(not d for d in self._block_is_double),
            self._base_out_features,
            orig.out_features * self.n_grid,
            self.n_grid,
            proj_out_init_mode,
        )

    @property
    def base_out_features(self) -> int:
        return self._base_out_features

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Any,
        double_stream_mod_img: torch.Tensor,
        double_stream_mod_txt: torch.Tensor,
        single_stream_mod: torch.Tensor,
        num_txt_tokens: int,
    ) -> torch.Tensor:
        """Run copied blocks → norm_out → expanded proj_out.

        Args:
            hidden_states: Image tokens (from hook).
            encoder_hidden_states: Text tokens (from hook).
            temb: Raw time embedding (for norm_out).
            image_rotary_emb: Concatenated RoPE (text + image).
            double_stream_mod_img/txt: Shared modulation for double blocks.
            single_stream_mod: Shared modulation for single blocks.
            num_txt_tokens: Number of text tokens (for stripping after single blocks).

        Returns:
            ``[B, img_seq, base_out_features * n_grid]``
        """
        for block, is_double in zip(self.blocks, self._block_is_double):
            if is_double:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                # Single blocks: concatenate text + image
                if encoder_hidden_states is not None:
                    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                    encoder_hidden_states = None  # Already concatenated
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=image_rotary_emb,
                )

        # Strip text tokens if concatenated (need to know how many were prepended)
        if num_txt_tokens > 0:
            hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# Flux2PivotBackend
# ---------------------------------------------------------------------------


class Flux2PivotBackend(PivotBackend):
    """Flux2 Klein backend for SimplifiedDistillation.

    Handles the Flux2-specific shared modulation architecture where:
    - Double blocks receive ``temb_mod_img`` + ``temb_mod_txt``
    - Single blocks receive ``temb_mod`` and concatenated hidden states
    - ``temb`` (raw time embedding) is used only for ``norm_out``
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
    ) -> Flux2Pivot:
        transformer = self.get_transformer(net)
        return Flux2Pivot(
            transformer,
            split_k,
            n_grid,
            proj_out_init_mode=proj_out_init_mode,
            proj_out_init_plan=proj_out_init_plan,
        )

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict) -> torch.utils.hooks.RemovableHook:
        """Register hook for Flux2 blocks.

        Captures: hidden_states, encoder_hidden_states, temb, image_rotary_emb,
        double_stream_mod_img, double_stream_mod_txt, single_stream_mod, num_txt_tokens.

        The modulation tensors and temb are captured by wrapping the transformer's
        classify_forward, while block-level data comes from the block hook.
        """
        transformer = self.get_transformer(net)

        # Wrap the patched forward to capture model-level tensors
        # Since classify_forward computes temb and modulations, we intercept them.
        _orig_forward = transformer.forward

        def wrapped_forward(*args, **kwargs):
            # The classify_forward stores temb/modulations before running blocks.
            # We intercept by temporarily patching the shared modulation modules.
            _orig_dmod_img = transformer.double_stream_modulation_img.forward
            _orig_dmod_txt = transformer.double_stream_modulation_txt.forward
            _orig_smod = transformer.single_stream_modulation.forward
            # diffusers: time_guidance_embed / context_embedder
            # BFL: time_in / txt_in
            _tge_mod = getattr(transformer, "time_guidance_embed", None) or getattr(transformer, "time_in", None)
            _ctx_mod = getattr(transformer, "context_embedder", None) or getattr(transformer, "txt_in", None)
            _orig_tge = _tge_mod.forward if _tge_mod else None
            _orig_ctx = _ctx_mod.forward if _ctx_mod else None

            def capture_tge(*a, **kw):
                result = _orig_tge(*a, **kw)
                capture_store["temb"] = result
                return result

            def capture_dmod_img(*a, **kw):
                result = _orig_dmod_img(*a, **kw)
                capture_store["double_stream_mod_img"] = result
                return result

            def capture_dmod_txt(*a, **kw):
                result = _orig_dmod_txt(*a, **kw)
                capture_store["double_stream_mod_txt"] = result
                return result

            def capture_smod(*a, **kw):
                result = _orig_smod(*a, **kw)
                capture_store["single_stream_mod"] = result
                return result

            def capture_ctx(*a, **kw):
                result = _orig_ctx(*a, **kw)
                # context_embedder output has shape [B, seq_len, hidden_size]
                capture_store["num_txt_tokens"] = result.shape[1]
                return result

            if _tge_mod:
                _tge_mod.forward = capture_tge
            transformer.double_stream_modulation_img.forward = capture_dmod_img
            transformer.double_stream_modulation_txt.forward = capture_dmod_txt
            transformer.single_stream_modulation.forward = capture_smod
            if _ctx_mod:
                _ctx_mod.forward = capture_ctx

            try:
                result = _orig_forward(*args, **kwargs)
            finally:
                if _tge_mod and _orig_tge:
                    _tge_mod.forward = _orig_tge
                transformer.double_stream_modulation_img.forward = _orig_dmod_img
                transformer.double_stream_modulation_txt.forward = _orig_dmod_txt
                transformer.single_stream_modulation.forward = _orig_smod
                if _ctx_mod and _orig_ctx:
                    _ctx_mod.forward = _orig_ctx

            return result

        transformer.forward = wrapped_forward
        capture_store["_orig_forward"] = _orig_forward

        # Also capture num_txt_tokens from encoder_hidden_states
        def hook_fn(module: nn.Module, input: Any, kwargs: dict, output: Any) -> Any:
            # Flux2 block output depends on block type and backend:
            # diffusers Double: (encoder_hidden_states, hidden_states)
            # diffusers Single: hidden_states
            # BFL Double forward_kv_extract: (img, txt, cache) — 3-tuple
            # BFL Single forward_kv_extract: (img, cache) — 2-tuple
            if isinstance(output, tuple) and len(output) == 3:
                # BFL Double block: (img, txt, cache)
                img, txt, _cache = output
                capture_store["hidden_states"] = img
                capture_store["encoder_hidden_states"] = txt
                capture_store["hook_on_single_block"] = False
                capture_store["num_txt_tokens"] = txt.shape[1]
            elif isinstance(output, tuple) and len(output) == 2:
                first, second = output
                if (
                    hasattr(first, "shape")
                    and hasattr(second, "shape")
                    and first.ndim == 3
                    and second.ndim == 3
                    and first.shape[-1] == second.shape[-1]
                ):
                    # diffusers Double block: (enc_hs, hs) — same hidden dim
                    capture_store["hidden_states"] = second
                    capture_store["encoder_hidden_states"] = first
                    capture_store["hook_on_single_block"] = False
                    capture_store["num_txt_tokens"] = first.shape[1]
                else:
                    # BFL Single block: (img, cache)
                    hs = first
                    capture_store["hidden_states"] = hs
                    capture_store["encoder_hidden_states"] = None
                    capture_store["hook_on_single_block"] = True
            else:
                # Single block: output is concatenated [txt+img] stream
                hs = output if not isinstance(output, tuple) else output[-1]
                capture_store["hidden_states"] = hs
                capture_store["encoder_hidden_states"] = None  # already concatenated
                capture_store["hook_on_single_block"] = True
                # num_txt_tokens was captured during the first double block or from
                # the context_embedder output; retrieve from earlier capture if available
                # For single blocks, we need the num_txt_tokens that was used during
                # the double→single transition (stored when classify_forward concatenates)

            # Extract image_rotary_emb from kwargs or positional args
            image_rotary_emb = kwargs.get("image_rotary_emb", None)
            if image_rotary_emb is None and len(input) > 3:
                image_rotary_emb = input[3]

            capture_store["image_rotary_emb"] = image_rotary_emb

            return None

        block_handle = block.register_forward_hook(hook_fn, with_kwargs=True)
        return _Flux2CaptureHandle(
            block_handle=block_handle,
            transformer=transformer,
            original_forward=_orig_forward,
            capture_store=capture_store,
        )

    def decode(self, pivot: nn.Module, hook_data: dict) -> torch.Tensor:
        enc_hs = hook_data.get("encoder_hidden_states")
        num_txt_tokens = hook_data.get("num_txt_tokens", 0)

        return pivot(
            hidden_states=hook_data["hidden_states"],
            encoder_hidden_states=enc_hs,
            temb=hook_data["temb"],
            image_rotary_emb=hook_data["image_rotary_emb"],
            double_stream_mod_img=hook_data["double_stream_mod_img"],
            double_stream_mod_txt=hook_data["double_stream_mod_txt"],
            single_stream_mod=hook_data["single_stream_mod"],
            num_txt_tokens=num_txt_tokens,
        )

    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        if hasattr(net, "_unpack_latents"):
            return net._unpack_latents(tokens, *input_shape[1:])
        B = tokens.shape[0]
        _C, H, W = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
        return tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    def rerun_output_layer(
        self,
        net: nn.Module,
        hook_data: dict,
        split_k: int,
    ) -> torch.Tensor:
        """Re-run student blocks after the hook + norm_out + proj_out."""
        transformer = self.get_transformer(net)
        hs = hook_data["hidden_states"].detach().requires_grad_(True)
        enc_hs = hook_data["encoder_hidden_states"]
        if enc_hs is not None:
            enc_hs = enc_hs.detach().requires_grad_(True)
        temb = hook_data["temb"]
        rope = hook_data["image_rotary_emb"]
        dmod_img = hook_data["double_stream_mod_img"]
        dmod_txt = hook_data["double_stream_mod_txt"]
        smod = hook_data["single_stream_mod"]

        all_blocks = _enumerate_flux_blocks(transformer)
        total = len(all_blocks)
        num_double = len(
            list(getattr(transformer, "transformer_blocks", None) or getattr(transformer, "double_blocks", []))
        )
        pos_k = split_k if split_k >= 0 else total + split_k

        for i in range(pos_k + 1, total):
            blk = all_blocks[i]
            if i < num_double:
                enc_hs, hs = blk(
                    hidden_states=hs,
                    encoder_hidden_states=enc_hs,
                    temb_mod_img=dmod_img,
                    temb_mod_txt=dmod_txt,
                    image_rotary_emb=rope,
                )
            else:
                if enc_hs is not None:
                    hs = torch.cat([enc_hs, hs], dim=1)
                    enc_hs = None
                hs = blk(
                    hidden_states=hs,
                    encoder_hidden_states=None,
                    temb_mod=smod,
                    image_rotary_emb=rope,
                )

        # Strip text tokens
        num_txt = hook_data.get("encoder_hidden_states", torch.empty(0)).shape[1] if enc_hs is None else 0
        if num_txt > 0:
            hs = hs[:, num_txt:, ...]

        # diffusers: norm_out(hs, temb) + proj_out(h); BFL: final_layer(hs, vec)
        if hasattr(transformer, "norm_out"):
            h = transformer.norm_out(hs, temb)
            return transformer.proj_out(h)
        else:
            return transformer.final_layer(hs, temb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
