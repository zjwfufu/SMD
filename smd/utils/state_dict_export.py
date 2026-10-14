# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-ready state dict export with LoRA merging and FSDP gathering.

This module walks a backbone module (optionally wrapped by PEFT LoRA) and
produces a flat state dict that looks like the original (un-PEFT-wrapped)
model, with every LoRA delta ``B @ A * scaling`` folded into the
corresponding base weight.

It handles three orthogonal concerns:

1. **PEFT FQN stripping** — PEFT inserts ``.base_layer.`` into every targeted
   module's parameter name; the merged dict drops that segment so downstream
   ``FluxTransformer2DModel.load_state_dict`` accepts it.
2. **LoRA delta computation** — replicates PEFT's own ``get_delta_weight``
   formula: ``delta = (lora_B @ lora_A) * scaling[adapter]`` (assumes
   ``fan_in_fan_out=False``, which is the default for ``nn.Linear``).
3. **FSDP2 DTensor gathering** — calls ``.full_tensor()`` so the returned dict
   contains plain, fully-replicated CPU tensors on rank 0.

The function is *read-only* on the live model: weights, lora_A, lora_B and
scaling are all gathered but never mutated.
"""

from __future__ import annotations


import torch
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    DTensor = None  # type: ignore[assignment]


def _gather_full(t: torch.Tensor) -> torch.Tensor:
    """Return a fully-replicated tensor, moving the local shard to CUDA first
    if needed (full_tensor performs an all_gather which requires a CUDA
    backend in NCCL setups)."""
    if DTensor is not None and isinstance(t, DTensor):
        if t.device.type == "cpu":
            return t.to("cuda").full_tensor()
        return t.full_tensor()
    return t


def _is_lora_layer(mod: nn.Module) -> bool:
    """Detect a PEFT LoRA wrapper. PEFT lora.Linear (and friends) all expose
    ``lora_A``, ``lora_B``, ``scaling``, and ``base_layer``."""
    return hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "scaling") and hasattr(mod, "base_layer")


# Wrapper-segment names that get auto-inserted into module FQNs at training
# time and must be stripped before the merged state dict is consumed by a
# clean HuggingFace model.
#
# - ``_checkpoint_wrapped_module``: inserted by
#   ``torch.distributed._composable.checkpoint.checkpoint`` (activation
#   checkpointing). Every wrapped submodule has its child params prefixed by
#   this segment.
# - ``_fsdp_wrapped_module``: inserted by the older FSDP1 wrapper. Harmless to
#   strip even though FSDP2 does not emit it.
_WRAPPER_SEGMENTS: tuple[str, ...] = (
    "_checkpoint_wrapped_module",
    "_fsdp_wrapped_module",
)


def _strip_wrapper_segments(fqn: str) -> str:
    """Remove any wrapper segments from ``fqn``, in place, preserving order."""
    if not any(seg in fqn for seg in _WRAPPER_SEGMENTS):
        return fqn
    parts = [p for p in fqn.split(".") if p not in _WRAPPER_SEGMENTS]
    return ".".join(parts)


def compute_inference_state_dict(
    module: nn.Module,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    skip_key_prefixes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Build a flat ``{fqn: tensor}`` dict equivalent to the inference-ready
    backbone, with LoRA deltas folded in and all FSDP2 shards gathered onto
    rank-0 CPU.

    Args:
        module: the backbone to export (e.g. ``net.get_inference_module()``).
            May or may not have PEFT LoRA adapters attached.
        target_dtype: dtype of the returned tensors. Defaults to bfloat16.
        skip_key_prefixes: tuple of FQN prefixes (matched against the
            *cleaned*, post-wrapper-strip name) to omit from the output. Use
            this to drop SMD-only heads that aren't part of the upstream
            HuggingFace model (e.g. ``logvar_linear``).

    Returns:
        ``dict[str, torch.Tensor]`` keyed by the *clean* (non-PEFT,
        non-wrapped) FQNs that the upstream HuggingFace model uses. No
        ``base_layer.``, no ``lora_A``/``lora_B``, no
        ``_checkpoint_wrapped_module.``. All tensors are on CPU and have
        dtype ``target_dtype``.

    Notes:
        - Only ``nn.Linear``-style LoRA layers are supported (the default for
          Flux). Conv/Embedding LoRA paths exist in PEFT but are not used by
          the current configs.
        - This is collective: every rank must call it (``.full_tensor()``
          requires participation). Callers can discard the result off rank 0.
        - Fails loud: any unexpected layer / missing scaling / shape mismatch
          raises a ``RuntimeError``.
    """
    merged: dict[str, torch.Tensor] = {}

    def _should_skip(clean_name: str) -> bool:
        return any(clean_name.startswith(p) for p in skip_key_prefixes)

    # First pass: identify every PEFT LoRA layer (by module path) so we can
    # skip their ``lora_A.*`` / ``lora_B.*`` parameters in the param iteration
    # below, and so we can rewrite their ``base_layer.weight`` FQNs.
    lora_module_paths: dict[str, nn.Module] = {name: mod for name, mod in module.named_modules() if _is_lora_layer(mod)}

    # Pre-compute merged tensors for each LoRA layer. We key these by the
    # *cleaned* (wrapper-segments-stripped) path so downstream lookup matches
    # the cleaned param-name path.
    lora_merged_weights: dict[str, torch.Tensor] = {}
    lora_merged_biases: dict[str, torch.Tensor] = {}
    for path, mod in lora_module_paths.items():
        clean_path = _strip_wrapper_segments(path)
        if _should_skip(clean_path):
            continue
        base_layer = mod.base_layer  # type: ignore[attr-defined]
        if not isinstance(base_layer, nn.Linear):
            raise RuntimeError(
                f"compute_inference_state_dict: LoRA layer at '{path}' wraps a "
                f"{type(base_layer).__name__}, but only nn.Linear is supported."
            )

        base_w = _gather_full(base_layer.weight.data).to(device="cpu", dtype=torch.float32)
        merged_w = base_w.clone()

        # Iterate active adapters (typically just "default") and fold each delta.
        active_adapters: list[str] = list(mod.lora_A.keys())  # type: ignore[attr-defined]
        if len(active_adapters) == 0:
            raise RuntimeError(f"compute_inference_state_dict: LoRA layer at '{path}' has no active adapters.")
        for adapter in active_adapters:
            weight_A = _gather_full(mod.lora_A[adapter].weight.data).to(  # type: ignore[attr-defined]
                device="cpu", dtype=torch.float32
            )
            weight_B = _gather_full(mod.lora_B[adapter].weight.data).to(  # type: ignore[attr-defined]
                device="cpu", dtype=torch.float32
            )
            scaling = float(mod.scaling[adapter])  # type: ignore[index]
            # fan_in_fan_out=False for nn.Linear → no transpose
            delta = (weight_B @ weight_A) * scaling
            if delta.shape != merged_w.shape:
                raise RuntimeError(
                    f"compute_inference_state_dict: shape mismatch at '{path}' "
                    f"adapter '{adapter}': base {tuple(merged_w.shape)} vs "
                    f"delta {tuple(delta.shape)}."
                )
            merged_w = merged_w + delta

        lora_merged_weights[clean_path] = merged_w.to(target_dtype)

        if base_layer.bias is not None:
            b = _gather_full(base_layer.bias.data).to(device="cpu", dtype=target_dtype)
            lora_merged_biases[clean_path] = b

    # Second pass: walk every parameter/buffer in the module and emit
    # cleaned FQNs.
    for name, param in module.named_parameters():
        # Skip LoRA-adapter params entirely; they're folded above.
        if ".lora_A." in name or ".lora_B." in name or ".lora_embedding_A." in name or ".lora_embedding_B." in name:
            continue
        clean_name = _strip_wrapper_segments(name)
        if _should_skip(clean_name):
            continue
        # Detect "<lora_path>.base_layer.weight|bias" → rewrite to "<lora_path>.weight|bias".
        rewritten = _strip_base_layer(clean_name)
        if rewritten is None:
            # Not a LoRA-touched param: gather and store as-is.
            merged[clean_name] = _gather_full(param.data).to(device="cpu", dtype=target_dtype)
            continue

        lora_path, suffix = rewritten
        if suffix == "weight":
            if lora_path not in lora_merged_weights:
                raise RuntimeError(
                    f"compute_inference_state_dict: encountered '{clean_name}' but no "
                    f"merged tensor was precomputed for '{lora_path}'."
                )
            merged[f"{lora_path}.weight"] = lora_merged_weights[lora_path]
        elif suffix == "bias":
            if lora_path in lora_merged_biases:
                merged[f"{lora_path}.bias"] = lora_merged_biases[lora_path]
        else:
            raise RuntimeError(
                f"compute_inference_state_dict: unexpected base_layer param suffix '{suffix}' on '{clean_name}'."
            )

    # Buffers (rotary caches, etc.) — FSDP2 doesn't shard them, so a direct copy works.
    for name, buf in module.named_buffers():
        clean_name = _strip_wrapper_segments(name)
        if _should_skip(clean_name):
            continue
        merged[clean_name] = buf.detach().to(device="cpu", dtype=target_dtype if buf.is_floating_point() else buf.dtype)

    return merged


def _strip_base_layer(fqn: str) -> tuple[str, str] | None:
    """If ``fqn`` looks like ``<path>.base_layer.<suffix>``, return
    ``(<path>, <suffix>)``; else None.

    PEFT only inserts a single ``base_layer`` segment per wrapped module, and
    only directly before the final parameter name."""
    parts = fqn.split(".")
    if len(parts) < 2:
        return None
    if parts[-2] != "base_layer":
        return None
    lora_path = ".".join(parts[:-2])
    suffix = parts[-1]
    return lora_path, suffix


__all__ = ["compute_inference_state_dict"]
