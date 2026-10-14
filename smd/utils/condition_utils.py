# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified condition manipulation for all architectures.

A "condition" is any nested structure of tensors: a single Tensor,
a tuple/list of Tensors, or a tuple/list containing nested tuples/lists
of Tensors.  All architectures produce conditions in this form:

    Flux:      (pooled [B,768], embeds [B,512,4096])
    Flux2:     Tensor [B,512,D]
    Lens:      ([layer0, layer1, layer2, layer3], mask [B,S])
    QwenImage: (embeds [B,S,3584], mask [B,S])
    Wan:       Tensor [B,512,D]
    TRELLIS:   Tensor [B,S,D]  (DINOv3 image features)

The tree operations (cond_batch_size, cond_slice, cond_repeat,
cond_repeat_for_vis) work on ANY nested structure — they walk the tree
and apply the operation to every tensor leaf.  New architectures that
return tensors in any nesting of tuples/lists need zero changes here.

cond_trim_padding is the exception: it needs to know which element
carries the attention mask.  See MASK_INDEX below.
"""

from __future__ import annotations

from torch import Tensor
from typing import Any, Union

CondTree = Union[Tensor, tuple, list, dict, Any]


# ---------------------------------------------------------------------------
# Tree operations (format-agnostic)
# ---------------------------------------------------------------------------


def cond_batch_size(cond: CondTree) -> int:
    """Return batch size by finding the first Tensor leaf."""
    if callable(getattr(cond, "batch_size", None)):
        return cond.batch_size()
    if isinstance(cond, Tensor):
        return cond.shape[0]
    if isinstance(cond, dict):
        for v in cond.values():
            if isinstance(v, (Tensor, tuple, list, dict)):
                return cond_batch_size(v)
    if isinstance(cond, (tuple, list)):
        return cond_batch_size(cond[0])
    raise TypeError(f"Cannot infer batch_size from {type(cond)}")


def cond_slice(cond: CondTree, n: int) -> CondTree:
    """Slice first *n* items from every Tensor in the structure."""
    if callable(getattr(cond, "slice_batch", None)):
        return cond.slice_batch(n)
    if isinstance(cond, Tensor):
        return cond[:n]
    if isinstance(cond, list):
        return [cond_slice(item, n) for item in cond]
    if isinstance(cond, tuple):
        return tuple(cond_slice(item, n) for item in cond)
    if isinstance(cond, dict):
        return {k: cond_slice(v, n) for k, v in cond.items()}
    return cond  # scalar / non-tensor passthrough


def cond_repeat(cond: CondTree, n: int) -> CondTree:
    """Repeat each item *n* times along batch dim (interleaved)."""
    if n <= 1:
        return cond
    if callable(getattr(cond, "repeat_batch", None)):
        return cond.repeat_batch(n)
    if isinstance(cond, Tensor):
        return cond.repeat_interleave(n, dim=0)
    if isinstance(cond, list):
        return [cond_repeat(item, n) for item in cond]
    if isinstance(cond, tuple):
        return tuple(cond_repeat(item, n) for item in cond)
    if isinstance(cond, dict):
        return {k: cond_repeat(v, n) if isinstance(v, (Tensor, tuple, list, dict)) else v for k, v in cond.items()}
    return cond


def cond_repeat_for_vis(cond: CondTree, n: int) -> CondTree:
    """Repeat for visualization (tile, not interleave)."""
    if n <= 1:
        return cond
    if isinstance(cond, Tensor):
        return cond.repeat(n, *([1] * (cond.ndim - 1)))
    if isinstance(cond, list):
        return [cond_repeat_for_vis(item, n) for item in cond]
    if isinstance(cond, tuple):
        return tuple(cond_repeat_for_vis(item, n) for item in cond)
    if isinstance(cond, dict):
        return {
            k: cond_repeat_for_vis(v, n) if isinstance(v, (Tensor, tuple, list, dict)) else v for k, v in cond.items()
        }
    return cond


# ---------------------------------------------------------------------------
# Trim padding (format-aware)
# ---------------------------------------------------------------------------

# Maps network_family → index of the attention mask within the condition tuple.
# Only architectures with variable-length text masking are listed.
# If your architecture isn't here, cond_trim_padding is a no-op.
#
# To add a new masked architecture:
#   1. Add an entry here.
#   2. That's it — tree operations already handle your format.
MASK_INDEX: dict[str, int] = {
    "lens": 1,  # (list[Tensor], mask) → mask is index 1
    "qwenimage": 1,  # (embeds, mask) → mask is index 1
    "qwen_image": 1,  # current QwenImage network_family
    "qwen_image_edit": 1,  # current QwenImageEdit network_family
}


def cond_trim_padding(cond: CondTree, arch: str) -> CondTree:
    """Remove trailing padding using the mask, if applicable.

    For architectures not in MASK_INDEX, returns *cond* unchanged.

    Args:
        cond: condition structure from text encoder
        arch: ``network_family`` string (e.g. ``"lens"``, ``"flux2"``)
    """
    if arch not in MASK_INDEX:
        return cond
    if not isinstance(cond, (tuple, list)) or len(cond) <= MASK_INDEX[arch]:
        return cond

    mask_idx = MASK_INDEX[arch]
    mask = cond[mask_idx]
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        return cond

    actual_len = int(mask.sum(dim=-1).max().item())
    if actual_len >= mask.shape[1]:
        return cond  # no padding to trim

    def _trim(item: Any) -> Any:
        if isinstance(item, Tensor) and item.ndim >= 2 and item.shape[1] == mask.shape[1]:
            return item[:, :actual_len]
        if isinstance(item, list):
            return [_trim(x) for x in item]
        if isinstance(item, tuple):
            return tuple(_trim(x) for x in item)
        return item

    cls = type(cond)
    return cls(_trim(item) for item in cond)


# ---------------------------------------------------------------------------
# Debug logging helper
# ---------------------------------------------------------------------------


def cond_shape_str(cond: Any) -> str:
    """Summarise condition shape for logging (handles Tensor, tuple/list, None)."""
    if cond is None:
        return "None"
    if isinstance(cond, Tensor):
        return f"Tensor{list(cond.shape)}"
    if isinstance(cond, (tuple, list)):
        tag = "tuple" if isinstance(cond, tuple) else "list"
        inner = ", ".join(cond_shape_str(c) for c in cond[:3])
        suffix = ", ..." if len(cond) > 3 else ""
        return f"{tag}({inner}{suffix})"
    if isinstance(cond, dict):
        return f"dict(keys={list(cond.keys())[:3]})"
    return f"{type(cond).__name__}"
