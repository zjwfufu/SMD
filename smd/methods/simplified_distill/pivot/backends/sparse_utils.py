# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SparseTensor utility helpers for SMD.

These functions enable SMD to work with both dense ``[B, C, ...]``
tensors and TRELLIS.2 ``SparseTensor`` (``feats: [N_total, C]``) without
requiring a TRELLIS.2 import at module load time.
"""

from __future__ import annotations

import torch
from torch import Tensor


def is_sparse(x) -> bool:
    """Return True if *x* is a TRELLIS.2 SparseTensor (duck-typed)."""
    return hasattr(x, "feats") and hasattr(x, "coords") and hasattr(x, "replace") and not isinstance(x, Tensor)


# ---------------------------------------------------------------------------
# PyTorch pytree registration for SparseTensor
# ---------------------------------------------------------------------------
# FSDP2 uses ``tree_flatten(output)`` to find output tensors on which to
# register backward hooks (for pre-backward all-gather).  If SparseTensor
# is not a registered pytree node, ``tree_flatten`` treats it as an opaque
# leaf and ``torch.is_tensor(leaf)`` returns False → FSDP2 skips the hook
# → parameters are never unsharded in backward → ``setStorage: size 0``.
#
# The fix: register SparseTensor so that ``tree_flatten`` extracts ``.feats``
# (a plain tensor on which FSDP2 can register hooks).

_PYTREE_REGISTERED = False


def register_sparse_pytree() -> None:
    """Register SparseTensor as a PyTorch pytree node (idempotent)."""
    global _PYTREE_REGISTERED
    if _PYTREE_REGISTERED:
        return
    try:
        from smd.networks.Trellis2.sparse import SparseTensor
    except ImportError:
        return

    from torch.utils._pytree import register_pytree_node, SUPPORTED_NODES

    # A legacy reference in the same process may already have registered this public type.
    if SparseTensor in SUPPORTED_NODES:
        _PYTREE_REGISTERED = True
        return

    def _sparse_flatten(sp):
        """Extract the differentiable part (.feats) as a child; keep the rest as context."""
        return [sp.feats], sp  # children=[feats], context=sp (for unflatten)

    def _sparse_unflatten(children, context):
        """Rebuild SparseTensor from (new feats, original SparseTensor context)."""
        (feats,) = children
        if feats is None:
            return context  # FSDP2 may call unflatten during reshard with None children
        return context.replace(feats)

    register_pytree_node(SparseTensor, _sparse_flatten, _sparse_unflatten)
    _PYTREE_REGISTERED = True
    print("[sparse_utils] Registered SparseTensor as pytree node", flush=True)


def expand_to_voxels(scalar: Tensor, sp) -> Tensor:
    """Expand a per-sample ``[B]`` scalar to per-voxel ``[N_total, 1]``.

    Uses the SparseTensor's ``batch_boardcast_map`` (batch index for each voxel).

    Parameters
    ----------
    scalar : Tensor  ``[B]`` or ``[B, 1, ...]``
        Per-sample scalar (e.g. sigma, dt).
    sp : SparseTensor
        The SparseTensor whose voxel structure to follow.

    Returns
    -------
    Tensor  ``[N_total, 1]``
        Scalar broadcast to every voxel.
    """
    # Flatten scalar to [B]
    s = scalar.reshape(-1) if scalar.ndim > 1 else scalar
    batch_map = sp.batch_boardcast_map  # [N_total] — batch index per voxel
    return s[batch_map].unsqueeze(-1)  # [N_total, 1]


def sparse_randn_like(sp) -> object:
    """Create a SparseTensor with ``randn`` feats and same coords."""
    return sp.replace(torch.randn_like(sp.feats))


def sparse_feats_or_tensor(x) -> Tensor:
    """Return ``x.feats`` if *x* is a SparseTensor, else *x* itself."""
    return x.feats if is_sparse(x) else x
