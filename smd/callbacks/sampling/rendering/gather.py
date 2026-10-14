# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect per-rank samples so previews show more than just rank 0.

Under FSDP/DDP every rank has to enter the sampling forward anyway (it is a
collective), so the samples the other ranks produce are free. Gathering them
gives:

  * ``BatchSampleCallback``  — tiles from *different training batches*
  * ``PromptSampleCallback`` — the same prompt under *different noise*

Shapes are validated before the gather; if ranks disagree (bucketed
resolutions, variable video length) we degrade to rank-0-only rather than
deadlock or crash.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from smd.utils import logging_utils as logger
from smd.utils.distributed import get_rank, is_rank0, world_size

#: Largest tensor rank we describe in the shape-agreement probe ([B, C, T, H, W] = 5).
_MAX_NDIM = 6


def _distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized() and world_size() > 1


def _shapes_agree(tensor: torch.Tensor) -> bool:
    """True when every rank holds a tensor of identical shape."""
    descriptor = torch.zeros(_MAX_NDIM + 1, dtype=torch.long, device=tensor.device)
    descriptor[0] = tensor.ndim
    descriptor[1 : tensor.ndim + 1] = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
    gathered = [torch.zeros_like(descriptor) for _ in range(world_size())]
    dist.all_gather(gathered, descriptor)
    return all(torch.equal(gathered[0], other) for other in gathered[1:])


def gather_samples(tensor: torch.Tensor | None, *, limit: int | None = None) -> torch.Tensor | None:
    """All-gather sample batches along dim 0; returns the batch on rank 0 only.

    Non-rank-0 processes always get ``None``. Every rank must call this — it
    runs collectives.
    """
    if tensor is None or tensor.numel() == 0:
        return None
    tensor = tensor.detach()

    if not _distributed_active():
        return tensor[:limit].cpu() if limit else tensor.cpu()

    if tensor.ndim > _MAX_NDIM:
        logger.warning(f"[VisGather] tensor rank {tensor.ndim} exceeds {_MAX_NDIM}; using rank-0 samples only")
        return tensor[:limit].cpu() if is_rank0() else None

    if not _shapes_agree(tensor):
        logger.warning("[VisGather] per-rank sample shapes disagree; using rank-0 samples only")
        return tensor[:limit].cpu() if is_rank0() else None

    contiguous = tensor.contiguous()
    buckets = [torch.empty_like(contiguous) for _ in range(world_size())]
    dist.all_gather(buckets, contiguous)
    if not is_rank0():
        return None
    gathered = torch.cat([bucket.cpu() for bucket in buckets], dim=0)
    return gathered[:limit] if limit else gathered


def gather_variable_samples(tensor: torch.Tensor, *, limit: int | None = None) -> list[torch.Tensor] | None:
    """Gather samples with different spatial shapes while preserving rank order.

    Each sample remains a ``[1, ...]`` CPU tensor. Object gathering is reserved
    for batch previews, where ``grid_size`` is small and rank-local image buckets
    may have different H/W. Fixed-prompt previews keep the faster tensor gather.
    """
    local = [sample.detach().to(device="cpu") for sample in tensor.split(1, dim=0)]
    if not _distributed_active():
        return local[:limit] if limit else local

    count = torch.tensor([len(local)], dtype=torch.long, device=tensor.device)
    counts = [torch.zeros_like(count) for _ in range(world_size())]
    dist.all_gather(counts, count)
    if limit is not None:
        offset = sum(int(value.item()) for value in counts[: get_rank()])
        local = local[: max(0, min(len(local), int(limit) - offset))]

    buckets: list[list[torch.Tensor] | None] | None = [None] * world_size() if is_rank0() else None
    dist.gather_object(local, object_gather_list=buckets, dst=0)
    if not is_rank0():
        return None
    assert buckets is not None
    gathered = [sample for bucket in buckets if bucket for sample in bucket]
    return gathered[:limit] if limit else gathered


def samples_per_rank(total: int) -> int:
    """How many samples each rank must draw so the job produces >= ``total``.

    Every rank draws the *same* count — FSDP requires matching forward shapes —
    so this rounds up and the surplus is trimmed after the gather.
    """
    return max(1, -(-int(total) // max(1, world_size())))
