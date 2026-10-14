# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-progress batching for variable-resolution image datasets."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import smd.utils.logging_utils as logger


def resize_chw_tensor(tensor: torch.Tensor, bucket_hw: tuple[int, int]) -> torch.Tensor:
    """Resize one normalized ``[C,H,W]`` image tensor to ``bucket_hw``."""
    if tuple(tensor.shape[-2:]) == tuple(bucket_hw):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=tuple(bucket_hw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


@dataclass
class BucketBatcherSettings:
    batch_size: int
    max_pending_samples: int = 64
    batch_timeout_s: float = 0.0
    log_prefix: str = "bucket"


class BucketedBatcher:
    """Batch by resolution while bounding pending-sample count.

    ``batch_timeout_s`` is kept as an explicit opt-in diagnostic escape hatch,
    but defaults to disabled.  Wall-clock time includes the period where Python
    generators are paused while the trainer computes a step, so timeout-based
    bucket fallback can otherwise become the normal path for slow large-model
    training.
    """

    def __init__(
        self,
        settings: BucketBatcherSettings,
        *,
        collate_fn: Callable[[list[dict[str, Any]], tuple[int, int]], dict[str, Any]],
        adapt_fn: Callable[[dict[str, Any], tuple[int, int]], dict[str, Any]] | None = None,
    ) -> None:
        if settings.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {settings.batch_size}")
        if 0 < settings.max_pending_samples < settings.batch_size:
            raise ValueError("max_pending_samples must be 0 or >= batch_size")
        if settings.batch_timeout_s < 0:
            raise ValueError(f"batch_timeout_s must be >= 0, got {settings.batch_timeout_s}")
        self.settings = settings
        self.collate_fn = collate_fn
        self.adapt_fn = adapt_fn or (lambda sample, _bucket_hw: sample)
        self.buffers: dict[tuple[int, int], deque[tuple[float, dict[str, Any]]]] = defaultdict(deque)
        self.first_pending_at: float | None = None
        self.exact_batches = 0
        self.fallback_batches = 0
        self.last_progress_at = time.monotonic()

    def add(self, sample: dict[str, Any], bucket_hw: tuple[int, int]) -> dict[str, Any] | None:
        bucket_hw = tuple(bucket_hw)
        self.buffers[bucket_hw].append((time.monotonic(), sample))
        self._refresh_first_pending_time()

        if len(self.buffers[bucket_hw]) >= self.settings.batch_size:
            return self._emit_exact(bucket_hw)

        reason = self._fallback_reason()
        if reason is not None:
            return self._emit_fallback(reason)
        return None

    def _emit_exact(self, bucket_hw: tuple[int, int]) -> dict[str, Any]:
        batch = [self.buffers[bucket_hw].popleft()[1] for _ in range(self.settings.batch_size)]
        self.exact_batches += 1
        self._refresh_first_pending_time()
        self._log_progress()
        return self.collate_fn(batch, bucket_hw)

    def _emit_fallback(self, reason: str) -> dict[str, Any] | None:
        pending = self._pending_count()
        if pending < self.settings.batch_size:
            return None

        target_bucket = max(self.buffers, key=lambda bucket: len(self.buffers[bucket]))
        oldest_bucket = min(
            (bucket for bucket, samples in self.buffers.items() if samples),
            key=lambda bucket: self.buffers[bucket][0][0],
        )
        counts = {bucket: len(samples) for bucket, samples in self.buffers.items() if samples}
        oldest_sample = self.buffers[oldest_bucket].popleft()[1]
        batch = [self.adapt_fn(oldest_sample, target_bucket)]

        while len(batch) < self.settings.batch_size and self.buffers[target_bucket]:
            batch.append(self.buffers[target_bucket].popleft()[1])
        while len(batch) < self.settings.batch_size:
            source_bucket = max(
                (bucket for bucket, samples in self.buffers.items() if samples),
                key=lambda bucket: len(self.buffers[bucket]),
                default=None,
            )
            if source_bucket is None:
                break
            sample = self.buffers[source_bucket].popleft()[1]
            batch.append(self.adapt_fn(sample, target_bucket))

        self.fallback_batches += 1
        self._refresh_first_pending_time()
        logger.warning(
            f"[bucket-fallback] loader={self.settings.log_prefix} reason={reason} "
            f"target_bucket={target_bucket} oldest_bucket={oldest_bucket} pending={pending} counts={counts}"
        )
        self._log_progress(force=True)
        return self.collate_fn(batch, target_bucket)

    def _fallback_reason(self) -> str | None:
        pending = self._pending_count()
        if self.settings.max_pending_samples > 0 and pending >= self.settings.max_pending_samples:
            return f"max_pending_samples={self.settings.max_pending_samples}"
        if self.settings.batch_timeout_s > 0 and self.first_pending_at is not None:
            elapsed = time.monotonic() - self.first_pending_at
            if elapsed >= self.settings.batch_timeout_s and pending >= self.settings.batch_size:
                return f"batch_timeout_s={self.settings.batch_timeout_s:g}"
        return None

    def seconds_until_fallback(self) -> float | None:
        """Return the remaining bucket deadline when enough samples can form a batch."""
        if (
            self.settings.batch_timeout_s <= 0
            or self.first_pending_at is None
            or self._pending_count() < self.settings.batch_size
        ):
            return None
        return max(0.0, self.settings.batch_timeout_s - (time.monotonic() - self.first_pending_at))

    def emit_due_fallback(self) -> dict[str, Any] | None:
        """Emit a fallback whose count/age threshold has become due."""
        reason = self._fallback_reason()
        return None if reason is None else self._emit_fallback(reason)

    def flush_full_batch(self, reason: str = "source_exhausted") -> dict[str, Any] | None:
        """Flush one full pending mixed-bucket batch when the source terminates."""
        if self._pending_count() < self.settings.batch_size:
            return None
        return self._emit_fallback(reason)

    def _pending_count(self) -> int:
        return sum(len(samples) for samples in self.buffers.values())

    def _refresh_first_pending_time(self) -> None:
        empty = [bucket for bucket, samples in self.buffers.items() if not samples]
        for bucket in empty:
            del self.buffers[bucket]
        self.first_pending_at = min((samples[0][0] for samples in self.buffers.values()), default=None)

    def _log_progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_progress_at < 60.0:
            return
        oldest_age = 0.0 if self.first_pending_at is None else now - self.first_pending_at
        logger.info(
            f"[bucket-progress] loader={self.settings.log_prefix} exact_batches={self.exact_batches} "
            f"fallback_batches={self.fallback_batches} pending={self._pending_count()} "
            f"buckets={len(self.buffers)} oldest_pending_s={oldest_age:.1f}"
        )
        self.last_progress_at = now
