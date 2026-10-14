# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for tar-backed streaming datasets."""

from __future__ import annotations

import os
import random
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

import smd.utils.logging_utils as logger


class BackgroundIteratorTimeout(TimeoutError):
    """Raised only when the caller's wait deadline expires."""


@dataclass(frozen=True)
class DataWorkerContext:
    rank: int
    world_size: int
    worker_id: int
    num_workers: int

    @property
    def slot(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    @property
    def total_slots(self) -> int:
        return self.world_size * self.num_workers


def get_data_worker_context() -> DataWorkerContext:
    """Return distributed and DataLoader worker identity with a spawn-safe fallback."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    num_workers = worker_info.num_workers if worker_info is not None else 1
    return DataWorkerContext(
        rank=rank,
        world_size=max(1, world_size),
        worker_id=worker_id,
        num_workers=max(1, num_workers),
    )


def resolve_num_workers(
    num_workers: int | None,
    *,
    loader_name: str,
    shard_count: int | None = None,
    auto_cap: int = 1,
) -> int:
    """Resolve worker count conservatively without silently disabling timeouts."""
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        resolved = int(num_workers)
    else:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        cpu_count = os.cpu_count() or 1
        resolved = max(1, min(max(1, cpu_count // world_size), auto_cap))

    if resolved > 0 and shard_count is not None:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        per_rank_shards = max(1, shard_count // world_size)
        capped = min(resolved, per_rank_shards)
        if capped != resolved:
            logger.warning(
                "[data-workers-capped] "
                f"loader={loader_name} requested={resolved} resolved={capped} "
                f"shards={shard_count} world_size={world_size}"
            )
        resolved = capped

    logger.info(
        f"[data-workers] loader={loader_name} requested={num_workers!r} resolved={resolved} shards={shard_count}"
    )
    return resolved


def assign_shards_to_worker(shards: Sequence[str]) -> list[str]:
    """Partition by rank then worker, replicating only when ranks outnumber shards."""
    if not shards:
        return []
    context = get_data_worker_context()
    rank_shards = list(shards[context.rank :: context.world_size])
    if not rank_shards:
        rank_shards = [shards[context.rank % len(shards)]]
    return rank_shards[context.worker_id :: context.num_workers]


def log_shard_assignment(loader_name: str, shards: Sequence[str]) -> None:
    context = get_data_worker_context()
    logger.info(
        "[data-shards] "
        f"loader={loader_name} rank={context.rank}/{context.world_size} "
        f"worker={context.worker_id}/{context.num_workers} assigned={len(shards)} "
        f"first={shards[0] if shards else '<none>'!r}"
    )


def create_streaming_dataloader(
    dataset: IterableDataset,
    *,
    loader_name: str,
    batch_size: int | None,
    num_workers: int,
    worker_timeout_s: float,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Build a DataLoader with consistent timeout and worker lifecycle settings."""
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if worker_timeout_s < 0:
        raise ValueError(f"worker_timeout_s must be >= 0, got {worker_timeout_s}")
    if prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

    effective_timeout = float(worker_timeout_s) if num_workers > 0 else 0.0
    if num_workers == 0 and worker_timeout_s > 0:
        logger.warning(
            "[data-safety-disabled] "
            f"loader={loader_name} num_workers=0 disables worker_timeout_s={worker_timeout_s:g}; "
            "blob-backed reads will run in the training process"
        )

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "timeout": effective_timeout,
        "generator": torch.Generator().manual_seed(getattr(dataset, 'seed', 0)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor

    logger.info(
        "[data-loader] "
        f"loader={loader_name} workers={num_workers} batch_size={batch_size} "
        f"timeout_s={effective_timeout:g} persistent_workers={num_workers > 0 and persistent_workers} "
        f"prefetch_factor={prefetch_factor if num_workers > 0 else 0}"
    )
    return DataLoader(**kwargs)


def reset_stream_rng(dataset):
    """Keep worker shuffling independent of model RNG and reproducible on replay."""
    dataset.rng = random.Random(getattr(dataset, 'seed', 0) + get_data_worker_context().slot)


class StreamingResume:
    """Resume at a global consumed-sample count by replaying complete local batches.

    Requires unchanged shards, seed, batch size and rank/worker topology. Replay
    trades startup reads for bounded memory; it does not retain prefetched data.
    """

    def __init__(self, dataset, batch_size, sampler_start_idx=None, seed=0):
        dataset.seed = int(seed)
        reset_stream_rng(dataset)
        stride = int(batch_size) * get_data_worker_context().world_size
        start = int(sampler_start_idx or 0)
        if stride < 1 or start < 0 or start % stride:
            raise ValueError('sampler_start_idx must align with a complete global batch')
        self.skip_batches = start // stride

    def iterate(self, batches):
        iterator = iter(batches)
        try:
            for _ in range(self.skip_batches):
                next(iterator)
            yield from iterator
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()


class DataProgressLogger:
    """Log first-sample latency and low-frequency worker progress."""

    def __init__(self, loader_name: str, *, interval_s: float = 60.0, every_n: int = 1000):
        self.loader_name = loader_name
        self.interval_s = interval_s
        self.every_n = every_n
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.samples = 0

    @staticmethod
    def _origin(sample: Any) -> str:
        if not isinstance(sample, dict):
            return f"type={type(sample).__name__}"
        shard = sample.get("shard", sample.get("__url__", "<unknown-shard>"))
        key = sample.get("fname", sample.get("__key__", "<unknown-sample>"))
        return f"shard={shard!r} sample={key!r}"

    def record_sample(self, sample: Any) -> None:
        self.samples += 1
        now = time.monotonic()
        context = get_data_worker_context()
        if self.samples == 1:
            logger.info(
                "[data-first-sample] "
                f"loader={self.loader_name} rank={context.rank} "
                f"worker={context.worker_id}/{context.num_workers} elapsed={now - self.started_at:.2f}s "
                f"{self._origin(sample)}"
            )
            self.last_log_at = now
            return
        if (self.every_n > 0 and self.samples % self.every_n == 0) or now - self.last_log_at >= self.interval_s:
            elapsed = max(now - self.started_at, 1e-9)
            logger.info(
                "[data-progress] "
                f"loader={self.loader_name} rank={context.rank} "
                f"worker={context.worker_id}/{context.num_workers} samples={self.samples} "
                f"rate={self.samples / elapsed:.2f}/s {self._origin(sample)}"
            )
            self.last_log_at = now


class BackgroundIterator:
    """Pull one item ahead so callers can enforce deadlines around a blocking iterator."""

    def __init__(self, iterable: Any):
        self._iterable = iterable
        self._iterator = iter(iterable)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        self._cancel = threading.Event()
        self._terminal: tuple[str, Any] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _publish(self, kind: str, value: Any) -> bool:
        while not self._cancel.is_set():
            try:
                self._queue.put((kind, value), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for item in self._iterator:
                if not self._publish("item", item):
                    return
        except BaseException as exc:
            self._publish("error", exc)
        else:
            self._publish("stop", None)

    def next(self, timeout_s: float | None = None) -> Any:
        if self._terminal is not None:
            kind, value = self._terminal
            if kind == "error":
                raise value
            raise StopIteration
        try:
            kind, value = self._queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise BackgroundIteratorTimeout("background iterator deadline expired") from exc
        if kind == "item":
            return value
        self._terminal = (kind, value)
        if kind == "error":
            raise value
        raise StopIteration

    def close(self) -> None:
        self._cancel.set()
        shutdown_workers = getattr(self._iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        if getattr(self._iterable, "_iterator", None) is self._iterator:
            self._iterable._iterator = None
        close = getattr(self._iterator, "close", None)
        if callable(close):
            try:
                close()
            except ValueError:
                pass
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
