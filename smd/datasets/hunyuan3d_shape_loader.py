# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-free loader for Hunyuan3D 2.1 Shape distillation.

Reads the shared 3D WebDataset shards and takes only the
conditioning image (``{sha}.cond_000.png``) from each sample — no shape latents.
Distillation is data-free: the student starts from noise and imitates the
teacher trajectory, conditioned solely on the image. So batches carry
``condition_images`` (raw PNG bytes) and NO ``real`` field.

The trainer's ``preprocess_data`` encodes ``condition_images`` on-GPU via
``net.encode_image`` (DINOv2-Large -> ``[1370, 1024]`` features) and fills
``condition`` / ``neg_condition`` (zeros for CFG). See ``smd/trainer.py``.
"""

from __future__ import annotations

import glob
import logging
import random
import tarfile
from typing import Any

from torch.utils.data import IterableDataset

from smd.datasets.streaming_runtime import (
    StreamingResume, reset_stream_rng,
    DataProgressLogger,
    assign_shards_to_worker,
    create_streaming_dataloader,
    log_shard_assignment,
    resolve_num_workers,
)
from smd.datasets.wds_utils import BoundedDataErrorHandler, DataErrorLimitExceeded

log = logging.getLogger(__name__)


def _image_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate raw PNG bytes into a data-free conditioning batch."""
    return {
        "condition_images": [s["cond_image_bytes"] for s in samples],
        "fname": [s.get("fname", "<unknown-sample>") for s in samples],
        "shard": [s.get("shard", "<unknown-shard>") for s in samples],
    }


class Hunyuan3DShapeImageDataset(IterableDataset):
    """Iterable dataset yielding only the condition image (PNG bytes) per sample.

    Parameters
    ----------
    shard_dir : str
        Directory containing ``shard-*.tar`` files (the TRELLIS 3D data shards).
    cond_key : str
        Which conditioning view to use (default ``"cond_000.png"`` — first view).
    shuffle : bool
        Shuffle shard order and per-shard samples.
    """

    def __init__(
        self,
        shard_dir: str,
        cond_key: str = "cond_000.png",
        shuffle: bool = True,
        max_consecutive_data_errors: int = 32,
        max_empty_cycles: int = 2,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.cond_key = cond_key
        self.shuffle = shuffle
        if max_empty_cycles < 1:
            raise ValueError(f"max_empty_cycles must be >= 1, got {max_empty_cycles}")
        self.max_empty_cycles = max_empty_cycles
        self._data_error_handler = BoundedDataErrorHandler(
            max_consecutive_errors=max_consecutive_data_errors,
            datatags=[shard_dir],
        )
        self._progress_logger = DataProgressLogger(self.__class__.__name__)
        self._shard_paths = sorted(glob.glob(f"{shard_dir}/shard-*.tar"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")
        log.info(f"Hunyuan3DShapeImageDataset: {len(self._shard_paths)} shards in {shard_dir} (cond_key={cond_key})")

    def __iter__(self):
        reset_stream_rng(self)
        shard_paths = assign_shards_to_worker(self._shard_paths)
        if not shard_paths:
            raise RuntimeError(f"No shards assigned to {self.__class__.__name__} worker")
        log_shard_assignment(self.__class__.__name__, shard_paths)
        if self.shuffle:
            self.rng.shuffle(shard_paths)
        empty_cycles = 0
        while True:  # infinite iteration for training
            yielded = 0
            for shard_path in shard_paths:
                try:
                    for sample in self._iter_shard(shard_path):
                        yielded += 1
                        self._progress_logger.record_sample(sample)
                        yield sample
                except Exception as exc:  # noqa: BLE001
                    if isinstance(exc, DataErrorLimitExceeded):
                        raise
                    self._data_error_handler.record_failure(exc, context=f"shard={shard_path!r}")
                    continue
            if yielded:
                empty_cycles = 0
            else:
                empty_cycles += 1
                log.error(
                    "[data-empty-cycle] loader=%s assigned_shards=%d empty_cycles=%d/%d",
                    self.__class__.__name__,
                    len(shard_paths),
                    empty_cycles,
                    self.max_empty_cycles,
                )
                if empty_cycles >= self.max_empty_cycles:
                    raise DataErrorLimitExceeded(
                        f"{self.__class__.__name__} produced no valid samples for {empty_cycles} complete shard cycles"
                    )
            if self.shuffle:
                self.rng.shuffle(shard_paths)

    def _iter_shard(self, shard_path: str):
        suffix = "." + self.cond_key
        images: dict[str, bytes] = {}
        with tarfile.open(shard_path, "r:*") as tar:
            for member in tar:
                if member.isdir() or not member.name.endswith(suffix):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                sha = member.name[: -len(suffix)]
                images[sha] = f.read()

        keys = list(images.keys())
        if self.shuffle:
            self.rng.shuffle(keys)
        for sha in keys:
            self._data_error_handler.record_success()
            yield {"cond_image_bytes": images[sha], "fname": sha, "shard": shard_path}


class Hunyuan3DShapeImageLoader:
    """DataLoader wrapper around :class:`Hunyuan3DShapeImageDataset`."""

    def __init__(
        self,
        shard_dir: str,
        batch_size: int = 8,
        cond_key: str = "cond_000.png",
        num_workers: int = 4,
        shuffle: bool = True,
        worker_timeout_s: float = 300.0,
        max_consecutive_data_errors: int = 32,
        max_empty_cycles: int = 2,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        sampler_start_idx: int | None = None,
        seed: int = 0,
    ):
        self.dataset = Hunyuan3DShapeImageDataset(
            shard_dir=shard_dir,
            cond_key=cond_key,
            shuffle=shuffle,
            max_consecutive_data_errors=max_consecutive_data_errors,
            max_empty_cycles=max_empty_cycles,
        )
        self._resume = StreamingResume(self.dataset, batch_size, sampler_start_idx, seed)
        num_workers = resolve_num_workers(
            num_workers,
            loader_name=self.__class__.__name__,
            shard_count=len(self.dataset._shard_paths),
            auto_cap=8,
        )
        self.shard_dir = shard_dir
        self.datatags = [shard_dir]
        self.num_workers = num_workers
        self.worker_timeout_s = float(worker_timeout_s)
        self.loader = create_streaming_dataloader(
            self.dataset,
            loader_name=self.__class__.__name__,
            batch_size=batch_size,
            collate_fn=_image_collate,
            num_workers=num_workers,
            pin_memory=False,
            worker_timeout_s=self.worker_timeout_s,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        return self._resume.iterate(self.loader)
