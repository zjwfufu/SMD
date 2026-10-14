# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebDataset loader for TRELLIS.2 Shape distillation data.

Each WDS shard contains per-sample files::

    {sha}.shape.npz   — ``feats: [N, 32]``, ``coords: [N, 3]`` (uint8)
    {sha}.cond_*.png   — conditioning images (multi-view)

The loader normalises voxel latents using TRELLIS.2 ``SHAPE_NORM`` and
produces batches with a :class:`SparseTensor` ``"real"`` field.
"""

from __future__ import annotations
from smd.networks.Trellis2.normalization import SHAPE_NORM_MEAN, SHAPE_NORM_STD

import io
import logging
from typing import Any

import numpy as np
import torch
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
from smd.datasets.shard_cache import local_shard

log = logging.getLogger(__name__)

# Per-channel normalisation constants from TRELLIS.2 (32 channels)



def _sparse_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate: concatenate per-sample voxel data into a SparseTensor.

    Each sample has ``feats: [N_i, 32]`` and ``coords: [N_i, 3]``.
    We prepend batch index to coords → ``[N_total, 4]``, and concatenate feats.
    """

    from smd.networks.Trellis2.sparse import SparseTensor

    all_coords = []
    all_feats = []
    cond_list = []
    cond_image_list = []  # raw PNG bytes for on-GPU encoding

    for i, s in enumerate(samples):
        feats = s["feats"]  # [N_i, 32]
        coords = s["coords"]  # [N_i, 3]
        batch_idx = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
        all_coords.append(torch.cat([batch_idx, coords], dim=1))  # [N_i, 4]
        all_feats.append(feats)
        if "cond" in s:
            cond_list.append(s["cond"])
        elif "cond_image_bytes" in s:
            cond_image_list.append(s["cond_image_bytes"])

    real = SparseTensor(
        feats=torch.cat(all_feats, dim=0),  # [N_total, 32]
        coords=torch.cat(all_coords, dim=0),  # [N_total, 4]
    )

    batch: dict[str, Any] = {
        "real": real,
        "fname": [s.get("fname", "<unknown-sample>") for s in samples],
        "shard": [s.get("shard", "<unknown-shard>") for s in samples],
    }

    if cond_list:
        batch["condition"] = cond_list  # list[Tensor], each [L, 1024] — variable length
    elif cond_image_list:
        # Raw PNG bytes — will be encoded by net.encode_image() in preprocess_data
        batch["condition_images"] = cond_image_list
        # Placeholder so downstream code doesn't crash before encoding
        B = len(samples)
        batch["condition"] = [torch.zeros(1, 1024) for _ in range(B)]
    else:
        # Dummy condition if no cond data is available
        B = len(samples)
        batch["condition"] = [torch.zeros(1, 1024) for _ in range(B)]

    # CFG requires neg_condition = zeros_like(cond) per TRELLIS.2 official pipeline
    if cond_list:
        batch["neg_condition"] = [torch.zeros_like(c) for c in cond_list]
    else:
        # Dummy cond was created above; create matching zero neg_cond
        batch["neg_condition"] = [torch.zeros_like(c) for c in batch["condition"]]

    return batch


class TrellisShapeWDSDataset(IterableDataset):
    """WebDataset-style iterable dataset for TRELLIS.2 Shape data.

    Parameters
    ----------
    shard_dir : str
        Path to directory containing ``shard-*.tar`` files.
    normalise : bool
        Whether to normalise feats using ``SHAPE_NORM``.
    max_voxels : int
        Maximum number of voxels per sample (skip if exceeded).
    shuffle : bool
        Whether to shuffle shards and samples.
    """

    def __init__(
        self,
        shard_dir: str,
        normalise: bool = True,
        max_voxels: int = 65536,
        shuffle: bool = True,
        max_consecutive_data_errors: int = 32,
        max_empty_cycles: int = 2,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.cache_dir = cache_dir
        self.normalise = normalise
        self.max_voxels = max_voxels
        self.shuffle = shuffle
        if max_empty_cycles < 1:
            raise ValueError(f"max_empty_cycles must be >= 1, got {max_empty_cycles}")
        self.max_empty_cycles = max_empty_cycles
        self._data_error_handler = BoundedDataErrorHandler(
            max_consecutive_errors=max_consecutive_data_errors,
            datatags=[shard_dir],
        )
        self._progress_logger = DataProgressLogger(self.__class__.__name__)

        # Discover shards
        import glob

        self._shard_paths = sorted(glob.glob(f"{shard_dir}/shard-*.tar"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")
        log.info(f"TrellisShapeWDSDataset: found {len(self._shard_paths)} shards in {shard_dir}")

    def __iter__(self):
        reset_stream_rng(self)
        import random

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
                except Exception as exc:
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
        import tarfile

        # Group files by sample key (sha hash)
        samples: dict[str, dict[str, Any]] = {}

        with local_shard(shard_path, self.cache_dir) as staged, tarfile.open(staged, "r:*") as tar:
            for member in tar:
                if member.isdir():
                    continue
                name = member.name
                # Parse: {sha}.{ext}
                parts = name.split(".", 1)
                if len(parts) < 2:
                    continue
                sha = parts[0]
                ext = parts[1]

                if sha not in samples:
                    samples[sha] = {}

                if ext == "shape.npz":
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    npz = np.load(io.BytesIO(f.read()))
                    samples[sha]["feats"] = torch.from_numpy(npz["feats"].copy()).float()
                    # coords are uint8 [N, 3] — convert to int32
                    samples[sha]["coords"] = torch.from_numpy(npz["coords"].copy().astype(np.int32))

                elif ext == "cond_dinov3.npy":
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    feat = np.load(io.BytesIO(f.read()))  # [N_patches, 1024]
                    samples[sha]["cond"] = torch.from_numpy(feat.copy()).float()

                elif ext == "cond_000.png" and "cond" not in samples.get(sha, {}):
                    # Fallback: store raw PNG bytes for on-GPU DINOv3 encoding
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    samples[sha]["cond_image_bytes"] = f.read()

        # Yield complete samples
        keys = list(samples.keys())
        if self.shuffle:
            import random

            self.rng.shuffle(keys)

        for sha in keys:
            s = samples[sha]
            if "feats" not in s or "coords" not in s:
                continue
            feats = s["feats"]
            coords = s["coords"]

            if feats.shape[0] > self.max_voxels:
                continue

            # Normalise
            if self.normalise:
                feats = (feats - SHAPE_NORM_MEAN) / SHAPE_NORM_STD

            extras = {}
            if "cond" in s:
                extras["cond"] = s["cond"]
            elif "cond_image_bytes" in s:
                extras["cond_image_bytes"] = s["cond_image_bytes"]

            self._data_error_handler.record_success()
            yield {"feats": feats, "coords": coords, "fname": sha, "shard": shard_path, **extras}


class TrellisShapeWDSLoader:
    """Loader wrapping :class:`TrellisShapeWDSDataset` with a ``DataLoader``.

    Parameters
    ----------
    shard_dir : str
        Path to WDS shard directory.
    batch_size : int
        Number of samples per batch.
    normalise : bool
        Whether to normalise using ``SHAPE_NORM``.
    max_voxels : int
        Maximum voxels per sample.
    num_workers : int
        DataLoader workers.
    shuffle : bool
        Shuffle shards and samples.
    """

    def __init__(
        self,
        shard_dir: str,
        batch_size: int = 2,
        normalise: bool = True,
        max_voxels: int = 65536,
        num_workers: int = 4,
        shuffle: bool = True,
        worker_timeout_s: float = 300.0,
        max_consecutive_data_errors: int = 32,
        max_empty_cycles: int = 2,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        sampler_start_idx: int | None = None,
        seed: int = 0,
        cache_dir: str | None = None,
    ):
        self.dataset = TrellisShapeWDSDataset(
            shard_dir=shard_dir,
            normalise=normalise,
            max_voxels=max_voxels,
            shuffle=shuffle,
            max_consecutive_data_errors=max_consecutive_data_errors,
            max_empty_cycles=max_empty_cycles,
            cache_dir=cache_dir,
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
            collate_fn=_sparse_collate,
            num_workers=num_workers,
            pin_memory=False,  # SparseTensor doesn't support pin_memory
            worker_timeout_s=self.worker_timeout_s,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        return self._resume.iterate(self.loader)
