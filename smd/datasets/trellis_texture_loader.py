# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebDataset loader for TRELLIS.2 Texture (PBR) distillation data.

Each WDS shard contains per-sample files::

    {sha}.shape.npz   — ``feats: [N, 32]``, ``coords: [N, 3]`` (uint8)
    {sha}.pbr.npz     — ``feats: [N, 32]``, ``coords: [N, 3]`` (uint8)
    {sha}.cond_*.png  — conditioning images (multi-view, optional)

The loader normalises both shape and PBR voxel latents using their
respective per-channel constants and produces batches with:

- ``"real"``:        SparseTensor(feats=[N_total, 32]) — PBR latent (normalised)
- ``"concat_cond"``: SparseTensor(feats=[N_total, 32]) — Shape latent (normalised)
- ``"condition"``:   list[Tensor] — DINOv3 embeddings (dummy if missing)
- ``"neg_condition"``: list[Tensor] — zero DINOv3 embeddings for official CFG

The shape and PBR coords are expected to be identical per sample; the
collate function builds two SparseTensors sharing the same coordinate layout.
"""

from __future__ import annotations
from smd.networks.Trellis2.normalization import PBR_NORM_MEAN, PBR_NORM_STD

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

# Reuse shape normalisation constants
from smd.datasets.trellis_shape_loader import SHAPE_NORM_MEAN, SHAPE_NORM_STD

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-channel PBR normalisation constants from TRELLIS.2 (32 channels)
# Source: pipeline.json → tex_slat_normalization (official TRELLIS.2-4B)
# ---------------------------------------------------------------------------



def _texture_sparse_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate: build two SparseTensors (real + concat_cond) with shared coords.

    Each sample has:
    - ``pbr_feats``:   [N_i, 32] — PBR (texture) latent features
    - ``shape_feats``: [N_i, 32] — Shape latent features (condition)
    - ``coords``:      [N_i, 3]  — shared voxel coordinates

    We prepend batch index to coords → ``[N_total, 4]``, and concatenate feats
    for each modality separately into two SparseTensors.
    """

    from smd.networks.Trellis2.sparse import SparseTensor

    all_coords = []
    all_pbr_feats = []
    all_shape_feats = []
    cond_list = []
    cond_image_list = []  # raw PNG bytes for on-GPU encoding

    for i, s in enumerate(samples):
        pbr_feats = s["pbr_feats"]  # [N_i, 32]
        shape_feats = s["shape_feats"]  # [N_i, 32]
        coords = s["coords"]  # [N_i, 3]
        batch_idx = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
        all_coords.append(torch.cat([batch_idx, coords], dim=1))  # [N_i, 4]
        all_pbr_feats.append(pbr_feats)
        all_shape_feats.append(shape_feats)
        if "cond" in s:
            cond_list.append(s["cond"])
        elif "cond_image_bytes" in s:
            cond_image_list.append(s["cond_image_bytes"])

    cat_coords = torch.cat(all_coords, dim=0)  # [N_total, 4]

    real = SparseTensor(
        feats=torch.cat(all_pbr_feats, dim=0),  # [N_total, 32]
        coords=cat_coords,
    )

    concat_cond = SparseTensor(
        feats=torch.cat(all_shape_feats, dim=0),  # [N_total, 32]
        coords=cat_coords,
    )

    batch: dict[str, Any] = {
        "real": real,
        "concat_cond": concat_cond,
        "fname": [s.get("fname", "<unknown-sample>") for s in samples],
        "shard": [s.get("shard", "<unknown-shard>") for s in samples],
    }

    if cond_list:
        batch["condition"] = cond_list  # list[Tensor], each [L, 1024]
    elif cond_image_list:
        batch["condition_images"] = cond_image_list
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
        batch["neg_condition"] = [torch.zeros_like(c) for c in batch["condition"]]

    return batch


class TrellisTextureWDSDataset(IterableDataset):
    """WebDataset-style iterable dataset for TRELLIS.2 Texture (PBR) data.

    Each sample must contain both ``{sha}.shape.npz`` and ``{sha}.pbr.npz``
    with identical coords.

    Parameters
    ----------
    shard_dir : str
        Path to directory containing ``shard-*.tar`` files.
    normalise : bool
        Whether to normalise feats using PBR_NORM / SHAPE_NORM.
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
    ):
        super().__init__()
        self.shard_dir = shard_dir
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
        log.info(f"TrellisTextureWDSDataset: found {len(self._shard_paths)} shards in {shard_dir}")

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

        with tarfile.open(shard_path, "r:*") as tar:
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
                    samples[sha]["shape_feats"] = torch.from_numpy(npz["feats"].copy()).float()
                    samples[sha]["shape_coords"] = torch.from_numpy(npz["coords"].copy().astype(np.int32))
                elif ext == "pbr.npz":
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    npz = np.load(io.BytesIO(f.read()))
                    samples[sha]["pbr_feats"] = torch.from_numpy(npz["feats"].copy()).float()
                    samples[sha]["pbr_coords"] = torch.from_numpy(npz["coords"].copy().astype(np.int32))
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

        # Yield complete samples (must have both shape and pbr)
        keys = list(samples.keys())
        if self.shuffle:
            import random

            self.rng.shuffle(keys)

        for sha in keys:
            s = samples[sha]
            if not all(k in s for k in ("shape_feats", "shape_coords", "pbr_feats", "pbr_coords")):
                continue

            shape_feats = s["shape_feats"]
            pbr_feats = s["pbr_feats"]
            coords = s["shape_coords"]  # use shape coords as canonical

            if shape_feats.shape[0] > self.max_voxels:
                continue

            # Verify coords match
            if shape_feats.shape[0] != pbr_feats.shape[0]:
                log.warning(
                    f"Skipping sample {sha}: shape ({shape_feats.shape[0]}) and "
                    f"pbr ({pbr_feats.shape[0]}) voxel counts differ"
                )
                continue

            # Normalise
            if self.normalise:
                shape_feats = (shape_feats - SHAPE_NORM_MEAN) / SHAPE_NORM_STD
                pbr_feats = (pbr_feats - PBR_NORM_MEAN) / PBR_NORM_STD

            extras = {}
            if "cond" in s:
                extras["cond"] = s["cond"]
            elif "cond_image_bytes" in s:
                extras["cond_image_bytes"] = s["cond_image_bytes"]

            self._data_error_handler.record_success()
            yield {
                "pbr_feats": pbr_feats,
                "shape_feats": shape_feats,
                "coords": coords,
                "fname": sha,
                "shard": shard_path,
                **extras,
            }


class TrellisTextureWDSLoader:
    """Loader wrapping :class:`TrellisTextureWDSDataset` with a ``DataLoader``.

    Parameters
    ----------
    shard_dir : str
        Path to WDS shard directory.
    batch_size : int
        Number of samples per batch.
    normalise : bool
        Whether to normalise using PBR_NORM / SHAPE_NORM.
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
    ):
        self.dataset = TrellisTextureWDSDataset(
            shard_dir=shard_dir,
            normalise=normalise,
            max_voxels=max_voxels,
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
            collate_fn=_texture_sparse_collate,
            num_workers=num_workers,
            pin_memory=False,  # SparseTensor doesn't support pin_memory
            worker_timeout_s=self.worker_timeout_s,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        return self._resume.iterate(self.loader)
