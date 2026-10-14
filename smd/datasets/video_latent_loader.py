# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered video latent WebDataset loading with global sample-offset resume."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import torch
from torch.utils.data import IterableDataset

from smd.datasets.streaming_runtime import (
    DataProgressLogger,
    create_streaming_dataloader,
    get_data_worker_context,
)
from smd.datasets.wds_utils import get_shards
from smd.datasets.shard_cache import ShardPrefetcher


class VideoLatentDataset(IterableDataset):
    """Read consecutive latent/embed/prompt triples, partitioning records by rank.

    Unselected tensor payloads are skipped by seeking through tar headers. A
    filename-to-count JSON optionally skips complete shards when resuming.
    Malformed records fail instead of silently changing the sample sequence.
    """

    suffixes = {"real": ".latent.pt", "condition": ".embed.pt", "prompt": ".prompt.txt"}

    def __init__(self, datatags, sampler_start_idx=None, shard_count_file=None, cache_dir=None):
        super().__init__()
        self.shards = get_shards(datatags)
        self.cache_dir = cache_dir
        self.start = int(sampler_start_idx or 0)
        if self.start < 0:
            raise ValueError("sampler_start_idx must be nonnegative")
        self.counts = None
        if shard_count_file:
            counts = json.loads(Path(shard_count_file).read_text())
            self.counts = [counts[Path(p).name] for p in self.shards]
            if any(type(n) is not int or n <= 0 for n in self.counts):
                raise ValueError("Shard counts must be positive integers")

    @classmethod
    def records(cls, archive):
        key, fields = None, {}
        for member in archive:
            if not member.isfile():
                continue
            match = next(((k, s) for k, s in cls.suffixes.items() if member.name.endswith(s)), None)
            if match is None:
                continue
            field, suffix = match
            current = member.name[: -len(suffix)]
            if key is not None and current != key:
                if fields.keys() != cls.suffixes.keys():
                    raise ValueError(f"Incomplete video record {key!r}: {sorted(fields)}")
                yield key, fields
                fields = {}
            key = current
            if field in fields:
                raise ValueError(f"Duplicate {field} in video record {key!r}")
            fields[field] = member
        if key is not None:
            if fields.keys() != cls.suffixes.keys():
                raise ValueError(f"Incomplete video record {key!r}: {sorted(fields)}")
            yield key, fields

    @staticmethod
    def decode(archive, fields):
        result = {}
        for field, member in fields.items():
            with archive.extractfile(member) as handle:
                payload = handle.read()
            result[field] = (
                payload.decode("utf-8")
                if field == "prompt"
                else torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            )
        for field, dimensions in (("real", 4), ("condition", 2)):
            value = result[field]
            if not torch.is_tensor(value) or value.ndim != dimensions or not value.is_floating_point():
                raise ValueError(f"{field} must be a floating tensor with {dimensions} dimensions")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {field} in cached video record")
        if not result["prompt"].strip():
            raise ValueError("Empty video prompt")
        return result

    def __iter__(self):
        with ShardPrefetcher(self.cache_dir) as prefetch:
            yield from self._iterate(prefetch)

    def _iterate(self, prefetch):
        context = get_data_worker_context()
        if context.num_workers != 1:
            raise ValueError("VideoLatentDataset supports one worker per rank")
        target = self.start + context.rank
        cursor = 0
        if self.counts:
            total = sum(self.counts)
            cursor = (target // total) * total
        progress = DataProgressLogger(type(self).__name__)
        while True:
            cycle_start = cursor
            for i, path in enumerate(self.shards):
                expected = self.counts[i] if self.counts else None
                if expected is not None and cursor + expected <= target:
                    cursor += expected
                    continue
                count = 0
                # The production format is an uncompressed, seekable .tar.
                next_path = self.shards[(i + 1) % len(self.shards)]
                with prefetch.open(path, next_path) as staged, tarfile.open(staged, "r:") as archive:
                    for key, fields in self.records(archive):
                        if cursor == target:
                            sample = self.decode(archive, fields)
                            sample.update(fname=key, shard=path, condition_raw=sample["prompt"])
                            progress.record_sample(sample)
                            yield sample
                            target += context.world_size
                        cursor += 1
                        count += 1
                if expected is not None and count != expected:
                    raise ValueError(f"Shard count changed for {path}: {count} != {expected}")
            if cursor == cycle_start:
                raise ValueError("Video dataset contains no complete records")


class VideoLatentLoader:
    """Load cached video tensors without re-encoding training prompts or pixels.

    Records are ordered by shard filename and tar member order. Use zero workers
    for local inspection or one worker with a timeout for training. Keep dataset,
    world size and batch size fixed across checkpoint continuation. ``cache_dir``
    optionally stages the current and next shard per worker on node-local storage.
    """

    def __init__(
        self,
        datatags,
        batch_size=2,
        num_workers=1,
        sampler_start_idx=None,
        sampler_base_offset=0,
        shard_count_file=None,
        worker_timeout_s=300,
        persistent_workers=True,
        prefetch_factor=2,
        cache_dir=None,
    ):
        if num_workers not in (0, 1):
            raise ValueError("VideoLatentLoader requires num_workers=0 or 1 for ordered resume")
        start = int(sampler_base_offset if sampler_start_idx is None else sampler_start_idx)
        stride = int(batch_size) * get_data_worker_context().world_size
        if stride < 1 or start < 0 or start % stride:
            raise ValueError("sampler_start_idx must align with a complete global batch")
        self.dataset = VideoLatentDataset(datatags, start, shard_count_file, cache_dir)
        self.batch_size = batch_size
        self.loader = create_streaming_dataloader(
            self.dataset,
            loader_name=type(self).__name__,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_timeout_s=worker_timeout_s,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        yield from self.loader
