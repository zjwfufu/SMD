# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tar-packed JSONL loader for paired image editing data.

Each tar shard contains multiple JSONL files, each with ~1000 records.
Each record has base64-encoded source_img and edited_img, plus multiple
instruction granularity fields. This loader:

- Discovers .tar shards via get_shards() (reuses WDS shard discovery)
- Extracts JSONL members, parses lines into individual records
- Decodes both source and edited images from base64
- Randomly selects from English instruction fields per record
- Supports aspect-ratio bucketed batching
- Distributes shards across workers/ranks

Output dict:
    {
        "real": edited_tensor,       # [B, 3, H, W] — target (for VAE encoding)
        "source": source_tensor,     # [B, 3, H, W] — source (for VAE encoding to latent concat)
        "source_bytes": [bytes, ...], # raw JPEG bytes — for VLM text encoder (Qwen2VLProcessor)
        "condition": [str, ...],     # edit instruction texts
        "bucket_hw": (H, W),        # bucket resolution
    }
"""

import base64
import io
import json
import math
import random
import tarfile
import time
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

from smd.datasets.bucketed_batching import BucketBatcherSettings, BucketedBatcher, resize_chw_tensor
from smd.datasets.bucketed_image_loader import crop_image_to_bucket, generate_crop_size_list, smart_crop_to_bucket
from smd.datasets.streaming_runtime import (
    StreamingResume, reset_stream_rng,
    BackgroundIterator,
    BackgroundIteratorTimeout,
    DataProgressLogger,
    assign_shards_to_worker,
    create_streaming_dataloader,
    log_shard_assignment,
    resolve_num_workers,
)
from smd.datasets.wds_utils import BoundedDataErrorHandler, DataErrorLimitExceeded, get_shards
import smd.utils.logging_utils as logger

# English instruction fields to randomly sample from
_DEFAULT_EDIT_CAPTION_KEYS = ("short_ins", "medium_ins", "long_ins", "ori_ins")


class EditImageTarDataset(IterableDataset):
    """IterableDataset that reads tar-packed JSONL edit data.

    Each tar contains ~20 JSONL files, each JSONL has ~1000 records.
    Records contain base64 source_img + edited_img + instruction fields.
    """

    def __init__(
        self,
        shard_urls: List[str],
        caption_keys: Tuple[str, ...] = _DEFAULT_EDIT_CAPTION_KEYS,
        buckets: Optional[List[Tuple[int, int]]] = None,
        norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        shuffle_buffer: int = 1000,
        shuffle_initial: int = 32,
        condition_image_size: int = 384 * 384,
        max_consecutive_data_errors: int = 32,
        shard_open_retries: int = 3,
        max_empty_cycles: int = 2,
    ):
        super().__init__()
        if shuffle_buffer < 0:
            raise ValueError(f"shuffle_buffer must be >= 0, got {shuffle_buffer}")
        if shuffle_initial < 1:
            raise ValueError(f"shuffle_initial must be >= 1, got {shuffle_initial}")
        if shard_open_retries < 1:
            raise ValueError(f"shard_open_retries must be >= 1, got {shard_open_retries}")
        if max_empty_cycles < 1:
            raise ValueError(f"max_empty_cycles must be >= 1, got {max_empty_cycles}")
        self.shard_urls = shard_urls
        self.caption_keys = caption_keys
        self.buckets = buckets
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = min(shuffle_initial, shuffle_buffer) if shuffle_buffer > 0 else 1
        self.condition_image_size = condition_image_size
        self.shard_open_retries = shard_open_retries
        self.max_empty_cycles = max_empty_cycles
        self._data_error_handler = BoundedDataErrorHandler(
            max_consecutive_errors=max_consecutive_data_errors,
            datatags=shard_urls[:1],
        )
        self._progress_logger = DataProgressLogger(self.__class__.__name__)
        self._tensor_transform = Compose(
            [
                Lambda(lambda img: img.convert("RGB")),
                ToTensor(),
                Normalize(norm_mean, norm_std),
            ]
        )

    def _worker_shard_slice(self) -> List[str]:
        """Return a balanced global rank/worker shard slice."""
        return assign_shards_to_worker(self.shard_urls)

    def _iter_records_from_tar(self, tar_path: str) -> Iterator[dict]:
        """Open a tar, iterate JSONL members, yield individual records."""
        for attempt in range(1, self.shard_open_retries + 1):
            emitted = 0
            started = time.monotonic()
            try:
                logger.info(
                    f"[data-shard-open] loader={self.__class__.__name__} shard={tar_path!r} "
                    f"attempt={attempt}/{self.shard_open_retries}"
                )
                with tarfile.open(tar_path, "r:*") as tf:
                    for member in tf:
                        if not member.name.endswith(".jsonl"):
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        text_stream = io.TextIOWrapper(f, encoding="utf-8")
                        for line_number, line in enumerate(text_stream, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                record["__data_shard__"] = tar_path
                                record["__data_member__"] = member.name
                                record["__data_key__"] = f"{member.name}:{line_number}"
                                emitted += 1
                                yield record
                            except json.JSONDecodeError as exc:
                                self._data_error_handler.record_failure(
                                    exc,
                                    context=(f"shard={tar_path!r} member={member.name!r} line={line_number}"),
                                )
                logger.info(
                    f"[data-shard-done] loader={self.__class__.__name__} shard={tar_path!r} "
                    f"records={emitted} elapsed={time.monotonic() - started:.2f}s"
                )
                return
            except Exception as exc:
                if isinstance(exc, DataErrorLimitExceeded):
                    raise
                self._data_error_handler.record_failure(
                    exc,
                    context=(
                        f"shard={tar_path!r} attempt={attempt}/{self.shard_open_retries} records_before_error={emitted}"
                    ),
                )
                if emitted > 0 or attempt == self.shard_open_retries:
                    return
                time.sleep(attempt)

    def _buffered_shuffle(self, iterator: Iterator, buffer_size: int, initial_size: int) -> Iterator:
        """Reservoir shuffle that yields before the full buffer is populated."""
        if buffer_size <= 1:
            yield from iterator
            return

        active_size = min(max(1, initial_size), buffer_size)
        buf = []
        for item in iterator:
            buf.append(item)
            if len(buf) >= active_size:
                idx = self.rng.randint(0, len(buf) - 1)
                yield buf[idx]
                buf[idx] = buf[-1]
                buf.pop()
                if active_size < buffer_size:
                    active_size += 1
        # Flush remaining
        self.rng.shuffle(buf)
        yield from buf

    def _calculate_condition_size(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate condition image size preserving aspect ratio.

        Matches QwenImageEditPlusPipeline.calculate_dimensions logic.
        """
        ratio = width / height
        cond_w = math.sqrt(self.condition_image_size * ratio)
        cond_h = cond_w / ratio
        cond_w = round(cond_w / 32) * 32
        cond_h = round(cond_h / 32) * 32
        return int(cond_w), int(cond_h)

    def _process_record(self, record):
        try:
            from smd.networks.QwenImageEdit.preprocessing import source_pixels
            source_bytes = base64.b64decode(record['source_img'], validate=True)
            edited_bytes = base64.b64decode(record['edited_img'], validate=True)
            with Image.open(BytesIO(source_bytes)) as image:
                source_pil = image.convert('RGB').copy()
            with Image.open(BytesIO(edited_bytes)) as image:
                edited_pil = image.convert('RGB').copy()
            captions = [record[key] for key in self.caption_keys if record.get(key)]
            if not captions:
                raise ValueError('No non-empty configured edit instruction')
            if self.buckets is None:
                source_tensor = source_pixels(source_pil)
                bucket_hw = tuple(source_tensor.shape[-2:])
                # The paired target shares source geometry. It is not an input
                # to the source-image text encoder.
                target = edited_pil.resize((bucket_hw[1], bucket_hw[0]), Image.BICUBIC)
                edited_tensor = self._tensor_transform(target)
            else:
                bucket_hw, source_crop = smart_crop_to_bucket(source_pil, self.buckets)
                source_tensor = self._tensor_transform(source_crop)
                edited_tensor = self._tensor_transform(crop_image_to_bucket(edited_pil, bucket_hw))
            return {'source': source_tensor, 'real': edited_tensor,
                    'source_pil': source_pil, 'condition': self.rng.choice(captions),
                    'bucket_hw': bucket_hw,
                    'fname': record.get('id', record.get('__data_key__', '<unknown-sample>')),
                    'shard': record.get('__data_shard__', '<unknown-shard>')}
        except Exception as error:
            self._data_error_handler.record_failure(error, context=f"shard={record.get('__data_shard__')!r}")
            return None


    def __iter__(self) -> Iterator[Dict[str, Any]]:
        reset_stream_rng(self)
        shards = self._worker_shard_slice()
        if not shards:
            raise RuntimeError("No shards assigned to EditImageTarDataset worker")
        log_shard_assignment(self.__class__.__name__, shards)

        # Infinite loop over shards for training
        empty_cycles = 0
        while True:
            if self.shuffle_buffer > 0:
                self.rng.shuffle(shards)
            all_records = (rec for url in shards for rec in self._iter_records_from_tar(url))
            shuffled = self._buffered_shuffle(all_records, self.shuffle_buffer, self.shuffle_initial)
            yielded = 0
            for record in shuffled:
                sample = self._process_record(record)
                if sample is not None:
                    self._data_error_handler.record_success()
                    self._progress_logger.record_sample(sample)
                    yielded += 1
                    yield sample
            if yielded:
                empty_cycles = 0
            else:
                empty_cycles += 1
                logger.error(
                    f"[data-empty-cycle] loader={self.__class__.__name__} assigned_shards={len(shards)} "
                    f"empty_cycles={empty_cycles}/{self.max_empty_cycles}"
                )
                if empty_cycles >= self.max_empty_cycles:
                    raise DataErrorLimitExceeded(
                        f"{self.__class__.__name__} produced no valid samples for {empty_cycles} complete shard cycles"
                    )


class EditImageTarLoader:
    """Bucketed loader for tar-packed JSONL edit data.

    Groups samples by aspect-ratio bucket and yields same-resolution batches.

    Args:
        datatags: List of "WDS:/path/to/tar/dir" or direct tar paths.
        batch_size: Samples per batch (per bucket).
        caption_keys: Tuple of instruction field names to randomly sample from.
        buckets: List of (H, W) bucket sizes. Default: preserve source aspect ratio at approximately one megapixel.
        norm_mean: Normalization mean for image tensors.
        norm_std: Normalization std for image tensors.
        num_workers: DataLoader workers.
        shuffle_buffer: Buffer size for record shuffling.
        condition_image_size: Target pixel area for VLM condition images (default 384*384).
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        caption_keys: Union[str, Sequence[str]] = _DEFAULT_EDIT_CAPTION_KEYS,
        buckets: Optional[List[Tuple[int, int]]] = None,
        norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        num_workers: Optional[int] = None,
        shuffle_buffer: int = 1000,
        shuffle_initial: int = 32,
        max_bucket_pending_samples: int = 64,
        bucket_batch_timeout_s: float = 0.0,
        condition_image_size: int = 384 * 384,
        worker_timeout_s: float = 300.0,
        max_consecutive_data_errors: int = 32,
        shard_open_retries: int = 3,
        max_empty_cycles: int = 2,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        sampler_start_idx: int | None = None,
        seed: int = 0,
        **kwargs,  # absorb extra BaseWDSLoader-style args
    ):
        self._batch_size = batch_size
        self._buckets = buckets
        self._bucket_batcher_settings = BucketBatcherSettings(
            batch_size=batch_size,
            max_pending_samples=max_bucket_pending_samples,
            batch_timeout_s=bucket_batch_timeout_s,
            log_prefix=self.__class__.__name__,
        )
        caption_keys = (caption_keys,) if isinstance(caption_keys, str) else tuple(caption_keys)

        # Discover shard URLs
        shard_urls = get_shards(datatags)
        if not shard_urls:
            raise ValueError(f"No .tar shards found for datatags={datatags}")
        logger.info(f"EditImageTarLoader: found {len(shard_urls)} shards from {datatags}")

        # Auto num_workers
        num_workers = resolve_num_workers(
            num_workers,
            loader_name=self.__class__.__name__,
            shard_count=len(shard_urls),
            auto_cap=8,
        )

        self._dataset = EditImageTarDataset(
            shard_urls=shard_urls,
            caption_keys=caption_keys,
            buckets=self._buckets,
            norm_mean=norm_mean,
            norm_std=norm_std,
            shuffle_buffer=shuffle_buffer,
            shuffle_initial=shuffle_initial,
            condition_image_size=condition_image_size,
            max_consecutive_data_errors=max_consecutive_data_errors,
            shard_open_retries=shard_open_retries,
            max_empty_cycles=max_empty_cycles,
        )
        if sampler_start_idx and bucket_batch_timeout_s > 0:
            raise ValueError("Resuming requires bucket_batch_timeout_s=0 for deterministic batching")
        self._resume = StreamingResume(self._dataset, batch_size, sampler_start_idx, seed)
        self.datatags = datatags
        self.num_workers = num_workers
        self.worker_timeout_s = float(worker_timeout_s)
        self._loader = create_streaming_dataloader(
            self._dataset,
            loader_name=self.__class__.__name__,
            batch_size=None,  # We handle batching ourselves
            num_workers=num_workers,
            pin_memory=True,
            worker_timeout_s=self.worker_timeout_s,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        return self._resume.iterate(self._iter_batches())

    def _iter_batches(self) -> Iterator[Dict[str, Any]]:
        """Bucketed batching: buffer samples by resolution, yield when full."""
        batcher = BucketedBatcher(
            self._bucket_batcher_settings,
            collate_fn=self._collate_bucket_batch,
            adapt_fn=self._adapt_sample_to_bucket,
        )
        source = BackgroundIterator(self._loader)
        try:
            while True:
                try:
                    sample = source.next(batcher.seconds_until_fallback())
                except BackgroundIteratorTimeout:
                    batch = batcher.emit_due_fallback()
                    if batch is not None:
                        yield batch
                    continue
                except StopIteration:
                    while (batch := batcher.flush_full_batch()) is not None:
                        yield batch
                    return

                bucket_hw = sample.get("bucket_hw")
                if bucket_hw is None:
                    continue
                bucket_hw = tuple(bucket_hw) if not isinstance(bucket_hw, tuple) else bucket_hw
                batch = batcher.add(sample, bucket_hw)
                if batch is not None:
                    yield batch
        finally:
            source.close()

    @staticmethod
    def _adapt_sample_to_bucket(sample: Dict[str, Any], bucket_hw: Tuple[int, int]) -> Dict[str, Any]:
        if tuple(sample.get("bucket_hw", ())) == tuple(bucket_hw):
            return sample
        adapted = dict(sample)
        adapted["real"] = resize_chw_tensor(adapted["real"], bucket_hw)
        adapted["source"] = resize_chw_tensor(adapted["source"], bucket_hw)
        adapted["bucket_hw"] = bucket_hw
        return adapted

    def _collate_bucket_batch(self, samples: List[Dict[str, Any]], bucket_hw: Tuple[int, int]) -> Dict[str, Any]:
        """Stack a list of same-bucket samples into a batch dict."""
        return {
            "real": torch.stack([s["real"] for s in samples]),
            "source": torch.stack([s["source"] for s in samples]),
            "source_pil": [s["source_pil"] for s in samples],
            "condition": [s["condition"] for s in samples],
            "bucket_hw": bucket_hw,
            "fname": [s.get("fname", "<unknown-sample>") for s in samples],
            "shard": [s.get("shard", "<unknown-shard>") for s in samples],
        }
