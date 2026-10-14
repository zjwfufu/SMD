# SPDX-License-Identifier: Apache-2.0

"""Target-free Ref2VA release loader for MiniMax H3 distillation."""

from __future__ import annotations

import errno
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Sampler

from smd.utils import logging_utils as logger

SCHEMA = "minimax_h3.ref2va-onpolicy-set/v1"
CONDITION_SCHEMA = "minimax_h3.multimodal-condition/v1"
COMPACT_SCHEMA = "minimax_h3.ref2va-compact-av/v1"
COMPACT_SET_SCHEMA = "minimax_h3.ref2va-compact-av-set/v1"
_TRANSIENT_IO_ERRNOS = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT}
_T = TypeVar("_T")


def _retry_transient_io(
    operation: Callable[[], _T],
    *,
    description: str,
    max_attempts: int = 5,
    initial_delay_seconds: float = 1.0,
) -> _T:
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}.")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except OSError as error:
            if error.errno not in _TRANSIENT_IO_ERRNOS or attempt == max_attempts:
                raise
            delay = initial_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                f"Transient Ref2VA payload I/O failure while {description} "
                f"(attempt {attempt}/{max_attempts}, errno={error.errno}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable Ref2VA payload retry state.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Ref2VA on-policy manifest requires source {name}.")
    path = (root / value).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_compact_reference(
    compact_root: Path,
    record: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, torch.Tensor]:
    return _retry_transient_io(
        lambda: _load_compact_reference_once(compact_root, record, verify_hashes=verify_hashes),
        description=f"loading compact reference for {record['instruction_id']}",
    )


def _load_compact_reference_once(
    compact_root: Path,
    record: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, torch.Tensor]:
    reference = record["references"][0]
    pointer = reference["compact_av"]
    reference_id = reference["reference_id"]
    payload_path = compact_root / "records" / f"{reference_id}.safetensors"
    metadata_path = compact_root / "records" / f"{reference_id}.json"
    if not payload_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing compact Ref2VA payload for {record['instruction_id']}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != COMPACT_SCHEMA
        or metadata.get("reference_id") != reference_id
        or metadata.get("clip_id") != record["clip_id"]
        or metadata.get("sha256") != pointer.get("sha256")
    ):
        raise ValueError(f"Compact Ref2VA metadata mismatch for {record['instruction_id']}.")
    if verify_hashes and _sha256(payload_path) != pointer["sha256"]:
        raise RuntimeError(f"Compact Ref2VA payload hash mismatch for {record['instruction_id']}.")
    tensors = load_file(str(payload_path), device="cpu")
    if set(tensors) != {"ref2va_video", "ref2va_audio"}:
        raise ValueError(f"Unexpected compact Ref2VA tensors for {record['instruction_id']}.")
    expected = reference["reference_geometry"]
    for name, shape in (
        ("ref2va_video", expected["video_latent_shape"]),
        ("ref2va_audio", expected["audio_latent_shape"]),
    ):
        tensor = tensors[name]
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.float16 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid compact {name} for {record['instruction_id']}.")
    return tensors


def _load_condition(
    condition_root: Path,
    record: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    return _retry_transient_io(
        lambda: _load_condition_once(condition_root, record, verify_hashes=verify_hashes),
        description=f"loading condition for {record['instruction_id']}",
    )


def _load_condition_once(
    condition_root: Path,
    record: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    pointer = record["condition"]
    payload_path = condition_root / pointer["condition_file"]
    metadata_path = condition_root / pointer["metadata_file"]
    if not payload_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Ref2VA condition for {record['instruction_id']}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != CONDITION_SCHEMA
        or metadata.get("pair_id") != record["instruction_id"]
        or metadata.get("clip_id") != record["clip_id"]
        or metadata.get("reference_clip_id") != record["clip_id"]
        or metadata.get("mode") != "ref2va_video_audio"
        or metadata.get("condition_sha256") != pointer.get("condition_sha256")
    ):
        raise ValueError(f"Ref2VA condition metadata mismatch for {record['instruction_id']}.")
    if verify_hashes and _sha256(payload_path) != pointer["condition_sha256"]:
        raise RuntimeError(f"Ref2VA condition hash mismatch for {record['instruction_id']}.")
    tensors = load_file(str(payload_path), device="cpu")
    if set(tensors) != {"prompt_embeds", "text_token_tags"}:
        raise ValueError(f"Unexpected Ref2VA condition tensors for {record['instruction_id']}.")
    embeds = tensors["prompt_embeds"]
    tags = tensors["text_token_tags"]
    if embeds.ndim != 2 or embeds.shape[1] != 5120 or embeds.dtype != torch.bfloat16:
        raise ValueError(f"Invalid Ref2VA prompt embeddings for {record['instruction_id']}.")
    if tags.shape != embeds.shape[:1] or tags.dtype != torch.int64:
        raise ValueError(f"Invalid Ref2VA text token tags for {record['instruction_id']}.")
    if not torch.isfinite(embeds).all() or not torch.all((tags == 0) | (tags == 1)) or not torch.any(tags == 0):
        raise ValueError(f"Ref2VA condition contains invalid values for {record['instruction_id']}.")
    return {"prompt_embeds": embeds, "text_token_tags": tags, "meta": metadata}


class MiniMaxH3Ref2VACacheDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        *,
        condition_root: str | None = None,
        compact_av_root: str | None = None,
        max_samples: int | None = None,
        instruction_ids: list[str] | None = None,
        instruction_ids_file: str | None = None,
        verify_payload_hashes: bool = False,
        duration_seconds: int | list[int] | None = None,
        task_types: str | list[str] | None = None,
        aspect_buckets: str | list[str] | None = None,
        resolution_tiers: str | list[str] | None = None,
    ) -> None:
        self.root = Path(data_path).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema") != SCHEMA
            or self.manifest.get("complete") is not True
            or self.manifest.get("target_free") is not True
        ):
            raise ValueError("MiniMax H3 Ref2VA on-policy view is incomplete.")
        records_path = self.root / self.manifest["records_file"]
        if _sha256(records_path) != self.manifest.get("records_sha256"):
            raise RuntimeError("MiniMax H3 Ref2VA instruction records hash mismatch.")
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        if len(records) != int(self.manifest.get("num_instructions", -1)):
            raise ValueError("MiniMax H3 Ref2VA instruction count mismatch.")
        if instruction_ids is not None and instruction_ids_file is not None:
            raise ValueError("Specify instruction_ids or instruction_ids_file, not both.")
        if instruction_ids_file is not None:
            selection = json.loads(Path(instruction_ids_file).read_text(encoding="utf-8"))
            requested = selection.get("instruction_ids")
            if not isinstance(requested, list) or not all(isinstance(value, str) for value in requested):
                raise ValueError("MiniMax H3 Ref2VA instruction_ids_file requires a string list.")
        else:
            requested = None if instruction_ids is None else list(instruction_ids)
        if requested is not None:
            by_id = {record["instruction_id"]: record for record in records}
            missing = [instruction_id for instruction_id in requested if instruction_id not in by_id]
            if missing:
                raise ValueError(f"Unknown MiniMax H3 Ref2VA instruction ids: {missing}.")
            records = [by_id[instruction_id] for instruction_id in requested]
        if duration_seconds is not None:
            values = {duration_seconds} if isinstance(duration_seconds, int) else set(duration_seconds)
            records = [record for record in records if int(record["output_geometry"]["duration_seconds"]) in values]
        if task_types is not None:
            values = {task_types} if isinstance(task_types, str) else set(task_types)
            records = [record for record in records if record["task_type"] in values]
        if aspect_buckets is not None:
            values = {aspect_buckets} if isinstance(aspect_buckets, str) else set(aspect_buckets)
            records = [record for record in records if record["output_geometry"]["aspect_bucket"] in values]
        if resolution_tiers is not None:
            values = {resolution_tiers} if isinstance(resolution_tiers, str) else set(resolution_tiers)
            records = [record for record in records if record["output_geometry"]["resolution_tier"] in values]
        self.records = records[:max_samples] if max_samples is not None else records
        if not self.records:
            raise ValueError("MiniMax H3 Ref2VA on-policy filters selected no instructions.")
        identities = [record["instruction_id"] for record in self.records]
        if len(identities) != len(set(identities)):
            raise ValueError("MiniMax H3 Ref2VA release contains duplicate instruction ids.")

        sources = self.manifest["sources"]
        self.condition_root = (
            Path(condition_root).expanduser().resolve()
            if condition_root is not None
            else _resolve(self.root, sources.get("condition_root"), "condition_root")
        )
        self.compact_av_root = (
            Path(compact_av_root).expanduser().resolve()
            if compact_av_root is not None
            else _resolve(self.root, sources.get("compact_av_root"), "compact_av_root")
        )
        if _sha256(self.condition_root / "manifest.json") != sources.get("condition_manifest_sha256"):
            raise RuntimeError("MiniMax H3 Ref2VA condition source changed.")
        compact_manifest = json.loads((self.compact_av_root / "manifest.json").read_text(encoding="utf-8"))
        if compact_manifest.get("schema") != COMPACT_SET_SCHEMA or compact_manifest.get("complete") is not True:
            raise ValueError("MiniMax H3 Ref2VA compact AV source is not finalized.")
        self.verify_payload_hashes = verify_payload_hashes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        media = _load_compact_reference(self.compact_av_root, record, verify_hashes=self.verify_payload_hashes)
        condition = _load_condition(self.condition_root, record, verify_hashes=self.verify_payload_hashes)
        geometry = record["output_geometry"]
        return {
            "video_latent_shape": geometry["video_latent_shape"],
            "audio_latent_shape": geometry["audio_latent_shape"],
            "reference_video": media["ref2va_video"],
            "reference_audio": media["ref2va_audio"],
            "prompt_embeds": condition["prompt_embeds"],
            "text_token_tags": condition["text_token_tags"],
            "fps": float(geometry["fps"]),
            "num_frames": int(geometry["num_frames"]),
            "meta": {
                "instruction_id": record["instruction_id"],
                "clip_id": record["clip_id"],
                "task_type": record["task_type"],
                "prompt": record["prompt"],
                "target_free": True,
                "reference_geometry": record["reference_geometry"],
                "output_geometry": geometry,
                "condition_sha256": record["condition"]["condition_sha256"],
                "compact_av_sha256": record["references"][0]["compact_av"]["sha256"],
            },
        }


class _GroupedInfiniteSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        seed: int,
        start_idx: int,
        context_parallel_size: int,
        condition_group_size: int,
    ) -> None:
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        cp_degree = int(context_parallel_size)
        condition_degree = int(condition_group_size)
        degree = cp_degree * condition_degree
        if degree < 1 or world % degree:
            raise ValueError(f"MiniMax H3 sample group size {degree} must divide world_size={world}.")
        if int(start_idx) % degree:
            raise ValueError("MiniMax H3 sampler_start_idx must be divisible by the sample group size.")
        self.dataset_size = len(dataset)
        self.num_replicas = world // degree
        self.rank = rank // degree
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.idx = int(start_idx) // degree + self.rank

    def __iter__(self):
        idx = self.idx
        epoch = None
        while True:
            current_epoch = idx // self.dataset_size
            if epoch != current_epoch:
                epoch = current_epoch
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    np.random.RandomState((self.seed + epoch) % (2**32 - 1)).shuffle(order)
            yield int(order[idx % self.dataset_size])
            idx += self.num_replicas


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    from smd.networks.MiniMaxH3.condition import MiniMaxH3Condition

    if len(items) != 1:
        raise ValueError("MiniMax H3 Ref2VA on-policy cache requires batch_size=1 per rank.")
    item = items[0]
    return {
        "video_latent_shape": torch.tensor([item["video_latent_shape"]], dtype=torch.int64),
        "audio_latent_shape": torch.tensor([item["audio_latent_shape"]], dtype=torch.int64),
        "condition": MiniMaxH3Condition(
            prompt_embeds=item["prompt_embeds"].unsqueeze(0),
            text_token_tags=item["text_token_tags"],
            mode="ref2va",
            video_latents=(item["reference_video"].unsqueeze(0),),
            audio_latents=(item["reference_audio"].unsqueeze(0),),
        ),
        "fps": torch.tensor([item["fps"]], dtype=torch.float32),
        "num_frames": torch.tensor([item["num_frames"]], dtype=torch.int64),
        "condition_raw": [item["meta"]["prompt"]],
        "meta": [item["meta"]],
    }


class MiniMaxH3Ref2VACacheLoader:
    def __init__(
        self,
        dataset_path: str,
        *,
        condition_root: str | None = None,
        compact_av_root: str | None = None,
        batch_size: int = 1,
        max_samples: int | None = None,
        instruction_ids: list[str] | None = None,
        instruction_ids_file: str | None = None,
        verify_payload_hashes: bool = False,
        duration_seconds: int | list[int] | None = None,
        task_types: str | list[str] | None = None,
        aspect_buckets: str | list[str] | None = None,
        resolution_tiers: str | list[str] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        sampler_start_idx: int | None = None,
        num_workers: int = 2,
        pin_memory: bool = True,
        context_parallel_size: int = 1,
        condition_group_size: int = 1,
        **_kwargs: Any,
    ) -> None:
        if batch_size != 1:
            raise ValueError("MiniMax H3 Ref2VA on-policy cache currently supports batch_size=1 only.")
        self.dataset = MiniMaxH3Ref2VACacheDataset(
            dataset_path,
            condition_root=condition_root,
            compact_av_root=compact_av_root,
            max_samples=max_samples,
            instruction_ids=instruction_ids,
            instruction_ids_file=instruction_ids_file,
            verify_payload_hashes=verify_payload_hashes,
            duration_seconds=duration_seconds,
            task_types=task_types,
            aspect_buckets=aspect_buckets,
            resolution_tiers=resolution_tiers,
        )
        sampler = _GroupedInfiniteSampler(
            self.dataset,
            shuffle=shuffle,
            seed=seed,
            start_idx=0 if sampler_start_idx is None else int(sampler_start_idx),
            context_parallel_size=context_parallel_size,
            condition_group_size=condition_group_size,
        )
        self.loader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=_collate,
        )

    def __iter__(self):
        return iter(self.loader)
