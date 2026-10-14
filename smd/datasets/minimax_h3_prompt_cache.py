# SPDX-License-Identifier: Apache-2.0

"""Sharded cache-only MiniMax H3 prompt-condition loader."""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

import smd.utils.logging_utils as logger
from smd.networks.MiniMaxH3.condition import MiniMaxH3Condition
from smd.networks.MiniMaxH3.geometry import align_num_frames, audio_latent_num_frames, video_latent_num_frames
from smd.utils.distributed import get_rank, world_size

H3_MIXED_NUM_FRAMES: Final[tuple[int, ...]] = (124, 192, 294, 362)
H3_PRODUCTION_HEIGHT: Final = 768
H3_PRODUCTION_WIDTH: Final = 1344


def build_minimax_h3_cache_policy(
    *,
    schema_version: int,
    model_id: str,
    model_revision: str | None,
    max_sequence_length: int,
    fps: float,
    height: int,
    width: int,
    num_frames: int | None = None,
    allowed_num_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the canonical encoder and geometry identity for cache reuse."""
    encoder: dict[str, Any] = {
        "model_id": str(model_id),
        "layer": 50,
        "dtype": "bfloat16",
        "hidden_dim": 5120,
        "max_sequence_length": int(max_sequence_length),
    }
    if model_revision:
        encoder["revision"] = str(model_revision)
    geometry: dict[str, Any] = {
        "fps": float(fps),
        "height": int(height),
        "width": int(width),
    }
    if schema_version == 1:
        if num_frames is None:
            raise ValueError("MiniMax H3 schema-v1 cache policy requires num_frames.")
        geometry["num_frames"] = int(num_frames)
    elif schema_version == 2:
        if allowed_num_frames is None:
            raise ValueError("MiniMax H3 schema-v2 cache policy requires allowed_num_frames.")
        geometry["allowed_num_frames"] = sorted(int(value) for value in allowed_num_frames)
    else:
        raise ValueError(f"Unsupported MiniMax H3 cache policy schema: {schema_version}.")
    return {
        "schema_version": int(schema_version),
        "encoder": encoder,
        "geometry": geometry,
    }


def minimax_h3_cache_policy_sha256(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=2)
def _load_shard(path: str) -> dict[str, torch.Tensor]:
    return load_file(path, device="cpu")


def _expected_shapes(num_frames: int, fps: float, height: int, width: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if align_num_frames(num_frames) != num_frames:
        raise ValueError(f"MiniMax H3 num_frames must satisfy 17*n+5 exactly, got {num_frames}.")
    return (
        (24, video_latent_num_frames(num_frames), height // 16, width // 16),
        (2, 32, audio_latent_num_frames(num_frames, fps=fps)),
    )


def _coerce_shape(record: dict[str, Any], key: str, expected: tuple[int, ...], index: int) -> None:
    value = record.get(key)
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"MiniMax H3 schema-v2 record {index} requires integer list {key!r}.")
    if tuple(value) != expected:
        raise ValueError(
            f"MiniMax H3 schema-v2 record {index} has inconsistent {key}: actual={tuple(value)}, expected={expected}."
        )


class MiniMaxH3PromptCacheDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int | None = None) -> None:
        self.root = Path(data_path).expanduser()
        manifest_path = self.root / "manifest.json"
        records_path = self.root / "records.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(
                f"MiniMax H3 prompt cache requires manifest.json and records.jsonl under {self.root}."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.schema_version = self.manifest.get("schema_version")
        if self.schema_version not in (1, 2):
            raise ValueError(f"Unsupported MiniMax H3 prompt-cache schema: {self.manifest.get('schema_version')!r}.")
        if self.manifest.get("hidden_dim") != 5120 or self.manifest.get("storage") != "ragged_bfloat16":
            raise ValueError("MiniMax H3 prompt cache must contain ragged BF16 layer-50 embeddings with dim 5120.")
        self.records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        if max_samples is not None:
            self.records = self.records[:max_samples]
        if not self.records:
            raise ValueError(f"No MiniMax H3 prompt-cache records found under {self.root}.")
        if (
            max_samples is None
            and "num_records" in self.manifest
            and int(self.manifest["num_records"]) != len(self.records)
        ):
            raise ValueError("MiniMax H3 prompt-cache manifest num_records does not match records.jsonl.")

        geometry = self.manifest.get("geometry")
        if not isinstance(geometry, dict):
            raise TypeError("MiniMax H3 prompt-cache manifest requires a geometry object.")
        self.fps = float(geometry["fps"])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"MiniMax H3 cache fps must be positive, got {self.fps}.")
        self.height = int(geometry["height"])
        self.width = int(geometry["width"])
        if self.height % 32 or self.width % 32:
            raise ValueError(f"MiniMax H3 cache geometry must be divisible by 32, got {self.height}x{self.width}.")
        if self.schema_version == 1:
            self.num_frames = align_num_frames(int(geometry["num_frames"]))
            self.video_latent_shape, self.audio_latent_shape = _expected_shapes(
                self.num_frames, self.fps, self.height, self.width
            )
            self.allowed_num_frames = (self.num_frames,)
        else:
            if (self.height, self.width) != (H3_PRODUCTION_HEIGHT, H3_PRODUCTION_WIDTH):
                raise ValueError(
                    "MiniMax H3 schema-v2 caches require production geometry "
                    f"{H3_PRODUCTION_HEIGHT}x{H3_PRODUCTION_WIDTH}, got {self.height}x{self.width}."
                )
            if "num_frames" in geometry:
                raise ValueError("MiniMax H3 schema-v2 geometry must use allowed_num_frames, not global num_frames.")
            allowed = geometry.get("allowed_num_frames")
            if not isinstance(allowed, list) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in allowed
            ):
                raise ValueError("MiniMax H3 schema-v2 geometry requires an integer allowed_num_frames list.")
            if tuple(sorted(set(allowed))) != H3_MIXED_NUM_FRAMES or len(allowed) != len(H3_MIXED_NUM_FRAMES):
                raise ValueError(
                    "MiniMax H3 schema-v2 allowed_num_frames must contain exactly "
                    f"{list(H3_MIXED_NUM_FRAMES)}, got {allowed}."
                )
            self.allowed_num_frames = H3_MIXED_NUM_FRAMES
            self.num_frames = None
            for index, record in enumerate(self.records):
                self._validate_v2_record(record, index)
        logger.info(f"Loaded {len(self.records)} cached MiniMax H3 prompt conditions from {self.root}")

    def _validate_v2_record(self, record: dict[str, Any], index: int) -> None:
        for key in ("item_id", "category", "duration_seconds", "num_frames", "source_provenance"):
            if key not in record:
                raise ValueError(f"MiniMax H3 schema-v2 record {index} is missing {key!r}.")
        if not isinstance(record["item_id"], (str, int)) or isinstance(record["item_id"], bool):
            raise TypeError(f"MiniMax H3 schema-v2 record {index} has an invalid item_id.")
        if not isinstance(record["category"], str) or not record["category"]:
            raise ValueError(f"MiniMax H3 schema-v2 record {index} has an invalid category.")
        if not isinstance(record["source_provenance"], dict) or not record["source_provenance"]:
            raise ValueError(f"MiniMax H3 schema-v2 record {index} has invalid source provenance.")
        if isinstance(record["duration_seconds"], bool) or not isinstance(record["duration_seconds"], (int, float)):
            raise TypeError(f"MiniMax H3 schema-v2 record {index} has invalid duration_seconds.")
        if isinstance(record["num_frames"], bool) or not isinstance(record["num_frames"], int):
            raise TypeError(f"MiniMax H3 schema-v2 record {index} has invalid num_frames.")
        duration_seconds = float(record["duration_seconds"])
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError(f"MiniMax H3 schema-v2 record {index} has non-positive duration_seconds.")
        num_frames = record["num_frames"]
        if num_frames not in self.allowed_num_frames:
            raise ValueError(
                f"MiniMax H3 schema-v2 record {index} frame count {num_frames} is not allowed by the manifest."
            )
        duration_frames = align_num_frames(round(duration_seconds * self.fps))
        if duration_frames != num_frames:
            raise ValueError(
                f"MiniMax H3 schema-v2 record {index} duration/frame mismatch: "
                f"duration_seconds={duration_seconds}, num_frames={num_frames}, aligned duration frames={duration_frames}."
            )
        video_shape, audio_shape = _expected_shapes(num_frames, self.fps, self.height, self.width)
        _coerce_shape(record, "video_latent_shape", video_shape, index)
        _coerce_shape(record, "audio_latent_shape", audio_shape, index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        shard = _load_shard(str(self.root / record["shard"]))
        start = int(record["token_start"])
        end = int(record["token_end"])
        if not 0 <= start < end <= shard["prompt_embeds"].shape[0]:
            raise RuntimeError(f"Invalid MiniMax H3 token slice [{start},{end}) for cache record {index}.")
        prompt = record.get("prompt")
        if prompt is not None:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if digest != record["prompt_sha256"]:
                raise RuntimeError(f"MiniMax H3 prompt-cache record {index} has a prompt hash mismatch.")
        embeds = shard["prompt_embeds"][start:end]
        tags = shard["text_token_tags"][start:end]
        if embeds.shape != (end - start, 5120) or embeds.dtype != torch.bfloat16 or tags.shape != (end - start,):
            raise RuntimeError(f"MiniMax H3 prompt-cache record {index} has inconsistent tensor shapes.")
        if tags.dtype != torch.int64 or not torch.all(tags == 1):
            raise RuntimeError(f"MiniMax H3 T2VA cache record {index} contains non-text token tags.")
        if self.schema_version == 1:
            num_frames = self.num_frames
            video_latent_shape = self.video_latent_shape
            audio_latent_shape = self.audio_latent_shape
            duration_seconds = float(num_frames) / self.fps
        else:
            num_frames = int(record["num_frames"])
            video_latent_shape = tuple(record["video_latent_shape"])
            audio_latent_shape = tuple(record["audio_latent_shape"])
            duration_seconds = float(record["duration_seconds"])
        meta = {
            "id": record.get("item_id", record.get("global_index", index)),
            "prompt_sha256": record["prompt_sha256"],
            "num_tokens": end - start,
            "duration_seconds": duration_seconds,
        }
        if self.schema_version == 2:
            meta.update(
                {
                    "item_id": record["item_id"],
                    "category": record["category"],
                    "source_provenance": record["source_provenance"],
                }
            )
        return {
            "prompt_embeds": embeds,
            "prompt": prompt,
            "text_token_tags": tags,
            "video_latent_shape": torch.tensor(video_latent_shape, dtype=torch.int64),
            "audio_latent_shape": torch.tensor(audio_latent_shape, dtype=torch.int64),
            "fps": self.fps,
            "num_frames": num_frames,
            "meta": meta,
        }


class MiniMaxH3ShardLocalInfiniteSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: MiniMaxH3PromptCacheDataset,
        shuffle: bool = True,
        seed: int = 0,
        start_idx: int = 0,
        condition_group_size: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_size = len(dataset)
        global_world_size = world_size()
        condition_group_size = int(condition_group_size)
        if condition_group_size < 1 or global_world_size % condition_group_size:
            raise ValueError(
                f"MiniMax H3 condition_group_size={condition_group_size} must divide world_size={global_world_size}."
            )
        if int(start_idx) % condition_group_size:
            raise ValueError(
                f"MiniMax H3 grouped sampler start_idx={start_idx} must be divisible by "
                f"condition_group_size={condition_group_size}."
            )
        self.num_replicas = global_world_size // condition_group_size
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.idx = int(start_idx) // condition_group_size + get_rank() // condition_group_size
        groups: dict[str, list[int]] = {}
        for index, record in enumerate(dataset.records):
            groups.setdefault(record["shard"], []).append(index)
        self.groups = list(groups.values())

    def _order(self, epoch: int) -> np.ndarray:
        random_state = np.random.RandomState((self.seed + epoch) % (2**32 - 1))
        group_order = np.arange(len(self.groups))
        if self.shuffle:
            random_state.shuffle(group_order)
        order: list[int] = []
        for group_index in group_order:
            indices = np.asarray(self.groups[int(group_index)], dtype=np.int64)
            if self.shuffle:
                indices = indices.copy()
                random_state.shuffle(indices)
            order.extend(indices.tolist())
        return np.asarray(order, dtype=np.int64)

    def __iter__(self):
        idx = self.idx
        epoch = None
        order = np.empty(0, dtype=np.int64)
        while True:
            current_epoch = idx // self.dataset_size
            if epoch != current_epoch:
                epoch = current_epoch
                order = self._order(epoch)
            yield int(order[idx % self.dataset_size])
            idx += self.num_replicas


def duration_route_for_step(seed: int, step: int, routes: Sequence[int] = H3_MIXED_NUM_FRAMES) -> int:
    """Return a rank-independent route; each four-step cycle is a seeded permutation."""
    if step < 0:
        raise ValueError(f"MiniMax H3 route step must be non-negative, got {step}.")
    stable_routes = tuple(sorted(int(route) for route in routes))
    if stable_routes != H3_MIXED_NUM_FRAMES:
        raise ValueError(f"MiniMax H3 mixed-duration routes must be exactly {H3_MIXED_NUM_FRAMES}.")
    cycle, offset = divmod(int(step), len(stable_routes))
    permutation = np.asarray(stable_routes, dtype=np.int64)
    np.random.RandomState((int(seed) + cycle) % (2**32 - 1)).shuffle(permutation)
    return int(permutation[offset])


def _route_consumption_before(seed: int, step: int, route: int) -> int:
    cycles, remainder = divmod(step, len(H3_MIXED_NUM_FRAMES))
    count = cycles
    for current_step in range(cycles * len(H3_MIXED_NUM_FRAMES), cycles * len(H3_MIXED_NUM_FRAMES) + remainder):
        count += int(duration_route_for_step(seed, current_step) == route)
    return count


class MiniMaxH3DurationRouter:
    """Pure step-indexed mixed-duration sampler with cross-rank route homogeneity."""

    def __init__(
        self,
        dataset: MiniMaxH3PromptCacheDataset,
        seed: int,
        sampler_start_idx: int,
        shuffle: bool = True,
        context_parallel_size: int = 1,
    ) -> None:
        if dataset.schema_version != 2:
            raise ValueError("MiniMaxH3DurationRouter requires a schema-v2 cache.")
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        global_world_size = world_size()
        context_parallel_size = int(context_parallel_size)
        if context_parallel_size < 1 or global_world_size % context_parallel_size:
            raise ValueError(
                f"MiniMax H3 context_parallel_size={context_parallel_size} must divide world_size={global_world_size}."
            )
        self.num_replicas = global_world_size // context_parallel_size
        self.rank = get_rank() // context_parallel_size
        if sampler_start_idx < 0 or sampler_start_idx % global_world_size:
            raise ValueError(
                "MiniMax H3 schema-v2 sampler_start_idx must be a non-negative multiple of world_size; "
                f"got {sampler_start_idx} for world_size={global_world_size}."
            )
        self.step = int(sampler_start_idx) // global_world_size
        self.route_indices: dict[int, tuple[int, ...]] = {}
        for route in H3_MIXED_NUM_FRAMES:
            indices = tuple(index for index, record in enumerate(dataset.records) if int(record["num_frames"]) == route)
            if len(indices) < self.num_replicas:
                raise ValueError(
                    f"MiniMax H3 route {route} has {len(indices)} records, fewer than world_size={self.num_replicas}."
                )
            self.route_indices[route] = indices

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"MiniMax H3 route step must be non-negative, got {step}.")
        self.step = int(step)

    def next_index(self) -> tuple[int, int]:
        route = duration_route_for_step(self.seed, self.step)
        consumption = _route_consumption_before(self.seed, self.step, route)
        indices = self.route_indices[route]
        records_per_block = len(indices) // self.num_replicas
        block, slot = divmod(consumption, records_per_block)
        order = np.asarray(indices, dtype=np.int64)
        route_seed = (self.seed + route * 1_000_003 + block) % (2**32 - 1)
        if self.shuffle:
            np.random.RandomState(route_seed).shuffle(order)
        index = int(order[slot * self.num_replicas + self.rank])
        self.step += 1
        return route, index


def _collate_minimax_h3_cache(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("MiniMax H3 cache-only training currently requires batch_size=1 per rank.")
    item = items[0]
    batch = {
        "condition": MiniMaxH3Condition(
            prompt_embeds=item["prompt_embeds"].unsqueeze(0),
            text_token_tags=item["text_token_tags"],
        ),
        "video_latent_shape": item["video_latent_shape"].unsqueeze(0),
        "audio_latent_shape": item["audio_latent_shape"].unsqueeze(0),
        "fps": torch.tensor([item["fps"]], dtype=torch.float32),
        "num_frames": torch.tensor([item["num_frames"]], dtype=torch.int64),
        "meta": [item["meta"]],
    }
    if item.get("prompt") is not None:
        batch["condition_raw"] = [item["prompt"]]
    return batch


class MiniMaxH3PromptCacheLoader:
    """Infinite H3 prompt loader which never initializes Qwen3-VL."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 1,
        max_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        sampler_start_idx: int | None = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        context_parallel_size: int = 1,
        condition_group_size: int = 1,
        **_kwargs: Any,
    ) -> None:
        if batch_size != 1:
            raise ValueError("MiniMax H3 cache loader currently supports batch_size=1 only.")
        sampler_start_idx = 0 if sampler_start_idx is None else int(sampler_start_idx)
        self.dataset = MiniMaxH3PromptCacheDataset(dataset_path, max_samples=max_samples)
        self._router: MiniMaxH3DurationRouter | None = None
        if self.dataset.schema_version == 1:
            if int(context_parallel_size) < 1 or int(condition_group_size) < 1:
                raise ValueError("MiniMax H3 parallel group sizes must be positive.")
            sampler = MiniMaxH3ShardLocalInfiniteSampler(
                dataset=self.dataset,
                shuffle=shuffle,
                seed=seed,
                start_idx=sampler_start_idx,
                condition_group_size=int(context_parallel_size) * int(condition_group_size),
            )
            self.loader: DataLoader | None = DataLoader(
                self.dataset,
                sampler=sampler,
                batch_size=1,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                persistent_workers=num_workers > 0,
                collate_fn=_collate_minimax_h3_cache,
            )
        else:
            if int(condition_group_size) != 1:
                raise ValueError(
                    "MiniMax H3 schema-v2 duration routing does not support condition_group_size; "
                    "use the fixed-duration view loader."
                )
            self.loader = None
            self._router = MiniMaxH3DurationRouter(
                self.dataset,
                seed=seed,
                sampler_start_idx=sampler_start_idx,
                shuffle=shuffle,
                context_parallel_size=context_parallel_size,
            )

    def __iter__(self):
        if self.loader is not None:
            return iter(self.loader)
        return self

    def __next__(self) -> dict[str, Any]:
        if self._router is None:
            raise TypeError("Schema-v1 MiniMaxH3PromptCacheLoader is iterable but is not its own iterator.")
        route, index = self._router.next_index()
        batch = _collate_minimax_h3_cache([self.dataset[index]])
        if int(batch["num_frames"][0]) != route:
            raise RuntimeError(f"MiniMax H3 duration router selected route {route} but loaded a different frame count.")
        return batch

    def set_step(self, step: int) -> None:
        """Map the Trainer's one-based update iteration to a zero-based route step."""
        if self._router is not None:
            if step < 1:
                raise ValueError(f"MiniMax H3 Trainer iteration must be positive, got {step}.")
            self._router.set_step(step - 1)
