# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, List

from torch.utils.data import DataLoader, Dataset

from smd.datasets.samplers import InfiniteSampler
import smd.utils.logging_utils as logger


def _choose_prompt_key(
    record: dict[str, Any],
    prompt_key: str | None = None,
    allow_empty_prompt: bool = False,
) -> str | None:
    if prompt_key is not None:
        value = record.get(prompt_key, None)
        if isinstance(value, str) and (allow_empty_prompt or value.strip()):
            return value
        return None

    for key in ("prompt", "text", "caption"):
        value = record.get(key, None)
        if isinstance(value, str) and (allow_empty_prompt or value.strip()):
            return value
    return None


def _load_prompts_from_jsonl(
    path: Path,
    prompt_key: str | None = None,
    allow_empty_prompt: bool = False,
) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = _choose_prompt_key(
                record,
                prompt_key=prompt_key,
                allow_empty_prompt=allow_empty_prompt,
            )
            if prompt is not None:
                yield prompt


def _load_prompts_from_json(
    path: Path,
    prompt_key: str | None = None,
    allow_empty_prompt: bool = False,
) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str) and (allow_empty_prompt or item.strip()):
                yield item
            elif isinstance(item, dict):
                prompt = _choose_prompt_key(
                    item,
                    prompt_key=prompt_key,
                    allow_empty_prompt=allow_empty_prompt,
                )
                if prompt is not None:
                    yield prompt
    elif isinstance(content, dict):
        prompt = _choose_prompt_key(
            content,
            prompt_key=prompt_key,
            allow_empty_prompt=allow_empty_prompt,
        )
        if prompt is not None:
            yield prompt


def _load_prompts_from_txt(path: Path, allow_empty_prompt: bool = False) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.rstrip("\r\n")
            if allow_empty_prompt or prompt.strip():
                yield prompt


def _load_prompts_from_parquet(
    path: Path,
    prompt_key: str | None = None,
    allow_empty_prompt: bool = False,
) -> Iterable[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            f"Reading parquet prompt files requires `pyarrow`, but it is not installed. File: {path}"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=4096):
        batch_dict = batch.to_pydict()
        if prompt_key is None:
            available = [k for k in ("prompt", "text", "caption") if k in batch_dict]
            if not available:
                raise KeyError(
                    f"Cannot infer prompt column from parquet file {path}. "
                    f"Expected one of ['prompt', 'text', 'caption'], got {list(batch_dict.keys())}."
                )
            key = available[0]
        else:
            if prompt_key not in batch_dict:
                raise KeyError(
                    f"Prompt key `{prompt_key}` not found in parquet file {path}. "
                    f"Available columns: {list(batch_dict.keys())}."
                )
            key = prompt_key

        for prompt in batch_dict[key]:
            if isinstance(prompt, str) and (allow_empty_prompt or prompt.strip()):
                yield prompt


class PromptOnlyDataset(Dataset):
    """Minimal prompt-only dataset for data-free distillation.

    Supported sources:
    - `.txt` (one prompt per line)
    - `.jsonl` (record with prompt/text/caption key)
    - `.json` (list of strings or list of records)
    - `.parquet` (requires optional `pyarrow`)
    """

    def __init__(
        self,
        data_path: str,
        max_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        prompt_key: str | None = None,
        file_glob: str = "*",
        allow_empty_prompt: bool = False,
    ):
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.prompt_key = prompt_key
        self.file_glob = file_glob
        self.allow_empty_prompt = allow_empty_prompt

        self.prompts = self._load_prompts()
        if len(self.prompts) == 0:
            raise ValueError(f"No prompts loaded from {self.data_path}")

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(self.prompts)

        logger.info(f"Loaded {len(self.prompts)} prompts from {self.data_path}")

    def _iter_files(self) -> List[Path]:
        if self.data_path.is_file():
            return [self.data_path]
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"data_path does not exist: {self.data_path}")
        files = sorted(self.data_path.glob(self.file_glob))
        files = [f for f in files if f.is_file()]
        if not files:
            raise FileNotFoundError(f"No files found under {self.data_path} with glob `{self.file_glob}`.")
        return files

    def _load_prompts(self) -> List[str]:
        prompts: List[str] = []
        for path in self._iter_files():
            suffix = path.suffix.lower()
            if suffix == ".txt":
                iterator = _load_prompts_from_txt(path, allow_empty_prompt=self.allow_empty_prompt)
            elif suffix == ".jsonl":
                iterator = _load_prompts_from_jsonl(
                    path,
                    prompt_key=self.prompt_key,
                    allow_empty_prompt=self.allow_empty_prompt,
                )
            elif suffix == ".json":
                iterator = _load_prompts_from_json(
                    path,
                    prompt_key=self.prompt_key,
                    allow_empty_prompt=self.allow_empty_prompt,
                )
            elif suffix == ".parquet":
                iterator = _load_prompts_from_parquet(
                    path,
                    prompt_key=self.prompt_key,
                    allow_empty_prompt=self.allow_empty_prompt,
                )
            else:
                continue

            for prompt in iterator:
                prompts.append(prompt)
                if self.max_samples is not None and len(prompts) >= self.max_samples:
                    return prompts
        return prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[idx]
        return {
            "condition": prompt,
            "meta": {"id": idx, "source": os.fspath(self.data_path)},
        }


class PromptOnlyLoader:
    """Infinite dataloader wrapper that matches SMD loader style."""

    dataset_cls = PromptOnlyDataset

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        max_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        prompt_key: str | None = None,
        sampler_start_idx: int | None = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        file_glob: str = "*",
        allow_empty_prompt: bool = False,
        **kwargs,
    ):
        _ = kwargs
        self.dataset = self.dataset_cls(
            data_path=dataset_path,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            prompt_key=prompt_key,
            file_glob=file_glob,
            allow_empty_prompt=allow_empty_prompt,
        )
        dataset_sampler = InfiniteSampler(
            dataset=self.dataset,
            shuffle=shuffle,
            seed=seed,
            start_idx=0 if sampler_start_idx is None else sampler_start_idx,
        )
        self.loader = DataLoader(
            dataset=self.dataset,
            sampler=dataset_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self):
        return iter(self.loader)
