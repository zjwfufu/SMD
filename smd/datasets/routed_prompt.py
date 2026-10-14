# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route-homogeneous prompt loader for hard-routed multi-teacher distillation.

Each training STEP draws a batch that is homogeneous in *route*: all samples come
from one of N route-specific prompt sets (e.g. ocr / geneval / clipscore /
preference). The route for a given step is a pure function of ``(seed, step)`` —
identical on every rank with zero communication — so that PEFT's global
``set_adapter()`` switch in the teacher stays consistent across all ranks under
FSDP/DDP.

Per-rank prompt diversity is preserved: each rank draws from the SAME route's
:class:`~smd.datasets.prompt_only.PromptOnlyDataset` but through its own
``InfiniteSampler`` offset (``start_idx + get_rank()``), so it sees different
prompts than other ranks while agreeing on the route.

The route name rides along as a single scalar string ``data["route"]`` attached
*after* collation, so the default collate never sees it and the trainer's
encode-only-known-keys path leaves it untouched.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import smd.utils.logging_utils as logger
from smd.datasets.prompt_only import PromptOnlyLoader


def pick_route(seed: int, step: int, names: List[str], weights: np.ndarray) -> str:
    """Pick a route for ``step``, deterministic and rank-independent.

    ``np.random.default_rng([seed, step])`` is seeded only by ``(seed, step)`` —
    NOT by rank — so every rank computes the same route for the same step. ``names``
    must be in a stable (sorted) order across ranks.
    """
    rng = np.random.default_rng([int(seed), int(step)])
    return names[int(rng.choice(len(names), p=weights))]


def _normalize(weights: List[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.min() < 0:
        raise ValueError(f"route weights must be non-negative, got {weights}")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"route weights must sum to > 0, got {weights}")
    return w / total


class RoutedPromptLoader:
    """Infinite loader emitting route-homogeneous prompt batches.

    Args:
        route_paths: ``{route_name: dataset_path}``. Each path is a prompt source
            (parquet/jsonl/json/txt dir) consumed by :class:`PromptOnlyLoader`.
        route_weights: ``{route_name: weight}``. Probability of drawing each route
            per step (normalized internally). Keys must match ``route_paths``.
        batch_size: Per-rank batch size (each step's batch is one route).
        seed: Base seed for both route selection and per-route shuffling.
        start_step: Initial step counter (for resume). Overridden each step by the
            trainer via :meth:`set_step` when available.
        num_workers: Per-route DataLoader workers. NOTE total worker procs =
            ``len(route_paths) * num_workers`` — keep small.
    """

    def __init__(
        self,
        route_paths: Dict[str, str],
        route_weights: Dict[str, float],
        batch_size: int,
        seed: int = 0,
        start_step: int = 0,
        num_workers: int = 1,
        **loader_kwargs,
    ):
        if not route_paths:
            raise ValueError("route_paths must have at least one route")
        missing = set(route_paths) - set(route_weights)
        if missing:
            raise ValueError(f"route_weights missing keys for routes: {sorted(missing)}")

        # Stable, rank-independent route order (critical for cross-rank agreement).
        self._names: List[str] = sorted(route_paths)
        self._weights = _normalize([route_weights[n] for n in self._names])
        self._seed = int(seed)
        self._step = int(start_step)

        self._loaders: Dict[str, PromptOnlyLoader] = {
            name: PromptOnlyLoader(
                dataset_path=route_paths[name],
                batch_size=batch_size,
                seed=seed,
                num_workers=num_workers,
                **loader_kwargs,
            )
            for name in self._names
        }
        self._iters: Dict[str, object] | None = None

        logger.info(
            f"RoutedPromptLoader: {len(self._names)} routes "
            f"({', '.join(f'{n}:{w:.2f}' for n, w in zip(self._names, self._weights))}), "
            f"batch_size={batch_size}, num_workers={num_workers}"
        )

    def set_step(self, step: int) -> None:
        """Sync route selection to the trainer's authoritative global iteration.

        Driving route choice from the trainer-owned ``iteration`` (rather than a
        free-running internal counter) keeps all ranks aligned even if any rank
        skips a step (NaN-skip / grad-accum) and makes resume exact.
        """
        self._step = int(step)

    def __iter__(self):
        self._iters = {name: iter(loader.loader) for name, loader in self._loaders.items()}
        return self

    def __next__(self) -> dict:
        if self._iters is None:
            self.__iter__()
        route = pick_route(self._seed, self._step, self._names, self._weights)
        self._step += 1
        batch = next(self._iters[route])  # type: ignore[index]
        # Attach AFTER collation as a scalar string: default collate never sees it,
        # and the trainer encodes only known keys, so it flows untouched to
        # single_train_step → _build_teacher_query → _select_teacher_name.
        batch["route"] = route
        return batch
