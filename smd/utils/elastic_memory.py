# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elastic memory management for variable-length SparseTensor training.

Ported from TRELLIS.2's ``elastic_utils.py``. Provides:

- :class:`MemoryController` — base class for GPU memory budgeting
- :class:`LinearMemoryController` — models ``peak_mem = k * N_total * mem_ratio + b``,
  predicts the optimal ``mem_ratio`` (fraction of transformer blocks to NOT checkpoint)
  that keeps peak GPU memory under ``target_ratio × available_memory``.

Usage in SMD sparse training::

    controller = LinearMemoryController(target_ratio=0.8)
    model.register_memory_controller(controller)

    # In training loop:
    with controller.record():
        loss = train_step(batch)
    # Controller automatically observes peak memory and adjusts predictions.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager

import numpy as np
import torch


class MemoryController:
    """Base class for GPU memory management during training."""

    _last_input_size = None
    _last_mem_ratio: list[float] = []

    @contextmanager
    def record(self):
        """Context manager wrapping a training step to observe peak memory."""
        yield

    def update_run_states(self, input_size: int | None = None, mem_ratio: float | None = None) -> None:
        """Called by the elastic module after each forward pass."""
        if self._last_input_size is None:
            self._last_input_size = input_size
        elif self._last_input_size != input_size:
            raise ValueError(f"Input size should not change for different ElasticModules.")
        self._last_mem_ratio.append(mem_ratio)

    @abstractmethod
    def get_mem_ratio(self, input_size: int) -> float:
        """Predict the optimal mem_ratio for the given input size."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        ...

    @abstractmethod
    def log(self) -> dict:
        ...


class LinearMemoryController(MemoryController):
    """Predict per-step memory budget via a linear model.

    Models GPU memory as::

        peak_memory = k * (input_size * mem_ratio) + b

    where ``input_size`` is the total number of voxels (``N_total``) and
    ``mem_ratio`` controls how many transformer blocks skip activation
    checkpointing (1.0 = no checkpointing, 0.0 = full checkpointing).

    The controller collects ``(peak_mem, input_size, mem_ratio)`` samples
    in a ring buffer and periodically refits the linear model. It then
    predicts the ``mem_ratio`` that keeps ``peak_mem`` under
    ``target_ratio × available_memory``.

    Parameters
    ----------
    buffer_size : int
        Number of samples in the ring buffer.
    update_every : int
        Refit the linear model every N steps.
    target_ratio : float
        Target fraction of total GPU memory to use (e.g. 0.8 = 80%).
    available_memory : float or None
        Total GPU memory in GiB. Auto-detected if None.
    max_mem_ratio_start : float
        Initial upper bound on ``mem_ratio`` (ramps up by +0.1 each update).
    params : tuple or None
        Initial (k, b) parameters. If None, starts at (0, 0).
    device : int or None
        CUDA device index. Auto-detected if None.
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        update_every: int = 500,
        target_ratio: float = 0.8,
        available_memory: float | None = None,
        max_mem_ratio_start: float = 0.1,
        params: tuple[float, float] | None = None,
        device: int | None = None,
    ):
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.target_ratio = target_ratio
        self.device = device if device is not None else torch.cuda.current_device()
        self.available_memory = (
            available_memory
            if available_memory is not None
            else torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        )

        self._memory = np.zeros(buffer_size, dtype=np.float32)
        self._input_size = np.zeros(buffer_size, dtype=np.float32)
        self._mem_ratio = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_ptr = 0
        self._buffer_length = 0
        self._params = tuple(params) if params is not None else (0.0, 0.0)
        self._max_mem_ratio = max_mem_ratio_start
        self.step = 0

        # Cached from last record()
        self._last_memory: float = 0.0

    def __repr__(self) -> str:
        return f"LinearMemoryController(target_ratio={self.target_ratio}, available_memory={self.available_memory:.1f}GiB)"

    def _add_sample(self, memory: float, input_size: float, mem_ratio: float) -> None:
        self._memory[self._buffer_ptr] = memory
        self._input_size[self._buffer_ptr] = input_size
        self._mem_ratio[self._buffer_ptr] = mem_ratio
        self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
        self._buffer_length = min(self._buffer_length + 1, self.buffer_size)

    @contextmanager
    def record(self):
        """Wrap a training step to observe peak GPU memory."""
        torch.cuda.reset_peak_memory_stats(self.device)
        self._last_input_size = None
        self._last_mem_ratio = []
        yield
        self._last_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self._last_mem_ratio = sum(self._last_mem_ratio) / max(len(self._last_mem_ratio), 1)
        if self._last_input_size is not None:
            self._add_sample(self._last_memory, self._last_input_size, self._last_mem_ratio)
        self.step += 1
        if self.step % self.update_every == 0 and self._buffer_length >= 10:
            self._max_mem_ratio = min(1.0, self._max_mem_ratio + 0.1)
            self._fit_params()

    def _fit_params(self) -> None:
        memory_usage = self._memory[: self._buffer_length]
        input_size = self._input_size[: self._buffer_length]
        mem_ratio = self._mem_ratio[: self._buffer_length]

        x = input_size * mem_ratio
        y = memory_usage
        k, b = np.polyfit(x, y, 1)
        self._params = (float(k), float(b))

    def get_mem_ratio(self, input_size: int) -> float:
        """Predict the optimal mem_ratio for a given input_size (N_total voxels)."""
        k, b = self._params
        if k == 0:
            return float(np.random.rand() * self._max_mem_ratio)
        pred = (self.available_memory * self.target_ratio - b) / (k * input_size)
        return min(self._max_mem_ratio, max(0.0, float(pred)))

    def state_dict(self) -> dict:
        return {"params": self._params}

    def load_state_dict(self, state_dict: dict) -> None:
        self._params = tuple(state_dict["params"])

    def log(self) -> dict:
        return {
            "elastic/params_k": self._params[0],
            "elastic/params_b": self._params[1],
            "elastic/peak_memory_gb": self._last_memory,
            "elastic/input_size": self._last_input_size or 0,
            "elastic/mem_ratio": self._last_mem_ratio if isinstance(self._last_mem_ratio, float) else 0.0,
            "elastic/max_mem_ratio": self._max_mem_ratio,
        }
