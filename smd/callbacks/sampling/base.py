# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for sample-visualization callbacks.

Every sampling callback follows the same shape:

  1. all ranks run the student forward (it is a collective — they must)
  2. the resulting latents are gathered to rank 0
  3. callbacks enqueue them into the trainer's job-wide preview service
  4. after every callback has run, one isolated worker renders the whole event

The worker is synchronous but process-isolated and bounded by a deadline shorter
than the distributed process-group timeout.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from smd.callbacks.callback import Callback
from smd.callbacks.sampling.rendering.decoder import DecoderSpec
from smd.callbacks.sampling.rendering.protocol import VisualizationItem
from smd.utils.distributed import is_rank0


def resolve_net(model: Any) -> Any:
    """Use the live student, including its distributed wrappers."""
    return model.net


@contextmanager
def sampling_mode(net: torch.nn.Module, *, precision_amp: torch.dtype | None, device_type: str):
    """FSDP-safe eval context for sampling.

    Uses ``no_grad`` rather than ``inference_mode``: inference tensors cannot be
    passed to NCCL collectives (we all-gather the latents afterwards) and break
    custom autograd kernels such as FLA's.
    """
    modes = [(module, module.training) for module in net.modules()]
    try:
        net.eval()
        with (
            torch.no_grad(),
            torch.autocast(device_type=device_type, dtype=precision_amp, enabled=precision_amp is not None),
        ):
            yield
    finally:
        for module, mode in modes:
            module.training = mode


@contextmanager
def preview_mode(model, net, seed: int):
    """Own all preview randomness and schedule changes, including preprocessing.

    Preview conditions can have a different resolution from a cached training
    trajectory. Work on a private noise scheduler so its shift/cache cannot
    change that trajectory's next segment.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    scheduler = getattr(net, "noise_scheduler", None)
    devices = (
        [model.device.index if model.device.index is not None else torch.cuda.current_device()]
        if model.device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            if scheduler is not None:
                net.noise_scheduler = copy.deepcopy(scheduler)
            with sampling_mode(net, precision_amp=model.precision_amp_infer, device_type=model.device.type):
                yield
    finally:
        if scheduler is not None:
            net.noise_scheduler = scheduler
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        # FSDP2 can retain the last forward's full parameter view. Release it
        # before the separate decoder allocates memory on the same devices.
        from torch.distributed.fsdp import FSDPModule

        for module in net.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


def sanitize_stem(text: str) -> str:
    """Filesystem-safe file stem derived from a W&B key."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


@dataclass
class SampleCallbackBase(Callback):
    """Schedule and isolated rendering shared by all sample callbacks.

    Attributes:
        every_n: Sample every N iterations (<= 0 disables the callback).
        seed: Base RNG seed. Rank and prompt index are mixed in so different
            ranks show genuinely different samples.
        run_on_start: Also sample once at the first training step.
        key_prefix: W&B key namespace, also the on-disk media folder name.
        grid_size: Maximum collected samples; AV/3D/edit samples use paired panels.
        fps: Frame rate for video grids.
    """

    every_n: int = 500
    seed: int = 0
    run_on_start: bool = True
    key_prefix: str = "samples"
    grid_size: int = 4
    fps: int = 16
    preview_timeout_s: float | None = None

    _first_iter: int | None = field(default=None, init=False, repr=False)
    _has_run_first: bool = field(default=False, init=False, repr=False)
    _decoder: DecoderSpec | None = field(default=None, init=False, repr=False)
    _decoder_probed: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def on_train_begin(self, model, iteration: int = 0) -> None:
        self._first_iter = iteration + 1
        if self.every_n > 0:
            self.prepare(model)

    def on_app_begin(self) -> None:
        if self.grid_size < 1 or self.fps < 1:
            raise ValueError("Preview grid_size and fps must be positive.")

    def prepare(self, model) -> None:
        """Eagerly warm up whatever the callback needs. Override as required."""

    def should_sample(self, iteration: int) -> bool:
        if self.every_n <= 0:
            return False
        if self.every_n > 0 and iteration % self.every_n == 0:
            return True
        if self.run_on_start and not self._has_run_first and iteration == self._first_iter:
            self._has_run_first = True
            return True
        return False

    def decoder_spec(self, net: Any) -> DecoderSpec | None:
        raise NotImplementedError

    def publish(self, items: Iterable[VisualizationItem], *, iteration: int, net: Any) -> None:
        """Queue this callback's request into the job-wide preview event."""
        items = list(items)
        self.trainer.preview_service.enqueue(
            label=self.key_prefix,
            items=items,
            iteration=iteration,
            decoder=self.decoder_spec(net) if is_rank0() else None,
            timeout_s=self.preview_timeout_s,
        )
