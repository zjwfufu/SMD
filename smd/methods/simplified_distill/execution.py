# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A trajectory cycle owns condition batches and detached endpoints, never graphs."""

from dataclasses import dataclass
from typing import Any, Callable
from smd.callbacks.callback import Callback


def backward_substep(model, loss, position, count, *, grad_accum_rounds, sync_grads, is_fsdp, on_backward_begin):
    """One shared backward protocol for diffusion segments and causal chunk replay."""
    from smd.utils.distributed.fsdp import fsdp_sync_grad

    if position == 0 and on_backward_begin is not None:
        on_backward_begin()
    scaled = model.grad_scaler.scale(loss / grad_accum_rounds)
    if is_fsdp:
        with fsdp_sync_grad(model, sync_grads and position == count - 1):
            scaled.backward()
    else:
        scaled.backward()


@dataclass
class TrajectoryBatch:
    data: dict
    prepared: Any = None
    state: Any = None
    pending_state: Any = None


class TrajectoryCycle:
    def __init__(self, length: int, batch_size: Callable):
        self.length = length
        self.batch_size = batch_size
        self.reset()
        self.samples_consumed = 0

    def reset(self):
        self.segment_index = 0
        self.batches: dict[int, TrajectoryBatch] = {}
        self.iteration = None
        self.visited = set()

    def acquire(self, iteration: int, accumulation_index: int, load_fresh: Callable):
        if self.iteration is not None and self.iteration != iteration:
            raise RuntimeError("Finish the previous optimizer iteration before advancing the cycle.")
        self.iteration = iteration
        if accumulation_index not in self.batches:
            if self.segment_index != 0:
                raise RuntimeError("Cannot introduce a new accumulation slot midway through a trajectory.")
            data = load_fresh()
            self.batches[accumulation_index] = TrajectoryBatch(data)
            self.samples_consumed += self.batch_size(data["condition"])
        return self.batches[accumulation_index].data

    def stage(self, slot: int, state):
        if slot in self.visited:
            raise RuntimeError("An accumulation slot was trained twice in one iteration.")
        self.batches[slot].pending_state = state.detach()
        self.visited.add(slot)

    def finish_iteration(self, iteration: int, accumulation_rounds: int):
        if self.iteration != iteration or self.visited != set(range(accumulation_rounds)):
            raise RuntimeError("All accumulation slots must complete before advancing trajectories.")
        if set(self.batches) != self.visited:
            raise RuntimeError("Gradient accumulation count changed during a trajectory cycle.")
        for entry in self.batches.values():
            entry.state, entry.pending_state = entry.pending_state, None
        self.segment_index += 1
        self.iteration = None
        self.visited.clear()
        if self.segment_index == self.length:
            self.reset()


class TrainingDataProgress(Callback):
    """Checkpoint only global consumed-condition count, not rank-local trajectory tensors."""

    def __init__(self):
        self.samples = None

    def on_save_checkpoint_start(self, model, iteration=0):
        from smd.utils.distributed import world_size

        self.samples = model.training_batch_reuse.samples_consumed * world_size()

    def state_dict(self):
        return {"samples_consumed": self.samples}

    def load_state_dict(self, state):
        self.samples = state["samples_consumed"]

    def on_load_checkpoint_end(self, model, iteration=0):
        from smd.utils.distributed import world_size

        legacy_count = (
            iteration * self.config.dataloader_train.batch_size * self.config.trainer.grad_accum_rounds * world_size()
        )
        samples = legacy_count if self.samples is None else self.samples
        self.trainer.training_samples_consumed = samples
        cycle = model.training_batch_reuse
        cycle.reset()
        cycle.samples_consumed = samples // world_size()
