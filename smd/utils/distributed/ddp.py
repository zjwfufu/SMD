# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, TYPE_CHECKING, Union
import os
import torch
import torch.distributed as dist
from datetime import timedelta
import smd.utils.logging_utils as logger

if TYPE_CHECKING:
    from smd.methods import SMDModel


DEFAULT_DISTRIBUTED_TIMEOUT_S = 600


def distributed_timeout_seconds() -> int:
    """Return the timeout used by the default distributed process group."""
    timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", str(DEFAULT_DISTRIBUTED_TIMEOUT_S)))
    if timeout_seconds <= 0:
        raise ValueError(f"NCCL_TIMEOUT must be positive, got {timeout_seconds}")
    return timeout_seconds


def init():
    """Initialize distributed data parallel."""
    if torch.distributed.is_available() and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        timeout_seconds = distributed_timeout_seconds()
        timeout = timedelta(seconds=timeout_seconds)

        # Use both gloo (CPU) and nccl (CUDA) backends so that FSDP2 with
        # cpu_offload can broadcast CPU tensors via gloo while keeping GPU
        # collectives on nccl.
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        logger.info(
            f"[{os.getpid()}] rank = {dist.get_rank()} ({local_rank}), world_size = {world_size}, timeout = {timeout_seconds}s"
        )
    else:
        logger.error("Distributed data parallel is not available")


def model_to_ddp(model: SMDModel) -> Union[SMDModel, torch.nn.parallel.DistributedDataParallel]:
    """Convert model to distributed data parallel."""
    if torch.distributed.is_available() and torch.cuda.is_available():
        model = DDPWrapper(
            model,
            device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
            output_device=int(os.environ.get("LOCAL_RANK", "0")),
            find_unused_parameters=model.config.ddp_find_unused_parameters,
        )
    else:
        raise RuntimeError("Distributed data parallel is not available")
    return model


class DDPWrapper(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def single_train_step(self, *args, **kwargs) -> Any:
        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            # The actual .single_train_step.
            return self.module.single_train_step(*_args, **_kwargs)

        # Patch the original_module's forward so we can redirect the arguments back to the real method.
        self.module.forward = wrapped_training_step
        # Call self, which implicitly calls self.forward() --> model.forward(), which is now model.training_step().
        # Without calling self.forward() or model.forward() explicitly, implicit hooks are also executed.
        return self(*args, **kwargs)


@contextmanager
def ddp_sync_grad(model: SMDModel, enabled: bool):
    r"""
    Context manager to enable/disable gradient synchronizations across DDP processes for DDP model.
    Modified from:
    https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
    Note that this is incompatible with static_graph=True and will be an no-op if static_graph=True.

    Within this context, gradients will be accumulated on module
    variables, which will later be synchronized in the first
    forward-backward pass exiting the context.

    .. warning::
        The forward pass should be included inside the context manager, or
        else gradients will still be synchronized.
    """
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                logger.warning("DDP static_graph=True is incompatible with sync_grad(). Performance will be reduced.")
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.require_backward_grad_sync = old_require_backward_grad_sync
