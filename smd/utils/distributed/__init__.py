# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Optional, Callable

import torch
import torch.distributed as dist

import smd.utils.logging_utils as logger


_LAUNCH_ENV_SIGNATURE = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def has_launch_env() -> bool:
    """Return True when a distributed launcher has populated the rank env."""
    return all(name in os.environ for name in _LAUNCH_ENV_SIGNATURE)


def launch_world_size() -> int:
    """Return the launcher-provided world size before process-group initialization."""
    if not has_launch_env():
        return 1
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        logger.warning("Ignoring invalid WORLD_SIZE; using 1.")
        return 1


def launch_rank() -> int:
    """Return the launcher-provided global rank before process-group initialization."""
    if not has_launch_env():
        return 0
    try:
        return max(0, int(os.environ.get("RANK", "0")))
    except ValueError:
        logger.warning("Ignoring invalid RANK; using 0.")
        return 0


def world_size():
    """Get the world size."""
    if dist.is_initialized() and torch.cuda.is_available():
        return dist.get_world_size()
    return 1


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def is_rank0() -> bool:
    """Return True if this is rank 0 (the primary loading rank)."""
    return get_rank() == 0


def synchronize():
    """
    Synchronize all devices.

    This method checks if the current running environment
    is distributed with a world-size greater than 1.
    If so, we use `dist.barrier` to synchronize
    all processes.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return
    logger.debug(f"Synchronizing all devices with world size {world_size}")
    dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
    logger.debug(f"Synchronized all devices with world size {world_size}")


def rank0_only(func: Callable) -> Callable:
    """Apply this function only to the master GPU.

    Example usage:
        @rank0_only
        def func(x):
            return x + 1

    Args:
        func (Callable): any function.

    Returns:
        (Callable): A function wrapper executing the function only on the master GPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_rank0():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def clean_up():
    if dist.is_available() and dist.is_initialized():
        try:
            logger.info("Distributed clean up.")
            dist.destroy_process_group()
        except ValueError as e:
            logger.error(f"Error destroying default process group: {e}")


def sync_all(local_all: bool, device: torch.device) -> bool:
    """Synchronize local all across distributed ranks.

    Args:
        local_all: all() in each rank
        device: Device for tensor operations

    Returns:
        global_all
    """
    global_all = torch.tensor([local_all], dtype=torch.uint8, device=device)

    if world_size() > 1:
        # MIN reduction: global_all is True only if all ranks have all samples in second stage
        torch.distributed.all_reduce(global_all, op=torch.distributed.ReduceOp.MIN)

    return global_all.to(torch.bool).item()


def sync_any(local_any: bool, device: torch.device) -> bool:
    """Synchronize local any across distributed ranks.

    Args:
        local_any: any() in each rank
        device: Device for tensor operations

    Returns:
        global_any
    """
    global_any = torch.tensor([local_any], dtype=torch.uint8, device=device)

    if world_size() > 1:
        # MAX reduction: global_any is True if any rank has any samples in second stage
        torch.distributed.all_reduce(global_any, op=torch.distributed.ReduceOp.MAX)

    return global_any.to(torch.bool).item()


def move_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    name: str = "module",
) -> None:
    """
    Move a module to the target device and precision, handling meta tensors.

    When using FSDP meta initialization, non-rank-0 processes have meta tensors
    that need to be materialized and synchronized from rank 0.

    Args:
        module: The module to move
        device: Target device
        dtype: Target dtype (optional)
        name: Name of the module for logging
    """
    # Check if ANY rank has meta tensors (need collective check for broadcast)
    is_meta = any(p.device.type == "meta" for p in module.parameters())
    any_meta = sync_any(is_meta, device=device) if world_size() > 1 else is_meta

    if any_meta:
        logger.info(f"{name}: some ranks have meta tensors, materializing and broadcasting from rank 0")
        if is_meta:
            # This rank has meta tensors - materialize them
            module.to_empty(device=device)
            if dtype is not None:
                module.to(dtype=dtype)
        else:
            # Rank 0 has real weights - just move to device
            module.to(dtype=dtype, device=device)

        # Broadcast weights from rank 0 to all other ranks (collective operation)
        if world_size() > 1:
            for param in module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)
    else:
        # No meta tensors on any rank - simple move
        module.to(dtype=dtype, device=device)

    synchronize()
