# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
from torch.distributed.tensor import DTensor

from smd.callbacks.callback import Callback
from smd.callbacks.wandb_utils import log_at_iteration
from smd.utils.distributed import is_rank0, world_size
import smd.utils.logging_utils as logger

if TYPE_CHECKING:
    from smd.methods import SMDModel


@contextmanager
def cast_gradients_dtype(model, dtype=torch.float32, enabled=True):
    if enabled:
        try:
            # Cast gradients to the desired dtype
            for param in model.parameters():
                if param.grad is not None and param.grad.dtype != dtype:
                    param.grad.data = param.grad.data.to(dtype)
            yield
        finally:
            # Restore original gradient dtypes
            for param in model.parameters():
                if param.grad is not None and param.grad.dtype != param.dtype:
                    param.grad.data = param.grad.data.to(param.dtype)
    else:
        yield


def get_grad_norm_fsdp(
    parameters,
    norm_type: float = 2.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute the global gradient norm for FSDP2 parameters.

    This implementation:
    1. Extracts local tensors from DTensors
    2. Computes local norms on native device (CPU or GPU)
    3. All-reduces the scalar norm to get global norm

    Args:
        parameters: Iterable of parameters with gradients
        norm_type: Type of norm (default: L2)
        device: Device for all-reduce tensor. If None, inferred from gradients or defaults to cuda.

    Returns:
        Total gradient norm (global across all ranks) as a regular tensor
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(p for p in parameters if p.grad is not None)

    if len(parameters) == 0:
        return torch.tensor(0.0, device=device)

    # Compute per-parameter norms on their native device (CPU or GPU)
    # We compute norm^norm_type to allow proper aggregation across ranks
    local_norm_sum = 0.0
    inferred_device = None
    for p in parameters:
        if isinstance(p.grad, DTensor):
            grad = p.grad._local_tensor
        else:
            grad = p.grad

        # Infer CUDA device from gradients (use first CUDA device found)
        if inferred_device is None and grad.device.type == "cuda":
            inferred_device = grad.device

        # Compute norm on the gradient's native device, accumulate as Python float
        local_norm_sum += torch.norm(grad.detach().float(), norm_type).item() ** norm_type

    # Use provided device, or inferred device, or fall back to current CUDA device
    if device is None:
        device = inferred_device if inferred_device is not None else torch.device("cuda")
    local_norm_sum = torch.tensor(local_norm_sum, device=device)

    # All-reduce to get global norm across all ranks
    if world_size() > 1:
        torch.distributed.all_reduce(local_norm_sum, op=torch.distributed.ReduceOp.SUM)

    # Compute final global norm
    return local_norm_sum ** (1.0 / norm_type)


def clip_grad_norm_fsdp(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Clip gradients for FSDP2 models with CPU offloading."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(p for p in parameters if p.grad is not None)
    total_norm = get_grad_norm_fsdp(parameters, norm_type=norm_type, device=device)

    # Compute clip coefficient (regular tensor division, no DTensor ops)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Apply clipping to gradients
    for p in parameters:
        if isinstance(p.grad, DTensor):
            # For DTensor, scale the local tensor directly
            local_grad = p.grad._local_tensor
            local_grad.mul_(clip_coef_clamped.to(local_grad.device))
        else:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return total_norm


class GradClipCallback(Callback):
    def __init__(
        self,
        grad_norm: float | None = 1.0,
        model_key: str = "net",
        posinf: float | None = None,
        neginf: float | None = None,
        precision_grad_clip: Optional[torch.dtype] = None,
        metric_prefix: str | None = None,
    ) -> None:
        self.grad_norm = grad_norm
        self.model_key = model_key
        self.posinf = posinf
        self.neginf = neginf
        self.precision_grad_clip = precision_grad_clip
        self.metric_prefix = metric_prefix

    def nan_to_num(self, module: torch.nn.Module) -> tuple[int, torch.dtype | None]:
        grad_dtype = None
        non_finite_grads_count = 0

        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_dtype = param.grad.dtype

                # Extract local tensor for DTensor (avoids triggering distributed ops on CPU)
                if isinstance(param.grad, DTensor):
                    grad = param.grad._local_tensor
                else:
                    grad = param.grad

                non_finite_grads = grad.numel() - grad.isfinite().sum().item()
                if non_finite_grads:
                    non_finite_grads_count += non_finite_grads
                    logger.debug(
                        f"Gradient of {name} (dtype {grad_dtype}) is not finite: "
                        f"Setting {grad.isnan().sum().item()} NaNs to 0 and {grad.isinf().sum().item()} Infs "
                        f"to {self.posinf} or {self.neginf}."
                    )
                    torch.nan_to_num(grad, nan=0.0, posinf=self.posinf, neginf=self.neginf, out=grad)

        return non_finite_grads_count, grad_dtype

    def on_optimizer_step_begin(self, model: SMDModel, iteration: int = 0) -> None:
        # unscale the optimizer related to the `model_key`
        assert self.model_key in model.optimizer_dict.keys(), (
            f"Keys in optimizer_dict: {list(model.optimizer_dict.keys())}."
        )
        optimizer = model.optimizer_dict[self.model_key]
        # Only unscale if grad_scaler should be used (checks enabled + float32 grads)
        if model.should_use_grad_scaler(optimizer):
            model.grad_scaler.unscale_(optimizer)

        # Save model device before selecting subnet (subnet may not have .device)
        model_device = model.device

        # select subnet if specified (by default, we only perform gradient clips on model.net)
        subnets = self.model_key.split(".")
        for subnet in subnets:
            model = getattr(model, subnet)

        # set nan to num for each parameter
        non_finite_grads_count, grad_dtype = self.nan_to_num(model)
        logger.debug(f"Gradient dtype of {self.model_key}: {grad_dtype}")
        if non_finite_grads_count > 0:
            logger.info(
                f"Number of parameters with non-finite gradients (of dtype {grad_dtype}): {non_finite_grads_count}"
            )
        nonfinite_key = (
            f"{self.metric_prefix}/non_finite_grads/{self.model_key}"
            if self.metric_prefix
            else f"optimizer/non_finite_grads_count (model_key {self.model_key})"
        )
        log_dict = {nonfinite_key: non_finite_grads_count}

        if self.grad_norm is not None:
            # Cast all gradients to precision_grad_clip for numerical stability during clipping
            cast_grads = (
                self.precision_grad_clip is not None
                and grad_dtype is not None
                and grad_dtype != self.precision_grad_clip
            )

            # log value at first iteration
            if iteration == 1 and cast_grads:
                logger.info(f"Casting gradients from {grad_dtype} to {self.precision_grad_clip} before clipping.")

            # Check if CPU offloading is enabled by looking for DTensor grads on CPU
            # CPU offloading = DTensor local tensors are on CPU
            # No CPU offloading = DTensor local tensors are on GPU (or no DTensors)
            use_fsdp_cpu_offload_clip = False
            for p in model.parameters():
                if p.grad is not None and isinstance(p.grad, DTensor):
                    if p.grad._local_tensor.device.type == "cpu":
                        use_fsdp_cpu_offload_clip = True
                        break

            with cast_gradients_dtype(model, dtype=self.precision_grad_clip, enabled=cast_grads):
                if use_fsdp_cpu_offload_clip:
                    # Use custom clipping for FSDP with CPU offloading
                    # Standard clip_grad_norm_ fails because DTensor ops trigger all_reduce on CPU
                    total_norm = clip_grad_norm_fsdp(model.parameters(), self.grad_norm, device=model_device)
                else:
                    # Standard clipping for non-FSDP or FSDP without CPU offloading
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm, foreach=True)

            norm_key = (
                f"{self.metric_prefix}/grad_norm/{self.model_key}"
                if self.metric_prefix
                else f"optimizer/grad_norm (model_key {self.model_key})"
            )
            log_dict[norm_key] = total_norm.item()

        if hasattr(self, "config"):
            # only wandb log when config exists
            if iteration % self.config.trainer.logging_iter == 0 and is_rank0():
                log_at_iteration(log_dict, iteration=iteration)
