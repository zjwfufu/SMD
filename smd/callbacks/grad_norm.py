# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from smd.callbacks.callback import Callback
from smd.callbacks.grad_clip import get_grad_norm_fsdp
from smd.callbacks.wandb_utils import log_at_iteration
from smd.utils.distributed import is_rank0

if TYPE_CHECKING:
    from smd.methods import SMDModel


class GradNormCallback(Callback):
    """Measure one optimizer-owned module's gradient norm without modifying gradients."""

    def __init__(self, model_key: str, norm_type: float = 2.0, metric_prefix: str | None = None) -> None:
        self.model_key = model_key
        self.norm_type = norm_type
        self.metric_prefix = metric_prefix

    def on_optimizer_step_begin(self, model: SMDModel, iteration: int = 0) -> None:
        if self.model_key not in model.optimizer_dict:
            raise KeyError(f"Unknown optimizer key {self.model_key!r}; available: {list(model.optimizer_dict)}")
        optimizer = model.optimizer_dict[self.model_key]
        if model.should_use_grad_scaler(optimizer):
            model.grad_scaler.unscale_(optimizer)

        model_device = model.device
        module = model
        for subnet in self.model_key.split("."):
            module = getattr(module, subnet)

        parameters = [parameter for parameter in module.parameters() if parameter.grad is not None]
        uses_cpu_offload = any(
            isinstance(parameter.grad, DTensor) and parameter.grad._local_tensor.device.type == "cpu"
            for parameter in parameters
        )
        if uses_cpu_offload:
            total_norm = get_grad_norm_fsdp(parameters, norm_type=self.norm_type, device=model_device)
        else:
            gradients = [parameter.grad for parameter in parameters]
            total_norm = torch.nn.utils.get_total_norm(
                gradients,
                norm_type=self.norm_type,
                foreach=True,
            )

        if iteration % self.config.trainer.logging_iter == 0 and is_rank0():
            log_at_iteration(
                {
                    (
                        f"{self.metric_prefix}/grad_norm/{self.model_key}"
                        if self.metric_prefix
                        else f"optimizer/grad_norm (model_key {self.model_key})"
                    ): total_norm.item()
                },
                iteration=iteration,
            )
