# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, TYPE_CHECKING, Optional

import torch

from smd.callbacks.callback import Callback
from smd.callbacks.wandb_utils import log_at_iteration
from smd.utils.basic_utils import get_batch_size_total
from smd.utils.distributed import synchronize, is_rank0
import smd.utils.logging_utils as logger

if TYPE_CHECKING:
    from smd.methods import SMDModel


class EMACallback(Callback):
    def __init__(
        self,
        type: str = "constant",
        # params for type=constant
        beta: float = 0.9999,
        # params for type=power
        gamma: float = 16.97,
        # params for type=halflife
        ema_halflife_kimg: float = 500,
        ema_rampup_ratio: Optional[float] = 0.05,
        start_iter: int = 0,
        ema_name: str = "ema",
        batch_size: int = 1,  # overwritten by self.config if it exists
        fsdp: bool = False,  # overwritten by self.config if it exists
    ):
        self.type = type
        self.beta = beta
        self.gamma = gamma
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio
        self.start_iter = start_iter
        self.ema_name = ema_name
        self.batch_size = batch_size
        self._is_fsdp = fsdp
        self._enabled = True

    def on_app_begin(self) -> None:
        if hasattr(self, "config"):
            # override using config
            self._is_fsdp = self.config.trainer.fsdp
            self.batch_size = get_batch_size_total(self.config)

    def on_model_init_end(
        self, model: SMDModel | torch.nn.parallel.DistributedDataParallel, iteration: int = 0
    ) -> None:
        # Unwrap DDP if needed to access the original model's attributes
        if hasattr(model, "module"):
            model = model.module

        # check ema initialization
        ema = getattr(model, self.ema_name, None)
        if ema is None:
            self._enabled = False
            logger.info(f"EMA {self.ema_name} is not enabled, skipping callback.")
            return

        assert ema.training is False, f"EMA {self.ema_name} should be in eval mode"
        for name, p_net in ema.named_parameters():
            assert not p_net.requires_grad, f"EMA parameter {name} should not require gradients"

    def _total_iteration(self, model: SMDModel, iteration: int) -> int:
        if hasattr(model, "resume_iter"):
            assert isinstance(model.resume_iter, int)
            iteration = iteration + model.resume_iter
        return iteration

    def _power_function_beta(self, iteration):
        beta = (1 - 1 / iteration) ** (self.gamma + 1)
        return beta

    def _get_cur_nimg(self, iteration):
        cur_nimg = iteration * self.batch_size
        return self.batch_size, cur_nimg

    def _halflife_beta(self, iteration):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        batch_size, cur_nimg = self._get_cur_nimg(iteration)
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        return ema_beta

    def _compute_beta(self, total_iteration: int) -> float:
        if self.type == "constant":
            return self.beta
        elif self.type == "power":
            return self._power_function_beta(total_iteration)
        elif self.type == "halflife":
            return self._halflife_beta(total_iteration)
        raise ValueError(f"Invalid {self.ema_name} type: {self.type}")

    def on_training_step_end(
        self,
        model: SMDModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict

        # Check if EMA is enabled
        if not self._enabled:
            return

        # Get total iteration and skip if before start_iter
        total_iteration = self._total_iteration(model, iteration)
        if total_iteration < self.start_iter:
            return
        elif total_iteration == self.start_iter:
            logger.info(f"Starting to update {self.ema_name} at iteration {total_iteration}.")
        if hasattr(model, "should_update_ema") and not model.should_update_ema(total_iteration):
            return

        beta = self._compute_beta(total_iteration)
        initialize = total_iteration == self.start_iter

        # Single unified model interface. The model owns the entire EMA update policy
        # (which modules to update, FSDP gather semantics, strict key matching). The
        # callback only computes beta and drives the schedule.
        with torch.no_grad():
            model.update_ema(self.ema_name, beta=beta, initialize=initialize, is_fsdp=self._is_fsdp)

            if hasattr(self, "config"):
                # only wandb log when config exists
                if iteration % self.config.trainer.logging_iter == 0 and is_rank0():
                    log_at_iteration({f"ema/{self.ema_name}_beta": beta}, iteration=iteration)
        synchronize()
