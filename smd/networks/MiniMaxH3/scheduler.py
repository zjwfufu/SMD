# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class MiniMaxH3SchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class MiniMaxH3Scheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._shift = float(shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor | None = None) -> torch.Tensor:
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | torch.Tensor | None = None,
    ) -> None:
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "`set_timesteps` requires either an explicit `sigmas` schedule or "
                    f"`num_inference_steps` >= 2, got {num_inference_steps}."
                )
            base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
            sigmas = self._shift * base / (1 + (self._shift - 1) * base)
            sigmas = torch.unique_consecutive(sigmas)
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
                raise ValueError("`sigmas` must hold at least two strictly decreasing values ending at 0.0.")

        self.sigmas = sigmas.to(device=device)
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `index_for_timestep`.")
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError("Passed `timestep` is not in `self.timesteps`.")
        return indices[0].item()

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> MiniMaxH3SchedulerOutput | tuple[torch.FloatTensor]:
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError("Pass actual timestep values from `scheduler.timesteps`, not integer indices.")
        if self.sigmas is None or self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")

        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
        prev_sample = prev_sample.to(dtype=sample.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)
