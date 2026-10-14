# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""factory for different noise schedules."""

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from scipy import stats
from torch.distributions import Normal
from diffusers import DDIMScheduler, CogVideoXDPMScheduler

import smd.utils.logging_utils as logger
from smd.utils import expand_like
from smd.utils.basic_utils import PRECISION_MAP
from smd.utils.hf_utils import get_hf_cache_dir


NET_PRED_TYPES = {"x0", "eps", "v", "flow"}


@dataclass(frozen=True)
class SegmentSchedule:
    """One Euler segment of a flow-matching sampling schedule.

    A full schedule of ``num_steps`` segments is the ordered list
    ``[seg_0, ..., seg_{num_steps-1}]`` covering raw-t ``1.0 → 0.0`` (noise → data).
    Segment ``i`` advances the chain from ``sigma_start`` to ``sigma_end``.

    Owned and produced exclusively by
    :meth:`BaseNoiseSchedule.build_segment_schedule`; downstream training /
    sampling loops consume it read-only.

    Attributes:
        index: 0-based segment index in ``[0, num_steps)``.
        raw_start: raw-t knot at the segment start — a Python ``float`` taken
            directly from the raw grid (exact, not warped).
        raw_end: raw-t knot at the segment end.
        sigma_start: warped + clamped sigma at ``raw_start`` — a **0-dim tensor**
            in the scheduler's ``t_precision``, ready for ``.expand(batch_size)``.
        sigma_end: warped + clamped sigma at ``raw_end`` — a 0-dim tensor. Equal
            in value to the next segment's ``sigma_start``.
        size: raw segment length ``raw_start - raw_end`` (Python ``float``), used
            as the loss/step weight (matches the pre-refactor ``seg_size``).
    """

    index: int
    raw_start: float
    raw_end: float
    sigma_start: torch.Tensor
    sigma_end: torch.Tensor
    size: float


class BaseNoiseSchedule(torch.nn.Module):
    """Abstract base noise schedule class.

    This class defines a noise schedule used for training/sampling from
    diffusion models.

    We follow the general format to define the forward process:

        x_t = alpha(t) x_0 + sigma(t) eps

    """

    def __init__(
        self,
        min_t: float,
        max_t: float,
        num_steps: int,
        clamp_min: float = 1e-6,
        t_precision: str = "float64",
        **kwargs,
    ):
        super().__init__()
        self._min_t = min_t
        self._max_t = max_t
        self.num_steps = num_steps
        self._supported_time_dist_types = ("lognormal", "logitnormal", "uniform", "polynomial", "shifted", "log_t")
        self.clamp_min = clamp_min
        self.t_precision = PRECISION_MAP[t_precision]

        # --- Segment-schedule time-shift policy (installed via configure_time_shift) ---
        # SMD flow schedules use build_segment_schedule() which
        # owns raw-knot construction, fixed/dynamic (resolution-dependent) shift, the
        # sigma warp, and the safe clamp. The policy is configured once (fixed shift or
        # dynamic base/max/seq-len/empirical params) and cached per input shape.
        self._time_shift_configured: bool = False
        self._time_shift_dynamic: bool = False
        self._time_shift_cache: dict[tuple[int, ...], float] = {}

    @property
    def min_t(self) -> float:
        """The minimum time step supported by this schedule.

        Returns:
            float: Minimum time step.
        """
        assert self._min_t >= 0, "min_t must be non-negative"
        return self._min_t

    @property
    def max_t(self) -> float:
        """The maximum time step supported by this schedule.

        Returns:
            float: Maximum time step.
        """
        assert self._max_t > self._min_t, "max_t must be greater than min_t"
        return self._max_t

    def latents(self, noise: torch.Tensor, t_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scale the noise by the sigma at the initial time step.

        Args:
            noise: Noise tensor.
            t_init: Initial time step. If None, use the maximum time step.

        Returns:
            torch.Tensor: Scaled noise tensor.
        """
        if t_init is None:
            t_init = torch.as_tensor(self.max_t, dtype=self.t_precision, device=noise.device)

        # we cast to double to be consistent with the `forward_process` method
        assert self.is_t_valid(t_init), f"t_init must be in [{self.min_t}, {self.max_t}], but got {t_init}"
        sigma = expand_like(self._sigma(t_init.to(torch.float64)), noise)
        return (noise.to(torch.float64) * sigma).to(noise.dtype)

    @staticmethod
    def safe_clamp(t: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
        """
        Clamp the timestep t such that min <= t <= max holds in floating point precision.

        Args:
            t (torch.Tensor): The timestep.
            min (float | None): The minimum time step.
            max (float | None): The maximum time step.

        Returns:
            torch.Tensor: The clamped timestep.
        """
        if min is not None:
            min_tensor = torch.as_tensor(min, dtype=t.dtype, device=t.device)
            if min_tensor.item() < min:
                # round up to the next representable value
                new_min = torch.nextafter(min_tensor, torch.tensor(float("inf"), dtype=t.dtype, device=t.device)).item()
                assert new_min >= min
                min = new_min

        if max is not None:
            max_tensor = torch.as_tensor(max, dtype=t.dtype, device=t.device)
            if max_tensor.item() > max:
                # round down to the next representable value
                new_max = torch.nextafter(
                    max_tensor, torch.tensor(float("-inf"), dtype=t.dtype, device=t.device)
                ).item()
                assert new_max <= max
                max = new_max

        return torch.clamp(t, min=min, max=max)

    def non_zero_clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clamp the tensor such that it is non-zero."""
        return torch.where(
            tensor >= 0,
            tensor.clamp(min=self.clamp_min),
            tensor.clamp(max=-self.clamp_min),
        )

    # ------------------------------------------------------------------
    # Segment schedule (flow-matching Euler sampling grid)
    #
    # These methods own everything the SMD segment loops and the
    # student/EMA samplers used to reimplement by mutating scheduler private
    # fields: raw-knot construction, fixed vs. dynamic (resolution-dependent)
    # shift, empirical-mu shift (Flux2/Lens), the sigma warp, the safe clamp, and
    # shape-keyed shift caching. Only warp-based flow schedules (those exposing
    # ``warp_t``) support this; ``configure_time_shift`` must run once first.
    # ------------------------------------------------------------------

    @staticmethod
    def compute_raw_grid_knots(num_steps: int, final_step_size_scale: float) -> list[float]:
        """Construct raw-t segment knots: ``num_steps + 1`` values from 1.0 → 0.0.

        The final Euler segment is scaled by ``final_step_size_scale``;
        ``final_step_size_scale == 1.0`` yields a uniform grid. The ``max(..., 1e-4)``
        floor preserves the exact pre-refactor numerics.

        Args:
            num_steps: Number of Euler segments (NFE); must be >= 1.
            final_step_size_scale: Last segment size as a multiple of the normal
                segment size.

        Returns:
            List of ``num_steps + 1`` raw-t knots, strictly decreasing from 1.0 to 0.0.
        """
        fss = max(final_step_size_scale, 1e-4)
        base_seg = 1.0 / (num_steps - 1 + fss)
        knots: list[float] = [1.0]
        for step_id in range(num_steps):
            seg = base_seg * (fss if step_id == num_steps - 1 else 1.0)
            knots.append(knots[-1] - seg)
        knots[-1] = 0.0
        return knots

    def configure_time_shift(
        self,
        *,
        dynamic: bool,
        fixed_shift: float | None = None,
        base_shift: float | None = None,
        max_shift: float | None = None,
        base_seq_len: int | None = None,
        max_seq_len: int | None = None,
        time_shift_type: str | None = None,
        empirical_steps: int | None = None,
    ) -> None:
        """Install the time-shift policy consumed by :meth:`build_segment_schedule`.

        This is the public, encapsulated replacement for reaching into scheduler
        private fields (``_base_shift``/``_max_shift``/``_shift``/...) from a training
        method. Call once (idempotent); it resets the per-shape shift cache.

        Fixed mode (``dynamic=False``) pins ``self._shift = fixed_shift``. Dynamic
        mode (``dynamic=True``) stores the resolution-dependent parameters and defers
        the actual shift to ``compute_shift(input_shape)`` inside
        ``build_segment_schedule`` (cached per shape). ``empirical_steps > 0`` selects
        the BFL empirical-mu formula on schedules that support it (Flux2/Lens).

        Raises:
            ValueError: On missing required parameters or an invalid
                ``time_shift_type``.
            TypeError: If ``dynamic=True`` on a schedule without ``compute_shift``, or
                ``empirical_steps`` is given to a schedule without empirical-mu support.
        """
        if dynamic:
            if not hasattr(self, "compute_shift"):
                raise TypeError(
                    f"{type(self).__name__} does not support dynamic (resolution-dependent) shift "
                    "(no compute_shift); use configure_time_shift(dynamic=False, fixed_shift=...)."
                )
            missing = [
                name
                for name, value in (
                    ("base_shift", base_shift),
                    ("max_shift", max_shift),
                    ("base_seq_len", base_seq_len),
                    ("max_seq_len", max_seq_len),
                    ("time_shift_type", time_shift_type),
                )
                if value is None
            ]
            if missing:
                raise ValueError("configure_time_shift(dynamic=True) requires " + ", ".join(missing) + ".")
            if time_shift_type not in ("linear", "exponential"):
                raise ValueError(f"time_shift_type must be 'linear' or 'exponential', got {time_shift_type!r}")
            self._base_shift = float(base_shift)
            self._max_shift = float(max_shift)
            self._base_seq_len = int(base_seq_len)
            self._max_seq_len = int(max_seq_len)
            self._time_shift_type = str(time_shift_type)
            if hasattr(self, "_empirical_steps"):
                self._empirical_steps = int(empirical_steps if empirical_steps is not None else 0)
            elif empirical_steps:
                raise TypeError(
                    f"{type(self).__name__} has no empirical-mu support but empirical_steps="
                    f"{empirical_steps} was provided."
                )
            self._time_shift_dynamic = True
        else:
            if fixed_shift is None:
                raise ValueError("configure_time_shift(dynamic=False) requires an explicit fixed_shift.")
            self._shift = float(fixed_shift)
            self._time_shift_dynamic = False
        self._time_shift_cache = {}
        self._time_shift_configured = True

    def _resolve_time_shift(self, input_shape) -> None:
        """Resolve ``self._shift`` for ``input_shape`` (dynamic → compute + cache)."""
        if not self._time_shift_dynamic:
            return  # fixed shift already pinned by configure_time_shift
        key = tuple(int(s) for s in input_shape)
        cached = self._time_shift_cache.get(key)
        if cached is not None:
            self._shift = cached
            return
        self.compute_shift(tuple(input_shape))  # mutates self._shift internally
        self._time_shift_cache[key] = float(self._shift)

    def resolve_time_shift(self, input_shape) -> None:
        """Resolve the configured fixed/dynamic shift before custom schedule construction."""
        if not self._time_shift_configured:
            raise RuntimeError(f"{type(self).__name__}.resolve_time_shift() requires configure_time_shift(...) first.")
        self._resolve_time_shift(input_shape)

    def build_segment_schedule(
        self,
        *,
        input_shape,
        num_steps: int,
        final_step_size_scale: float,
        device: Optional[torch.device] = None,
    ) -> list[SegmentSchedule]:
        """Build the flow-matching Euler segment schedule (noise → data).

        Owns the full construction: shift resolution (fixed/dynamic/empirical),
        raw-knot grid, sigma warp, and safe clamp. Numerically identical to the
        pre-refactor ``_update_noise_scheduler`` + ``_build_sigma_grid`` pair.

        Args:
            input_shape: latent shape (excluding batch), e.g. ``[128, 64, 64]``. Used
                for the resolution-dependent shift; ignored for fixed shift.
            num_steps: Number of Euler segments (NFE); must be >= 1.
            final_step_size_scale: Last-segment size multiplier; must be finite and > 0.
            device: Device for the sigma tensors (defaults to the sigma array device).

        Returns:
            Ordered list of ``num_steps`` :class:`SegmentSchedule` objects.

        Raises:
            RuntimeError: If ``configure_time_shift`` has not been called.
            TypeError: If the schedule is not warp-based (no ``warp_t``).
            ValueError: If ``num_steps < 1`` or ``final_step_size_scale`` is invalid.
        """
        if not self._time_shift_configured:
            raise RuntimeError(
                f"{type(self).__name__}.build_segment_schedule() requires configure_time_shift(...) "
                "to install the shift policy first."
            )
        if not hasattr(self, "warp_t"):
            raise TypeError(
                f"{type(self).__name__} is not a flow-matching schedule (no warp_t); "
                "build_segment_schedule is only defined for warp-based flow schedules."
            )
        num_steps = int(num_steps)
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1 to yield at least one segment, got {num_steps}")
        fss = float(final_step_size_scale)
        if not math.isfinite(fss) or fss <= 0.0:
            raise ValueError(f"final_step_size_scale must be finite and > 0, got {final_step_size_scale}")

        self.resolve_time_shift(input_shape)

        device = device if device is not None else self.sigmas.device
        raw_grid = torch.tensor(
            self.compute_raw_grid_knots(num_steps, fss),
            device=device,
            dtype=self.t_precision,
        )
        sigma_grid = self.safe_clamp(
            self.warp_t(raw_grid).to(dtype=self.t_precision),
            min=self.min_t,
            max=self.max_t,
        )

        segments: list[SegmentSchedule] = []
        for i in range(num_steps):
            segments.append(
                SegmentSchedule(
                    index=i,
                    raw_start=float(raw_grid[i]),
                    raw_end=float(raw_grid[i + 1]),
                    sigma_start=sigma_grid[i],
                    sigma_end=sigma_grid[i + 1],
                    size=float(raw_grid[i] - raw_grid[i + 1]),
                )
            )
        return segments

    @property
    @abstractmethod
    def max_sigma(self) -> float:
        """The maximum noise scale, i.e., sigma(max_t).

        Returns:
            float: Maximum noise scale.
        """

    def rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale the timestep t to the range that is actually used by the model.

        Returns:
            torch.Tensor: rescaled timestep.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._rescale_t(t)

    @abstractmethod
    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """The data scale alpha(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The scaling value at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._alpha(t)

    @abstractmethod
    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise scale sigma(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The noise at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._sigma(t)

    @abstractmethod
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative of alpha(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The time derivative of alpha(t) at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._alpha_prime(t)

    @abstractmethod
    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative of sigma(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The time derivative of sigma(t) at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._sigma_prime(t)

    @abstractmethod
    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigmas(self) -> torch.Tensor:
        """The noise scales sigma(t) at all time steps, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Returns:
            torch.Tensor: The noise scales.
        """

    @abstractmethod
    def sample_t(
        self,
        n: int,
        time_dist_type: str = "uniform",
        min_t: Optional[float] = None,
        max_t: Optional[float] = None,
        device: Optional[torch.device] = None,
        **sample_kwargs,
    ) -> torch.Tensor:
        """Sample random time steps for sampling the teacher model for student model distillation

        Args:
            n (int): Batch size.
            time_dist_type (str): The type of time t distribution to sample from.
            min_t: Minimum time step. Defaults to None.
            max_t: Maximum time step. Defaults to None.
            device: Device for the output tensor. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing time steps, of shape [n].
        """

    def get_t_list(self, sample_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample a list of timesteps [max_t, ..., 0] for multistep student model distillation

        Args:
            sample_steps: Number of timesteps to return
            device: Target device for the output tensor

        Returns:
            torch.Tensor: List of `num_steps + 1` timesteps uniformly sampled in decreasing order
        """
        device = device or self.sigmas.device
        t_list = torch.linspace(self.max_t, 0, sample_steps + 1, device=device, dtype=self.t_precision)
        return self.safe_clamp(t_list, max=self.max_t)

    def sample_from_t_list(
        self,
        n: int,
        sample_steps: int,
        t_list: Optional[list] = None,
        return_ids: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Sample timestep from the t_list tensor. If t_list is not given, it uses self.get_t_list() to
            get the default t_list tensor.

        Args:
            n (int): Batch size.
            sample_steps: (int) Number of timesteps to return
            t_list (torch.Tensor): list of finite time steps
            return_ids (bool): If True, return timestep indices instead of time step indices.
            device: torch device

        Return:
            torch.Tensor: Tensor containing time steps, of shape [n].
        """
        if t_list is None:
            device = device or self.sigmas.device
            t_list = self.get_t_list(sample_steps=sample_steps, device=device)
        else:
            t_list = torch.tensor(t_list, device=device, dtype=self.t_precision)

        ids = torch.randint(0, len(t_list) - 1, (n,)).to(device=device)  # Do not train on clean data (t = 0)
        if return_ids:
            return t_list[ids], ids
        return t_list[ids]

    def next_in_t_list(
        self,
        ids: torch.Tensor,
        sample_steps: int,
        t_list: torch.Tensor | None,
        device: Optional[torch.device] = None,
        stride: int = 1,
    ) -> torch.Tensor:
        """
        Get the next time step in the t_list based on the ids.

        Args:
            ids: Tensor of shape [batch_size] containing the ids of the time steps to get the next time step for.
            sample_steps: Number of timesteps in the t_list.
            t_list: Tensor of shape [sample_steps + 1] containing the time steps.
            device: Device to store the output tensor. Defaults to None.
            stride: Stride to use to get the next time step. Defaults to 1.

        Returns:
            Tensor of shape [batch_size] containing the next time steps.
        """
        if t_list is None:
            t_list = self.get_t_list(sample_steps=sample_steps, device=device)
        else:
            t_list = torch.as_tensor(t_list, device=device, dtype=self.t_precision)
            assert t_list.shape == (sample_steps + 1,), (
                f"t_list must be of shape (sample_steps + 1,), but got {t_list.shape}"
            )

        next_ids = ids + stride
        if next_ids.max() > sample_steps:
            raise ValueError(
                f"Clamping next ids to sample steps. ids: {ids}, next_ids: {next_ids}, sample_steps: {sample_steps}"
            )
        return t_list[next_ids]

    def sample_t_inhom(
        self,
        n: int,
        seq_len: int,
        chunk_size: int,
        sample_steps: int,
        t_list: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample block-wise independent t variables for diffusion forcing in distillation.

        Args:
            n: batch size
            seq_len: sequence length
            chunk_size: chunk size
            t_list: list of finite time steps
            device: torch device

        Return:
            Tuple[torch.Tensor, torch.Tensor]: The first tensor contains time steps and
                the second tensor contains the indices, both of shape [n, seq_len].
        """
        if t_list is None:
            target_device = device or self.sigmas.device
            t_list = self.get_t_list(sample_steps, device=target_device)
        else:
            t_list = torch.tensor(t_list, device=device, dtype=self.t_precision)

        num_chunks = seq_len // chunk_size
        remaining_size = seq_len % chunk_size
        unique_id = torch.randint(0, len(t_list), (n, num_chunks)).to(device=device)

        # First chunk’s t_id is repeated for the first (chunk_size + remaining_size) elements
        first_col = unique_id[:, :1].repeat(1, chunk_size + remaining_size)

        # Remaining chunks: each t_id repeated chunk_size times
        rest_cols = unique_id[:, 1:].repeat_interleave(chunk_size, dim=1)

        # Concatenate: [first chunk | remaining chunks] → shape (n, seq_len)
        ids = torch.cat([first_col, rest_cols], dim=1)
        return t_list[ids], ids

    def sample_t_inhom_sft(self, n: int, seq_len: int, chunk_size: int, **sample_t_kwargs) -> torch.Tensor:
        """
        Sample block-wise independent t variables for diffusion forcing in SFT.

        Args:
            n: batch size
            seq_len: sequence length
            chunk_size: chunk size

        Return:
            torch.Tensor: time steps of shape [n, seq_len].
        """
        num_chunks = seq_len // chunk_size
        remaining_size = seq_len % chunk_size
        flat_t = self.sample_t(n * num_chunks, **sample_t_kwargs)  # shape (n*num_chunks,)

        t = flat_t.view(n, num_chunks)  # shape (n, num_chunks)

        first_col = t[:, :1].repeat(1, chunk_size + remaining_size)
        rest_cols = t[:, 1:].repeat_interleave(chunk_size, dim=1)

        t_inhom = torch.cat([first_col, rest_cols], dim=1)
        return t_inhom

    def is_t_valid(self, t: torch.Tensor) -> torch.bool:
        if t.dtype in (
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):  # nextafter only supports these dtypes
            dtype = t.dtype
        else:
            dtype = torch.float32
        min_t = torch.as_tensor(self.min_t, dtype=dtype, device=t.device)
        max_t = torch.as_tensor(self.max_t, dtype=dtype, device=t.device)
        # deal with dtype issue
        lower = torch.nextafter(min_t, torch.tensor(float("-inf"), dtype=dtype, device=t.device))
        upper = torch.nextafter(max_t, torch.tensor(float("inf"), dtype=dtype, device=t.device))
        return torch.all((lower <= t) & (t <= upper))

    def forward_process(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The diffusion forward process for this noise schedule.

        Args:
            x (torch.Tensor): Input sample from the data distribution.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: Sample from the forward process at time `t` given `x`.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = x.dtype

        # Convert input tensors to double precision
        t = t.to(torch.float64)
        x = x.to(torch.float64)
        eps = eps.to(torch.float64)

        # Perform calculation in double precision
        alpha_t = expand_like(self._alpha(t), x)
        sigma_t = expand_like(self._sigma(t), eps)
        forward_process = x.mul(alpha_t).add(eps.mul(sigma_t))
        return forward_process.to(original_dtype)

    def cond_forward_process(
        self,
        x_src: torch.Tensor,
        eps_new: torch.Tensor,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional forward process: re-noise a state from t_src to a noisier t_dst.

        Given x_src at noise level t_src, produce x_dst at noise level t_dst > t_src
        by scaling the existing state and injecting fresh independent noise.

        For the general path x(t) = a(t)*x0 + b(t)*eps:
            Signal constraint:  c = a(t_dst) / a(t_src)
            Noise constraint:   d = sqrt(b(t_dst)^2 - c^2 * b(t_src)^2)
            Result:             x_dst = c * x_src + d * eps_new

        Special cases:
            VE (a=1, b=sigma):    x_dst = x_src + sqrt(sigma_dst^2 - sigma_src^2) * eps
            VP (a^2+b^2=1):       x_dst = (a_dst/a_src)*x_src + sqrt(1-(a_dst/a_src)^2) * eps
            RF (a=1-t, b=t):      x_dst = [(1-t_d)/(1-t_s)]*x_src + sqrt(...) * eps

        Args:
            x_src: State at t_src.  [B, C, ...]
            eps_new: Fresh Gaussian noise, same shape as x_src.
            t_src: Source time (less noisy).  [B] or scalar.
            t_dst: Destination time (more noisy, t_dst > t_src).  [B] or scalar.

        Returns:
            x_dst at noise level t_dst.
        """
        original_dtype = x_src.dtype
        t_src = t_src.to(torch.float64)
        t_dst = t_dst.to(torch.float64)
        x_src = x_src.to(torch.float64)
        eps_new = eps_new.to(torch.float64)

        alpha_src = expand_like(self._alpha(t_src), x_src)
        sigma_src = expand_like(self._sigma(t_src), x_src)
        alpha_dst = expand_like(self._alpha(t_dst), x_src)
        sigma_dst = expand_like(self._sigma(t_dst), x_src)

        # Signal scaling: rescale x_src so its x0 coefficient becomes alpha_dst
        signal_scale = alpha_dst / alpha_src.clamp(min=1e-8)
        # Fresh noise std: match total noise variance at t_dst
        noise_var = (sigma_dst**2 - (signal_scale**2) * (sigma_src**2)).clamp(min=0.0)
        noise_std = torch.sqrt(noise_var)

        cond_forward = signal_scale * x_src + noise_std * eps_new
        return cond_forward.to(original_dtype)

    def cond_velocity(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The instantaneous velocity for this noise schedule: dx/dt := alpha'(t) x_0 + sigma'(t) eps.

        Args:
            x (torch.Tensor): Input sample from the data distribution.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: Sample from the forward process at time `t` given `x`.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = x.dtype

        # Convert input tensors to double precision
        t = t.to(torch.float64)
        x = x.to(torch.float64)
        eps = eps.to(torch.float64)

        # Perform calculation in double precision
        alpha_prime_t = expand_like(self._alpha_prime(t), x)
        sigma_prime_t = expand_like(self._sigma_prime(t), eps)
        cond_vel = x.mul(alpha_prime_t).add(eps.mul(sigma_prime_t))

        return cond_vel.to(original_dtype)

    def closest_sigma_idx(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Find the closest `sigma` in `self.sigmas` that matches the target `sigma` value

        Args:
            sigma_t (torch.Tensor): The target `sigma`.

        Returns:
            torch.Tensor: Index of the closest `sigma` in `self.sigmas` that matches the target `sigma`.
        """
        shape = sigma_t.shape
        if sigma_t.ndim != 0:
            sigma_t = sigma_t.squeeze(list(range(1, sigma_t.ndim)))

        sigmas = self.sigmas.to(sigma_t)
        assert sigmas.ndim == 1

        # Find the right and left indices
        indices_right = torch.searchsorted(sigmas, sigma_t, side="right")
        indices_left = (indices_right - 1).clamp(min=0)
        indices_right = indices_right.clamp(max=sigmas.numel() - 1)

        # Compute the differences and return the closest index
        right_diff = torch.abs(sigmas[indices_right] - sigma_t)
        left_diff = torch.abs(sigmas[indices_left] - sigma_t)
        indices = torch.where(right_diff < left_diff, indices_right, indices_left)
        return indices.view(shape)

    @abstractmethod
    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Find `t` that matches the target `sigma` value based on `sigma_idx`.

        Args:
            sigma_idx (torch.Tensor): The index of the tensor `self.sigmas`.

        Returns:
            torch.Tensor: The timestep
        """

    def sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the square root of the Signal-to-Noise Ratio (SNR) at time `t`.
        sqrt_snr(t) = alpha(t) / sigma(t)
        """
        assert self.is_t_valid(t)
        # Calculate in double precision
        t = t.to(torch.float64)
        alpha = self._alpha(t)
        sigma = self._sigma(t)

        sqrt_snr_val = alpha / self.non_zero_clamp(sigma)
        return sqrt_snr_val.to(t.dtype)

    @abstractmethod
    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find `t` that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        Args:
            sqrt_snr_t (torch.Tensor): The square root of the SNR at time `t`.

        Returns:
            torch.Tensor: The timestep
        """

    def x0_to_eps(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the eps-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The eps-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)
        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Perform calculation in double precision
        eps = (xt - x0.mul(alpha_t)).div(self.non_zero_clamp(sigma_t))

        # Convert back to original dtype
        return eps.to(original_dtype)

    def eps_to_x0(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the eps-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            eps (torch.Tensor): The eps-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        eps = eps.to(torch.float64)

        # Perform calculation in double precision
        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Perform calculation in double precision
        x0 = (xt - eps.mul(sigma_t)).div(self.non_zero_clamp(alpha_t))

        # Convert back to original dtype
        return x0.to(original_dtype)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)
        alpha_prime_t = expand_like(self._alpha_prime(t), xt)
        sigma_prime_t = expand_like(self._sigma_prime(t), xt)

        # Perform calculation in double precision
        xt_coeff = sigma_prime_t / self.non_zero_clamp(sigma_t)
        x0_coeff = xt_coeff * alpha_t - alpha_prime_t
        x0 = (xt * xt_coeff - v) / self.non_zero_clamp(x0_coeff)

        # Convert back to original dtype
        return x0.to(original_dtype)

    @abstractmethod
    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        return self.cond_velocity(x0, self.x0_to_eps(xt, x0, t), t)

    def convert_model_output(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        src_pred_type: str = "x0",
        target_pred_type: str = "eps",
    ) -> torch.Tensor:
        """
        Convert the model output from src_pred_type to target_pred_type.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            model_output (`torch.Tensor`): The direct output from the learned diffusion model.
            t (torch.Tensor): The time step in the forward process.
            src_pred_type (`str`): The type of src prediction, choose from 'x0', 'eps', 'v', 'flow'.
            target_pred_type (`str`): The type of target prediction, choose from 'x0', 'eps', 'v', 'flow'.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        # If source and target are the same, return as-is
        if src_pred_type == target_pred_type:
            return model_output

        # Supported prediction types
        if src_pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported src_pred_type '{src_pred_type}'. Supported types: {NET_PRED_TYPES}")
        if target_pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported target_pred_type '{target_pred_type}'. Supported types: {NET_PRED_TYPES}")

        if src_pred_type == "v" or target_pred_type == "v":
            assert torch.allclose(self.alpha(t) ** 2 + self.sigma(t) ** 2, torch.ones_like(t), rtol=1e-2, atol=1e-2), (
                "Only AlphaNoiseSchedule supports v-prediction!"
            )

        # Convert through intermediate representations
        # First convert src to x0 (if needed)
        if src_pred_type == "x0":
            x0 = model_output
        elif src_pred_type == "eps":
            x0 = self.eps_to_x0(xt, model_output, t)
        elif src_pred_type == "v":
            x0 = self.v_to_x0(xt, model_output, t)
        elif src_pred_type == "flow":
            x0 = self.flow_to_x0(xt, model_output, t)
        else:
            raise ValueError(f"Conversion from '{src_pred_type}' not implemented")

        # Then convert x0 to target (if needed)
        if target_pred_type == "x0":
            return x0
        elif target_pred_type == "eps":
            return self.x0_to_eps(xt, x0, t)
        elif target_pred_type == "v":
            return self.x0_to_v(xt, x0, t)
        elif target_pred_type == "flow":
            return self.x0_to_flow(xt, x0, t)
        else:
            raise ValueError(f"Conversion to '{target_pred_type}' not implemented")


class EDMNoiseSchedule(BaseNoiseSchedule):
    """
    The noise schedule defined in the EDM paper [Karras et al. 2022].

    x_t = x_0 + sigma(t) eps,

    where t in [0.002, 80]

    """

    def __init__(
        self,
        min_t: float = 0.002,
        max_t: float = 80.0,
        rho: float = 7.0,
        min_step_percent: float = 0.002,
        max_step_percent: float = 0.998,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        if not 0.002 <= min_t < max_t <= 80.0:
            logger.warning(f"EDM min_t and max_t should be between 0.002 and 80.0, but got {min_t} and {max_t}.")
        ramp = torch.linspace(0, 1, num_steps, dtype=self.t_precision)
        min_inv_rho = min_t ** (1 / rho)
        max_inv_rho = max_t ** (1 / rho)
        sigmas_flipped = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        self._sigmas = torch.flip(sigmas_flipped, [0])
        # Map time values to step indices for polynomial sampling
        self._min_step = int(min_step_percent * num_steps)
        self._max_step = int(max_step_percent * num_steps)

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    @property
    def max_sigma(self) -> float:
        """In EDM, sigma and t are the same, so as _max_t and max_sigma"""
        return self._max_t

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.

        For EDM, sigma(t) = t, so this directly returns the sigma values.
        Note: No roundtrip assertion needed since sigma(t) = t is identity.

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values (which equal sigma values for EDM)
        """
        assert sigma_idx.dtype == torch.long
        return self.sigmas.to(device=sigma_idx.device)[sigma_idx]

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find timestep that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        Args:
            sqrt_snr_t: The square root of the SNR at time t

        Returns:
            torch.Tensor: The timestep

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        t = 1 / self.non_zero_clamp(sqrt_snr_t)
        return t.to(original_dtype)

    def _truncated_lognormal_sample(
        self, n: int, mean: float, std, min_t: float, max_t: float, device: Optional[torch.device] = None
    ):
        # Do all calculations on CPU in t_precision for numerical stability, then move to device
        device = device or torch.device("cpu")

        # Transform bounds to log space (on CPU)
        min_t_val = max(min_t, self.clamp_min)
        log_min_t = torch.tensor(min_t_val, dtype=self.t_precision).log()
        log_max_t = torch.tensor(max_t, dtype=self.t_precision).log()

        # Compute CDF values for truncation bounds
        normal = Normal(
            torch.tensor(mean, dtype=self.t_precision),
            torch.tensor(std, dtype=self.t_precision),
        )
        cdf_min = normal.cdf(log_min_t)
        cdf_max = normal.cdf(log_max_t)

        # Sample uniformly between cdf_min and cdf_max, apply icdf, then exp
        u = torch.rand(n, dtype=self.t_precision) * (cdf_max - cdf_min) + cdf_min
        t = normal.icdf(u).exp()

        # Move to device at the end
        return t.to(device=device)

    def _truncated_log_t_sample(
        self,
        n: int,
        mean: float,
        std,
        min_t: float,
        max_t: float,
        df: float = 2.0,
        device: Optional[torch.device] = None,
    ):
        # transform bounds to log space
        min_t = max(min_t, self.clamp_min)
        log_min_t = np.log(min_t)
        log_max_t = np.log(max_t)

        # Sample from truncated, log-transformed student_t
        lower_adjusted = (log_min_t - mean) / std
        upper_adjusted = (log_max_t - mean) / std

        # Get the CDF values for the adjusted bounds
        a = stats.t.cdf(lower_adjusted, df)
        b = stats.t.cdf(upper_adjusted, df)

        # Sample from the uniform distribution within the adjusted CDF bounds
        u = np.random.uniform(a, b, n)

        # Get the PPF and then adjust it to match the desired mean and scale
        log_t = stats.t.ppf(u, df) * std + mean

        # Exponentiate and move to device
        t = torch.from_numpy(np.exp(log_t)).to(device=device, dtype=self.t_precision)
        return t

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "polynomial",
        train_p_mean: float = -1.2,
        train_p_std: float = 1.2,
        min_t: Optional[float] = 0.002,
        max_t: Optional[float] = 80.0,
        log_t_df: float = 0.01,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample random timesteps based on the EDM noise sampling schedule
        (at inference) for training.

        EDM adopts a parameterized scheme where the time steps are defined
        according to a sequence of noise levels (i.e., sigmas)
        """
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        # Use provided device or default to _sigmas properties
        target_device = device or self._sigmas.device

        if time_dist_type == "lognormal":
            t = self._truncated_lognormal_sample(
                n,
                train_p_mean,
                train_p_std,
                min_t=min_t,
                max_t=max_t,
                device=target_device,
            )
        elif time_dist_type == "log_t":
            t = self._truncated_log_t_sample(
                n,
                train_p_mean,
                train_p_std,
                min_t=min_t,
                max_t=max_t,
                df=log_t_df,
                device=target_device,
            )
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        elif time_dist_type == "polynomial":
            indices = torch.randint(self._min_step, self._max_step + 1, (n,), device=self._sigmas.device)
            t = self._sigmas[indices].to(device=target_device, dtype=self.t_precision)
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in EDMNoiseSchedule."
                f"Currently only supports polynomial, uniform, and lognormal distributions."
            )
        return self.safe_clamp(t, min_t, max_t)

    def get_t_list(self, sample_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get a uniformly spaced list of timesteps from the EDM sigma schedule.

        Timesteps are returned in decreasing order (high noise to low noise) for typical
        diffusion sampling workflows.

        Args:
            sample_steps: Number of timesteps to return
            device: Target device for the output tensor

        Returns:
            torch.Tensor: List of `num_steps+1` timesteps uniformly sampled from sigmas in decreasing order
        """
        device = device or self._sigmas.device

        # Sample num_steps points from sigmas in reverse order (largest to smallest)
        indices = torch.linspace(
            self._max_step,  # Start from max (largest sigma)
            self._min_step,  # End at min (smallest sigma)
            sample_steps + 1,
            device=self._sigmas.device,
        ).long()

        # Get timesteps from sigma indices
        t_list = self._sigmas[indices]

        # Replace the smallest timestep with exact zero for clean final step
        t_list = t_list.clone()  # Make a copy to avoid modifying the original
        t_list[-1] = 0.0

        # Move to target device if needed
        t_list = t_list.to(device=device, dtype=self.t_precision)
        return self.safe_clamp(t_list, max=self.max_t)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        # Perform calculation in double precision
        x0 = xt - v.mul(t_expanded)

        # Convert back to original dtype
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)

        # Perform calculation in double precision
        flow = (xt - x0).div(self.non_zero_clamp(t_expanded))

        # Convert back to original dtype
        return flow.to(original_dtype)


class AlphasNoiseSchedule(BaseNoiseSchedule):
    """
    The SD noise schedule.

    x_t = alpha_cumprod(t).sqrt() x_0 + (1 - alpha_cumprod(t)).sqrt() eps,

    where t is in [0, 1], and alpha_cumprod(t) is from a predefined noise scheduler.

    """

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        min_t: float = 0.0,
        max_t: float = 0.999,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        assert 0 <= min_t < max_t <= 0.999, "Alphas min_t and max_t must be between 0 and 0.999"
        self._alphas_cumprod = alphas_cumprod.to(self.t_precision)
        assert len(self._alphas_cumprod) == num_steps, (
            f"alphas_cumprod's length {len(self._alphas_cumprod)} is not equal to num_steps {num_steps}"
        )

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_steps * t

    @property
    def max_sigma(self) -> float:
        t_max_scale = int(self.num_steps * self.max_t)
        assert 0 <= t_max_scale < len(self._alphas_cumprod)
        return (1 - self._alphas_cumprod[t_max_scale]).sqrt().item()

    @property
    def sigmas(self) -> torch.Tensor:
        return (1 - self._alphas_cumprod).sqrt()

    def _t_to_idx(self, t: torch.Tensor) -> torch.Tensor:
        t_rescale = self._rescale_t(t).to(torch.long)
        # Clamp to prevent floating point precision issues at boundaries
        t_rescale = torch.clamp(t_rescale, 0, len(self._alphas_cumprod) - 1)
        return t_rescale

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        t_rescale = self._t_to_idx(t)
        return self._alphas_cumprod.to(t)[t_rescale].sqrt()

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        t_rescale = self._t_to_idx(t)
        return (1 - self._alphas_cumprod.to(t)[t_rescale]).sqrt()

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.

        This is the inverse of rescale_t: t = sigma_idx / num_steps

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values
        """
        assert sigma_idx.dtype == torch.long
        return sigma_idx.to(self.t_precision) / self.num_steps

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Find timestep that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        Args:
            sqrt_snr_t: The square root of the SNR at time t

        Returns:
            torch.Tensor: The timestep

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        sigma_t = 1 / (1 + sqrt_snr_t**2)
        sigma_idx = self.closest_sigma_idx(sigma_t)
        return self.sigma_idx_to_t(sigma_idx).to(original_dtype)

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "logitnormal",
        train_p_mean: float = 0,
        train_p_std: float = 1.0,
        min_t: Optional[float] = 0.001,
        max_t: Optional[float] = 0.999,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"_expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        # Use provided device or default to _alphas_cumprod properties
        target_device = device if device is not None else self._alphas_cumprod.device

        if time_dist_type == "logitnormal":
            t = (
                torch.sigmoid(torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean)
                * (max_t - min_t)
                + min_t
            )
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in AlphasNoiseSchedule."
                f"Currently only supports logitnormal and uniform."
            )
        return self.safe_clamp(t, min_t, max_t)

    def v_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the v-prediction (x0 = alpha(t) xt - sigma(t) v).
        Only AlphaNoise has v-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The v-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert input tensors to double precision
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)
        t = t.to(torch.float64)

        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Use higher tolerance for bfloat16 due to lower precision
        rtol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3
        atol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3

        # Perform constraint check in double precision
        assert torch.allclose(alpha_t**2 + sigma_t**2, torch.ones_like(alpha_t), rtol=rtol, atol=atol), (
            "v-prediction only supports the case of alpha_t^2+sigma_t^2 = 1."
        )

        # Perform calculation in double precision
        x0 = xt.mul(alpha_t) - v.mul(sigma_t)

        # Convert back to original dtype
        return x0.to(original_dtype)

    def x0_to_v(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the v-prediction from the x0-prediction (v = (alpha(t) xt - x0) / sigma(t)).
        Only AlphaNoise has v-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The v-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert input tensors to double precision
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)
        t = t.to(torch.float64)

        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Use higher tolerance for bfloat16 due to lower precision
        rtol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3
        atol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3

        # Perform constraint check in double precision
        assert torch.allclose(alpha_t**2 + sigma_t**2, torch.ones_like(alpha_t), rtol=rtol, atol=atol), (
            "v-prediction only supports the case of alpha_t^2+sigma_t^2 = 1."
        )

        # Perform calculation in double precision
        v = (xt.mul(alpha_t) - x0).div(self.non_zero_clamp(sigma_t))

        # Convert back to original dtype
        return v.to(original_dtype)


class SDNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, **kwargs):
        scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler",
            cache_dir=get_hf_cache_dir("runwayml/stable-diffusion-v1-5"),
        )
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class SDXLNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, **kwargs):
        scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler",
            cache_dir=get_hf_cache_dir("stabilityai/stable-diffusion-xl-base-1.0"),
        )
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class CogVideoXNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, model_id="THUDM/CogVideoX-5b", **kwargs):
        """Initialize the CogVideoX noise schedule."""

        # Store config for later access (e.g., config["num_train_timesteps"])
        if model_id == "THUDM/CogVideoX-5b":
            self.config = {
                "num_train_timesteps": 1000,
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "trained_betas": None,
                "clip_sample": False,
                "set_alpha_to_one": True,
                "steps_offset": 0,
                "prediction_type": "v_prediction",
                "clip_sample_range": 1.0,
                "sample_max_value": 1.0,
                "timestep_spacing": "trailing",
                "rescale_betas_zero_snr": True,
                "snr_shift_scale": 1.0,
            }
        else:
            self.config = {
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "trained_betas": None,
                "clip_sample": False,
                "clip_sample_range": 1.0,
                "num_train_timesteps": 1000,
                "prediction_type": "v_prediction",
                "rescale_betas_zero_snr": True,
                "sample_max_value": 1.0,
                "set_alpha_to_one": True,
                "snr_shift_scale": 3.0,
                "steps_offset": 0,
                "timestep_spacing": "trailing",
            }
        scheduler = CogVideoXDPMScheduler.from_config(self.config)
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class RFNoiseSchedule(BaseNoiseSchedule):
    """Rectified Flow noise schedule: x_t = (1-t)*x_0 + t*noise.

    Convention: t=0 is data, t=1 is noise.
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = 0.999,
        num_steps: int = 1000,
        shift: float = 1.0,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        time_shift_type: str = "linear",
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        self._supported_time_dist_types = self._supported_time_dist_types + ("shifted",)
        assert 0 <= min_t < max_t <= 1.0, "RF min_t and max_t must be between 0 and 1"
        self._sigmas = torch.linspace(min_t, max_t, num_steps, dtype=self.t_precision)

        # Resolution-dependent shift parameters
        self._shift = float(shift)
        self._base_shift = float(base_shift)
        self._max_shift = float(max_shift)
        self._base_seq_len = int(base_seq_len)
        self._max_seq_len = int(max_seq_len)
        assert time_shift_type in ("linear", "exponential"), (
            f"time_shift_type must be 'linear' or 'exponential', got '{time_shift_type}'"
        )
        self._time_shift_type = time_shift_type

    # ------------------------------------------------------------------
    # Resolution-dependent time shift (Flux-style)
    # ------------------------------------------------------------------

    @property
    def shift(self) -> float:
        """Current shift value (may be updated by ``compute_shift``)."""
        return self._shift

    def compute_shift(self, spatial_shape: tuple[int, ...]) -> float:
        """Compute resolution-dependent shift from spatial dimensions.

        Linearly interpolates ``base_shift``/``max_shift`` by packed sequence length,
        then — when ``time_shift_type == "exponential"`` — exponentiates so that
        ``_shift`` stores the *actual* shift used in the linear warp formula
        ``sigma = shift * t / (1 + (shift - 1) * t)``.

        Equivalently, interpolate the logarithms of the endpoint shifts,
        then compute ``shift = exp(logshift)``.

        Args:
            spatial_shape: latent spatial dims, e.g. (128, 128) or (96, 170).
                Computes packed seq_len = (H//2)*(W//2) for Flux 2×2 packing.

        Returns:
            The computed shift value (already exponentiated if exponential mode).
        """
        import math

        h, w = spatial_shape[-2], spatial_shape[-1]
        image_seq_len = (h // 2) * (w // 2)
        m = (self._max_shift - self._base_shift) / (self._max_seq_len - self._base_seq_len)
        b = self._base_shift - m * self._base_seq_len
        mu = max(image_seq_len * m + b, 0.01)
        if self._time_shift_type == "exponential":
            self._shift = math.exp(mu)
        else:
            self._shift = mu
        return self._shift

    def warp_t(self, raw_t: torch.Tensor) -> torch.Tensor:
        """Map raw_t ∈ [0, 1] → sigma via current shift.

        Always uses the linear formula: ``sigma = shift * t / (1 + (shift - 1) * t)``.
        When ``time_shift_type == "exponential"``, the exponentiation is already
        folded into ``_shift`` by ``compute_shift`` (i.e. ``_shift = exp(mu)``).
        """
        raw_t = torch.clamp(raw_t, min=0.0, max=1.0)
        sigma = self._shift * raw_t / (1.0 + (self._shift - 1.0) * raw_t)
        return torch.clamp(sigma, min=0.0, max=1.0)

    def unwarp_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Inverse of ``warp_t``: sigma → raw_t via current shift.

        ``t = sigma / (shift + (1 - shift) * sigma)``
        """
        sigma = torch.clamp(sigma, min=0.0, max=1.0)
        raw_t = sigma / (self._shift + (1.0 - self._shift) * sigma)
        return torch.clamp(raw_t, min=0.0, max=1.0)

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_steps * t

    @property
    def max_sigma(self) -> float:
        t_max_scale = min(int(self.num_steps * self.max_t), len(self._sigmas) - 1)
        assert 0 <= t_max_scale < len(self._sigmas)
        return self._sigmas[t_max_scale].item()

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.

        For Rectified Flow, sigma(t) = t, so this maps indices back to timesteps.

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values
        """
        assert sigma_idx.dtype == torch.long
        return sigma_idx.to(self.t_precision) / self.num_steps

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Find timestep that matches the target sqrt(SNR) = alpha/sigma = (1-t)/t.

        Solving: sqrt_snr = (1-t)/t
        => sqrt_snr * t = 1 - t
        => t * (sqrt_snr + 1) = 1
        => t = 1 / (sqrt_snr + 1)

        Args:
            sqrt_snr_t: The square root of the SNR at time `t`, i.e., alpha_t / sigma_t.

        Returns:
            torch.Tensor: The timestep
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        t = 1 / (sqrt_snr_t + 1)
        return t.to(original_dtype)

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "logitnormal",
        train_p_mean: float = 0,
        train_p_std: float = 1.0,
        min_t: Optional[float] = 0.001,
        max_t: Optional[float] = 0.999,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"_expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        # Use provided device or default to _sigmas properties
        target_device = device or self._sigmas.device

        if time_dist_type == "logitnormal":
            t = (
                torch.sigmoid(torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean)
                * (max_t - min_t)
                + min_t
            )
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        elif time_dist_type == "shifted":
            shift = kwargs.get("shift", 5.0)
            assert shift >= 1, f"shift must be >= 1, got {shift}"
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
            t = t * shift / (t * (shift - 1) + 1)
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in RFNoiseSchedule."
                f"Currently only supports logitnormal, uniform, and shifted."
            )
        return self.safe_clamp(t, min_t, max_t)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        # Perform calculation in double precision
        x0 = xt - v.mul(t_expanded)

        # Convert back to original dtype
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)

        # Perform calculation in double precision
        flow = (xt - x0).div(self.non_zero_clamp(t_expanded))

        # Convert back to original dtype
        return flow.to(original_dtype)


class TrellisFlowMatchSchedule(BaseNoiseSchedule):
    """TRELLIS.2 flow matching schedule with ``σ_min`` offset.

    Forward process::

        x_t = (1 - t) * x_0 + (σ_min + (1 - σ_min) * t) * ε

    Velocity (constant w.r.t. *t*)::

        v = dx_t/dt = (1 - σ_min) * ε − x_0

    Convention: *t* = 0 is data, *t* = 1 is (approximately) noise.

    Differences from :class:`RFNoiseSchedule` (``σ_min = 0``):

    * The noise coefficient is ``σ_min + (1 - σ_min) * t`` instead of ``t``.
    * ``alpha(0) = 1``, ``sigma(0) = σ_min`` (not exactly 0).
    * ``flow_to_x0`` and ``x0_to_flow`` account for the offset.

    Official TRELLIS.2 applies its Möbius ``rescale_t`` warp to the Euler
    sampler time sequence.  SMD mirrors that by exposing ``warp_t()`` on
    this schedule and feeding the already-warped value to the backbone.
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = 1.0,
        num_steps: int = 1000,
        sigma_min: float = 1e-5,
        shift: float = 1.0,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        self._sigma_min = float(sigma_min)
        self._sigmas = torch.linspace(min_t, max_t, num_steps, dtype=self.t_precision)
        self._shift = float(shift)

    # ------------------------------------------------------------------
    # Time warp (Möbius reparametrization)
    # ------------------------------------------------------------------

    @property
    def shift(self) -> float:
        return self._shift

    def warp_t(self, raw_t: torch.Tensor) -> torch.Tensor:
        """``sigma = shift * t / (1 + (shift - 1) * t)``"""
        raw_t = torch.clamp(raw_t, min=0.0, max=1.0)
        sigma = self._shift * raw_t / (1.0 + (self._shift - 1.0) * raw_t)
        return torch.clamp(sigma, min=0.0, max=1.0)

    def unwarp_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Inverse of ``warp_t``."""
        sigma = torch.clamp(sigma, min=0.0, max=1.0)
        raw_t = sigma / (self._shift + (1.0 - self._shift) * sigma)
        return torch.clamp(raw_t, min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Core schedule functions
    # ------------------------------------------------------------------

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        """``α(t) = 1 − t``"""
        return 1.0 - t

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        """``σ(t) = σ_min + (1 − σ_min) * t``"""
        return self._sigma_min + (1.0 - self._sigma_min) * t

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self._sigma_min) * torch.ones_like(t)

    # ------------------------------------------------------------------
    # Sigma array / rescale
    # ------------------------------------------------------------------

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_steps * t

    @property
    def max_sigma(self) -> float:
        t_max_scale = int(self.num_steps * self.max_t)
        t_max_scale = min(t_max_scale, len(self._sigmas) - 1)
        return self._sigmas[t_max_scale].item()

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        assert sigma_idx.dtype == torch.long
        return sigma_idx.to(self.t_precision) / self.num_steps

    def sqrt_snr_to_t(self, sqrt_snr: torch.Tensor) -> torch.Tensor:
        """``sqrt_snr = (1-t) / (σ_min + (1-σ_min)*t)`` → solve for ``t``."""
        s = self._sigma_min
        # t = (1 - s * sqrt_snr) / (sqrt_snr * (1 - s) + 1)
        t = (1.0 - s * sqrt_snr) / (sqrt_snr * (1.0 - s) + 1.0)
        return t.to(self.t_precision)

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def sample_t(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        time_dist_type: str = "uniform",
        **kwargs,
    ) -> torch.Tensor:
        """Sample random timesteps (default: uniform on [min_t, max_t])."""
        if time_dist_type == "uniform":
            t = torch.rand(batch_size, device=device, dtype=self.t_precision)
            t = t * (self.max_t - self.min_t) + self.min_t
        elif time_dist_type == "logitnormal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            normal = torch.randn(batch_size, device=device, dtype=self.t_precision)
            t = torch.sigmoid(normal * std + mean)
            t = t * (self.max_t - self.min_t) + self.min_t
        else:
            raise ValueError(f"Unsupported time_dist_type={time_dist_type!r} for TrellisFlowMatchSchedule")
        return t

    # ------------------------------------------------------------------
    # Flow ↔ x0 conversions (accounts for σ_min offset)
    # ------------------------------------------------------------------

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Recover ``x_0`` from flow velocity.

        From ``x_t = (1-t)*x_0 + (σ_min + (1-σ_min)*t) * ε`` and
        ``v = (1-σ_min)*ε − x_0``, we get:

        ``x_0 = ((1-σ_min)*x_t − (σ_min + (1-σ_min)*t) * v) / (1 − σ_min * t)``
        """
        original_dtype = xt.dtype
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)
        t_exp = expand_like(t.to(torch.float64), xt)
        s = self._sigma_min
        numer = (1.0 - s) * xt - (s + (1.0 - s) * t_exp) * v
        denom = 1.0 - s * t_exp
        # Clamp denom away from zero
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom) * 1e-8, denom)
        x0 = numer / denom
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Compute flow velocity from ``x_0``.

        ``v = (1-σ_min)*ε − x_0`` where ``ε = (x_t − (1-t)*x_0) / (σ_min + (1-σ_min)*t)``
        """
        original_dtype = xt.dtype
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)
        t_exp = expand_like(t.to(torch.float64), xt)
        s = self._sigma_min
        sigma_t = s + (1.0 - s) * t_exp
        sigma_t = torch.where(sigma_t.abs() < 1e-8, torch.ones_like(sigma_t) * 1e-8, sigma_t)
        eps = (xt - (1.0 - t_exp) * x0) / sigma_t
        flow = (1.0 - s) * eps - x0
        return flow.to(original_dtype)


class TrigNoiseSchedule(BaseNoiseSchedule):
    """
    The Trigonometric Flow noise schedule, typically used in Geometric/Consistency flows.
    x_t = cos(t) * x_0 + sin(t) * eps

    where t is in [0, pi/2].
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = torch.pi / 2,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        assert 0 <= min_t < max_t, "Trig min_t must be non-negative and less than max_t"
        self._sigmas = torch.sin(torch.linspace(min_t, max_t, num_steps, dtype=self.t_precision))

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        # In this parameterization, t acts directly as the angle.
        return t

    @property
    def max_sigma(self) -> float:
        return torch.sin(torch.tensor(self.max_t)).item()

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t)

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.sin(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.
        Since we linspace t to create sigmas, we can map indices linearly back to t.

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values
        """
        assert sigma_idx.dtype == torch.long
        t = sigma_idx.to(self.t_precision) / (self.num_steps - 1) * (self.max_t - self.min_t) + self.min_t
        return t

    def sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the square root of the Signal-to-Noise Ratio (SNR) at time `t`.
        sqrt_snr(t) = cos(t) / sin(t) = cot(t)
        """
        assert self.is_t_valid(t)
        t = t.to(torch.float64)
        # We clamp tan(t) to avoid dividing by zero at t=0
        tan_t = self.non_zero_clamp(torch.tan(t))
        return 1.0 / tan_t

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find timestep that matches the target square root of SNR value.
        sqrt_snr = alpha/sigma = cos(t)/sin(t) = cot(t).
        t = arccot(SNR) = arctan(1/SNR).

        Args:
            sqrt_snr_t: The square root of the SNR at time `t`.

        Returns:
            torch.Tensor: The timestep
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        # Use atan2(1, sqrt_snr_t) which is a robust equivalent of arccot(sqrt_snr_t)
        t = torch.atan2(torch.ones_like(sqrt_snr_t), sqrt_snr_t)
        return t.to(original_dtype)

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "uniform",
        train_p_mean: float = 0,
        train_p_std: float = 1.0,
        min_t: Optional[float] = 0.0,
        max_t: Optional[float] = torch.pi / 2,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"_expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        target_device = device or self._sigmas.device

        if time_dist_type == "logitnormal":
            # Map sigmoid output [0, 1] to [min_t, max_t]
            norm_t = torch.sigmoid(
                torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean
            )
            t = norm_t * (max_t - min_t) + min_t
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in TrigNoiseSchedule."
                f"Currently only supports logitnormal and uniform."
            )
        return self.safe_clamp(t, min_t, max_t)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute x0 from flow prediction v.
        v_t = -sin(t)x0 + cos(t)eps
        x_t = cos(t)x0 + sin(t)eps

        Algebraic rearrangement yields:
        x0 = cos(t)x_t - sin(t)v
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        original_dtype = xt.dtype

        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        cos_t = expand_like(torch.cos(t), xt)
        sin_t = expand_like(torch.sin(t), xt)

        x0 = xt * cos_t - v * sin_t
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute flow prediction v from x0.
        v = (cos(t)x_t - x_0) / sin(t)
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        original_dtype = xt.dtype

        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)

        cos_t = expand_like(torch.cos(t), xt)
        sin_t = expand_like(torch.sin(t), xt)

        flow = (xt * cos_t - x0) / self.non_zero_clamp(sin_t)
        return flow.to(original_dtype)


# Global registry of noise schedules for easy access by string name
NOISE_SCHEDULES = {
    "edm": EDMNoiseSchedule,
    "alphas": AlphasNoiseSchedule,
    "sd": SDNoiseSchedule,
    "sdxl": SDXLNoiseSchedule,
    "cogvideox": CogVideoXNoiseSchedule,
    "rf": RFNoiseSchedule,
    "rectified_flow": RFNoiseSchedule,  # Alias for RF
    "trig": TrigNoiseSchedule,
    "trellis_flow": TrellisFlowMatchSchedule,
}


def get_noise_schedule(name: str, **kwargs):
    """
    Get a noise schedule class by name.

    Args:
        name (str): Name of the noise schedule. Available options:
            - "edm": EDMNoiseSchedule (Karras et al. 2022)
            - "alphas": AlphasNoiseSchedule (base class)
            - "sd": SDNoiseSchedule (Stable Diffusion v1.5)
            - "sdxl": SDXLNoiseSchedule (Stable Diffusion XL)
            - "cogvideox": CogVideoXNoiseSchedule (CogVideoX)
            - "rf" or "rectified_flow": RFNoiseSchedule (Rectified Flow)
        **kwargs: Arguments to pass to the noise schedule constructor.

    Returns:
        BaseNoiseSchedule: Instantiated noise schedule.

    Raises:
        KeyError: If the noise schedule name is not found.

    Example:
        >>> schedule = get_noise_schedule("edm", min_t=0.002, max_t=80.0)
        >>> schedule = get_noise_schedule("sd")
    """
    if name not in NOISE_SCHEDULES:
        available = ", ".join(sorted(NOISE_SCHEDULES.keys()))
        raise KeyError(f"Unknown noise schedule '{name}'. Available schedules: {available}")

    schedule_class = NOISE_SCHEDULES[name]
    return schedule_class(**kwargs)


def list_noise_schedules():
    """
    List all available noise schedule names.

    Returns:
        list: List of available noise schedule names.
    """
    return sorted(NOISE_SCHEDULES.keys())


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Sanity: instantiate each no-download schedule, test forward_process / alpha / sigma.
    # ------------------------------------------------------------------
    import torch

    print("=== Noise schedule sanity ===")

    # Schedules that can be instantiated without HF downloads
    test_cases = {
        "edm": {"min_t": 0.002, "max_t": 80.0},
        "rf": {"min_t": 0.0, "max_t": 0.999},
        "trig": {"min_t": 0.0, "max_t": torch.pi / 2},
    }

    for name, kwargs in test_cases.items():
        print(f"  testing {name}...")
        sched = get_noise_schedule(name, **kwargs)

        B = 4
        # Sample t in valid range
        t = torch.rand(B, dtype=torch.float64) * (sched.max_t - sched.min_t) + sched.min_t
        x0 = torch.randn(B, 3, 8, 8, dtype=torch.float64)
        eps = torch.randn_like(x0)

        # alpha / sigma
        a = sched.alpha(t)
        s = sched.sigma(t)
        assert torch.isfinite(a).all(), f"{name}: non-finite alpha"
        assert torch.isfinite(s).all(), f"{name}: non-finite sigma"

        # forward_process
        x_t = sched.forward_process(x0, eps, t)
        assert x_t.shape == x0.shape, f"{name}: shape mismatch"
        assert torch.isfinite(x_t).all(), f"{name}: non-finite x_t"

        print(f"    alpha range=[{a.min():.4f}, {a.max():.4f}], sigma range=[{s.min():.4f}, {s.max():.4f}]")

    print("=== Noise schedule sanity PASSED ===")

    # ------------------------------------------------------------------
    # Sanity: SegmentSchedule API (scheduler-owned segment schedule)
    #   fixed shift, dynamic shift, empirical shift, TRELLIS endpoints,
    #   final_step_size_scale, and validation of invalid inputs.
    # ------------------------------------------------------------------
    print("=== SegmentSchedule sanity ===")

    def _assert_segments(segs, num_steps, name):
        assert len(segs) == num_steps, f"{name}: expected {num_steps} segments, got {len(segs)}"
        assert segs[0].index == 0 and segs[-1].index == num_steps - 1, f"{name}: bad indices"
        assert segs[0].raw_start == 1.0, f"{name}: raw grid must start at 1.0"
        assert segs[-1].raw_end == 0.0, f"{name}: raw grid must end at 0.0"
        for seg in segs:
            assert seg.raw_start > seg.raw_end, f"{name}: raw knots not decreasing at {seg.index}"
            assert float(seg.sigma_start) >= float(seg.sigma_end), f"{name}: sigma not decreasing at {seg.index}"
            assert abs(seg.size - (seg.raw_start - seg.raw_end)) < 1e-12, f"{name}: size mismatch at {seg.index}"
        for a_seg, b_seg in zip(segs[:-1], segs[1:]):
            assert float(a_seg.sigma_end) == float(b_seg.sigma_start), f"{name}: non-contiguous sigma endpoints"

    # --- fixed shift ---
    rf_fixed = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    rf_fixed.configure_time_shift(dynamic=False, fixed_shift=3.0)
    segs = rf_fixed.build_segment_schedule(input_shape=[16, 32, 32], num_steps=4, final_step_size_scale=1.0)
    _assert_segments(segs, 4, "fixed")
    # raw knots must match the standalone knot builder exactly
    raw_knots = BaseNoiseSchedule.compute_raw_grid_knots(4, 1.0)
    got_knots = [segs[0].raw_start] + [s.raw_end for s in segs]
    assert got_knots == raw_knots, f"fixed: raw knots {got_knots} != {raw_knots}"
    assert abs(rf_fixed.shift - 3.0) < 1e-12, "fixed: shift must be pinned to 3.0"
    print(f"  fixed: shift={rf_fixed.shift:.4f}, sigma0={float(segs[0].sigma_start):.4f}")

    # --- dynamic shift (resolution-dependent, linear) ---
    rf_dyn = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    rf_dyn.configure_time_shift(
        dynamic=True,
        base_shift=0.5,
        max_shift=1.15,
        base_seq_len=256,
        max_seq_len=4096,
        time_shift_type="linear",
    )
    segs_a = rf_dyn.build_segment_schedule(input_shape=[16, 64, 64], num_steps=4, final_step_size_scale=1.0)
    shift_a = rf_dyn.shift
    _assert_segments(segs_a, 4, "dynamic")
    # different resolution → different shift; shift cache is keyed per shape
    segs_b = rf_dyn.build_segment_schedule(input_shape=[16, 32, 32], num_steps=4, final_step_size_scale=1.0)
    shift_b = rf_dyn.shift
    assert shift_a != shift_b, "dynamic: distinct resolutions must yield distinct shifts"
    # re-resolving the first shape must reproduce the cached shift bit-exactly
    rf_dyn.build_segment_schedule(input_shape=[16, 64, 64], num_steps=4, final_step_size_scale=1.0)
    assert rf_dyn.shift == shift_a, "dynamic: cached shift must be reproducible"
    print(f"  dynamic: shift(64x64)={shift_a:.4f}, shift(32x32)={shift_b:.4f}")

    # --- empirical shift branch (Flux2/Lens-style step-dependent mu) ---
    class _EmpiricalRF(RFNoiseSchedule):
        def __init__(self, empirical_steps: int = 0, **kw):
            super().__init__(**kw)
            self._empirical_steps = int(empirical_steps)

        def compute_shift(self, spatial_shape):
            if self._empirical_steps > 0:
                # step-dependent: fewer steps → larger shift (monotone stand-in
                # for the real BFL empirical-mu formula, which the LOCAL golden
                # parity harness validates bit-exactly against Flux2/Lens).
                self._shift = 1.0 + 100.0 / float(self._empirical_steps)
                return self._shift
            return super().compute_shift(spatial_shape)

    emp = _EmpiricalRF(min_t=0.0, max_t=0.999)
    emp.configure_time_shift(
        dynamic=True,
        base_shift=0.5,
        max_shift=1.15,
        base_seq_len=256,
        max_seq_len=4096,
        time_shift_type="linear",
        empirical_steps=50,
    )
    segs_emp = emp.build_segment_schedule(input_shape=[16, 64, 64], num_steps=4, final_step_size_scale=1.0)
    _assert_segments(segs_emp, 4, "empirical")
    assert abs(emp.shift - (1.0 + 100.0 / 50.0)) < 1e-12, "empirical: shift must use empirical branch"
    # a schedule without empirical support must reject empirical_steps
    try:
        get_noise_schedule("rf", min_t=0.0, max_t=0.999).configure_time_shift(
            dynamic=True,
            base_shift=0.5,
            max_shift=1.15,
            base_seq_len=256,
            max_seq_len=4096,
            time_shift_type="linear",
            empirical_steps=50,
        )
        raise AssertionError("empirical: expected TypeError for schedule without empirical support")
    except TypeError:
        pass
    print(f"  empirical: shift(steps=50)={emp.shift:.4f}")

    # --- TRELLIS official endpoints (uniform raw grid, shift=3 warp) ---
    trellis = TrellisFlowMatchSchedule(min_t=0.0, max_t=1.0, num_steps=1000, sigma_min=1e-5, shift=3.0)
    trellis.configure_time_shift(dynamic=False, fixed_shift=3.0)
    segs_tr = trellis.build_segment_schedule(input_shape=[8, 16, 16], num_steps=4, final_step_size_scale=1.0)
    _assert_segments(segs_tr, 4, "trellis")
    tr_sigmas = [float(segs_tr[0].sigma_start)] + [float(s.sigma_end) for s in segs_tr]
    expected_tr = [1.0, 0.9, 0.75, 0.5, 0.0]
    assert all(abs(g - e) < 1e-7 for g, e in zip(tr_sigmas, expected_tr)), (
        f"trellis: sigma endpoints {tr_sigmas} != {expected_tr}"
    )
    print(f"  trellis: sigma endpoints={[round(x, 4) for x in tr_sigmas]}")

    # --- final_step_size_scale: last segment shrinks vs uniform ---
    rf_fss = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    rf_fss.configure_time_shift(dynamic=False, fixed_shift=1.0)
    segs_u = rf_fss.build_segment_schedule(input_shape=[16, 32, 32], num_steps=4, final_step_size_scale=1.0)
    segs_s = rf_fss.build_segment_schedule(input_shape=[16, 32, 32], num_steps=4, final_step_size_scale=0.5)
    assert abs(segs_u[-1].size - segs_u[0].size) < 1e-12, "fss=1.0 must be uniform"
    assert segs_s[-1].size < segs_s[0].size, "fss<1 must shrink the final segment"
    assert abs(segs_s[-1].size / segs_s[0].size - 0.5) < 1e-9, "fss=0.5 ratio must be 0.5"
    print(f"  fss: uniform last={segs_u[-1].size:.4f}, scaled last={segs_s[-1].size:.4f}")

    # --- validation: invalid inputs raise explicit exceptions ---
    rf_bad = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    try:
        rf_bad.build_segment_schedule(input_shape=[16, 32, 32], num_steps=4, final_step_size_scale=1.0)
        raise AssertionError("expected RuntimeError before configure_time_shift")
    except RuntimeError:
        pass
    rf_bad.configure_time_shift(dynamic=False, fixed_shift=1.0)
    for bad_steps in (0, -1):
        try:
            rf_bad.build_segment_schedule(input_shape=[16, 32, 32], num_steps=bad_steps, final_step_size_scale=1.0)
            raise AssertionError(f"expected ValueError for num_steps={bad_steps}")
        except ValueError:
            pass
    for bad_fss in (0.0, -0.5, float("inf"), float("nan")):
        try:
            rf_bad.build_segment_schedule(input_shape=[16, 32, 32], num_steps=4, final_step_size_scale=bad_fss)
            raise AssertionError(f"expected ValueError for final_step_size_scale={bad_fss}")
        except ValueError:
            pass
    try:
        rf_bad.configure_time_shift(dynamic=False)  # missing fixed_shift
        raise AssertionError("expected ValueError for missing fixed_shift")
    except ValueError:
        pass
    print("  validation: invalid num_steps / final_step_size_scale / missing shift all rejected")

    print("=== SegmentSchedule sanity PASSED ===")
