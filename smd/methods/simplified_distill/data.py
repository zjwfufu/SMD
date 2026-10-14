# SPDX-License-Identifier: Apache-2.0
"""Explicit inputs and results of a student segment; no execution ownership."""

from dataclasses import dataclass
from typing import Any
import torch
from smd.networks.noise_schedule import SegmentSchedule


@dataclass(kw_only=True)
class TrainingBatch:
    x0: torch.Tensor | None
    condition: Any
    neg_condition: Any
    batch_size: int
    input_shape: list[int]
    segments: list[SegmentSchedule]
    iteration: int = 0
    is_unconditional_step: bool = False
    teacher_condition: Any = None
    teacher_neg_condition: Any = None
    route: str | None = None


@dataclass
class SegmentContext:
    base_velocity: torch.Tensor
    pivot_coeffs: dict[str, torch.Tensor]
    logvar: torch.Tensor | None = None


@dataclass
class SegmentState:
    schedule: SegmentSchedule
    x: torch.Tensor
    sigma_start: torch.Tensor
    sigma_end: torch.Tensor
    delta: torch.Tensor
    queries: list[tuple[torch.Tensor, torch.Tensor]]
    guided_context: SegmentContext
    unconditional_context: SegmentContext | None
    query_alpha: torch.Tensor | None = None
