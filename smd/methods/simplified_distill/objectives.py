# SPDX-License-Identifier: Apache-2.0
"""Independent tensor objectives; no teacher calls or execution state."""

from dataclasses import dataclass
import torch
from .utils import per_sample_velocity_distance


def velocity_distance(left, right, beta=1.0, eps=1e-6):
    return per_sample_velocity_distance(left, right, left.shape[0], beta, eps)


@dataclass(frozen=True)
class ImitationLoss:
    beta: float = 1.0
    eps: float = 1e-6

    def __call__(self, student, target, *, distance=velocity_distance):
        return distance(student, target, self.beta, self.eps)


@dataclass(frozen=True)
class TeacherRepulsionLoss:
    beta: float = 1.0
    eps: float = 1e-6
    adaptive: bool = False

    def __call__(self, student, reference, *, anchor=None, distance=velocity_distance):
        value = distance(student, reference, self.beta, self.eps)
        if self.adaptive:
            if anchor is None:
                raise ValueError("Adaptive teacher repulsion requires an anchor teacher velocity.")
            positive = distance(student, anchor, 1.0, 0.0)
            negative = distance(student, reference, 1.0, 0.0)
            gap = distance(anchor, reference, 1.0, 0.0)
            gate = ((negative - positive) / (gap + self.eps)).clamp(0, 1).detach()
        else:
            gate = torch.ones_like(value)
        return value * gate, gate


@dataclass(frozen=True)
class LossWeights:
    imitation: float
    teacher_repulsion: float
    normalization: float

    def __call__(self, imitation, teacher_repulsion):
        return self.normalization * (
            self.imitation * imitation
            - self.teacher_repulsion * teacher_repulsion
        )
