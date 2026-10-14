# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lens-specific noise schedule with empirical mu formula.

Lens uses the same BFL ``compute_empirical_mu`` approach as Flux2 Klein, with
*identical* coefficients — both were fitted by BFL for the Flux2 semantic VAE
latent space that Lens shares.

Reference: ``lens/pipeline.py :: compute_empirical_mu``
"""

from __future__ import annotations

import math

from smd.networks.noise_schedule import RFNoiseSchedule


class LensNoiseSchedule(RFNoiseSchedule):
    """RFNoiseSchedule with Lens / Flux2-style empirical mu formula.

    ``empirical_steps`` should be set to the **teacher's default inference
    steps** (50 for Lens-Base, 20 for Lens-RL). This ensures the training
    schedule matches the sigma distribution the teacher was optimized for.

    When ``empirical_steps == 0`` (default), behaviour is identical to
    :class:`RFNoiseSchedule` — full backward compatibility.
    """

    def __init__(self, empirical_steps: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._empirical_steps = int(empirical_steps)

    def compute_shift(self, spatial_shape: tuple[int, ...]) -> float:
        """Compute shift using Lens/Flux2 empirical mu formula.

        For Lens the latents are already in patchified space [B, 128, H, W],
        so ``image_seq_len = H * W`` (not ``(H//2) * (W//2)``).
        """
        if self._empirical_steps > 0:
            h, w = spatial_shape[-2], spatial_shape[-1]
            image_seq_len = h * w
            mu = compute_empirical_mu(image_seq_len, self._empirical_steps)
            if self._time_shift_type == "exponential":
                self._shift = math.exp(mu)
            else:
                self._shift = mu
            return self._shift

        return super().compute_shift(spatial_shape)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical mu for FlowMatchEulerDiscreteScheduler dynamic shift.

    Constants are calibrated identically to Flux2 Klein / Lens — both share
    the same Flux2 semantic VAE latent space.

    Reference: ``lens/pipeline.py::compute_empirical_mu``
    """
    a1, b1 = 8.73809524e-05, 1.89833333  # 10-step anchor
    a2, b2 = 0.00016927, 0.45666666  # 200-step anchor

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)
