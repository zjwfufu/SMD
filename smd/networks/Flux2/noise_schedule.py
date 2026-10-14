# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux2-specific noise schedule with BFL's empirical mu formula.

Flux2 Klein base models use a resolution-AND-step-dependent time shift
that differs from Flux1's purely resolution-dependent shift. BFL fitted
two anchor lines (10-step and 200-step) relating packed sequence length
to the optimal mu parameter, then linearly interpolates between them
for any given ``num_steps``.

The ``scheduler_config.json`` values (``base_shift=0.5, max_shift=1.15``)
correspond exactly to the 200-step anchor line and are NOT the final shift
used at inference — the official Flux2 pipeline always calls
``compute_empirical_mu(seq_len, num_steps)`` to get the actual mu.

Reference: ``https://github.com/black-forest-labs/flux2``
    ``src/flux2/sampling.py :: compute_empirical_mu, get_schedule``
"""

from __future__ import annotations

import math

from smd.networks.noise_schedule import RFNoiseSchedule


class Flux2NoiseSchedule(RFNoiseSchedule):
    """RFNoiseSchedule with Flux2's empirical mu formula.

    When ``empirical_steps > 0``, :meth:`compute_shift` uses BFL's
    ``compute_empirical_mu(seq_len, empirical_steps)`` instead of the
    Flux1-style linear interpolation of ``base_shift / max_shift``.

    ``empirical_steps`` should be set to the **teacher's default inference
    steps** (50 for Klein base models). This ensures the training schedule
    matches the sigma distribution the teacher was optimized for.

    When ``empirical_steps == 0`` (default), behaviour is identical to
    :class:`RFNoiseSchedule` — full backward compatibility with Flux1.
    """

    def __init__(self, empirical_steps: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._empirical_steps = int(empirical_steps)

    # ------------------------------------------------------------------
    # Override: compute_shift with empirical mu
    # ------------------------------------------------------------------

    def compute_shift(self, spatial_shape: tuple[int, ...]) -> float:
        """Compute shift, optionally using Flux2's empirical mu formula.

        If ``empirical_steps > 0``, uses :func:`compute_empirical_mu` which
        bilinearly interpolates over ``(seq_len, num_steps)`` — fewer steps
        → higher mu → schedule biased toward high noise.

        Otherwise falls back to the parent's Flux1-style linear interpolation.
        """
        if self._empirical_steps > 0:
            h, w = spatial_shape[-2], spatial_shape[-1]
            image_seq_len = (h // 2) * (w // 2)
            mu = compute_empirical_mu(image_seq_len, self._empirical_steps)
            if self._time_shift_type == "exponential":
                self._shift = math.exp(mu)
            else:
                self._shift = mu
            return self._shift

        # Flux1-style fallback
        return super().compute_shift(spatial_shape)


# ---------------------------------------------------------------------------
# Standalone function — usable without instantiating a schedule object
# ---------------------------------------------------------------------------


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Flux2 empirical mu formula from BFL's official repository.

    Bilinear interpolation over ``(image_seq_len, num_steps)`` between two
    experimentally fitted anchor lines:

    - **10-step anchor:**  ``mu_10(seq)  = a1 * seq + b1``
    - **200-step anchor:** ``mu_200(seq) = a2 * seq + b2``

    For ``image_seq_len > 4300`` (very high resolution), only the 200-step
    anchor is used.

    The warp formula is ``sigma = exp(mu) * t / (1 + (exp(mu) - 1) * t)``
    (equivalently ``generalized_time_snr_shift(t, mu, sigma=1)``).

    Reference: ``flux2_official/src/flux2/sampling.py::compute_empirical_mu``

    Args:
        image_seq_len: Packed sequence length, e.g. 4096 for 1024×1024.
        num_steps: Number of inference steps (teacher's default, e.g. 50).

    Returns:
        mu value (log-space shift parameter).
    """
    # Coefficients fitted by BFL
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
