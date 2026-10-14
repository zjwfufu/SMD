# smd/methods/simplified_distill/basis/spectral_zero_mean.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spectral zero-mean residual basis (Category B).

Parameterizes the velocity residual directly:
    r(alpha) = sum_{k=1}^{K} c_k * psi_k(alpha)

where each psi_k has zero integral over [0, 1].

Supported families:
  - cosine: psi_k(alpha) = cos(k * pi * alpha)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import ResidualBasis


def _residual_scale_factor(num_terms: int, mode: str, scale: float) -> float:
    if num_terms <= 0:
        return 1.0
    if mode == "sqrt_n":
        denom = max(num_terms**0.5, 1e-12)
    elif mode == "fixed":
        denom = 1.0
    elif mode == "none":
        denom = 1.0
        scale = 1.0
    else:
        raise ValueError(f"Unknown scale_mode: {mode!r}")
    return scale / denom


class SpectralZeroMeanResidual(ResidualBasis):
    """Spectral zero-mean velocity residual.

    Args:
        num_terms: Number of basis functions K (= n_grid).
        family: "cosine" (others: future work).
        scale_mode: "sqrt_n", "fixed", or "none".
        residual_scale: Multiplicative scale.
    """

    def __init__(
        self,
        num_terms: int = 4,
        family: str = "cosine",
        scale_mode: str = "sqrt_n",
        residual_scale: float = 1.0,
    ):
        self._num_terms = num_terms
        self._family = family
        self._scale = _residual_scale_factor(num_terms, scale_mode, residual_scale)

    @property
    def n_grid(self) -> int:
        return self._num_terms

    def build_residual_coeffs(
        self,
        flow: Tensor,
        x_i: Tensor,
        sigma_i: Tensor,
        seg_size: Tensor,
    ) -> dict[str, Tensor]:
        return {"c_k": flow * self._scale}

    def evaluate_r(self, coeffs: dict[str, Tensor], alpha: Tensor) -> Tensor:
        """r(alpha) = sum c_k * psi_k(alpha)."""
        c_k = coeffs["c_k"]  # [B, K, C, H, W]
        result = torch.zeros_like(c_k[:, 0])
        if self._family == "cosine":
            for k_idx in range(self._num_terms):
                k = k_idx + 1
                result = result + c_k[:, k_idx] * torch.cos(k * math.pi * alpha)
        else:
            raise ValueError(f"Unknown family: {self._family!r}")
        return result

    def evaluate_q(self, coeffs: dict[str, Tensor], alpha: Tensor, dt: Tensor) -> Tensor:
        """q(alpha) = dt * integral_0^alpha r(s) ds.

        For cosine: integral_0^alpha cos(k*pi*s) ds = sin(k*pi*alpha) / (k*pi).
        """
        c_k = coeffs["c_k"]
        result = torch.zeros_like(c_k[:, 0])
        if self._family == "cosine":
            for k_idx in range(self._num_terms):
                k = k_idx + 1
                result = result + c_k[:, k_idx] * torch.sin(k * math.pi * alpha) / (k * math.pi)
        else:
            raise ValueError(f"Unknown family: {self._family!r}")
        return dt * result

    def windowed_velocity(
        self,
        v_base: Tensor,
        coeffs: dict[str, Tensor],
        alpha_start: Tensor,
        alpha_end: Tensor,
        eps: float = 1e-6,
    ) -> Tensor:
        """Analytic windowed average.

        For cosine: (1/d) * integral_{a}^{a+d} cos(k*pi*s) ds
                  = [sin(k*pi*(a+d)) - sin(k*pi*a)] / (k*pi*d)
        """
        d_alpha = (alpha_end - alpha_start).clamp(min=0)
        small_mask = d_alpha.squeeze(-1).squeeze(-1).squeeze(-1) < eps

        if small_mask.all():
            return v_base + self.evaluate_r(coeffs, alpha_start)

        c_k = coeffs["c_k"]
        r_avg = torch.zeros_like(v_base)
        if self._family == "cosine":
            for k_idx in range(self._num_terms):
                k = k_idx + 1
                freq = k * math.pi
                sin_end = torch.sin(freq * alpha_end)
                sin_start = torch.sin(freq * alpha_start)
                r_avg = r_avg + c_k[:, k_idx] * (sin_end - sin_start) / (freq * d_alpha.clamp(min=eps))

        if small_mask.any():
            r_instant = self.evaluate_r(coeffs, alpha_start)
            mask_nd = small_mask.reshape(-1, *([1] * (v_base.ndim - 1)))
            r_avg = torch.where(mask_nd, r_instant, r_avg)

        return v_base + r_avg
