# smd/methods/simplified_distill/basis/bubble_path.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bubble path residual basis (Category C).

Parameterizes the path residual q(alpha) with boundary-zero basis functions:
    q(alpha) = sum_{k=1}^{K} b_k * phi_k(alpha),   phi_k(0) = phi_k(1) = 0

Then  r(alpha) = q'(alpha) / dt.

Supported families:
  - sin_path:    phi_k(alpha) = sin(k * pi * alpha)
  - bubble_poly: phi_k(alpha) = alpha^k * (1 - alpha)
  - taylor_path: phi_k(alpha) = alpha * (1 - alpha) * (2*alpha - 1)^{k-1}
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import ResidualBasis


def _residual_scale_factor(num_terms: int, mode: str, scale: float) -> float:
    """Compute the per-coefficient scaling factor."""
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


class BubblePathResidual(ResidualBasis):
    """Bubble path residual: q(alpha) with phi_k(0) = phi_k(1) = 0.

    Args:
        num_terms: Number of basis functions K (= n_grid).
        family: One of "sin_path", "bubble_poly", "taylor_path".
        scale_mode: "sqrt_n", "fixed", or "none".
        residual_scale: Multiplicative scale.
    """

    def __init__(
        self,
        num_terms: int = 4,
        family: str = "sin_path",
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
        # flow: [B, K, C, H, W] — each slot is a path coefficient b_k
        return {"b_k": flow * self._scale}

    # -- Family dispatchers --

    def _phi(self, k: int, alpha: Tensor) -> Tensor:
        """Path basis phi_k(alpha). k is 1-indexed."""
        if self._family == "sin_path":
            return torch.sin(k * math.pi * alpha)
        elif self._family == "bubble_poly":
            return (alpha**k) * (1.0 - alpha)
        elif self._family == "taylor_path":
            return alpha * (1.0 - alpha) * (2.0 * alpha - 1.0) ** (k - 1)
        else:
            raise ValueError(f"Unknown family: {self._family!r}")

    def _phi_prime(self, k: int, alpha: Tensor) -> Tensor:
        """Derivative d/d_alpha phi_k(alpha). k is 1-indexed."""
        if self._family == "sin_path":
            return k * math.pi * torch.cos(k * math.pi * alpha)
        elif self._family == "bubble_poly":
            # d/da [a^k * (1-a)] = k*a^{k-1}*(1-a) - a^k
            return k * (alpha ** (k - 1)) * (1.0 - alpha) - alpha**k
        elif self._family == "taylor_path":
            # d/da [a*(1-a)*(2a-1)^{k-1}]
            # Let u = a*(1-a), v = (2a-1)^{k-1}
            # u' = 1 - 2a, v' = 2*(k-1)*(2a-1)^{k-2}
            u = alpha * (1.0 - alpha)
            u_prime = 1.0 - 2.0 * alpha
            v = (2.0 * alpha - 1.0) ** (k - 1) if k > 1 else torch.ones_like(alpha)
            v_prime = 2.0 * (k - 1) * (2.0 * alpha - 1.0) ** max(k - 2, 0) if k > 1 else torch.zeros_like(alpha)
            return u_prime * v + u * v_prime
        else:
            raise ValueError(f"Unknown family: {self._family!r}")

    def _phi_antideriv(self, k: int, alpha: Tensor) -> Tensor:
        """Antiderivative integral_0^alpha phi_k(s) ds. k is 1-indexed."""
        if self._family == "sin_path":
            # integral sin(k*pi*s) ds = (1 - cos(k*pi*alpha)) / (k*pi)
            return (1.0 - torch.cos(k * math.pi * alpha)) / (k * math.pi)
        elif self._family == "bubble_poly":
            # integral_0^a s^k*(1-s) ds = a^{k+1}/(k+1) - a^{k+2}/(k+2)
            return alpha ** (k + 1) / (k + 1) - alpha ** (k + 2) / (k + 2)
        elif self._family == "taylor_path":
            # No simple closed form — use 4-point GL quadrature as fallback
            from .quadrature import gauss_legendre_nodes_weights

            nodes, weights = gauss_legendre_nodes_weights(8)
            nodes = nodes.to(dtype=alpha.dtype, device=alpha.device)
            weights = weights.to(dtype=alpha.dtype, device=alpha.device)
            # Map [0,1] GL to [0, alpha]
            result = torch.zeros_like(alpha)
            for i in range(len(nodes)):
                s = alpha * nodes[i]
                phi_val = s * (1.0 - s) * (2.0 * s - 1.0) ** (k - 1) if k > 1 else s * (1.0 - s)
                result = result + weights[i] * phi_val * alpha  # Jacobian = alpha
            return result
        else:
            raise ValueError(f"Unknown family: {self._family!r}")

    # -- Core interface --

    def evaluate_r(self, coeffs: dict[str, Tensor], alpha: Tensor) -> Tensor:
        """r(alpha) = (1/dt) * q'(alpha) = (1/dt) * sum b_k * phi_k'(alpha).

        Note: We return r(alpha) WITHOUT the 1/dt factor, because in the
        full velocity v(alpha) = v_base + r(alpha), r must be in dx/d_sigma
        units (same as v_base). The dt factor cancels:
            q(alpha) = dt * integral_0^alpha r(s) ds
            q'(alpha) = dt * r(alpha)
            r(alpha) = q'(alpha) / dt = sum b_k * phi_k'(alpha) / dt

        But since flow coefficients b_k are already in "dx/d_sigma * dt" units
        (from the decoder), we compute r = sum b_k * phi_k'(alpha) directly.
        The training loop ensures consistent units.

        Actually, for cleaner separation: b_k has arbitrary units from the
        decoder. r(alpha) = sum b_k * phi_k'(alpha). The dt scaling lives
        in evaluate_q where q = dt * integral r, i.e. q = sum b_k * phi_k
        and we let the user pass dt separately.

        Simplification: since phi_k(0) = phi_k(1) = 0, we have
        integral_0^1 phi_k'(alpha) d_alpha = phi_k(1) - phi_k(0) = 0.
        So r(alpha) = sum b_k * phi_k'(alpha) is automatically zero-mean.
        """
        b_k = coeffs["b_k"]  # [B, K, C, H, W]
        result = torch.zeros_like(b_k[:, 0])  # [B, C, H, W]
        for k_idx in range(self._num_terms):
            k = k_idx + 1  # 1-indexed
            result = result + b_k[:, k_idx] * self._phi_prime(k, alpha)
        return result

    def evaluate_q(self, coeffs: dict[str, Tensor], alpha: Tensor, dt: Tensor) -> Tensor:
        """q(alpha) = dt * integral_0^alpha r(s) ds = sum b_k * phi_k(alpha) * dt_factor.

        Wait -- we need to be careful about units.

        q(alpha) = dt * integral_0^alpha r(s) ds

        If r(s) = sum b_k * phi_k'(s), then:
            integral_0^alpha r(s) ds = sum b_k * [phi_k(alpha) - phi_k(0)]
                                     = sum b_k * phi_k(alpha)   (since phi_k(0) = 0)

        So q(alpha) = dt * sum b_k * phi_k(alpha).
        """
        b_k = coeffs["b_k"]  # [B, K, C, H, W]
        result = torch.zeros_like(b_k[:, 0])  # [B, C, H, W]
        for k_idx in range(self._num_terms):
            k = k_idx + 1
            result = result + b_k[:, k_idx] * self._phi(k, alpha)
        return dt * result

    def windowed_velocity(
        self,
        v_base: Tensor,
        coeffs: dict[str, Tensor],
        alpha_start: Tensor,
        alpha_end: Tensor,
        eps: float = 1e-6,
    ) -> Tensor:
        """Analytic windowed average for BubblePath.

        v_bar = v_base + (1/d_alpha) * integral_{a_s}^{a_e} r(s) ds
              = v_base + (1/d_alpha) * sum b_k * [phi_k(a_e) - phi_k(a_s)]
        """
        d_alpha = (alpha_end - alpha_start).clamp(min=0)
        small_mask = d_alpha.squeeze(-1).squeeze(-1).squeeze(-1) < eps

        if small_mask.all():
            return v_base + self.evaluate_r(coeffs, alpha_start)

        b_k = coeffs["b_k"]
        r_avg = torch.zeros_like(v_base)
        for k_idx in range(self._num_terms):
            k = k_idx + 1
            delta_phi = self._phi(k, alpha_end) - self._phi(k, alpha_start)
            r_avg = r_avg + b_k[:, k_idx] * delta_phi / d_alpha.clamp(min=eps)

        if small_mask.any():
            r_instant = self.evaluate_r(coeffs, alpha_start)
            mask_nd = small_mask.reshape(-1, *([1] * (v_base.ndim - 1)))
            r_avg = torch.where(mask_nd, r_instant, r_avg)

        return v_base + r_avg
