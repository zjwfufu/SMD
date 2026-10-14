# smd/methods/simplified_distill/basis/base.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified abstract base class for all pivot trajectory parameterizations.

``PivotBasis`` is the single ABC that every basis type implements,
eliminating the need for ``_is_residual`` branching in model.py.

``ResidualBasis`` is an intermediate ABC for zero-mean residual bases
(BubblePath, Spectral, CenteredDX, Collocation).  It implements the
``PivotBasis`` interface in terms of the existing ``evaluate_r``/``evaluate_q``
interface, so existing subclasses require no changes.

Core invariant for ResidualBasis subclasses:

    v(alpha) = v_base + r(alpha)       where integral_0^1 r(alpha) d_alpha = 0
    x(alpha) = x_i + alpha * dt * v_base + q(alpha)
                                        where q(alpha) = dt * integral_0^alpha r(s) ds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class PivotBasis(ABC):
    """Unified ABC for all pivot trajectory parameterizations.

    Every basis implements these operations so model.py needs no branching:
    - build_coeffs: parse decoder output -> coefficient dict
    - evaluate_velocity: instantaneous velocity v(alpha)
    - windowed_velocity: average velocity over [alpha_start, alpha_end]
    - trajectory_state: state x(alpha) at any alpha in [0, 1]
    - average_velocity: segment-average velocity (= alignment target for student)
    - n_grid: number of output slots
    """

    @property
    @abstractmethod
    def n_grid(self) -> int:
        """Number of output slots needed from the expanded proj_out."""

    @property
    def proj_out_init_mode(self) -> str:
        """How Pivot should initialize expanded proj_out.

        'zero'   -- residual output (r(alpha) starts at 0).
        'copy'   -- absolute velocity output (tile student weights).
        'hybrid' -- mixed; see proj_out_init_plan for per-slot details.
        """
        return "copy"

    @property
    def proj_out_init_plan(self) -> list[str] | None:
        """Per-slot init plan for hybrid mode. None for non-hybrid."""
        return None

    @property
    def supports_decomposed(self) -> bool:
        """Whether this basis supports decomposed imitation mode.

        Only meaningful for residual bases that have a natural r(alpha)
        decomposition.
        """
        return False

    @abstractmethod
    def build_coeffs(
        self,
        flow: Tensor,
        x_i: Tensor,
        sigma_i: Tensor,
        seg_size: Tensor,
    ) -> dict[str, Any]:
        """Parse decoder output [B, n_grid, C, H, W] into coefficient dict.

        Args:
            flow: [B, n_grid, C, H, W] from pivot (already in flow space).
            x_i: [B, C, H, W] segment start state.
            sigma_i: [B] segment start sigma.
            seg_size: [B] segment size in raw_t space.

        Returns:
            Dict of named values specific to this basis type (Tensors, policies, lists, etc.).
        """

    @abstractmethod
    def evaluate_velocity(
        self,
        v_base: Tensor,
        coeffs: dict[str, Any],
        alpha: Tensor,
    ) -> Tensor:
        """Full velocity v(alpha).

        For residual: v_base + r(alpha).

        Args:
            v_base: [B, C, H, W] student baseline velocity.
            coeffs: From build_coeffs().
            alpha: [B, 1, 1, 1] in [0, 1].

        Returns:
            [B, C, H, W] velocity at alpha.
        """

    @abstractmethod
    def windowed_velocity(
        self,
        v_base: Tensor,
        coeffs: dict[str, Any],
        alpha_start: Tensor,
        alpha_end: Tensor,
        eps: float = 1e-6,
    ) -> Tensor:
        """Average velocity over [alpha_start, alpha_end].

        Args:
            v_base: [B, C, H, W] student baseline velocity.
            coeffs: From build_coeffs().
            alpha_start: [B, 1, 1, 1] window start.
            alpha_end: [B, 1, 1, 1] window end.
            eps: Threshold for small windows (fallback to instantaneous).

        Returns:
            [B, C, H, W] windowed average velocity.
        """

    @abstractmethod
    def trajectory_state(
        self,
        x_i: Tensor,
        dt: Tensor,
        v_base: Tensor,
        coeffs: dict[str, Any],
        alpha: Tensor,
    ) -> Tensor:
        """State x(alpha). Analytic or numerical depending on basis.

        Args:
            x_i: [B, C, H, W] segment start state.
            dt: [B, 1, 1, 1] = sigma_j - sigma_i (negative for denoising).
            v_base: [B, C, H, W] student baseline velocity.
            coeffs: From build_coeffs().
            alpha: [B, 1, 1, 1] in [0, 1].

        Returns:
            [B, C, H, W] state at alpha.
        """

    @abstractmethod
    def average_velocity(
        self,
        x_i: Tensor,
        dt: Tensor,
        v_base: Tensor,
        coeffs: dict[str, Any],
    ) -> Tensor:
        """Average velocity over full segment [0, 1]. Used as alignment target.

        Args:
            x_i: [B, C, H, W] segment start state.
            dt: [B, 1, 1, 1] = sigma_j - sigma_i.
            v_base: [B, C, H, W] student baseline velocity.
            coeffs: From build_coeffs().

        Returns:
            [B, C, H, W] segment-average velocity.
        """


class ResidualBasis(PivotBasis):
    """Intermediate ABC for zero-mean residual bases (BubblePath, Spectral, etc.).

    Bridges the gap between ``PivotBasis`` and the existing ``evaluate_r``/``evaluate_q``
    interface.  Existing subclasses only need to implement ``build_residual_coeffs``,
    ``evaluate_r``, and ``evaluate_q`` — the ``PivotBasis`` methods are implemented
    here via these residual operations.
    """

    @property
    def proj_out_init_mode(self) -> str:
        return "zero"

    @property
    def supports_decomposed(self) -> bool:
        return True

    # --- Residual-specific abstract interface (unchanged from before) ---

    @abstractmethod
    def build_residual_coeffs(
        self,
        flow: Tensor,
        x_i: Tensor,
        sigma_i: Tensor,
        seg_size: Tensor,
    ) -> dict[str, Tensor]:
        """Parse decoder output [B, n_grid, C, H, W] into coefficient dict.

        Args:
            flow: [B, n_grid, C, H, W] from pivot (already in flow space).
            x_i: [B, C, H, W] segment start state.
            sigma_i: [B] segment start sigma.
            seg_size: [B] segment size in raw_t space.

        Returns:
            Dict of named tensors specific to this basis type.
        """

    @abstractmethod
    def evaluate_r(
        self,
        coeffs: dict[str, Tensor],
        alpha: Tensor,
    ) -> Tensor:
        """Velocity residual r(alpha) at given alpha.

        Args:
            coeffs: From build_residual_coeffs().
            alpha: [B, 1, 1, 1] in [0, 1].

        Returns:
            [B, C, H, W] velocity residual.
        """

    @abstractmethod
    def evaluate_q(
        self,
        coeffs: dict[str, Tensor],
        alpha: Tensor,
        dt: Tensor,
    ) -> Tensor:
        """Path residual q(alpha) = dt * integral_0^alpha r(s) ds.

        Args:
            coeffs: From build_residual_coeffs().
            alpha: [B, 1, 1, 1] in [0, 1].
            dt: [B, 1, 1, 1] = sigma_j - sigma_i (negative for denoising).

        Returns:
            [B, C, H, W] path residual.
        """

    # --- PivotBasis implementation via residual interface ---

    def build_coeffs(self, flow, x_i, sigma_i, seg_size):
        return self.build_residual_coeffs(flow, x_i, sigma_i, seg_size)

    def evaluate_velocity(self, v_base, coeffs, alpha):
        return v_base + self.evaluate_r(coeffs, alpha)

    def trajectory_state(self, x_i, dt, v_base, coeffs, alpha):
        """x(alpha) = x_i + alpha * dt * v_base + q(alpha).

        Default implementation; subclasses may override for efficiency.
        """
        return x_i + alpha * dt * v_base + self.evaluate_q(coeffs, alpha, dt)

    def average_velocity(self, x_i, dt, v_base, coeffs):
        """Zero-mean residual => average = v_base."""
        return v_base

    def windowed_velocity(self, v_base, coeffs, alpha_start, alpha_end, eps=1e-6):
        """v_base + (1/d_alpha) * integral_{alpha_start}^{alpha_end} r(s) ds.

        Falls back to v_base + r(alpha_start) when d_alpha < eps.
        Default uses two-point midpoint approximation; subclasses should
        override with analytic integrals.
        """
        d_alpha = (alpha_end - alpha_start).clamp(min=0)
        small_mask = d_alpha.squeeze(-1).squeeze(-1).squeeze(-1) < eps

        if small_mask.all():
            return v_base + self.evaluate_r(coeffs, alpha_start)

        # Default: midpoint approximation
        alpha_mid = (alpha_start + alpha_end) * 0.5
        r_mid = self.evaluate_r(coeffs, alpha_mid)

        if small_mask.any():
            r_instant = self.evaluate_r(coeffs, alpha_start)
            mask_nd = small_mask.reshape(-1, *([1] * (v_base.ndim - 1)))
            r_mid = torch.where(mask_nd, r_instant, r_mid)

        return v_base + r_mid
