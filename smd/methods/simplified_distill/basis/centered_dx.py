# smd/methods/simplified_distill/basis/centered_dx.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centered DX residual basis.

The Pivot predicts residual slots on uniform alpha nodes. Trapezoidal
centering enforces zero mean, the residual is piecewise linear, and its path
integral is evaluated exactly as a piecewise quadratic.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import ResidualBasis
from .quadrature import uniform_nodes_weights


class CenteredDXResidual(ResidualBasis):
    """Centered DX residual: grid velocities with zero-mean projection.

    Args:
        n_nodes: Number of grid nodes (= n_grid).
        The slots directly parameterize residual node values. Unlike analytic
        bases, CenteredDX applies no K-dependent or user-configured rescaling.
    """

    def __init__(self, n_nodes: int = 10):
        self._n_nodes = n_nodes
        # Precompute trapezoidal nodes and weights
        nodes, weights = uniform_nodes_weights(n_nodes)
        self._nodes = nodes.float()  # [n_nodes] in [0, 1]
        self._weights = weights.float()  # [n_nodes] trapezoidal weights

    @property
    def n_grid(self) -> int:
        return self._n_nodes

    def build_residual_coeffs(
        self,
        flow: Tensor,
        x_i: Tensor,
        sigma_i: Tensor,
        seg_size: Tensor,
    ) -> dict[str, Tensor]:
        del x_i, sigma_i, seg_size
        if flow.ndim < 3:
            raise ValueError(f"flow must have shape [B, K, ...], got {tuple(flow.shape)}")
        if flow.shape[1] != self._n_nodes:
            raise ValueError(f"flow has K={flow.shape[1]} slots, expected {self._n_nodes}")

        raw = flow.float()
        w = self._weights.to(device=raw.device, dtype=torch.float32)
        w_nd = w.reshape(1, -1, *([1] * (raw.ndim - 2)))
        weighted_mean = (w_nd * raw).sum(dim=1, keepdim=True)
        centered = raw - weighted_mean
        return {
            "r_nodes": centered,
            "nodes": self._nodes.to(device=raw.device, dtype=torch.float32),
            "weights": w,
            "dtype_ref": flow.new_zeros(()),
        }

    @staticmethod
    def _batch_values(value: Tensor | float, reference: Tensor, name: str) -> Tensor:
        """Normalize a scalar or batch-wise value to shape ``[B]``."""
        tensor = torch.as_tensor(value, device=reference.device)
        batch_size = reference.shape[0]
        if tensor.numel() == 1:
            return tensor.reshape(1).expand(batch_size)
        if tensor.shape[0] == batch_size and all(size == 1 for size in tensor.shape[1:]):
            return tensor.reshape(batch_size)
        raise ValueError(
            f"{name} must be scalar or batch-wise with singleton trailing dimensions; "
            f"got shape {tuple(tensor.shape)} for batch size {batch_size}"
        )

    @staticmethod
    def _as_data_broadcast(values: Tensor, data: Tensor) -> Tensor:
        return values.to(dtype=data.dtype).reshape(data.shape[0], *([1] * (data.ndim - 1)))

    def _interval_coordinates(self, r_nodes: Tensor, alpha: Tensor | float) -> tuple[Tensor, Tensor, Tensor]:
        """Return left interval indices, local fractions, and clamped alpha."""
        alpha_batch = self._batch_values(alpha, r_nodes, "alpha")
        alpha_locator = alpha_batch.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
        position_locator = alpha_locator * (self._n_nodes - 1)
        idx_lo = torch.floor(position_locator).to(dtype=torch.long)
        idx_lo = idx_lo.clamp(0, self._n_nodes - 2)

        alpha_math = alpha_batch.to(dtype=torch.float32).clamp(0.0, 1.0)
        position = alpha_math * (self._n_nodes - 1)
        fraction = position - idx_lo.to(dtype=torch.float32)
        return idx_lo, fraction, alpha_locator

    @staticmethod
    def _gather_nodes(values: Tensor, indices: Tensor) -> Tensor:
        batch_indices = torch.arange(values.shape[0], device=values.device)
        return values[batch_indices, indices]

    def evaluate_r(self, coeffs: dict[str, Tensor], alpha: Tensor) -> Tensor:
        """Piecewise linear interpolation of r at alpha."""
        r_nodes = coeffs["r_nodes"]
        idx_lo, fraction, _ = self._interval_coordinates(r_nodes, alpha)
        r_lo = self._gather_nodes(r_nodes, idx_lo)
        r_hi = self._gather_nodes(r_nodes, idx_lo + 1)
        fraction_nd = self._as_data_broadcast(fraction, r_lo)
        residual = r_lo + fraction_nd * (r_hi - r_lo)
        return residual.to(dtype=coeffs["dtype_ref"].dtype)

    def _integral_r(self, coeffs: dict[str, Tensor], alpha: Tensor | float) -> Tensor:
        """Evaluate ``integral_0^alpha r(s) ds`` exactly."""
        r_nodes = coeffs["r_nodes"]
        idx_lo, fraction, alpha_locator = self._interval_coordinates(r_nodes, alpha)
        h = r_nodes.new_tensor(1.0 / (self._n_nodes - 1))

        interval_areas = 0.5 * h * (r_nodes[:, :-1] + r_nodes[:, 1:])
        prefix_nodes = torch.cat(
            [torch.zeros_like(interval_areas[:, :1]), interval_areas.cumsum(dim=1)],
            dim=1,
        )
        prefix = self._gather_nodes(prefix_nodes, idx_lo)
        r_lo = self._gather_nodes(r_nodes, idx_lo)
        r_hi = self._gather_nodes(r_nodes, idx_lo + 1)
        fraction_nd = self._as_data_broadcast(fraction, r_lo)
        local_integral = h * (fraction_nd * r_lo + 0.5 * fraction_nd.square() * (r_hi - r_lo))
        integral = prefix + local_integral
        endpoint = (alpha_locator == 0.0) | (alpha_locator == 1.0)
        if endpoint.any():
            endpoint_nd = endpoint.reshape(endpoint.shape[0], *([1] * (integral.ndim - 1)))
            integral = torch.where(endpoint_nd, torch.zeros_like(integral), integral)
        return integral

    def evaluate_q(self, coeffs: dict[str, Tensor], alpha: Tensor, dt: Tensor) -> Tensor:
        """Return ``dt * integral_0^alpha r(s) ds``."""
        integral = self._integral_r(coeffs, alpha)
        dt_batch = self._batch_values(dt, integral, "dt")
        path = self._as_data_broadcast(dt_batch, integral) * integral
        return path.to(dtype=coeffs["dtype_ref"].dtype)

    def trajectory_state(self, x_i, dt, v_base, coeffs, alpha):
        alpha_batch = self._batch_values(alpha, x_i, "alpha")
        dt_batch = self._batch_values(dt, x_i, "dt")
        alpha_nd = self._as_data_broadcast(alpha_batch, x_i)
        dt_nd = self._as_data_broadcast(dt_batch, x_i)
        return x_i + alpha_nd * dt_nd * v_base + self.evaluate_q(coeffs, alpha, dt)

    def windowed_velocity(self, v_base, coeffs, alpha_start, alpha_end, eps=1e-6):
        start = self._batch_values(alpha_start, v_base, "alpha_start")
        end = self._batch_values(alpha_end, v_base, "alpha_end")
        width = (end - start).clamp(min=0.0)
        small = width < eps

        integral_delta = self._integral_r(coeffs, alpha_end) - self._integral_r(coeffs, alpha_start)
        width_safe = width.clamp(min=eps)
        average_r = integral_delta / self._as_data_broadcast(width_safe, integral_delta)
        if small.any():
            instantaneous = self.evaluate_r(coeffs, alpha_start).float()
            small_nd = small.reshape(small.shape[0], *([1] * (average_r.ndim - 1)))
            average_r = torch.where(small_nd, instantaneous, average_r)
        return v_base + average_r.to(dtype=v_base.dtype)
