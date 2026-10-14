# smd/methods/simplified_distill/basis/collocation.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collocation / quadrature residual basis (Category D).

Outputs residual values at M prescribed quadrature nodes, then
reconstructs the full curve via Lagrange interpolation.
Zero-mean enforced via weighted projection.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .quadrature import gauss_legendre_nodes_weights, uniform_collocation_nodes_weights
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


def _lagrange_basis(nodes: Tensor, alpha: Tensor) -> Tensor:
    """Compute Lagrange basis weights at query point alpha.

    Args:
        nodes: [M] quadrature nodes.
        alpha: [B] query points.

    Returns:
        [B, M] Lagrange basis values at alpha.
    """
    columns = []
    for j in range(nodes.numel()):
        value = torch.ones_like(alpha)
        for m in range(nodes.numel()):
            if m != j:
                value = value * (alpha - nodes[m]) / (nodes[j] - nodes[m])
        columns.append(value)
    return torch.stack(columns, dim=1)


class CollocationResidual(ResidualBasis):
    """Collocation residual: node values + Lagrange interpolation.

    Args:
        n_nodes: Number of quadrature nodes M (= n_grid).
        scheme: "gauss_legendre" or "uniform".
        scale_mode: "sqrt_n", "fixed", or "none".
        residual_scale: Multiplicative scale.
    """

    def __init__(
        self,
        n_nodes: int = 4,
        scheme: str = "gauss_legendre",
        scale_mode: str = "sqrt_n",
        residual_scale: float = 1.0,
    ):
        self._n_nodes = n_nodes
        self._scale = _residual_scale_factor(n_nodes, scale_mode, residual_scale)
        if scheme == "gauss_legendre":
            nodes, weights = gauss_legendre_nodes_weights(n_nodes)
        elif scheme == "uniform":
            nodes, weights = uniform_collocation_nodes_weights(n_nodes)
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")
        self._nodes = nodes.float()
        self._weights = weights.float()
        integration_order = max(1, (n_nodes + 1) // 2)
        integration_nodes, integration_weights = gauss_legendre_nodes_weights(integration_order)
        self._integration_nodes = integration_nodes.float()
        self._integration_weights = integration_weights.float()

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

        raw = flow.float() * self._scale
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
        tensor = torch.as_tensor(value, device=reference.device, dtype=torch.float32)
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

    @staticmethod
    def _basis_broadcast(values: Tensor, r_nodes: Tensor) -> Tensor:
        return values.reshape(r_nodes.shape[0], r_nodes.shape[1], *([1] * (r_nodes.ndim - 2)))

    def evaluate_r(self, coeffs: dict[str, Tensor], alpha: Tensor) -> Tensor:
        """Lagrange interpolation of r at alpha."""
        r_nodes = coeffs["r_nodes"]
        nodes = coeffs["nodes"]
        alpha_batch = self._batch_values(alpha, r_nodes, "alpha").clamp(0.0, 1.0)
        basis_values = self._basis_broadcast(_lagrange_basis(nodes, alpha_batch), r_nodes)
        residual = (basis_values * r_nodes).sum(dim=1)
        return residual.to(dtype=coeffs["dtype_ref"].dtype)

    def _integral_r(self, coeffs: dict[str, Tensor], alpha: Tensor | float) -> Tensor:
        """Evaluate ``integral_0^alpha r(s) ds`` with exact-order GL quadrature."""
        r_nodes = coeffs["r_nodes"]
        nodes = coeffs["nodes"]
        alpha_batch = self._batch_values(alpha, r_nodes, "alpha").clamp(0.0, 1.0)
        integration_nodes = self._integration_nodes.to(device=r_nodes.device, dtype=torch.float32)
        integration_weights = self._integration_weights.to(device=r_nodes.device, dtype=torch.float32)

        integral = torch.zeros_like(r_nodes[:, 0])
        for node, weight in zip(integration_nodes, integration_weights, strict=True):
            sample = alpha_batch * node
            basis_values = self._basis_broadcast(_lagrange_basis(nodes, sample), r_nodes)
            integral = integral + weight * (basis_values * r_nodes).sum(dim=1)
        integral = self._as_data_broadcast(alpha_batch, integral) * integral

        endpoint = (alpha_batch == 0.0) | (alpha_batch == 1.0)
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
        average_r = integral_delta / self._as_data_broadcast(width.clamp(min=eps), integral_delta)
        if small.any():
            instantaneous = self.evaluate_r(coeffs, alpha_start).float()
            small_nd = small.reshape(small.shape[0], *([1] * (average_r.ndim - 1)))
            average_r = torch.where(small_nd, instantaneous, average_r)
        return v_base + average_r.to(dtype=v_base.dtype)
