# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quadrature nodes and weights on [0, 1].

Provides Gauss-Legendre and uniform (trapezoidal) quadrature rules
for use by residual basis classes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def gauss_legendre_nodes_weights(n: int) -> tuple[Tensor, Tensor]:
    """Return (nodes, weights) for n-point Gauss-Legendre quadrature on [0, 1].

    Uses numpy for the standard [-1, 1] rule, then maps to [0, 1].
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    # numpy computes GL on [-1, 1]
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    # Map [-1, 1] -> [0, 1]: x = (t + 1) / 2, w = w / 2
    nodes_01 = (nodes_np + 1.0) / 2.0
    weights_01 = weights_np / 2.0
    return (
        torch.tensor(nodes_01, dtype=torch.float64),
        torch.tensor(weights_01, dtype=torch.float64),
    )


def chebyshev_nodes_weights(n: int) -> tuple[Tensor, Tensor]:
    """Return (nodes, weights) for n-point Chebyshev quadrature on [0, 1].

    Uses Chebyshev nodes of the first kind mapped to [0, 1] with
    Fejér first rule weights (Waldvogel 2006).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Chebyshev nodes of first kind on [-1, 1]: cos(pi*(2j+1)/(2n))
    j = np.arange(n)
    nodes_11 = np.cos(np.pi * (2 * j + 1) / (2 * n))
    # Map [-1, 1] -> [0, 1]
    nodes_01 = (nodes_11 + 1.0) / 2.0

    # Fejér first rule weights on [-1, 1] for Chebyshev nodes of the first kind
    # w_j = (2/n) * [1 - 2 * sum_{k=1}^{floor(n/2)} cos(2k*theta_j) / (4k^2-1)]
    # where theta_j = pi*(2j+1)/(2n)
    # Reference: Waldvogel (2006), "Fast Construction of the Fejér and
    # Clenshaw-Curtis Quadrature Rules"
    theta = np.pi * (2 * j + 1) / (2 * n)
    weights_11 = np.ones(n)
    for k in range(1, n // 2 + 1):
        factor = 2.0 / (4.0 * k * k - 1.0)
        if 2 * k == n:
            factor /= 2.0  # halve last term when n is even
        weights_11 -= factor * np.cos(2 * k * theta)
    weights_11 *= 2.0 / n

    # Map weights [-1, 1] -> [0, 1]: w01 = w11 / 2
    weights_01 = weights_11 / 2.0

    # Sort by node position (Chebyshev nodes come in decreasing order)
    order = np.argsort(nodes_01)
    return (
        torch.tensor(nodes_01[order], dtype=torch.float64),
        torch.tensor(weights_01[order], dtype=torch.float64),
    )


def uniform_nodes_weights(n: int) -> tuple[Tensor, Tensor]:
    """Return (nodes, weights) for n-point trapezoidal quadrature on [0, 1].

    Nodes are uniformly spaced including endpoints.
    Weights follow the trapezoidal rule (half-weight at endpoints).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for trapezoidal rule, got {n}")
    nodes = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    h = 1.0 / (n - 1)
    weights = torch.full((n,), h, dtype=torch.float64)
    weights[0] = h / 2.0
    weights[-1] = h / 2.0
    return nodes, weights


def interpolatory_quadrature_weights(nodes: Tensor) -> Tensor:
    """Return weights that exactly integrate the Lagrange interpolant on [0, 1]."""
    if nodes.ndim != 1 or nodes.numel() < 1:
        raise ValueError(f"nodes must be a non-empty 1D tensor, got {tuple(nodes.shape)}")
    nodes64 = nodes.to(dtype=torch.float64)
    degrees = torch.arange(nodes64.numel(), device=nodes64.device, dtype=torch.float64)
    vandermonde = nodes64.unsqueeze(0) ** degrees.unsqueeze(1)
    moments = 1.0 / (degrees + 1.0)
    return torch.linalg.solve(vandermonde, moments)


def uniform_collocation_nodes_weights(n: int) -> tuple[Tensor, Tensor]:
    """Return uniform nodes with weights for their global Lagrange interpolant."""
    nodes, _ = uniform_nodes_weights(n)
    return nodes, interpolatory_quadrature_weights(nodes)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Sanity harness: verify Gauss-Legendre, Chebyshev, and uniform
    # quadrature rules satisfy basic mathematical invariants.
    # ------------------------------------------------------------------
    print("=== quadrature sanity ===")

    for name, fn, n_range in [
        ("gauss_legendre", gauss_legendre_nodes_weights, [2, 3, 4, 5, 8]),
        ("chebyshev", chebyshev_nodes_weights, [2, 3, 4, 5, 8]),
        ("uniform", uniform_nodes_weights, [2, 3, 5, 10]),
    ]:
        for n in n_range:
            nodes, weights = fn(n)
            # weights sum to 1 (integral of constant 1 over [0,1])
            w_sum = weights.sum().item()
            assert abs(w_sum - 1.0) < 1e-10, f"{name}({n}): weights sum = {w_sum}"
            # nodes in [0, 1]
            assert (nodes >= 0.0).all() and (nodes <= 1.0).all(), f"{name}({n}): nodes out of [0,1]"
            print(f"  {name}(n={n}): weights_sum={w_sum:.15f}, nodes_range=[{nodes.min():.6f}, {nodes.max():.6f}]")

    # GL polynomial exactness: n-point rule integrates degree <= 2n-1 exactly
    for n in [2, 3, 4, 5]:
        nodes, weights = gauss_legendre_nodes_weights(n)
        for d in range(2 * n):
            numerical = (weights * nodes**d).sum().item()
            exact = 1.0 / (d + 1)
            assert abs(numerical - exact) < 1e-10, f"GL({n}) degree {d}: {numerical} != {exact}"
    print("  GL polynomial exactness OK (up to degree 2n-1)")

    # Uniform trapezoidal: integrates linear exactly
    nodes, weights = uniform_nodes_weights(10)
    numerical = (weights * (2.0 * nodes + 1.0)).sum().item()
    assert abs(numerical - 2.0) < 1e-10
    print("  uniform trapezoidal linear exactness OK")

    # Uniform: equally spaced
    diffs = nodes[1:] - nodes[:-1]
    assert torch.allclose(diffs, diffs[0:1].expand_as(diffs), atol=1e-12)
    print("  uniform equal spacing OK")

    # Uniform collocation uses the closed Newton-Cotes weights associated with
    # the global Lagrange interpolant, not trapezoidal weights.
    for n in [2, 3, 4, 6, 8]:
        nodes, weights = uniform_collocation_nodes_weights(n)
        for d in range(n):
            numerical = (weights * nodes**d).sum().item()
            exact = 1.0 / (d + 1)
            assert abs(numerical - exact) < 1e-10, f"uniform collocation ({n}) degree {d}: {numerical} != {exact}"
    print("  uniform collocation polynomial exactness OK (up to degree n-1)")

    print("=== quadrature sanity PASSED ===")
