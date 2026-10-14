# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable basis evaluation functions for on-policy pivot distillation.

This module provides autograd-compatible functions for evaluating residual
velocity r(alpha), path integral q(alpha), trajectory states x(alpha), and
predicted velocities v(alpha) for all five basis types:

  - BubblePath / sin_path:   q(alpha) = sum_k b_k * sin(k*pi*alpha)
  - BubblePath / bubble_poly: q(alpha) = sum_k b_k * alpha^k * (1 - alpha)
  - Spectral / cosine:       r(alpha) = sum_k c_k * cos(k*pi*alpha)
  - CenteredDX / (grid):     r at uniform nodes, piecewise linear interpolation
  - Collocation / GL:         r at Gauss-Legendre nodes, Lagrange interpolation

All functions preserve the autograd graph so gradients flow back through
nn.Parameters (v_base, coeffs).

Dimension conventions:
  - coeffs: [K, *S]  where S is the spatial shape (e.g. B, C, H, W)
  - alpha:  [P]      query points in [0, 1]
  - output: [P, *S]  values at each query point
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Zero-mean projection
# ---------------------------------------------------------------------------


def project_zero_mean(coeffs: Tensor, weights: Tensor) -> Tensor:
    """Zero-mean projection: c' = c - sum_j(w_j * c_j).

    Args:
        coeffs: [K, *S] coefficient tensor.
        weights: [K] quadrature weights (must sum to ~1 for proper zero-mean).

    Returns:
        [K, *S] projected coefficients with weighted mean subtracted.
    """
    # weights: [K] -> [K, 1, 1, ...] for broadcasting
    ndim_extra = coeffs.ndim - 1
    w = weights.reshape(-1, *([1] * ndim_extra))
    weighted_mean = (w * coeffs).sum(dim=0, keepdim=True)  # [1, *S]
    return coeffs - weighted_mean


# ---------------------------------------------------------------------------
# evaluate_r: velocity residual r(alpha) at P query points
# ---------------------------------------------------------------------------


def evaluate_r(basis_type: str, coeffs: Tensor, alpha: Tensor, **kwargs) -> Tensor:
    """Evaluate r(alpha) at P query points.

    r(alpha) is the zero-mean velocity residual such that v(alpha) = v_base + r(alpha).

    Args:
        basis_type: One of "bubble_path", "spectral_zero_mean", "centered_dx", "collocation".
        coeffs: [K, *S] learnable coefficients.
        alpha: [P] query points in [0, 1].
        **kwargs:
            family: "sin_path" or "bubble_poly" (for bubble_path).
            nodes: [K] quadrature nodes (for centered_dx or collocation).
            weights: [K] quadrature weights (for centered_dx or collocation).

    Returns:
        [P, *S] residual velocity values.
    """
    if basis_type == "bubble_path":
        return _evaluate_r_bubble_path(coeffs, alpha, family=kwargs.get("family", "sin_path"))
    elif basis_type == "spectral_zero_mean":
        return _evaluate_r_spectral(coeffs, alpha)
    elif basis_type == "centered_dx":
        nodes = kwargs["nodes"]
        weights = kwargs["weights"]
        return _evaluate_r_centered_dx(coeffs, alpha, nodes, weights)
    elif basis_type == "collocation":
        nodes = kwargs["nodes"]
        weights = kwargs["weights"]
        return _evaluate_r_collocation(coeffs, alpha, nodes, weights)
    else:
        raise ValueError(
            f"Unknown basis_type={basis_type!r}. "
            "Supported: 'bubble_path', 'spectral_zero_mean', 'centered_dx', 'collocation'."
        )


def _evaluate_r_bubble_path(coeffs: Tensor, alpha: Tensor, family: str = "sin_path") -> Tensor:
    """BubblePath r(alpha) = sum_k b_k * phi_k'(alpha).

    sin_path:    r(alpha) = sum_k b_k * k*pi*cos(k*pi*alpha)
    bubble_poly: r(alpha) = sum_k b_k * [k*alpha^{k-1}*(1-alpha) - alpha^k]
    """
    K = coeffs.shape[0]
    S = coeffs.shape[1:]
    P = alpha.shape[0]

    # alpha: [P] -> [P, 1, 1, ...]
    a = alpha.reshape(P, *([1] * len(S)))
    result = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)

    for k_idx in range(K):
        k = k_idx + 1  # 1-indexed
        bk = coeffs[k_idx]  # [*S]
        if family == "sin_path":
            # phi_k'(alpha) = k*pi*cos(k*pi*alpha)
            phi_prime = k * math.pi * torch.cos(k * math.pi * a)
        elif family == "bubble_poly":
            # phi_k(alpha) = alpha^k * (1 - alpha)
            # phi_k'(alpha) = k*alpha^{k-1}*(1-alpha) - alpha^k
            phi_prime = k * (a ** max(k - 1, 0)) * (1.0 - a) - a**k
        elif family == "taylor_path":
            # phi_k(alpha) = alpha * (1 - alpha) * (2*alpha - 1)^{k-1}
            # phi_k'(alpha) = (1 - 2*alpha) * (2*alpha - 1)^{k-1}
            #                + alpha * (1 - alpha) * 2*(k-1) * (2*alpha - 1)^{max(k-2,0)}
            u = a * (1.0 - a)
            u_prime = 1.0 - 2.0 * a
            v = (2.0 * a - 1.0) ** max(k - 1, 0)
            if k == 1:
                phi_prime = u_prime  # v=1, v'=0
            else:
                v_prime = 2.0 * (k - 1) * (2.0 * a - 1.0) ** max(k - 2, 0)
                phi_prime = u_prime * v + u * v_prime
        else:
            raise ValueError(f"Unknown BubblePath family: {family}")
        result = result + phi_prime * bk.unsqueeze(0)

    return result


def _evaluate_r_spectral(coeffs: Tensor, alpha: Tensor) -> Tensor:
    """Spectral r(alpha) = sum_k c_k * cos(k*pi*alpha)."""
    K = coeffs.shape[0]
    S = coeffs.shape[1:]
    P = alpha.shape[0]

    a = alpha.reshape(P, *([1] * len(S)))
    result = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)

    for k_idx in range(K):
        k = k_idx + 1
        ck = coeffs[k_idx]  # [*S]
        result = result + ck.unsqueeze(0) * torch.cos(k * math.pi * a)

    return result


def _evaluate_r_centered_dx(coeffs: Tensor, alpha: Tensor, nodes: Tensor, weights: Tensor) -> Tensor:
    """CenteredDX: project to zero-mean, then piecewise linear interpolation.

    coeffs: [K, *S], nodes: [K] uniform in [0, 1], weights: [K] trapezoidal.
    """
    # Project to zero mean (differentiable)
    c = project_zero_mean(coeffs, weights)

    K = c.shape[0]
    S = c.shape[1:]
    P = alpha.shape[0]

    # Piecewise linear interpolation
    # idx_float = alpha * (K - 1)
    a = alpha.clamp(0.0, 1.0)  # [P]
    idx_float = a * (K - 1)
    idx_lo = idx_float.long().clamp(0, K - 2)  # [P]
    frac = (idx_float - idx_lo.float()).clamp(0.0, 1.0)  # [P]
    idx_hi = idx_lo + 1

    # Gather: c[idx_lo] and c[idx_hi]
    # c: [K, *S], idx_lo: [P] -> c_lo: [P, *S]
    c_lo = c[idx_lo]  # [P, *S]
    c_hi = c[idx_hi]  # [P, *S]

    frac_nd = frac.reshape(P, *([1] * len(S)))
    return c_lo + frac_nd * (c_hi - c_lo)


def _evaluate_r_collocation(coeffs: Tensor, alpha: Tensor, nodes: Tensor, weights: Tensor) -> Tensor:
    """Collocation: project to zero-mean, then Lagrange interpolation.

    coeffs: [K, *S], nodes: [K] GL nodes, weights: [K] GL weights.
    """
    # Project to zero mean (differentiable)
    c = project_zero_mean(coeffs, weights)

    K = c.shape[0]
    S = c.shape[1:]
    P = alpha.shape[0]

    # Lagrange interpolation: r(alpha) = sum_j L_j(alpha) * c[j]
    # L_j(alpha) = prod_{m != j} (alpha - node_m) / (node_j - node_m)
    a = alpha  # [P]

    result = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)
    for j in range(K):
        L_j = torch.ones(P, device=alpha.device, dtype=alpha.dtype)
        for m in range(K):
            if m != j:
                L_j = L_j * (a - nodes[m]) / (nodes[j] - nodes[m])
        # L_j: [P] -> [P, 1, 1, ...]
        L_j_nd = L_j.reshape(P, *([1] * len(S)))
        result = result + L_j_nd * c[j].unsqueeze(0)

    return result


# ---------------------------------------------------------------------------
# evaluate_q: path integral q(alpha) = integral_0^alpha r(s) ds
# ---------------------------------------------------------------------------


def evaluate_q(basis_type: str, coeffs: Tensor, alpha: Tensor, **kwargs) -> Tensor:
    """Evaluate q(alpha) = integral_0^alpha r(s) ds at P query points.

    Note: does NOT include the dt factor. The caller multiplies by dt.

    Args:
        basis_type: One of "bubble_path", "spectral_zero_mean", "centered_dx", "collocation".
        coeffs: [K, *S] learnable coefficients.
        alpha: [P] query points in [0, 1].
        **kwargs: Same as evaluate_r.

    Returns:
        [P, *S] path integral values (without dt factor).
    """
    if basis_type == "bubble_path":
        return _evaluate_q_bubble_path(coeffs, alpha, family=kwargs.get("family", "sin_path"))
    elif basis_type == "spectral_zero_mean":
        return _evaluate_q_spectral(coeffs, alpha)
    elif basis_type == "centered_dx":
        nodes = kwargs["nodes"]
        weights = kwargs["weights"]
        return _evaluate_q_centered_dx(coeffs, alpha, nodes, weights)
    elif basis_type == "collocation":
        nodes = kwargs["nodes"]
        weights = kwargs["weights"]
        return _evaluate_q_collocation(coeffs, alpha, nodes, weights)
    else:
        raise ValueError(
            f"Unknown basis_type={basis_type!r}. "
            "Supported: 'bubble_path', 'spectral_zero_mean', 'centered_dx', 'collocation'."
        )


def _evaluate_q_bubble_path(coeffs: Tensor, alpha: Tensor, family: str = "sin_path") -> Tensor:
    """BubblePath q(alpha) = sum_k b_k * phi_k(alpha).

    sin_path:    q(alpha) = sum_k b_k * sin(k*pi*alpha)
    bubble_poly: q(alpha) = sum_k b_k * alpha^k * (1 - alpha)
    """
    K = coeffs.shape[0]
    S = coeffs.shape[1:]
    P = alpha.shape[0]

    a = alpha.reshape(P, *([1] * len(S)))
    result = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)

    for k_idx in range(K):
        k = k_idx + 1
        bk = coeffs[k_idx]  # [*S]
        if family == "sin_path":
            phi = torch.sin(k * math.pi * a)
        elif family == "bubble_poly":
            phi = (a**k) * (1.0 - a)
        elif family == "taylor_path":
            # phi_k(alpha) = alpha * (1 - alpha) * (2*alpha - 1)^{k-1}
            phi = a * (1.0 - a) * (2.0 * a - 1.0) ** max(k - 1, 0)
        else:
            raise ValueError(f"Unknown BubblePath family: {family}")
        result = result + bk.unsqueeze(0) * phi

    return result


def _evaluate_q_spectral(coeffs: Tensor, alpha: Tensor) -> Tensor:
    """Spectral q(alpha) = sum_k c_k * sin(k*pi*alpha) / (k*pi)."""
    K = coeffs.shape[0]
    S = coeffs.shape[1:]
    P = alpha.shape[0]

    a = alpha.reshape(P, *([1] * len(S)))
    result = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)

    for k_idx in range(K):
        k = k_idx + 1
        ck = coeffs[k_idx]  # [*S]
        result = result + ck.unsqueeze(0) * torch.sin(k * math.pi * a) / (k * math.pi)

    return result


def _evaluate_q_centered_dx(coeffs: Tensor, alpha: Tensor, nodes: Tensor, weights: Tensor) -> Tensor:
    """CenteredDX q(alpha): cumulative trapezoidal integration of zero-mean projected r."""
    # Project to zero mean
    c = project_zero_mean(coeffs, weights)

    K = c.shape[0]
    S = c.shape[1:]
    P = alpha.shape[0]

    # Compute cumulative trapezoidal integral at each node
    h = 1.0 / (K - 1)  # uniform spacing
    # cum_q[0] = 0, cum_q[k] = cum_q[k-1] + 0.5*(c[k-1]+c[k])*h
    cum_q = [torch.zeros(S, device=coeffs.device, dtype=coeffs.dtype)]
    for k in range(1, K):
        cum_q.append(cum_q[-1] + 0.5 * (c[k - 1] + c[k]) * h)
    cum_q = torch.stack(cum_q, dim=0)  # [K, *S]

    # Interpolate at query alpha via piecewise linear
    a = alpha.clamp(0.0, 1.0)
    idx_float = a * (K - 1)
    idx_lo = idx_float.long().clamp(0, K - 2)
    frac = (idx_float - idx_lo.float()).clamp(0.0, 1.0)
    idx_hi = idx_lo + 1

    q_lo = cum_q[idx_lo]  # [P, *S]
    q_hi = cum_q[idx_hi]  # [P, *S]
    frac_nd = frac.reshape(P, *([1] * len(S)))
    return q_lo + frac_nd * (q_hi - q_lo)


def _evaluate_q_collocation(coeffs: Tensor, alpha: Tensor, nodes: Tensor, weights: Tensor) -> Tensor:
    """Collocation q(alpha): GL sub-quadrature on [0, alpha].

    q(alpha) = alpha * sum_i w_i * r(alpha * s_i)
    where (s_i, w_i) are GL nodes/weights on [0, 1].
    """
    from .quadrature import gauss_legendre_nodes_weights

    K = coeffs.shape[0]
    S = coeffs.shape[1:]
    P = alpha.shape[0]

    # Project to zero mean
    c = project_zero_mean(coeffs, weights)

    # Sub-integration GL rule
    n_sub = max(1, (K + 1) // 2)
    sub_nodes, sub_weights = gauss_legendre_nodes_weights(n_sub)
    sub_nodes = sub_nodes.to(device=alpha.device, dtype=alpha.dtype)
    sub_weights = sub_weights.to(device=alpha.device, dtype=alpha.dtype)

    result = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)
    for i in range(n_sub):
        # s_i maps to alpha * s_i on [0, alpha]
        s = alpha * sub_nodes[i]  # [P]

        # Evaluate r at s via Lagrange interpolation (using projected c)
        r_s = torch.zeros(P, *S, device=coeffs.device, dtype=coeffs.dtype)
        for j in range(K):
            L_j = torch.ones(P, device=alpha.device, dtype=alpha.dtype)
            for m in range(K):
                if m != j:
                    L_j = L_j * (s - nodes[m]) / (nodes[j] - nodes[m])
            L_j_nd = L_j.reshape(P, *([1] * len(S)))
            r_s = r_s + L_j_nd * c[j].unsqueeze(0)

        # alpha * w_i * r(alpha * s_i)  (Jacobian from change of variable)
        a_nd = alpha.reshape(P, *([1] * len(S)))
        result = result + sub_weights[i] * r_s * a_nd

    return result


# ---------------------------------------------------------------------------
# trajectory_state: x(alpha) = x_i + alpha * dt * v_base + dt * q(alpha)
# ---------------------------------------------------------------------------


def trajectory_state(
    x_i: Tensor,
    v_base: Tensor,
    coeffs: Tensor,
    alpha: Tensor,
    dt: float,
    basis_type: str,
    **kwargs,
) -> Tensor:
    """Compute on-policy trajectory state x(alpha).

    x(alpha) = x_i + alpha * dt * v_base + dt * q(alpha)

    Args:
        x_i: [*S] starting state.
        v_base: [*S] learnable base velocity.
        coeffs: [K, *S] learnable coefficients.
        alpha: [P] query points in [0, 1].
        dt: scalar (sigma_j - sigma_i).
        basis_type: Basis type string.
        **kwargs: Passed to evaluate_q.

    Returns:
        [P, *S] trajectory states.
    """
    P = alpha.shape[0]
    S = x_i.shape

    # alpha: [P] -> [P, 1, 1, ...]
    a_nd = alpha.reshape(P, *([1] * len(S)))

    # q: [P, *S]
    q = evaluate_q(basis_type, coeffs, alpha, **kwargs)

    # x(alpha) = x_i + alpha * dt * v_base + dt * q(alpha)
    return x_i.unsqueeze(0) + a_nd * dt * v_base.unsqueeze(0) + dt * q


# ---------------------------------------------------------------------------
# evaluate_v_pred: v(alpha) = v_base + r(alpha)
# ---------------------------------------------------------------------------


def evaluate_v_pred(
    v_base: Tensor,
    coeffs: Tensor,
    alpha: Tensor,
    basis_type: str,
    **kwargs,
) -> Tensor:
    """Compute predicted velocity v(alpha) = v_base + r(alpha).

    Args:
        v_base: [*S] learnable base velocity.
        coeffs: [K, *S] learnable coefficients.
        alpha: [P] query points in [0, 1].
        basis_type: Basis type string.
        **kwargs: Passed to evaluate_r.

    Returns:
        [P, *S] predicted velocities.
    """
    r = evaluate_r(basis_type, coeffs, alpha, **kwargs)  # [P, *S]
    return v_base.unsqueeze(0) + r


# ---------------------------------------------------------------------------
# Utility: get nodes and weights for a given basis type
# ---------------------------------------------------------------------------


def get_nodes_weights(basis_type: str, K: int, **kwargs) -> dict:
    """Return nodes/weights dict needed by evaluate_r/evaluate_q for grid-based bases.

    For bubble_path and spectral_zero_mean, returns only ``family``.
    For centered_dx and collocation, returns ``nodes``, ``weights`` tensors.

    Args:
        basis_type: Basis type string.
        K: Number of basis terms.
        **kwargs: ``family`` for bubble_path, ``scheme`` for collocation.

    Returns:
        dict of keyword arguments to pass to evaluate_r / evaluate_q.
    """
    if basis_type == "bubble_path":
        return {"family": kwargs.get("family", "sin_path")}
    elif basis_type == "spectral_zero_mean":
        return {}
    elif basis_type == "centered_dx":
        from .quadrature import uniform_nodes_weights

        nodes, weights = uniform_nodes_weights(K)
        return {"nodes": nodes.float(), "weights": weights.float()}
    elif basis_type == "collocation":
        scheme = kwargs.get("scheme", "gauss_legendre")
        if scheme == "gauss_legendre":
            from .quadrature import gauss_legendre_nodes_weights

            nodes, weights = gauss_legendre_nodes_weights(K)
        elif scheme == "chebyshev":
            from .quadrature import chebyshev_nodes_weights

            nodes, weights = chebyshev_nodes_weights(K)
        elif scheme == "uniform":
            from .quadrature import uniform_collocation_nodes_weights

            nodes, weights = uniform_collocation_nodes_weights(K)
        else:
            raise ValueError(f"Unknown collocation scheme: {scheme!r}")
        return {"nodes": nodes.float(), "weights": weights.float()}
    else:
        raise ValueError(f"Unknown basis_type={basis_type!r}")


# ---------------------------------------------------------------------------
# All basis configurations (mirrors fitting.ALL_BASIS_CONFIGS)
# ---------------------------------------------------------------------------

ALL_BASIS_CONFIGS_ONPOLICY = [
    ("bubble_path", "sin_path"),
    ("bubble_path", "bubble_poly"),
    ("bubble_path", "taylor_path"),
    ("spectral_zero_mean", "cosine"),
    ("centered_dx", "(grid)"),
    ("collocation", "GL"),
    ("collocation", "chebyshev"),
]
"""List of (basis_type, family) tuples for on-policy experiments."""


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Sanity harness for on-policy basis fitting functions.
    #
    # Checks: evaluate_r/q shapes, known analytic values, gradient flow,
    # trajectory_state endpoints, project_zero_mean idempotence,
    # small optimization convergence, consistency with fitting.py.
    # Pure CPU, no GPU.
    # ------------------------------------------------------------------
    import math

    import torch
    from smd.methods.simplified_distill.basis.fitting import build_basis_matrix_spectral

    print("=== fitting_onpolicy sanity ===")

    def _make_basis_kwargs(basis_type, family, K):
        if family == "chebyshev":
            nw = get_nodes_weights(basis_type, K, family=family, scheme="chebyshev")
        else:
            nw = get_nodes_weights(basis_type, K, family=family, scheme="gauss_legendre")
        if basis_type == "bubble_path":
            return {"family": family}
        elif basis_type == "spectral_zero_mean":
            return {}
        else:
            return nw

    # 1. evaluate_r shape for all basis types
    for basis_type, family in ALL_BASIS_CONFIGS_ONPOLICY:
        K, B, C, H, W, P = 4, 2, 3, 4, 5, 10
        if basis_type == "centered_dx":
            K = max(K, 2)
        coeffs = torch.randn(K, B, C, H, W)
        alpha = torch.linspace(0.1, 0.9, P)
        kwargs = _make_basis_kwargs(basis_type, family, K)
        r = evaluate_r(basis_type, coeffs, alpha, **kwargs)
        assert r.shape == (P, B, C, H, W), f"{basis_type}/{family}: shape {r.shape}"
    print("  evaluate_r shapes OK (all 7 configs)")

    # 2. Known analytic values
    # bubble/sin: K=1, b=1 => r(0.5) = pi*cos(pi*0.5) = 0
    c1 = torch.ones(1, 1)
    r_val = evaluate_r("bubble_path", c1, torch.tensor([0.5]), family="sin_path").item()
    assert abs(r_val - math.pi * math.cos(math.pi * 0.5)) < 1e-6
    # bubble/sin: q(0.5) = sin(pi*0.5) = 1
    q_val = evaluate_q("bubble_path", c1, torch.tensor([0.5]), family="sin_path").item()
    assert abs(q_val - 1.0) < 1e-6
    # spectral: K=1, c=1 => r(0) = cos(0) = 1
    r_sp = evaluate_r("spectral_zero_mean", c1, torch.tensor([0.0])).item()
    assert abs(r_sp - 1.0) < 1e-6
    # taylor_path: K=1, b=1 => r(0.5) = 1-2*0.5 = 0
    r_tp = evaluate_r("bubble_path", c1, torch.tensor([0.5]), family="taylor_path").item()
    assert abs(r_tp) < 1e-6
    print("  known analytic values OK")

    # 3. q(0) = 0 for all basis types
    for basis_type, family in ALL_BASIS_CONFIGS_ONPOLICY:
        K_ = 4 if basis_type != "centered_dx" else max(4, 2)
        c_ = torch.randn(K_, 2, 3)
        kw = _make_basis_kwargs(basis_type, family, K_)
        q_ = evaluate_q(basis_type, c_, torch.tensor([0.0]), **kw)
        assert q_.abs().max().item() < 1e-6, f"q(0)!=0 for {basis_type}/{family}"
    print("  q(0)=0 OK (all configs)")

    # 4. Gradient flow for all basis types
    for basis_type, family in ALL_BASIS_CONFIGS_ONPOLICY:
        K_ = 4 if basis_type != "centered_dx" else max(4, 2)
        vb = torch.randn(2, 3, requires_grad=True)
        cc = torch.randn(K_, 2, 3, requires_grad=True)
        alpha_ = torch.tensor([0.2, 0.5, 0.8])
        kw = _make_basis_kwargs(basis_type, family, K_)
        vp = evaluate_v_pred(vb, cc, alpha_, basis_type, **kw)
        vp.pow(2).sum().backward()
        assert vb.grad is not None and vb.grad.abs().sum() > 0, f"no grad v_base {basis_type}/{family}"
        assert cc.grad is not None and cc.grad.abs().sum() > 0, f"no grad coeffs {basis_type}/{family}"
    print("  gradient flow OK (all configs)")

    # 5. trajectory_state endpoints
    x_i = torch.randn(2, 3)
    vb = torch.randn(2, 3)
    cc = torch.zeros(4, 2, 3)
    dt = -0.5
    x0 = trajectory_state(x_i, vb, cc, torch.tensor([0.0]), dt=dt, basis_type="spectral_zero_mean")
    x1 = trajectory_state(x_i, vb, cc, torch.tensor([1.0]), dt=dt, basis_type="spectral_zero_mean")
    torch.testing.assert_close(x0.squeeze(0), x_i, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x1.squeeze(0), x_i + dt * vb, atol=1e-6, rtol=1e-6)
    print("  trajectory_state endpoints OK")

    # 6. project_zero_mean idempotent
    from smd.methods.simplified_distill.basis.quadrature import gauss_legendre_nodes_weights

    _, ww = gauss_legendre_nodes_weights(6)
    ww = ww.float()
    cc2 = torch.randn(6, 3, 4)
    p1 = project_zero_mean(cc2, ww)
    p2 = project_zero_mean(p1, ww)
    torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)
    print("  project_zero_mean idempotent OK")

    # 7. Consistency with fitting.py (spectral)
    L_, K2_ = 50, 4
    alphas_ = torch.linspace(0.0, 1.0, L_, dtype=torch.float64)
    Phi_ = build_basis_matrix_spectral(alphas_, K2_, family="cosine")
    cc3 = torch.randn(K2_, 6, dtype=torch.float64)
    r_fit = Phi_ @ cc3
    r_onp = evaluate_r("spectral_zero_mean", cc3, alphas_)
    torch.testing.assert_close(r_onp, r_fit, atol=1e-10, rtol=1e-10)
    print("  consistency with fitting.py OK (spectral)")

    # 8. Small optimization convergence
    torch.manual_seed(42)
    D_ = 8
    v0_true = torch.randn(D_)
    A_true = 0.5 * torch.randn(D_)
    v_base_p = torch.nn.Parameter(torch.zeros(D_))
    coeffs_p = torch.nn.Parameter(torch.zeros(1, D_))
    opt = torch.optim.Adam([v_base_p, coeffs_p], lr=1e-2)
    losses = []
    for step in range(200):
        ak = torch.rand(16)
        vp = evaluate_v_pred(v_base_p, coeffs_p, ak, "spectral_zero_mean")
        vt = (v0_true.unsqueeze(0) + A_true.unsqueeze(0) * torch.cos(math.pi * ak.reshape(-1, 1))).detach()
        loss = (vp - vt).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    init_l = sum(losses[:5]) / 5
    final_l = sum(losses[-5:]) / 5
    assert final_l < init_l / 10, f"optimization did not converge: {init_l:.4e} -> {final_l:.4e}"
    print(f"  optimization convergence OK ({init_l:.4e} -> {final_l:.4e}, {init_l / final_l:.0f}x)")

    print("=== fitting_onpolicy sanity PASSED ===")
