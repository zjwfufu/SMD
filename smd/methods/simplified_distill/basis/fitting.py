# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure math / basis fitting library for pivot distillation diagnostics.

This module provides model-agnostic functions for:
- Constructing basis matrices (bubble path, spectral, centered DX, collocation)
- Least-squares fitting of residual velocities
- Evaluating fit quality (residual MSE, trajectory MSE, endpoint error)
- Computing v_base and residuals from dense teacher trajectories

All functions operate on plain tensors and require no model or GPU.
"""

from __future__ import annotations

import logging
import math
import time

import torch
from torch import Tensor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v_base and residual computation
# ---------------------------------------------------------------------------


def compute_vbase_and_residuals(traj: dict) -> dict:
    """Compute v_base_gt (trapezoidal weighted mean) and residuals dv_T.

    v_base_gt = (1 / (sigma_j - sigma_i)) * integral_{sigma_i}^{sigma_j} v(x(s), s) ds
              ~= trapezoidal weighted mean of vs

    But since alphas are uniformly spaced, this simplifies to trapezoidal
    weights on the alpha grid for the velocity samples.

    Actually, v_base should be the average velocity that makes the trajectory
    exact at the endpoints: v_base = (x_j - x_i) / (sigma_j - sigma_i).
    This is the "numerical" v_base. We also compute the trapezoidal weighted
    mean as a sanity check.
    """
    xs = traj["xs"]  # [N+1, B, C, H, W]
    vs = traj["vs"]  # [N, B, C, H, W]
    _ = traj["alphas"]  # [N+1] (used by caller)
    dt = traj["dt"]  # scalar

    N = vs.shape[0]

    # Numerical v_base from endpoints
    x_i = xs[0]
    x_j = xs[-1]
    # v_base = (x_j - x_i) / (sigma_j - sigma_i) = (x_j - x_i) / dt
    # Note: dt = sigma_j - sigma_i is negative, so this gives the correct sign
    v_base_numerical = (x_j - x_i) / dt

    # Trapezoidal weighted mean of velocities
    # Weights: half at first, full in middle, half at last
    # But vs has N entries (one per step, not N+1). The velocity at step l
    # is measured at alpha_l, and we have N of them (for l=0..N-1).
    # Use trapezoidal weights on [0, (N-1)/N] grid
    trap_weights = torch.ones(N, device=vs.device, dtype=torch.float64) / N
    trap_weights[0] /= 2.0
    trap_weights[-1] /= 2.0
    # Re-normalize
    trap_weights = trap_weights / trap_weights.sum()
    w = trap_weights.reshape(N, *([1] * (vs.ndim - 1)))
    v_base_trap = (w.to(dtype=vs.dtype) * vs).sum(dim=0)  # [B, C, H, W]

    # Residuals: dv_T_l = v_l - v_base for each step
    dv_numerical = vs - v_base_numerical.unsqueeze(0)  # [N, B, C, H, W]
    dv_trap = vs - v_base_trap.unsqueeze(0)  # [N, B, C, H, W]

    return {
        "v_base_numerical": v_base_numerical,  # [B, C, H, W]
        "v_base_trap": v_base_trap,  # [B, C, H, W]
        "dv_numerical": dv_numerical,  # [N, B, C, H, W]
        "dv_trap": dv_trap,  # [N, B, C, H, W]
        "N": N,
    }


# ---------------------------------------------------------------------------
# Basis matrix construction
# ---------------------------------------------------------------------------


def build_basis_matrix(basis_type: str, alphas: Tensor, K: int, **kwargs) -> Tensor:
    """Unified dispatcher for basis matrix construction.

    Args:
        basis_type: One of "bubble_path", "spectral_zero_mean", "centered_dx", "collocation".
        alphas: [L] alpha grid values.
        K: Number of basis terms.
        **kwargs: Passed to the per-type builder (e.g., ``family=``, ``scheme=``).

    Returns:
        [L, K] basis matrix (float64).
    """
    if basis_type == "bubble_path":
        return build_basis_matrix_bubble_path(alphas, K, **kwargs)
    elif basis_type == "spectral_zero_mean":
        return build_basis_matrix_spectral(alphas, K, **kwargs)
    elif basis_type == "centered_dx":
        return build_basis_matrix_centered_dx(alphas, K)
    elif basis_type == "collocation":
        return build_basis_matrix_collocation(alphas, K, **kwargs)
    else:
        raise ValueError(
            f"Unknown basis_type={basis_type!r}. "
            "Supported: 'bubble_path', 'spectral_zero_mean', 'centered_dx', 'collocation'."
        )


def build_basis_matrix_bubble_path(
    alphas: Tensor,
    K: int,
    family: str = "sin_path",
) -> Tensor:
    """Construct Phi[l, k] = phi_k'(alpha_l) for BubblePath basis.

    The velocity residual r(alpha) = sum_k b_k * phi_k'(alpha) where
    phi_k are path-basis functions with phi_k(0) = phi_k(1) = 0.

    Returns: [L, K] matrix (float64).
    """
    L = len(alphas)
    Phi = torch.zeros(L, K, dtype=torch.float64, device=alphas.device)
    for k_idx in range(K):
        k = k_idx + 1  # 1-indexed
        if family == "sin_path":
            Phi[:, k_idx] = k * math.pi * torch.cos(k * math.pi * alphas.double())
        elif family == "bubble_poly":
            a = alphas.double()
            Phi[:, k_idx] = k * (a ** (k - 1)) * (1.0 - a) - a**k
        elif family == "taylor_path":
            a = alphas.double()
            u = a * (1.0 - a)
            u_prime = 1.0 - 2.0 * a
            if k > 1:
                v = (2.0 * a - 1.0) ** (k - 1)
                v_prime = 2.0 * (k - 1) * (2.0 * a - 1.0) ** max(k - 2, 0)
            else:
                v = torch.ones_like(a)
                v_prime = torch.zeros_like(a)
            Phi[:, k_idx] = u_prime * v + u * v_prime
        else:
            raise ValueError(f"Unknown BubblePath family: {family}")
    return Phi


def build_basis_matrix_spectral(
    alphas: Tensor,
    K: int,
    family: str = "cosine",
) -> Tensor:
    """Construct Phi[l, k] = cos(k * pi * alpha_l) for SpectralZeroMean basis.

    Returns: [L, K] matrix (float64).
    """
    L = len(alphas)
    Phi = torch.zeros(L, K, dtype=torch.float64, device=alphas.device)
    for k_idx in range(K):
        k = k_idx + 1
        if family == "cosine":
            Phi[:, k_idx] = torch.cos(k * math.pi * alphas.double())
        else:
            raise ValueError(f"Unknown Spectral family: {family}")
    return Phi


def build_basis_matrix_centered_dx(
    alphas: Tensor,
    K: int,
) -> Tensor:
    """Construct Phi[l, k] for CenteredDX: piecewise linear hats on a uniform grid.

    The K grid nodes are uniformly spaced in [0, 1]. Phi[l, k] is the hat
    function value at alpha_l centered at node k. After construction, we
    project each column to zero mean (weighted centering with trapezoidal weights).

    Returns: [L, K] matrix (float64).
    """
    from smd.methods.simplified_distill.basis.quadrature import uniform_nodes_weights

    nodes, weights = uniform_nodes_weights(K)
    nodes = nodes.to(device=alphas.device)
    weights = weights.to(device=alphas.device)
    a = alphas.double()

    L = len(a)
    Phi = torch.zeros(L, K, dtype=torch.float64, device=a.device)

    # Piecewise linear interpolation: each column k is the hat function at node k
    # For uniform nodes with spacing h = 1/(K-1), the hat at node k is:
    #   1 at node k, linearly decreasing to 0 at nodes k-1 and k+1
    for k_idx in range(K):
        # Map alpha to index space
        idx_float = a * (K - 1)
        # Hat function: max(0, 1 - |idx_float - k_idx|)
        Phi[:, k_idx] = torch.clamp(1.0 - torch.abs(idx_float - k_idx), min=0.0)

    # Zero-mean centering: project each column so integral_0^1 phi_k(alpha) dalpha ~= 0.
    # Use trapezoidal weights on the dense alpha query grid for the integral.
    h_query = 1.0 / (L - 1) if L > 1 else 1.0
    query_w = torch.full((L,), h_query, dtype=torch.float64, device=a.device)
    query_w[0] /= 2.0
    query_w[-1] /= 2.0
    # Per-column weighted integral: [K]
    col_integral = (query_w.reshape(L, 1) * Phi).sum(dim=0, keepdim=True)  # [1, K]
    # Subtract so each basis function has zero mean
    Phi = Phi - col_integral  # broadcast [L, K] - [1, K]

    return Phi


def build_basis_matrix_collocation(
    alphas: Tensor,
    K: int,
    scheme: str = "gauss_legendre",
) -> Tensor:
    """Construct Phi[l, k] for Collocation: Lagrange interpolation at quadrature nodes.

    Returns: [L, K] matrix (float64).
    """
    from smd.methods.simplified_distill.basis.quadrature import gauss_legendre_nodes_weights, uniform_nodes_weights

    if scheme == "gauss_legendre":
        nodes, weights = gauss_legendre_nodes_weights(K)
    elif scheme == "uniform":
        nodes, weights = uniform_nodes_weights(K)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    nodes = nodes.to(device=alphas.device)
    weights = weights.to(device=alphas.device)

    a = alphas.double()
    L = len(a)
    Phi = torch.zeros(L, K, dtype=torch.float64, device=a.device)

    # Lagrange basis: L_j(alpha) = prod_{m != j} (alpha - node_m) / (node_j - node_m)
    for j in range(K):
        basis_j = torch.ones(L, dtype=torch.float64, device=a.device)
        for m in range(K):
            if m != j:
                basis_j = basis_j * (a - nodes[m]) / (nodes[j] - nodes[m])
        Phi[:, j] = basis_j

    # Zero-mean centering: project each column so integral_0^1 L_k(alpha) dalpha ~= 0.
    # Use trapezoidal weights on the dense alpha query grid.
    h_query = 1.0 / (L - 1) if L > 1 else 1.0
    query_w = torch.full((L,), h_query, dtype=torch.float64, device=a.device)
    query_w[0] /= 2.0
    query_w[-1] /= 2.0
    col_integral = (query_w.reshape(L, 1) * Phi).sum(dim=0, keepdim=True)  # [1, K]
    Phi = Phi - col_integral

    return Phi


# ---------------------------------------------------------------------------
# Least-squares fitting
# ---------------------------------------------------------------------------


def fit_basis(
    Phi: Tensor,
    dv_flat: Tensor,
) -> Tensor:
    """Solve least-squares: min ||Phi @ c - dv_flat||^2.

    Args:
        Phi: [L, K] basis matrix (float64).
        dv_flat: [L, D] residual velocity samples, flattened spatial dims (float64).

    Returns:
        coeffs: [K, D] fitted coefficients (float64).
    """
    # torch.linalg.lstsq: A @ X = B -> X = lstsq(A, B).solution
    result = torch.linalg.lstsq(Phi, dv_flat)
    return result.solution  # [K, D]


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------


def evaluate_fit(
    Phi: Tensor,
    coeffs: Tensor,
    dv_flat: Tensor,
    xs_gt: Tensor,
    alphas: Tensor,
    x_i: Tensor,
    dt: float,
    v_base: Tensor,
) -> dict:
    """Evaluate the quality of a basis fit.

    Args:
        Phi: [L, K] basis matrix for velocity residual.
        coeffs: [K, D] fitted coefficients.
        dv_flat: [L, D] ground-truth residual velocities.
        xs_gt: [N+1, B, C, H, W] ground-truth trajectory.
        alphas: [N+1] alpha grid.
        x_i: [B, C, H, W] segment start.
        dt: sigma_j - sigma_i (scalar).
        v_base: [B, C, H, W] average velocity.

    Returns:
        dict with residual_mse, trajectory_mse, endpoint_error, x_fitted.
    """
    L = Phi.shape[0]

    # Reconstructed residual velocity at sample points
    r_fitted_flat = Phi @ coeffs  # [L, D]
    residual_mse = float(((r_fitted_flat - dv_flat) ** 2).mean().item())

    # Reconstruct trajectory via x(alpha) = x_i + alpha * dt * v_base + q(alpha)
    # where q(alpha) = dt * integral_0^alpha r(s) ds
    B_shape = x_i.shape  # [B, C, H, W]

    # Reshape fitted residual for trajectory reconstruction.
    # Use float32 for the cumulative integration to avoid bf16 accumulation errors.
    compute_dtype = torch.float32
    r_fitted_reshaped = r_fitted_flat.reshape(L, *B_shape).to(dtype=compute_dtype)

    # Cumulative trapezoidal integration for q at trajectory points.
    # r is sampled at alpha_0..alpha_{N-1} (L = N points).
    # q[0] = 0
    # q[l+1] = q[l] + 0.5 * (r[l] + r[l+1]) * h    for l < N-1 (true trapezoidal)
    # q[N]   = q[N-1] + r[N-1] * h                   last step uses rectangle rule
    #          (no r[N] sample available)
    h = float(alphas[1] - alphas[0])  # uniform spacing
    N = L
    q_traj = torch.zeros(N + 1, *B_shape, device=x_i.device, dtype=compute_dtype)
    for step in range(N):
        if step < N - 1:
            # Trapezoidal: average of consecutive r values
            q_traj[step + 1] = q_traj[step] + 0.5 * (r_fitted_reshaped[step] + r_fitted_reshaped[step + 1]) * h
        else:
            # Last sub-interval: only have r[N-1], use rectangle rule
            q_traj[step + 1] = q_traj[step] + r_fitted_reshaped[step] * h

    # Fitted trajectory: x(alpha_l) = x_i + alpha_l * dt * v_base + dt * q(alpha_l)
    alpha_col = alphas.to(dtype=compute_dtype).reshape(N + 1, *([1] * len(B_shape)))
    x_i_f = x_i.to(dtype=compute_dtype)
    v_base_f = v_base.to(dtype=compute_dtype)
    x_fitted = x_i_f.unsqueeze(0) + alpha_col * dt * v_base_f.unsqueeze(0) + dt * q_traj

    # Trajectory MSE: average over all points
    traj_mse = float(((x_fitted - xs_gt.to(dtype=compute_dtype)) ** 2).mean().item())

    # Endpoint error: ||x_fitted(1) - x_gt(1)||
    endpoint_err = float(((x_fitted[-1] - xs_gt[-1].to(dtype=compute_dtype)) ** 2).mean().sqrt().item())

    # Cast back to input dtype for downstream use (e.g. VAE decode)
    x_fitted = x_fitted.to(dtype=x_i.dtype)

    return {
        "residual_mse": residual_mse,
        "trajectory_mse": traj_mse,
        "endpoint_error": endpoint_err,
        "x_fitted": x_fitted,
    }


# ---------------------------------------------------------------------------
# Basis configurations and main fitting loop
# ---------------------------------------------------------------------------

# Basis configurations: (type, family/scheme, matrix_builder)
ALL_BASIS_CONFIGS = [
    ("bubble_path", "sin_path", build_basis_matrix_bubble_path),
    ("bubble_path", "bubble_poly", build_basis_matrix_bubble_path),
    ("spectral_zero_mean", "cosine", build_basis_matrix_spectral),
    ("centered_dx", "(grid)", build_basis_matrix_centered_dx),
    ("collocation", "GL", build_basis_matrix_collocation),
]


def run_fitting(
    traj: dict,
    decomp: dict,
    K_values: list[int],
    use_numerical_vbase: bool = True,
) -> tuple[list[dict], Tensor | None]:
    """Run basis fitting for all configurations and K values.

    Returns:
        results: list of result dicts (one per basis_type × K combination).
        best_x_fitted: [N+1, B, C, H, W] fitted trajectory from the
            configuration with the lowest residual_mse, or ``None`` if
            no configurations were evaluated.
    """
    vs = traj["vs"]  # [N, B, C, H, W]
    xs = traj["xs"]  # [N+1, B, C, H, W]
    alphas = traj["alphas"]  # [N+1]
    dt = float(traj["dt"])

    N = vs.shape[0]

    v_base = decomp["v_base_numerical"] if use_numerical_vbase else decomp["v_base_trap"]
    dv = decomp["dv_numerical"] if use_numerical_vbase else decomp["dv_trap"]

    x_i = xs[0]  # [B, C, H, W]
    B_shape = x_i.shape

    # Alpha grid for velocity samples: alphas[0:N]
    alpha_v = alphas[:N]  # [N]

    # Flatten dv: [N, B, C, H, W] -> [N, D]
    D = int(torch.tensor(B_shape).prod().item())
    dv_flat = dv.reshape(N, D).double()

    results = []
    best_residual_mse = float("inf")
    best_x_fitted: Tensor | None = None

    for basis_type, family, matrix_builder in ALL_BASIS_CONFIGS:
        for K in K_values:
            # CenteredDX needs at least 2 nodes
            if basis_type == "centered_dx" and K < 2:
                continue

            t0 = time.time()

            # Build basis matrix
            if basis_type == "bubble_path":
                Phi = matrix_builder(alpha_v, K, family=family)
            elif basis_type == "spectral_zero_mean":
                Phi = matrix_builder(alpha_v, K, family=family)
            elif basis_type == "centered_dx":
                Phi = matrix_builder(alpha_v, K)
            elif basis_type == "collocation":
                scheme = "gauss_legendre" if family == "GL" else "uniform"
                Phi = matrix_builder(alpha_v, K, scheme=scheme)
            else:
                raise ValueError(f"Unknown basis type: {basis_type}")

            # Least-squares fit
            coeffs = fit_basis(Phi, dv_flat)  # [K, D]

            elapsed = time.time() - t0

            # Evaluate quality
            metrics = evaluate_fit(
                Phi=Phi,
                coeffs=coeffs,
                dv_flat=dv_flat,
                xs_gt=xs,
                alphas=alphas,
                x_i=x_i,
                dt=dt,
                v_base=v_base,
            )

            # Track best-fit trajectory (lowest residual_mse)
            if metrics["residual_mse"] < best_residual_mse:
                best_residual_mse = metrics["residual_mse"]
                best_x_fitted = metrics["x_fitted"].clone()

            results.append(
                {
                    "basis_type": basis_type,
                    "family": family,
                    "K": K,
                    "residual_mse": metrics["residual_mse"],
                    "trajectory_mse": metrics["trajectory_mse"],
                    "endpoint_error": metrics["endpoint_error"],
                    "time_s": elapsed,
                }
            )
            log.info(
                f"  {basis_type:20s} {family:12s} K={K:3d}  "
                f"resid={metrics['residual_mse']:.3e}  "
                f"traj={metrics['trajectory_mse']:.3e}  "
                f"endpt={metrics['endpoint_error']:.3e}  "
                f"({elapsed:.2f}s)"
            )

    return results, best_x_fitted


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Sanity harness for basis fitting (offline / dense-trajectory mode).
    #
    # 1. Build a synthetic nonlinear trajectory (sinusoidal velocity residual)
    # 2. compute_vbase_and_residuals + run_fitting over all basis types, K=2..8
    # 3. Verify: exact recovery with spectral basis, monotone residual_mse in K,
    #    endpoint consistency, zero-mean column integrals for centered_dx/collocation
    # ------------------------------------------------------------------
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== fitting sanity ===")

    # ---- helpers ----
    def _make_nonlinear_traj(N=100, B=1, C=1, H=2, W=2):
        torch.manual_seed(42)
        dt = -0.5
        alphas = torch.linspace(0.0, 1.0, N + 1, dtype=torch.float32)
        sigmas = 1.0 + alphas * dt
        x_i = torch.randn(B, C, H, W)
        v_base_ = torch.randn(B, C, H, W)
        A = 0.3 * torch.randn(B, C, H, W)
        xs = torch.zeros(N + 1, B, C, H, W)
        vs = torch.zeros(N, B, C, H, W)
        xs[0] = x_i
        for idx in range(N):
            a = alphas[idx]
            vs[idx] = v_base_ + A * math.sin(2 * math.pi * float(a))
            d_sigma = float(sigmas[idx + 1] - sigmas[idx])
            xs[idx + 1] = xs[idx] + vs[idx] * d_sigma
        return {"alphas": alphas, "sigmas": sigmas, "xs": xs, "vs": vs, "dt": dt}

    # ---- 1. v_base endpoint consistency ----
    traj = _make_nonlinear_traj(N=50)
    decomp = compute_vbase_and_residuals(traj)
    x_i = traj["xs"][0]
    x_j = traj["xs"][-1]
    x_j_recon = x_i + traj["dt"] * decomp["v_base_numerical"]
    err = (x_j_recon - x_j).abs().max().item()
    assert err < 1e-4, f"v_base endpoint error: {err}"
    print(f"  v_base endpoint consistency: max_err={err:.2e}")

    # ---- 2. lstsq exact recovery (spectral) ----
    L, K_ = 100, 3
    alphas_ = torch.linspace(0.0, 1.0, L, dtype=torch.float64)
    Phi_ = build_basis_matrix_spectral(alphas_, K_, family="cosine")
    torch.manual_seed(123)
    true_c = torch.randn(K_, 8, dtype=torch.float64)
    dv_flat = Phi_ @ true_c
    rec_c = fit_basis(Phi_, dv_flat)
    torch.testing.assert_close(rec_c, true_c, atol=1e-8, rtol=1e-8)
    print("  lstsq exact recovery OK (spectral/cosine K=3)")

    # ---- 3. zero-mean column integrals for centered_dx / collocation ----
    for btype, bfn in [
        ("centered_dx", build_basis_matrix_centered_dx),
        ("collocation", build_basis_matrix_collocation),
    ]:
        L2 = 200
        alphas2 = torch.linspace(0.0, 1.0, L2, dtype=torch.float64)
        if btype == "centered_dx":
            Phi2 = bfn(alphas2, 4)
        else:
            Phi2 = bfn(alphas2, 4, scheme="gauss_legendre")
        h = 1.0 / (L2 - 1)
        w = torch.full((L2,), h, dtype=torch.float64)
        w[0] /= 2.0
        w[-1] /= 2.0
        col_int = (w.unsqueeze(1) * Phi2).sum(dim=0)
        assert col_int.abs().max().item() < 1e-10, f"{btype} columns not zero-mean: {col_int}"
        print(f"  {btype} zero-mean column integrals OK")

    # ---- 4. run_fitting monotone residual_mse ----
    traj2 = _make_nonlinear_traj(N=100)
    decomp2 = compute_vbase_and_residuals(traj2)
    K_values = [2, 4, 8]
    results, _ = run_fitting(traj2, decomp2, K_values=K_values, use_numerical_vbase=True)
    by_basis = {}
    for r in results:
        by_basis.setdefault((r["basis_type"], r["family"]), []).append(r)
    for key, rlist in by_basis.items():
        sr = sorted(rlist, key=lambda r: r["K"])
        for i in range(1, len(sr)):
            assert sr[i]["residual_mse"] <= sr[i - 1]["residual_mse"] + 1e-10, (
                f"{key} K={sr[i]['K']} worse than K={sr[i - 1]['K']}"
            )
    print(f"  run_fitting monotone residual_mse OK ({len(results)} configs)")

    print("=== fitting sanity PASSED ===")
