# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity harness for all residual basis classes.

Verifies mathematical invariants (zero-mean, boundary, degeneracy,
trajectory endpoints, windowed velocity convergence) for every basis
type using the real SMD config parameters.

Usage:
    python -m smd.methods.simplified_distill.basis
"""

import torch

from smd.methods.simplified_distill.basis import create_basis, is_residual_basis
from smd.methods.simplified_distill.basis.quadrature import gauss_legendre_nodes_weights

B, C, H, W = 2, 3, 4, 4
K = 4


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


basis_configs = [
    _Cfg(
        pivot_parameterization="residual",
        residual_basis="bubble_path",
        basis_family="sin_path",
        num_basis_terms=K,
        residual_scale_mode="sqrt_n",
        pivot_residual_scale=1.0,
    ),
    _Cfg(
        pivot_parameterization="residual",
        residual_basis="bubble_path",
        basis_family="bubble_poly",
        num_basis_terms=K,
        residual_scale_mode="sqrt_n",
        pivot_residual_scale=1.0,
    ),
    _Cfg(
        pivot_parameterization="residual",
        residual_basis="spectral_zero_mean",
        basis_family="cosine",
        num_basis_terms=K,
        residual_scale_mode="sqrt_n",
        pivot_residual_scale=1.0,
    ),
    _Cfg(
        pivot_parameterization="residual",
        residual_basis="centered_dx",
        num_basis_terms=10,
        residual_scale_mode="sqrt_n",
        pivot_residual_scale=1.0,
    ),
    _Cfg(
        pivot_parameterization="residual",
        residual_basis="collocation",
        num_basis_terms=6,
        collocation_scheme="gauss_legendre",
        residual_scale_mode="sqrt_n",
        pivot_residual_scale=1.0,
    ),
]

gl_nodes, gl_weights = gauss_legendre_nodes_weights(64)
gl_nodes, gl_weights = gl_nodes.float(), gl_weights.float()

print("=== basis sanity ===")
for cfg in basis_configs:
    label = f"{cfg.residual_basis}"
    if hasattr(cfg, "basis_family"):
        label += f"/{cfg.basis_family}"
    basis = create_basis(cfg)
    assert is_residual_basis(basis), f"{label}: not a ResidualBasis"
    n = basis.n_grid
    print(f"\n  [{label}] n_grid={n}")

    # -- random coeffs --
    flow_rand = torch.randn(B, n, C, H, W)
    x_i = torch.randn(B, C, H, W)
    sigma_i = torch.full((B,), 0.8)
    seg_size = torch.full((B,), 0.2)
    coeffs = basis.build_residual_coeffs(flow_rand, x_i, sigma_i, seg_size)

    # 1. Zero-mean
    integral = torch.zeros(B, C, H, W)
    for i in range(len(gl_nodes)):
        alpha = gl_nodes[i].reshape(1, 1, 1, 1).expand(B, 1, 1, 1)
        integral = integral + gl_weights[i] * basis.evaluate_r(coeffs, alpha)
    zm_err = integral.abs().max().item()
    tol = 1e-2 if cfg.residual_basis in ("centered_dx", "collocation") else 1e-4
    assert zm_err < tol, f"{label}: zero-mean violated, max|integral|={zm_err}"
    print(f"    zero-mean: max|integral|={zm_err:.2e} (tol={tol})")

    # 2. Boundary q(0) = q(1) = 0
    dt = torch.full((B, 1, 1, 1), -0.3)
    q0 = basis.evaluate_q(coeffs, torch.zeros(B, 1, 1, 1), dt).abs().max().item()
    q1 = basis.evaluate_q(coeffs, torch.ones(B, 1, 1, 1), dt).abs().max().item()
    q_tol = 1e-3 if cfg.residual_basis in ("centered_dx", "collocation") else 1e-5
    assert q0 < q_tol, f"{label}: q(0)={q0}"
    assert q1 < q_tol, f"{label}: q(1)={q1}"
    print(f"    boundary: q(0)={q0:.2e}, q(1)={q1:.2e}")

    # -- zero coeffs --
    flow_zero = torch.zeros(B, n, C, H, W)
    coeffs_z = basis.build_residual_coeffs(flow_zero, x_i, sigma_i, seg_size)

    # 3. Degeneracy
    alpha_rand = torch.rand(B, 1, 1, 1)
    r_z = basis.evaluate_r(coeffs_z, alpha_rand).abs().max().item()
    q_z = basis.evaluate_q(coeffs_z, alpha_rand, dt).abs().max().item()
    assert r_z < 1e-6, f"{label}: degeneracy r={r_z}"
    assert q_z < 1e-6, f"{label}: degeneracy q={q_z}"
    print(f"    degeneracy: r={r_z:.2e}, q={q_z:.2e}")

    # 4. Trajectory endpoints
    v_base = torch.randn(B, C, H, W)
    x0 = basis.trajectory_state(x_i, dt, v_base, coeffs_z, torch.zeros(B, 1, 1, 1))
    x1 = basis.trajectory_state(x_i, dt, v_base, coeffs_z, torch.ones(B, 1, 1, 1))
    torch.testing.assert_close(x0, x_i, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x1, x_i + dt * v_base, atol=1e-5, rtol=1e-5)
    print("    trajectory endpoints OK")

    # 5. Windowed velocity convergence
    alpha_s = torch.full((B, 1, 1, 1), 0.3)
    v_inst = v_base + basis.evaluate_r(coeffs, alpha_s)
    for d in [0.1, 0.01, 0.001]:
        v_win = basis.windowed_velocity(v_base, coeffs, alpha_s, alpha_s + d)
        err = (v_win - v_inst).abs().max().item()
    assert err < 0.1, f"{label}: windowed not converging at d=0.001, err={err}"
    print(f"    windowed velocity convergence OK (err@d=0.001={err:.4f})")

print("\n=== basis sanity PASSED ===")
