# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable trajectory basis for pivot distillation.

Use ``create_basis(config)`` to instantiate the correct basis from config.
All bases implement the unified ``PivotBasis`` interface — model.py
needs no branching by basis type.

Available parameterizations:
  - "residual"   -> dispatches to ResidualBasis subclass via config.residual_basis
"""

from __future__ import annotations

from typing import Any


def create_basis(config: Any):
    """Factory: create a basis object from config.

    Args:
        config: Model config with ``pivot_parameterization`` field.
            Reads ``residual_basis``, ``basis_family``,
            ``num_basis_terms``, ``residual_scale_mode``, ``pivot_residual_scale``.

    Returns:
        A ``PivotBasis`` subclass instance.
    """
    name = str(getattr(config, "pivot_parameterization", "residual"))

    if name == "residual":
        return _create_residual_basis(config)

    else:
        raise ValueError(f"Unknown pivot_parameterization={name!r}. Supported: 'residual'.")


def _create_residual_basis(config: Any):
    """Dispatch to the correct ResidualBasis subclass."""
    basis_type = str(getattr(config, "residual_basis", "bubble_path"))
    family = str(getattr(config, "basis_family", "sin_path"))
    num_terms = int(getattr(config, "num_basis_terms", 4))
    scale_mode = str(getattr(config, "residual_scale_mode", "sqrt_n"))
    residual_scale = float(getattr(config, "pivot_residual_scale", 1.0))

    if basis_type == "bubble_path":
        from .bubble_path import BubblePathResidual

        return BubblePathResidual(
            num_terms=num_terms,
            family=family,
            scale_mode=scale_mode,
            residual_scale=residual_scale,
        )

    elif basis_type == "spectral_zero_mean":
        from .spectral_zero_mean import SpectralZeroMeanResidual

        return SpectralZeroMeanResidual(
            num_terms=num_terms,
            family=family,
            scale_mode=scale_mode,
            residual_scale=residual_scale,
        )

    elif basis_type == "centered_dx":
        from .centered_dx import CenteredDXResidual

        return CenteredDXResidual(n_nodes=num_terms)

    elif basis_type == "collocation":
        scheme = str(getattr(config, "collocation_scheme", "gauss_legendre"))
        from .collocation import CollocationResidual

        return CollocationResidual(
            n_nodes=num_terms,
            scheme=scheme,
            scale_mode=scale_mode,
            residual_scale=residual_scale,
        )

    else:
        raise ValueError(
            f"Unknown residual_basis={basis_type!r}. "
            "Supported: 'bubble_path', 'spectral_zero_mean', 'centered_dx', 'collocation'."
        )


def is_residual_basis(basis: Any) -> bool:
    """Check if a basis is a ResidualBasis (zero-mean residual interface).

    With the unified PivotBasis framework, this is mainly useful for tests
    and diagnostics.  model.py no longer branches on this.
    """
    from .base import ResidualBasis

    return isinstance(basis, ResidualBasis)


def get_proj_out_init_mode(basis: Any) -> str:
    """Get proj_out initialization mode from a basis object."""
    return getattr(basis, "proj_out_init_mode", "copy")


def get_proj_out_init_plan(basis: Any) -> list[str] | None:
    """Get per-slot init plan for hybrid mode. None for non-hybrid."""
    if getattr(basis, "proj_out_init_mode", "copy") == "hybrid":
        return basis.proj_out_init_plan
    return None
