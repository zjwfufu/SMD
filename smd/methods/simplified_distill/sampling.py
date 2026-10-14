# SPDX-License-Identifier: Apache-2.0
"""Stateless segment query sampling, preserving the legacy RNG sequence."""

import torch


def sample_query_alphas(
    mode: str,
    *,
    iteration: int,
    segment_index: int,
    batch_size: int,
    num_query_points: int,
    device: torch.device,
    dtype: torch.dtype,
    shared_batch: bool = False,
) -> torch.Tensor:
    """Sample per-segment query alphas without changing the default RNG path."""
    if mode == "random":
        sample_batch = 1 if shared_batch else batch_size
        values = torch.rand(sample_batch, num_query_points, device=device, dtype=dtype)
        return values.expand(batch_size, -1) if shared_batch else values
    if mode != "random_endpoint":
        raise ValueError(f"query_alpha_mode must be 'random' or 'random_endpoint', got {mode!r}")
    if num_query_points != 1:
        raise ValueError(
            f"query_alpha_mode='random_endpoint' requires pivot_num_query_points=1, got {num_query_points}"
        )
    bucket = (int(iteration) * 2246822519 + int(segment_index) * 3266489917) % 1000
    endpoint = 1.0 if bucket >= 500 else 0.0
    return torch.full((batch_size, 1), endpoint, device=device, dtype=dtype)
