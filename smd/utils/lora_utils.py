# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LoRA utilities used across architectures."""

from __future__ import annotations

from typing import List

import torch


def freeze_for_lora(module: torch.nn.Module, allowlist: List[str]) -> None:
    """Freeze everything in ``module`` except LoRA params and any param whose
    dotted name contains a substring in ``allowlist``.

    Args:
        module: The module to freeze in place.
        allowlist: Substrings that, if found in a parameter's dotted name,
            keep that parameter trainable (e.g. ``["norm_out"]``).
    """
    for name, p in module.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            continue
        if any(key in name for key in allowlist):
            p.requires_grad_(True)
            continue
        p.requires_grad_(False)
