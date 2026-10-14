# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping

import wandb


def log_at_iteration(values: Mapping[str, Any], *, iteration: int) -> None:
    """Append one W&B history row using the replay-safe training iteration axis."""
    if not wandb.run:
        return
    payload = dict(values)
    configured_iteration = payload.get("optimizer/iteration")
    if configured_iteration is not None and int(configured_iteration) != int(iteration):
        raise ValueError(
            f"W&B payload optimizer/iteration={configured_iteration} does not match iteration={iteration}."
        )
    payload["optimizer/iteration"] = int(iteration)
    wandb.log(payload)
