# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-wide isolated synchronous preview rendering."""

from smd.callbacks.sampling.rendering.decoder import DecoderSpec, build_decoder, describe_decoder
from smd.callbacks.sampling.rendering.gather import gather_samples, gather_variable_samples, samples_per_rank
from smd.callbacks.sampling.rendering.protocol import (
    VisualizationGroup,
    VisualizationItem,
    VisualizationPayload,
)
from smd.callbacks.sampling.rendering.service import PreviewService, preview_timeout_seconds

__all__ = [
    "DecoderSpec",
    "PreviewService",
    "VisualizationGroup",
    "VisualizationItem",
    "VisualizationPayload",
    "build_decoder",
    "describe_decoder",
    "gather_samples",
    "gather_variable_samples",
    "preview_timeout_seconds",
    "samples_per_rank",
]
