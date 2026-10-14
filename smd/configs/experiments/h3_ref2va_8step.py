# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 REF2VA, 8-step training configuration."""

from smd.configs.experiments.h3_ref2va_4step import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks


def create_config():
    config = create_parent_config()

    config.model.student_sample_steps = 8

    config.log_config.name = "h3_ref2va_8step"

    return configure_callbacks(config)
