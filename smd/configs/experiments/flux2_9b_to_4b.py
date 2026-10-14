# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 Klein 9B to 4B training configuration."""

from smd.configs.experiments.flux2_4b import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config()

    config.model.teacher.model_id = "black-forest-labs/FLUX.2-klein-base-9B"

    config.model_class = L("smd.methods.simplified_distill.cross.CrossSimplifiedDistillation")(
        config=None,
    )

    config.log_config.name = "flux2_9b_to_4b"

    return configure_callbacks(config)
