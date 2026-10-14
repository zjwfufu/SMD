# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 Klein 9B to Lens training configuration."""

from smd.configs.experiments.lens import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config()

    config.model.teacher = L("smd.networks.Flux2.network.Flux2")(
        model_id="black-forest-labs/FLUX.2-klein-base-9B",
        use_lora=False,
    )
    config.model.guidance_scale = 4.0
    config.model.cfg_norm_rescale = False

    config.model_class = L("smd.methods.simplified_distill.cross.CrossSimplifiedDistillation")(
        config=None,
    )

    config.log_config.name = "flux2_9b_to_lens"

    return configure_callbacks(config)
