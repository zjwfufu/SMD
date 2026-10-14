# SPDX-License-Identifier: Apache-2.0
"""FLUX.1-dev training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_base_config("flux1_dev")

    config.model.net = L("smd.networks.Flux.network.Flux")(
        lora_alpha=256,
        lora_dropout=0.0,
        lora_rank=256,
        use_lora=True,
    )
    config.model.teacher = L("smd.networks.Flux.network.Flux")(
        use_lora=False,
    )
    config.model.sample_t_cfg.shift = 3.2
    config.model.sample_t_cfg.min_t = 0.0001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.input_shape = [16, 96, 170]
    config.model.teacher_repulsion_weight = 0.4
    config.model.teacher_cfg_scale = 3.5
    config.model.teacher_repulsion_cfg_scale = 1.0
    config.model.embedded_guidance_scale = 3.5
    config.model.dynamic_shift = True
    config.model.time_shift_type = "exponential"
    config.model.tie_backbone = True

    config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
        dataset_path="assets/examples/text_to_image/prompts.parquet",
        prompt_key="prompt",
        batch_size=4,
        num_workers=0,
        seed=0,
        shuffle=True,
    )

    return configure_callbacks(config)
