# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_base_config("qwen_image")

    config.model.net = L("smd.networks.QwenImage.network.QwenImage")(
        lora_alpha=256,
        lora_dropout=0.0,
        lora_rank=256,
        use_lora=True,
    )
    config.model.teacher = L("smd.networks.QwenImage.network.QwenImage")(
        use_lora=False,
    )
    config.model.guidance_scale = 4.0
    config.model.sample_t_cfg.shift = 3.2
    config.model.sample_t_cfg.min_t = 0.0001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.input_shape = [16, 128, 128]
    config.model.teacher_repulsion_weight = 0.2
    config.model.residual_basis = "centered_dx"
    config.model.teacher_cfg_scale = 5.0
    config.model.teacher_repulsion_cfg_scale = 1.0
    config.model.dynamic_shift = True
    config.model.dynamic_shift_max_shift = 0.9
    config.model.dynamic_shift_max_seq_len = 8192
    config.model.time_shift_type = "exponential"
    config.model.tie_backbone = True

    config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
        dataset_path="assets/examples/text_to_image/prompts.parquet",
        prompt_key="prompt",
        batch_size=2,
        num_workers=0,
        seed=0,
        shuffle=True,
    )

    return configure_callbacks(config)
