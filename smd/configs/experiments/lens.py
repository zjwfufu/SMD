# SPDX-License-Identifier: Apache-2.0
"""Lens training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_base_config("lens")

    config.model.net = L("smd.networks.Lens.network.Lens")(
        lora_alpha=256,
        lora_dropout=0.05,
        lora_rank=256,
        model_id="zjwfufu/SMD-Lens",
        use_lora=True,
    )
    config.model.teacher = L("smd.networks.Lens.network.Lens")(
        model_id="zjwfufu/SMD-Lens",
        use_lora=False,
    )
    config.model.guidance_scale = 5.0
    config.model.sample_t_cfg.shift = 3.0
    config.model.sample_t_cfg.min_t = 0.0001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.input_shape = [128, 64, 64]
    config.model.load_student_weights = False
    config.model.pivot_num_query_points = 4
    config.model.cfg_norm_rescale = True
    config.model.dynamic_shift = True
    config.model.dynamic_shift_empirical_steps = 50
    config.model.time_shift_type = "exponential"

    config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
        dataset_path="assets/examples/text_to_image/prompts.parquet",
        prompt_key="prompt",
        batch_size=4,
        num_workers=0,
        seed=0,
        shuffle=True,
    )

    config.trainer.max_iter = 10000
    config.trainer.fsdp_cpu_offload = True

    return configure_callbacks(config)
