# SPDX-License-Identifier: Apache-2.0
"""SD3.5-Medium self-distillation training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config(model=None):
    config = create_base_config("sd35_self", model=model)

    config.model.net = L("smd.networks.SD35Medium.network.SD35Medium")(
        model_id="stabilityai/stable-diffusion-3.5-medium",
    )
    config.model.guidance_scale = 5.0
    config.model.net_optimizer.lr = 1e-05
    config.model.sample_t_cfg.shift = 3.0
    config.model.sample_t_cfg.min_t = 0.0001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.input_shape = [16, 128, 128]
    config.model.teacher_cfg_scale = 5.0

    config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
        dataset_path="assets/sd35/combined_all.parquet",
        batch_size=4,
        num_workers=0,
        seed=0,
        shuffle=True,
    )

    config.trainer.max_iter = 10000

    return configure_callbacks(config)
