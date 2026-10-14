# SPDX-License-Identifier: Apache-2.0
"""Wan2.1 14B training configuration."""

from smd.configs.experiments.wan13 import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config()

    config.model.net = L("smd.networks.Wan.network.Wan")(
        disable_efficient_attn=False,
        disable_grad_ckpt=False,
        enable_logvar_linear=False,
        lora_alpha=256,
        lora_dropout=0.0,
        lora_rank=256,
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        r_timestep=False,
        use_lora=True,
        use_wan_official_sinusoidal=False,
    )
    config.model.teacher = L("smd.networks.Wan.network.Wan")(
        disable_efficient_attn=False,
        disable_grad_ckpt=False,
        enable_logvar_linear=False,
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        r_timestep=False,
        use_lora=False,
        use_wan_official_sinusoidal=False,
    )
    config.model.guidance_scale = 5.0
    config.model.skip_layers = [11]
    config.model.sample_t_cfg.shift = 5.0
    config.model.pivot_optimizer.lr = 3e-05
    config.model.teacher_repulsion_weight = 0.2
    config.model.pivot_residual_scale = 0.0
    config.model.teacher_cfg_scale = 6.0
    config.model.teacher_repulsion_cfg_scale = 1.0

    config.log_config.name = "wan14"

    return configure_callbacks(config)
