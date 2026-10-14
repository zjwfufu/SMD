# SPDX-License-Identifier: Apache-2.0
"""Wan2.1 1.3B training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config(model=None):
    config = create_base_config("wan13", model=model)

    config.model.net = L("smd.networks.Wan.network.Wan")(
        enable_logvar_linear=False,
        lora_alpha=256,
        lora_dropout=0.0,
        lora_rank=256,
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        r_timestep=False,
        use_lora=True,
        use_wan_official_sinusoidal=False,
    )
    config.model.teacher = L("smd.networks.Wan.network.Wan")(
        enable_logvar_linear=False,
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        r_timestep=False,
        use_lora=False,
        use_wan_official_sinusoidal=False,
    )
    config.model.guidance_scale = 6.0
    config.model.neg_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
        "the background, walking backwards"
    )
    config.model.skip_layers = [9]
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 10.0
    config.model.input_shape = [16, 21, 60, 104]
    config.model.final_step_size_scale = 0.3
    config.model.query_point_sampling = "sigma"
    config.model.teacher_repulsion_weight = 0.1
    config.model.teacher_cfg_scale = 5.0
    config.model.backbone_unfreeze_allowlist = []
    config.model.tie_backbone = True

    config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
        dataset_path="assets/examples/text_to_video/prompts.parquet",
        prompt_key="prompt",
        batch_size=2,
        num_workers=0,
        seed=0,
        shuffle=True,
    )

    return configure_callbacks(config)
