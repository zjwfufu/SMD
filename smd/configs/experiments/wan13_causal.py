# SPDX-License-Identifier: Apache-2.0
"""Causal Wan2.1 1.3B training configuration."""

import os

from smd.configs.experiments.wan13 import create_config as create_parent_config
from smd.configs.distillation import CausalModelConfig, configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config(model=CausalModelConfig())

    config.model.net = L("smd.networks.Wan.network_causal.CausalWan")(
        chunk_size=3,
        enable_logvar_linear=False,
        lora_alpha=256,
        lora_dropout=0.0,
        lora_rank=256,
        max_t=1.0,
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        r_timestep=False,
        total_num_frames=21,
        use_lora=True,
        use_wan_official_sinusoidal=True,
    )
    config.model.teacher = L("smd.networks.Wan.network_causal.CausalWan")(
        chunk_size=3,
        enable_logvar_linear=False,
        max_t=1.0,
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        r_timestep=False,
        total_num_frames=21,
        use_lora=False,
        use_wan_official_sinusoidal=False,
    )
    config.model.guidance_scale = 3.0
    config.model.neg_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.sample_t_cfg.shift = 5.0
    config.model.student_sample_type = "sde"
    config.model.pivot_optimizer.lr = 3e-05
    config.model.pivot_scheduler.warm_up_steps = [0]
    config.model.student_init_checkpoint = (
        os.environ.get("SMD_MODEL_ROOT", "./models") + "/causal_forcing/causal_ode.pt"
    )
    config.model.final_step_size_scale = 1.0
    config.model.teacher_repulsion_weight = 0.2
    config.model.tie_backbone = False
    config.model.causal_correction_strong_teacher = {
        "_target_": "smd.networks.Wan.network.Wan",
        "disable_efficient_attn": False,
        "disable_grad_ckpt": False,
        "enable_logvar_linear": False,
        "max_t": 1.0,
        "model_id_or_local_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "r_timestep": False,
        "use_lora": False,
        "use_wan_official_sinusoidal": False,
    }
    config.model.causal_correction_strong_skip_layers = [11]
    config.model.causal_teacher_type = "causal"
    config.model.causal_teacher_checkpoint = (
        os.environ.get("SMD_MODEL_ROOT", "./models") + "/causal_forcing/ar_diffusion.pt"
    )
    config.model.causal_objective_mode = "hybrid_causal_correction"
    config.model.causal_replay_chunks_per_step = 7

    config.model_class = L("smd.methods.simplified_distill.causal.CausalSimplifiedDistillation")(
        config=None,
    )

    config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
        dataset_path="assets/examples/text_to_video/prompts.parquet",
        prompt_key="prompt",
        batch_size=1,
        num_workers=0,
        seed=0,
        shuffle=True,
    )

    config.trainer.max_iter = 1000
    config.trainer.grad_accum_rounds = 2
    config.trainer.fsdp_cpu_offload = True

    config.log_config.name = "wan13_causal"

    return configure_callbacks(config)
