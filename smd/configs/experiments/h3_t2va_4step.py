# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 T2VA, 4-step training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import H3ModelConfig, configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_base_config("h3_t2va_4step", model=H3ModelConfig())

    config.model.net = L("smd.networks.MiniMaxH3.network.MiniMaxH3")(
        model_id="MiniMaxAI/MiniMax-H3",
        model_revision="939557dc319dd91227e30195a763f272ba7f8765",
        transformer_model_id=None,
        transformer_subfolder="transformer",
        load_pretrained=True,
        use_lora=True,
        lora_rank=128,
        lora_alpha=128,
        lora_dropout=0.0,
        gradient_checkpointing=True,
        ff_chunk_size=4096,
        qk_norm_chunk_size=4096,
        video_shift=12.0,
        audio_shift=3.0,
        condition_video_timestep=0.999,
        context_parallel_size=2,
    )
    config.model.guidance_scale = 1.0
    config.model.neg_prompt = None
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 1.0
    config.model.input_shape = [24, 37, 48, 84]
    config.model.enable_preprocessors = False
    config.model.fsdp_meta_init = True
    config.model.pivot_hook_layer = -8
    config.model.teacher_repulsion_weight = 0.2
    config.model.num_basis_terms = 32
    config.model.backbone_unfreeze_allowlist = []
    config.model.pivot_train_mode = "lora"
    config.model.tie_backbone = True

    config.model_class = L("smd.methods.simplified_distill.minimax_h3.MiniMaxH3Distillation")(
        config=None,
    )

    config.dataloader_train = L("smd.datasets.minimax_h3_prompt_cache.MiniMaxH3PromptCacheLoader")(
        dataset_path="assets/examples/h3/t2va",
        batch_size=1,
        num_workers=0,
        sampler_start_idx=None,
        seed=0,
        shuffle=True,
        context_parallel_size=2,
    )

    config.trainer.fsdp_cpu_offload = True

    return configure_callbacks(config)
