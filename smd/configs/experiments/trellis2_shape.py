# SPDX-License-Identifier: Apache-2.0
"""TRELLIS.2 Shape training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import SparseModelConfig, configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_base_config("trellis2_shape", model=SparseModelConfig())

    config.model.net = L("smd.networks.Trellis2.network.Trellis2")(
        ckpt_name="slat_flow_img2shape_dit_1_3B_1024_bf16",
        cond_channels=1024,
        latent_channels=32,
        lora_rank=128,
        max_t=1.0,
        min_t=0.0,
        model_id="microsoft/TRELLIS.2-4B",
        net_pred_type="flow",
        rescale_t=3.0,
        schedule_type="trellis_flow",
        shift=3.0,
        sigma_min=1e-05,
        use_lora=True,
    )
    config.model.teacher = L("smd.networks.Trellis2.network.Trellis2")(
        ckpt_name="slat_flow_img2shape_dit_1_3B_1024_bf16",
        cond_channels=1024,
        latent_channels=32,
        lora_rank=128,
        max_t=1.0,
        min_t=0.0,
        model_id="microsoft/TRELLIS.2-4B",
        net_pred_type="flow",
        rescale_t=3.0,
        schedule_type="trellis_flow",
        shift=3.0,
        sigma_min=1e-05,
        use_lora=False,
    )
    config.model.guidance_scale = 7.5
    config.model.neg_prompt = None
    config.model.net_optimizer.lr = 1e-05
    config.model.sample_t_cfg.shift = 3.0
    config.model.sample_t_cfg.min_t = 0.0
    config.model.sample_t_cfg.max_t = 1.0
    config.model.input_shape = [32]
    config.model.final_step_size_scale = 1.0
    config.model.pivot_num_query_points = 4
    config.model.teacher_cfg_scale = 7.5
    config.model.guidance_interval = [0.6, 1.0]
    config.model.guidance_rescale = 0.5

    config.model_class = L("smd.methods.simplified_distill.sparse.SparseSimplifiedDistillation")(
        config=None,
    )

    config.dataloader_train = L("smd.datasets.trellis_shape_loader.TrellisShapeWDSLoader")(
        shard_dir="assets/examples/objects3d/shards",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        sampler_start_idx=None,
        seed=0,
    )

    config.trainer.max_iter = 10000
    config.trainer.fsdp_cpu_offload = True

    return configure_callbacks(config)
