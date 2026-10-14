# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D-2.1 Shape training configuration."""

from smd.configs.defaults import create_base_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_base_config("hunyuan3d_shape")

    config.model.net = L("smd.networks.Hunyuan3DShape.network.Hunyuan3DShape")(
        cond_channels=1024,
        latent_channels=64,
        latent_tokens=4096,
        lora_rank=128,
        model_id="tencent/Hunyuan3D-2.1",
        net_pred_type="flow",
        schedule_type="rf",
        shift=1.0,
        use_lora=True,
    )
    config.model.teacher = L("smd.networks.Hunyuan3DShape.network.Hunyuan3DShape")(
        cond_channels=1024,
        latent_channels=64,
        latent_tokens=4096,
        lora_rank=128,
        model_id="tencent/Hunyuan3D-2.1",
        net_pred_type="flow",
        schedule_type="rf",
        shift=1.0,
        use_lora=False,
    )
    config.model.guidance_scale = 5.0
    config.model.neg_prompt = None
    config.model.net_optimizer.lr = 1e-05
    config.model.sample_t_cfg.shift = 1.0
    config.model.sample_t_cfg.min_t = 0.0001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.input_shape = [4096, 64]
    config.model.final_step_size_scale = 1.0
    config.model.pivot_num_query_points = 4
    config.model.teacher_cfg_scale = 5.0
    config.model.backbone_unfreeze_allowlist = ["final_layer.norm_final"]

    config.dataloader_train = L("smd.datasets.hunyuan3d_shape_loader.Hunyuan3DShapeImageLoader")(
        shard_dir="assets/examples/objects3d/shards",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        sampler_start_idx=None,
        seed=0,
    )

    config.trainer.max_iter = 5000

    return configure_callbacks(config)
