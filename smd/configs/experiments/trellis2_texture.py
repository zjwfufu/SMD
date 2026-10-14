# SPDX-License-Identifier: Apache-2.0
"""TRELLIS.2 Texture training configuration."""

from smd.configs.experiments.trellis2_shape import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config()

    config.model.net.ckpt_name = "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
    config.model.teacher.ckpt_name = "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
    config.model.guidance_scale = 1.0
    config.model.teacher_cfg_scale = 1.0
    config.model.guidance_interval = [0.6, 0.9]
    config.model.guidance_rescale = 0.0

    config.dataloader_train = L("smd.datasets.trellis_texture_loader.TrellisTextureWDSLoader")(
        shard_dir="assets/examples/objects3d/shards",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        sampler_start_idx=None,
        seed=0,
    )

    config.log_config.name = "trellis2_texture"

    return configure_callbacks(config)
