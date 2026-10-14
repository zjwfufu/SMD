# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 training configuration."""

from smd.configs.experiments.qwen_image import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config()

    config.model.net = L("smd.networks.QwenImageEdit.network.QwenImageEdit")(
        lora_alpha=256,
        lora_dropout=0.0,
        lora_rank=256,
        use_lora=True,
    )
    config.model.teacher = L("smd.networks.QwenImageEdit.network.QwenImageEdit")(
        use_lora=False,
    )
    config.model.neg_prompt = " "
    config.model.sample_t_cfg.shift = 2.0
    config.model.teacher_repulsion_weight = 0.4
    config.model.residual_basis = "bubble_path"
    config.model.num_basis_terms = 8
    config.model.cfg_norm_rescale = True

    config.dataloader_train = L("smd.datasets.edit_image_loader.EditImageTarLoader")(
        datatags=["WDS:assets/examples/image_edit/shards"],
        batch_size=1,
        caption_keys=["short_ins"],
        num_workers=0,
        shuffle_buffer=0,
        sampler_start_idx=None,
        seed=0,
    )

    config.log_config.name = "qwen_image_edit"

    return configure_callbacks(config)
