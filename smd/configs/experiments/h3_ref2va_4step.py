# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 REF2VA, 4-step training configuration."""

from smd.configs.experiments.h3_t2va_4step import create_config as create_parent_config
from smd.configs.distillation import configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config()

    config.model.net.transformer_subfolder = "transformer_ref"
    config.model.net.context_parallel_size = 4
    config.model.h3_condition_mode = "ref2va"

    config.dataloader_train = L("smd.datasets.minimax_h3_ref2va_cache.MiniMaxH3Ref2VACacheLoader")(
        dataset_path="assets/examples/h3/ref2va/release",
        batch_size=1,
        num_workers=0,
        sampler_start_idx=None,
        seed=0,
        shuffle=True,
        context_parallel_size=4,
        verify_payload_hashes=True,
    )

    config.log_config.name = "h3_ref2va_4step"

    return configure_callbacks(config)
