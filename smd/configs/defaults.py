# SPDX-License-Identifier: Apache-2.0
"""Shared optimizer, precision and trainer settings for the experiments."""

from copy import deepcopy

from smd.configs.distillation import Config, ModelConfig
from smd.utils import LazyCall as L


def create_base_config(name, model=None):
    """Return fresh settings; attach callbacks after choosing the model and data."""
    config = deepcopy(Config(model=model if model is not None else ModelConfig()))
    config.model_class = L("smd.methods.simplified_distill.distillation.SimplifiedDistillation")(config=None)

    config.model.student_sample_steps = 4
    config.model.student_sample_type = "ode"
    config.model.pivot_num_query_points = 2
    config.model.precision = "bfloat16"
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.model.precision_fsdp = "bfloat16"
    config.model.neg_prompt = ""
    config.model.net_optimizer = L("smd.configs.opt.get_optimizer")(
        optim_type="adamw",
        lr=1e-4,
        weight_decay=0.0,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=False,
    )
    config.model.pivot_optimizer = deepcopy(config.model.net_optimizer)
    config.model.net_scheduler = L("smd.utils.lr_scheduler.LambdaLinearScheduler")(
        warm_up_steps=[100],
        cycle_lengths=[10000000000],
        f_start=[1e-6],
        f_max=[1.0],
        f_min=[1.0],
    )
    config.model.pivot_scheduler = deepcopy(config.model.net_scheduler)

    config.trainer.fsdp = True
    config.trainer.ddp = False
    config.trainer.max_iter = 9000
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 100
    config.trainer.validation_iter = 1000000

    config.log_config.project = "simplified_distill"
    config.log_config.group = "reference"
    config.log_config.name = name
    config.log_config.wandb_mode = "offline"
    config.log_config.wandb_credential = ""
    return config
