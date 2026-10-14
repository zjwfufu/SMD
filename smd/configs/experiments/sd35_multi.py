# SPDX-License-Identifier: Apache-2.0
"""SD3.5-Medium multi-teacher distillation training configuration."""

import os

from smd.configs.experiments.sd35_self import create_config as create_parent_config
from smd.configs.distillation import MultiTeacherModelConfig, configure_callbacks
from smd.utils import LazyCall as L


def create_config():
    config = create_parent_config(model=MultiTeacherModelConfig())

    model_root = os.environ.get("SMD_MODEL_ROOT", "./models")
    config.model.teacher_adapters = [
        {"name": name, "path": f"{model_root}/sd35_teachers/{name}"}
        for name in ("clipscore", "geneval", "ocr", "pickscore")
    ]
    config.model.route_to_lora = {
        "clipscore": ["clipscore"],
        "geneval": ["geneval"],
        "ocr": ["ocr"],
        "preference": ["pickscore"],
    }
    config.model.base_teacher_ratio = 0.0

    config.model_class = L("smd.methods.simplified_distill.multi_teacher.MultiTeacherSimplifiedDistillation")(
        config=None,
    )

    config.dataloader_train = L("smd.datasets.routed_prompt.RoutedPromptLoader")(
        route_paths={
            "clipscore": "assets/sd35/clipscore.parquet",
            "geneval": "assets/sd35/geneval.parquet",
            "ocr": "assets/sd35/ocr.parquet",
            "preference": "assets/sd35/preference.parquet",
        },
        route_weights={"clipscore": 0.25, "geneval": 0.25, "ocr": 0.25, "preference": 0.25},
        batch_size=4,
        seed=0,
        num_workers=0,
    )

    config.log_config.name = "sd35_multi"

    return configure_callbacks(config)
