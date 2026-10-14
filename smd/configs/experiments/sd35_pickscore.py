# SPDX-License-Identifier: Apache-2.0
"""SD3.5-Medium distillation with the pickscore teacher."""

from smd.configs.experiments.sd35_multi import create_config as create_multi_config


def create_config():
    config = create_multi_config()
    config.model.teacher_adapters = [
        adapter for adapter in config.model.teacher_adapters if adapter["name"] == "pickscore"
    ]
    config.model.route_to_lora = {route: ["pickscore"] for route in config.dataloader_train.route_paths}
    config.log_config.name = "sd35_pickscore"
    return config
