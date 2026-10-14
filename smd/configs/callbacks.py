# SPDX-License-Identifier: Apache-2.0
"""W&B and sampling callback defaults for the reference experiments."""

import json
from pathlib import Path

from smd.utils import LazyCall as L
from smd.callbacks.simplified_wandb import SimplifiedWandbCallback

from smd.callbacks.sampling.samplers import BatchSampleCallback, PromptSampleCallback

WANDB_CALLBACK = dict(wandb=L(SimplifiedWandbCallback)(sample_logging_iter=None))

_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "prompt_samples"


def fixed_preview_source(config):
    """Select versioned preview assets independently of the training loader."""
    target = str(config.model.net.get("_target_", ""))
    if ".MiniMaxH3." in target:
        name = "h3_" + config.model.h3_condition_mode
    elif ".QwenImageEdit." in target:
        name = "qwen_image_edit"
    elif ".Hunyuan3DShape." in target:
        name = "hunyuan3d_shape"
    elif ".Trellis2." in target:
        name = "trellis2_texture" if "tex" in config.model.net.ckpt_name else "trellis2_shape"
    else:
        name = "text_to_video" if ".Wan." in target or ".CausalWan." in target else "text_to_image"
    asset = json.loads((_ASSETS / f"{name}.json").read_text(encoding="utf-8"))
    if "assets" in asset:
        return {"assets": asset["assets"]}
    if "prompts" in asset:
        prompts = asset["prompts"]
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError(f"Invalid fixed preview prompts in {name}.json")
        return {"prompts": prompts}
    return {"dataset": asset["dataset"], "case_indices": asset["case_indices"]}


def fixed_preview_inputs(config):
    from smd.callbacks.sampling.inputs.cache import cache_path, signature

    source = fixed_preview_source(config)
    return {"cache": cache_path(config), "cache_signature": signature(config, source)}


def preview_callbacks(config):
    target = str(config.model.net.get("_target_", ""))
    h3 = ".MiniMaxH3." in target
    wan = ".Wan." in target or ".CausalWan." in target
    edit = ".QwenImageEdit." in target
    shape_3d = ".Hunyuan3DShape." in target or ".Trellis2." in target
    common = {
        "every_n": 500,
        "run_on_start": True,
        "grid_size": 2 if h3 else 4,
        "seed": 0,
        "fps": 24 if h3 else 16,
        "preview_timeout_s": 2000 if h3 or wan or shape_3d else None,
    }
    fixed = dict(common, **fixed_preview_inputs(config))
    if shape_3d or edit:
        # Keep one stable result per original fixed image. The batch callback
        # still collects four samples from the current training conditions.
        fixed.update(grid_size=1, seed=42)
    return {"samples": L(BatchSampleCallback)(**common), "prompt_samples": L(PromptSampleCallback)(**fixed)}
