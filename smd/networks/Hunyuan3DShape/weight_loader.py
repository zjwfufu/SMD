# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace weight loading for the vendored Hunyuan3D 2.1 Shape model.

The HF repo ``tencent/Hunyuan3D-2.1`` lays the shape components out as
subfolders of a single snapshot::

    hunyuan3d-dit-v2-1/{config.yaml, model.fp16.ckpt}   # DiT denoiser (7.4 GB)
    hunyuan3d-vae-v2-1/{config.yaml, model.fp16.ckpt}   # ShapeVAE (656 MB)

Both ``model.fp16.ckpt`` files are *pickle* checkpoints (``torch.load``), not
safetensors. We resolve the snapshot dir via the shared ``hf_utils`` cache
helpers (rank-0 download, ``/hub`` path robustness — the same machinery TRELLIS
uses) and parse the YAML ``params`` to instantiate our vendored modules.
"""

from __future__ import annotations

import os

import torch
import yaml

import smd.utils.logging_utils as logger
from smd.utils.hf_utils import get_hf_search_dirs, get_hf_cache_dir

from .dit import HunYuanDiTPlain
from .vae import ShapeVAE

MODEL_ID = "tencent/Hunyuan3D-2.1"
DIT_SUBFOLDER = "hunyuan3d-dit-v2-1"
VAE_SUBFOLDER = "hunyuan3d-vae-v2-1"
CKPT_NAME = "model.fp16.ckpt"
CONFIG_NAME = "config.yaml"


def _resolve_snapshot_dir(model_id: str = MODEL_ID) -> str:
    """Return the local snapshot directory for *model_id*, downloading if absent.

    Mirrors ``Trellis2._resolve_from_hf``: probe every HF search dir for an
    existing snapshot; if none is found, trigger a (rank-0-safe) download via
    ``snapshot_download`` into the resolved cache dir.
    """
    dir_name = f"models--{model_id.replace('/', '--')}"
    for base in (os.path.join(d, dir_name) for d in get_hf_search_dirs()):
        snapshots_dir = os.path.join(base, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        snaps = sorted(os.listdir(snapshots_dir))
        for snap in snaps:
            cand = os.path.join(snapshots_dir, snap)
            # require at least the DiT subfolder config to be present
            if os.path.exists(os.path.join(cand, DIT_SUBFOLDER, CONFIG_NAME)):
                return cand

    # Not cached: download the two shape subfolders only.
    from huggingface_hub import snapshot_download

    cache_dir = get_hf_cache_dir(model_id)
    logger.info(f"Hunyuan3DShape: downloading {model_id} (dit+vae subfolders) into {cache_dir}")
    snap_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=[f"{DIT_SUBFOLDER}/*", f"{VAE_SUBFOLDER}/*"],
    )
    return snap_path


def _load_ckpt(path: str, component: str) -> dict:
    """Load a pickle ``.ckpt`` and return the sub-state-dict for *component*.

    The released ``model.fp16.ckpt`` files bundle multiple components at the top
    level (``{"model": ..., "vae": ..., "conditioner": ...}``). We select the
    requested one; if the checkpoint is already a flat state dict (no such
    wrapper), it is returned as-is.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hunyuan3DShape checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and component in ckpt and isinstance(ckpt[component], dict):
        return ckpt[component]
    return ckpt


def _read_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dit(model_id: str = MODEL_ID, snapshot_dir: str | None = None) -> HunYuanDiTPlain:
    """Instantiate ``HunYuanDiTPlain`` from the HF config and load fp16 weights."""
    snapshot_dir = snapshot_dir or _resolve_snapshot_dir(model_id)
    cfg_path = os.path.join(snapshot_dir, DIT_SUBFOLDER, CONFIG_NAME)
    ckpt_path = os.path.join(snapshot_dir, DIT_SUBFOLDER, CKPT_NAME)

    cfg = _read_config(cfg_path)
    params = cfg["model"]["params"] if "model" in cfg else cfg["params"]
    model = HunYuanDiTPlain(**params)

    state_dict = _load_ckpt(ckpt_path, "model")
    info = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in info.missing_keys if not k.startswith("_")]
    if missing:
        logger.warning(f"Hunyuan3DShape DiT missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if info.unexpected_keys:
        logger.warning(
            f"Hunyuan3DShape DiT unexpected keys: {info.unexpected_keys[:8]}"
            f"{' ...' if len(info.unexpected_keys) > 8 else ''}"
        )
    logger.info(f"Hunyuan3DShape: loaded DiT from {ckpt_path}")
    return model


def load_vae(model_id: str = MODEL_ID, snapshot_dir: str | None = None) -> ShapeVAE:
    """Instantiate the decode-only ``ShapeVAE`` and load weights (strict=False).

    The encoder / pre_kl checkpoint keys are expected as *unexpected* (we vendor
    decode-only); the decode keys (post_kl / transformer / geo_decoder) must all
    be present.
    """
    snapshot_dir = snapshot_dir or _resolve_snapshot_dir(model_id)
    cfg_path = os.path.join(snapshot_dir, VAE_SUBFOLDER, CONFIG_NAME)
    ckpt_path = os.path.join(snapshot_dir, VAE_SUBFOLDER, CKPT_NAME)

    cfg = _read_config(cfg_path)
    params = cfg["params"] if "params" in cfg else cfg["vae"]["params"]
    vae = ShapeVAE(**params)

    state_dict = _load_ckpt(ckpt_path, "vae")
    info = vae.load_state_dict(state_dict, strict=False)
    # Decode-path keys that MUST be present:
    decode_missing = [
        k for k in info.missing_keys if k.startswith(("post_kl", "transformer", "geo_decoder"))
    ]
    if decode_missing:
        raise RuntimeError(
            f"Hunyuan3DShape VAE missing required decode keys: {decode_missing[:8]}"
            f"{' ...' if len(decode_missing) > 8 else ''}"
        )
    logger.info(
        f"Hunyuan3DShape: loaded VAE decode path from {ckpt_path} "
        f"(ignored {len(info.unexpected_keys)} encoder/pre_kl keys)"
    )
    return vae
