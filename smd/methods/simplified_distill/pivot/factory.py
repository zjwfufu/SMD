# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch.nn as nn

from .common import PivotBackend
from .backends.flux import FluxPivotBackend
from .backends.wan import WanPivotBackend
from .backends.causal_wan import CausalWanPivotBackend
from .backends.qwen_image import QwenImagePivotBackend, QwenImageEditPivotBackend
from .backends.lens import LensPivotBackend
from .backends.flux2 import Flux2PivotBackend
from .backends.sd35 import SD35PivotBackend


def create_pivot_backend(net: nn.Module) -> PivotBackend:
    """Auto-detect architecture and return the correct backend.

    Duck-typing detection:
    - TRELLIS.2: ``model.blocks`` + ``model.input_layer``, block[0] has "Sparse" in name
    - Wan: ``hasattr(transformer, "blocks")`` and no ``transformer_blocks``
    - SD3.5: ``transformer_blocks`` whose first block is a ``JointTransformerBlock`` (diffusers)
    - QwenImage: ``hasattr(transformer, "transformer_blocks")`` and no ``single_transformer_blocks``
    - Flux2: ``hasattr(transformer, "double_stream_modulation_img")`` (shared modulation)
    - Flux: ``hasattr(transformer, "transformer_blocks")`` and ``single_transformer_blocks``
    """
    # TRELLIS.2 detection: net.model has .blocks + .input_layer, block type contains "Sparse"
    model = getattr(net, "model", None)
    if model is not None and hasattr(model, "blocks") and hasattr(model, "input_layer"):
        blocks = list(model.blocks)
        if blocks and "Sparse" in type(blocks[0]).__name__:
            from .backends.trellis import TrellisPivotBackend

            return TrellisPivotBackend()

    # Hunyuan3D Shape detection: net.model has .blocks whose type is HunYuanDiTBlock
    # (U-Net DiT, dense vec2set tokens). Checked alongside TRELLIS since both expose
    # net.model.blocks, but Hunyuan blocks are not Sparse and there is no input_layer.
    if model is not None and hasattr(model, "blocks") and hasattr(model, "final_layer"):
        blocks = list(model.blocks)
        if blocks and "HunYuanDiTBlock" in type(blocks[0]).__name__:
            from .backends.hunyuan3d import Hunyuan3DPivotBackend

            return Hunyuan3DPivotBackend()

    transformer = getattr(net, "transformer", None)
    if transformer is not None:
        if getattr(net, "network_family", None) == "minimax_h3":
            from .backends.minimax_h3 import MiniMaxH3PivotBackend

            return MiniMaxH3PivotBackend()
        if getattr(net, "network_family", None) in {"zimage", "sana_wm", "ltx2"}:
            raise NotImplementedError(f"SimplifiedDistill has no backend for {net.network_family}.")
        # Wan: has `blocks` but NOT `transformer_blocks`
        if hasattr(transformer, "blocks") and not hasattr(transformer, "transformer_blocks"):
            from smd.networks.network import CausalSMDNetwork

            if isinstance(net, CausalSMDNetwork):
                return CausalWanPivotBackend()
            return WanPivotBackend()
        # Flux2: has shared modulation modules (distinctive marker)
        if hasattr(transformer, "double_stream_modulation_img"):
            return Flux2PivotBackend()
        # Distinguish QwenImage (no single_transformer_blocks) from Flux (has both)
        if hasattr(transformer, "transformer_blocks"):
            if hasattr(transformer, "single_transformer_blocks"):
                return FluxPivotBackend()
            # SD3.5: JointTransformerBlock from diffusers.models.attention
            # (QwenImage uses QwenImageJointTransformerBlock, which is different)
            try:
                from diffusers.models.attention import JointTransformerBlock as _JTB

                blocks = list(transformer.transformer_blocks)
                if blocks and isinstance(blocks[0], _JTB):
                    return SD35PivotBackend()
            except ImportError:
                pass
            # Lens: LensTransformerBlock (check before QwenImage fallback)
            try:
                from smd.networks.Lens.transformer import LensTransformerBlock as _LTB

                blocks = list(transformer.transformer_blocks)
                if blocks and isinstance(blocks[0], _LTB):
                    return LensPivotBackend()
            except ImportError:
                pass
            # QwenImageEdit: has is_edit attribute
            if getattr(net, "is_edit", False):
                return QwenImageEditPivotBackend()
            return QwenImagePivotBackend()
    raise NotImplementedError(
        f"No PivotBackend for {type(net).__name__}. Supported: Flux, Flux2, Lens, "
        "QwenImage, QwenImageEdit, SD3.5, Wan, CausalWan, TRELLIS and Hunyuan3D."
    )
