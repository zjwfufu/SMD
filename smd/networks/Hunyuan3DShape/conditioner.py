# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored & adapted (slim) from Tencent Hunyuan3D-2.1
#   hy3dshape/hy3dshape/models/conditioner.py
# Original: Copyright (C) 2024 THL A29 Limited (Tencent), TENCENT HUNYUAN
# NON-COMMERCIAL LICENSE AGREEMENT.
#
# Only the single-image DINOv2 conditioner is vendored (CLIP / multi-view /
# dual-encoder variants dropped). The DINOv2-Large backbone is FROZEN and
# loaded from canonical `facebook/dinov2-large` — Tencent does not fine-tune the
# encoder, so this matches their training (the official training yaml uses
# `version: facebook/dinov2-large`) and avoids depending on whether encoder
# weights are bundled in the shape checkpoint.

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel

from smd.utils.hf_utils import get_hf_cache_dir


class DinoImageEncoder(nn.Module):
    """Frozen DINOv2-Large image encoder -> patch+CLS token features.

    image_size=518 -> 37x37 patches + 1 CLS = 1370 tokens, each 1024-dim.
    Input images are expected in value_range (default [-1, 1]).
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, version="facebook/dinov2-large", image_size=518, use_cls_token=True, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(version, cache_dir=get_hf_cache_dir(version))
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_cls_token = use_cls_token
        self.size = image_size // 14
        self.num_patches = (image_size // 14) ** 2
        if self.use_cls_token:
            self.num_patches += 1
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    @torch.no_grad()
    def forward(self, image, mask=None, value_range=(-1, 1), **kwargs):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)
        image = image.to(self.model.device, dtype=self.model.dtype)
        inputs = self.transform(image)
        outputs = self.model(inputs)
        last_hidden_state = outputs.last_hidden_state
        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]
        return last_hidden_state

    def unconditional_embedding(self, batch_size, **kwargs):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return torch.zeros(batch_size, self.num_patches, self.model.config.hidden_size, device=device, dtype=dtype)


class SingleImageEncoder(nn.Module):
    """Wraps DinoImageEncoder, returning the Hunyuan contexts dict ``{"main": ...}``."""

    def __init__(self, version="facebook/dinov2-large", image_size=518, use_cls_token=True, **kwargs):
        super().__init__()
        self.main_image_encoder = DinoImageEncoder(version=version, image_size=image_size, use_cls_token=use_cls_token)

    def forward(self, image, mask=None, **kwargs):
        return {"main": self.main_image_encoder(image, mask=mask, **kwargs)}

    def unconditional_embedding(self, batch_size, **kwargs):
        return {"main": self.main_image_encoder.unconditional_embedding(batch_size, **kwargs)}
