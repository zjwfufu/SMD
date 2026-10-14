# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored & adapted from Tencent Hunyuan3D-2.1 hy3dshape/hy3dshape/preprocessors.py
# Original: Copyright (C) 2024 THL A29 Limited (Tencent), TENCENT HUNYUAN
# NON-COMMERCIAL LICENSE AGREEMENT.
#
# ImageProcessorV2: RGBA -> recenter (alpha-tight crop, pad to border_ratio) ->
# white-composite -> resize -> tensor in [-1, 1], layout [B, 3, size, size].
# Multi-view variant omitted (single-image Shape conditioning only).

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image


def array_to_tensor(np_array):
    image_pt = torch.tensor(np_array).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts


class ImageProcessorV2:
    def __init__(self, size=512, border_ratio=0.15):
        self.size = size
        self.border_ratio = border_ratio

    @staticmethod
    def recenter(image, border_ratio: float = 0.2):
        """Recenter an RGBA/RGB image leaving `border_ratio` empty margin, composited on white."""
        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]

        H, W, C = image.shape
        size = max(H, W)
        result = np.zeros((size, size, C), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        if h == 0 or w == 0:
            raise ValueError("input image is empty")
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (size - w2) // 2
        y2_max = y2_min + w2
        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
        )

        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255
        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)
        mask = mask * 255
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 255).astype(np.uint8)
        return result, mask

    def load_image(self, image, border_ratio=0.15, to_tensor=True):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image, mask = self.recenter(image, border_ratio=border_ratio)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = image.convert("RGBA")
            image = np.asarray(image)
            image, mask = self.recenter(image, border_ratio=border_ratio)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]
        if to_tensor:
            image = array_to_tensor(image)
            mask = array_to_tensor(mask)
        return image, mask

    def __call__(self, image, border_ratio=0.15, to_tensor=True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
        image, mask = self.load_image(image, border_ratio=border_ratio, to_tensor=to_tensor)
        return {"image": image, "mask": mask}
