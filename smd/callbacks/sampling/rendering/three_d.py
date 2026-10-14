# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone 3D preview renderers rebuilt inside the isolated worker."""

from __future__ import annotations

from typing import Any

import torch


class Hunyuan3DPreviewRenderer:
    def __init__(self, model_id: str, octree_resolution: int, render_size: int):
        from smd.networks.Hunyuan3DShape.weight_loader import load_vae

        self.vae = load_vae(model_id).eval()
        self.octree_resolution = int(octree_resolution)
        self.render_size = int(render_size)

    def to(self, *args: Any, **kwargs: Any):
        self.vae.to(*args, **kwargs)
        return self

    def render(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from smd.callbacks.sampling.rendering.hunyuan3d import decode_and_render_shape_hy

        normal, clay = decode_and_render_shape_hy(
            data["student_latents"],
            self.vae,
            octree_resolution=self.octree_resolution,
            render_size=self.render_size,
        )
        if normal is None or clay is None:
            return {}
        return {"mesh": torch.cat([normal, clay], dim=2).clamp(0, 1)}


def _sparse(data: dict[str, torch.Tensor], prefix: str):
    from smd.networks.Trellis2.sparse import SparseTensor

    feats = data[f"{prefix}_feats"]
    coords = data[f"{prefix}_coords"].to(dtype=torch.int32)
    return SparseTensor(feats=feats, coords=coords)


class TrellisShapePreviewRenderer:
    def __init__(self, shape_decoder_path: str, decode_resolution: int, render_size: int):
        self.shape_decoder_path = shape_decoder_path
        self.decode_resolution = int(decode_resolution)
        self.render_size = int(render_size)
        self.device = torch.device("cpu")

    def to(self, device: torch.device, **kwargs: Any):
        del kwargs
        self.device = torch.device(device)
        return self

    def render(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from smd.callbacks.sampling.rendering.trellis import decode_and_render_shape

        torch.cuda.set_device(self.device)
        outputs = {}
        for tag in ("student", "gt"):
            if f"{tag}_feats" not in data:
                continue
            views, pbr, _, _ = decode_and_render_shape(
                _sparse(data, tag),
                self.shape_decoder_path,
                self.decode_resolution,
                self.render_size,
            )
            if views is not None:
                outputs[f"{tag}_views"] = views
            if pbr is not None:
                outputs[f"{tag}_pbr"] = pbr
        return outputs


class TrellisTexturePreviewRenderer:
    def __init__(
        self,
        shape_decoder_path: str,
        tex_decoder_path: str,
        decode_resolution: int,
        render_size: int,
    ):
        self.shape_decoder_path = shape_decoder_path
        self.tex_decoder_path = tex_decoder_path
        self.decode_resolution = int(decode_resolution)
        self.render_size = int(render_size)
        self.device = torch.device("cpu")

    def to(self, device: torch.device, **kwargs: Any):
        del kwargs
        self.device = torch.device(device)
        return self

    def render(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from smd.callbacks.sampling.rendering.trellis import decode_and_render_texture

        torch.cuda.set_device(self.device)
        shape = _sparse(data, "shape")
        outputs = {}
        for tag in ("student", "gt"):
            if f"{tag}_feats" not in data:
                continue
            views, pbr = decode_and_render_texture(
                _sparse(data, tag),
                shape,
                self.shape_decoder_path,
                self.tex_decoder_path,
                self.decode_resolution,
                self.render_size,
            )
            if views is not None:
                outputs[f"{tag}_views"] = views
            if pbr is not None:
                outputs[f"{tag}_pbr"] = pbr
        return outputs
