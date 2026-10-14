# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode and render Hunyuan3D shape latents."""
import torch
from smd.utils import logging_utils as logger

from smd.utils.rendering.panels import _make_trellis_mesh, _render_clay_panel

@torch.no_grad()
def decode_and_render_shape_hy(
    latent: torch.Tensor,
    vae,
    *,
    octree_resolution: int = 256,
    render_size: int = 512,
    num_chunks: int = 20000,
):
    """Decode a Hunyuan dense latent ``[1, 4096, 64]`` -> mesh -> (normal, clay).

    Returns ``(normal_panel, clay_panel)`` each ``[3, 2H, 2H]`` in [0,1], or
    ``(None, None)`` if marching cubes produced no surface.
    """
    # Un-scale by the VAE scale_factor (mirror of the official pipeline export).
    lat = latent.float() / vae.scale_factor
    lat = lat.to(next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    decoded = vae.decode(lat)  # [1, 4096, width]
    outputs = vae.latents2mesh(
        decoded, bounds=1.01, mc_level=0.0, num_chunks=num_chunks, octree_resolution=octree_resolution
    )
    out = outputs[0]
    if out is None or out.mesh_v is None or len(out.mesh_v) == 0:
        # Debug: log grid stats to diagnose why marching cubes failed
        try:
            grid = vae.volume_decoder(
                decoded, vae.geo_decoder, bounds=1.01, num_chunks=num_chunks, octree_resolution=octree_resolution
            )
            g = grid[0]
            logger.info(
                f"[Hunyuan3DPromptSample] MC failed — grid stats: "
                f"shape={tuple(g.shape)} min={g.min():.4f} max={g.max():.4f} "
                f"mean={g.mean():.4f} median={g.median():.4f} "
                f"nan={g.isnan().sum()} inf={g.isinf().sum()} "
                f"frac_pos={(g > 0).float().mean():.3f}"
            )
        except Exception:
            pass
        return None, None
    mesh = _make_trellis_mesh(out.mesh_v, out.mesh_f)
    from smd.utils.rendering.panels import render_mesh_normal_panel

    normal_panel = render_mesh_normal_panel(mesh, render_size)
    clay_panel = _render_clay_panel(mesh, render_size)
    return normal_panel, clay_panel
