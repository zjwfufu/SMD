# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering utilities shared by previews and standalone inference.

TRELLIS.2 and Hunyuan3D both turn latents into meshes/voxels and then into
image panels. These functions have no callback state and are shared by the
preview worker, the Hunyuan3D renderer and standalone inference.

Unlike the VAE-based families, these renderers need a live GPU rasterizer. The
preview worker imports this module only after it has isolated training and
reconstructed the required decoder on its assigned GPU.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from smd.utils.rendering import PbrMeshRenderer, EnvMap
from smd.utils.rendering.voxel_renderer import VoxelRenderer
from smd.networks.Trellis2.representations.mesh import Mesh, MeshWithVoxel
from smd.networks.Trellis2.representations.voxel import Voxel
from smd.utils.rendering.render_utils import (
    get_renderer,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)
from smd.utils import logging_utils as logger



def _render_voxel_panel(
    coords_xyz: torch.Tensor,
    colors: torch.Tensor,
    resolution: int = 64,
    render_size: int = 512,
) -> torch.Tensor:
    """Render 4-view voxel panel using TRELLIS.2 VoxelRenderer.

    Parameters
    ----------
    coords_xyz : Tensor [N, 3]
        Integer voxel coordinates (no batch idx).
    colors : Tensor [N, 3]
        Per-voxel RGB colors in [0, 1].
    resolution : int
        Voxel grid resolution (for voxel_size computation).
    render_size : int
        Per-view render resolution.

    Returns
    -------
    Tensor [3, 2*render_size, 2*render_size]
        Composited 2×2 panel image, values in [0, 1].
    """
    device = coords_xyz.device

    voxel = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=coords_xyz.int(),
        attrs=colors.float(),
        layout={"color": slice(0, 3)},
    )

    # 4 turntable views (matching TRELLIS.2 render_snapshot: r=10, fov=8)
    yaw_offset = -16 / 180 * math.pi
    yaws = [i * math.pi / 2 + yaw_offset for i in range(4)]
    pitchs = [20 / 180 * math.pi] * 4
    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 10, 8)

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_size
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 4

    H = render_size
    panel = torch.ones(3, 2 * H, 2 * H, device=device)  # white background
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(voxel, ext, intr, colors_overwrite=colors.float())
        row, col = j // 2, j % 2
        rendered = res["color"]  # [3, H, H] in [0, 1]
        alpha = res.get("alpha", None)
        if alpha is not None:
            # Composite over white background
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            rendered = rendered * alpha + (1 - alpha)
        panel[:, row * H : (row + 1) * H, col * H : (col + 1) * H] = rendered

    return panel


def _render_matplotlib_fallback(
    coords_xyz: torch.Tensor,
    colors: torch.Tensor,
    render_size: int = 512,
) -> torch.Tensor:
    """Fallback: matplotlib 3D scatter plot when VoxelRenderer is unavailable."""
    matplotlib.use("Agg")

    coords_np = coords_xyz.cpu().float().numpy()
    colors_np = colors.cpu().float().numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})
    for i, ax in enumerate(axes):
        elev, azim = 20, i * 90 - 16
        ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], c=colors_np, s=0.5, marker=".")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(0, coords_np.max())
        ax.set_ylim(0, coords_np.max())
        ax.set_zlim(0, coords_np.max())
        ax.axis("off")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return img_t


def render_sparse_panel(
    sp,
    resolution: int = 64,
    render_size: int = 512,
    color_mode: str = "xyz",
) -> torch.Tensor:
    """Render a SparseTensor as a multi-view panel image.

    Parameters
    ----------
    sp : SparseTensor
        Single-sample SparseTensor (batch_size=1).
    resolution : int
        Voxel grid resolution.
    render_size : int
        Per-view render resolution.
    color_mode : str
        ``"xyz"`` for coordinate colormap, ``"pca"`` for PCA of feats.

    Returns
    -------
    Tensor [3, H, W]
        Panel image in [0, 1].
    """
    coords = sp.coords
    # Drop batch index if present (col 0)
    if coords.shape[1] == 4:
        coords_xyz = coords[:, 1:]
    else:
        coords_xyz = coords

    if color_mode == "xyz":
        colors = coords_xyz.float() / float(resolution)
    elif color_mode == "pca":
        feats = sp.feats.float()
        if feats.shape[1] >= 3:
            # Simple PCA: center, SVD, take top 3 components
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                projected = centered @ Vh[:3].T  # [N, 3]
            except Exception:
                projected = feats[:, :3]
            # Normalize to [0, 1]
            pmin = projected.min(dim=0).values
            pmax = projected.max(dim=0).values
            colors = (projected - pmin) / (pmax - pmin + 1e-6)
        else:
            colors = feats[:, :3] if feats.shape[1] >= 3 else feats.repeat(1, 3)[:, :3]
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
    else:
        colors = coords_xyz.float() / float(resolution)

    colors = colors.to(coords_xyz.device)

    # ``VoxelRenderer`` is a hard import at module scope, so it is always present
    # if this module loaded at all; only a runtime failure (no GPU rasterizer,
    # degenerate voxel set) needs the matplotlib fallback.
    try:
        return _render_voxel_panel(coords_xyz, colors, resolution, render_size)
    except Exception as exc:  # noqa: BLE001 - a preview is never worth a crash
        logger.warning(f"VoxelRenderer failed ({type(exc).__name__}: {exc}); using matplotlib fallback")
        return _render_matplotlib_fallback(coords_xyz, colors, render_size)


def get_camera_views(render_size: int = 512):
    """Build 4 turntable camera views matching TRELLIS.2 ``render_snapshot``.

    Uses ``r=10, fov=8`` (near-orthographic) — the same defaults as
    ``trellis2.utils.render_utils.render_snapshot``.
    """
    yaw_offset = -16 / 180 * math.pi
    yaws = [i * math.pi / 2 + yaw_offset for i in range(4)]
    pitchs = [20 / 180 * math.pi] * 4
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 10, 8)


def render_mesh_normal_panel(mesh, render_size: int = 512) -> torch.Tensor:
    """Render a Mesh as 4-view normal map panel using nvdiffrast MeshRenderer.

    The caller must ensure the mesh is decoded at the correct resolution
    (``decode_resolution`` matching the decoder's internal coordinate grid,
    typically 1024) so vertices are in [-0.5, 0.5] world space—matching
    the voxel latent colormap camera.

    The mesh is simplified to ``NVDIFFRAST_FACE_LIMIT`` faces before
    rendering to stay within nvdiffrast's rasterization limits.

    Returns [3, 2*H, 2*H] tensor in [0, 1].
    """
    # nvdiffrast limit (same as app.py: mesh.simplify(16777216))
    NVDIFFRAST_FACE_LIMIT = 16_777_216
    if mesh.faces.shape[0] > NVDIFFRAST_FACE_LIMIT:
        mesh.simplify(NVDIFFRAST_FACE_LIMIT)

    exts, ints = get_camera_views(render_size)
    renderer = get_renderer(mesh, resolution=render_size)

    H = render_size
    panel = torch.zeros(3, 2 * H, 2 * H, device="cuda")
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(mesh, ext, intr)
        row, col = j // 2, j % 2
        normal = res["normal"]  # [3, H, H] in [0, 1]
        panel[:, row * H : (row + 1) * H, col * H : (col + 1) * H] = normal

    return panel


def _render_pbr_panel(mesh_with_voxel, render_size: int = 512) -> torch.Tensor:
    """Render MeshWithVoxel as 4-view PBR-shaded panel using default envmap.

    Returns [3, 2*H, 2*H] tensor in [0, 1].
    """
    renderer, envmap = _get_pbr_renderer_and_envmap(render_size)
    if renderer is None:
        return render_mesh_normal_panel(Mesh(mesh_with_voxel.vertices, mesh_with_voxel.faces), render_size)

    exts, ints = get_camera_views(render_size)
    H = render_size
    panel = torch.zeros(3, 2 * H, 2 * H, device="cuda")
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(mesh_with_voxel, ext, intr, envmap=envmap)
        row, col = j // 2, j % 2
        shaded = res["shaded"]  # [3, H, H]
        panel[:, row * H : (row + 1) * H, col * H : (col + 1) * H] = shaded

    return panel


def _get_pbr_renderer_and_envmap(render_size: int = 512):
    """Create PBR renderer and load environment map. Returns (renderer, envmap) or (None, None)."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # This module lives at <repo>/smd/utils/rendering/.
    envmap_path = str(Path(__file__).resolve().parents[3] / "assets" / "hdri" / "forest.exr")

    if not os.path.exists(envmap_path):
        logger.warning(f"[TrellisSample] envmap not found at {envmap_path}")
        return None, None

    envmap = EnvMap(
        torch.tensor(
            cv2.cvtColor(cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device="cuda",
        )
    )

    renderer = PbrMeshRenderer()
    renderer.rendering_options.resolution = render_size
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 2
    renderer.rendering_options.peel_layers = 8

    return renderer, envmap


def _compose_pbr_diagnostic(res: dict, render_size: int) -> torch.Tensor:
    """Compose a single-view PBR diagnostic panel from renderer output.

    Layout (matching TRELLIS.2 ``make_pbr_vis_frames``)::

        ┌──────────┬──────────┐
        │  shaded  │  normal  │   ← render_size × render_size each
        ├────┬─────┼────┬─────┤
        │base│metal│rough│alpha│   ← render_size/2 × render_size/2 each
        └────┴─────┴────┴─────┘

    Returns [3, 1.5*H, 2*H] tensor in [0, 1].
    """
    H = render_size
    half = H // 2

    shaded = res["shaded"]  # [3, H, H]
    normal = res["normal"]  # [3, H, H]

    # Base color: already [3, H, H], resize to half
    base_color = res.get("base_color", torch.zeros(3, H, H, device="cuda"))
    base_color = F.interpolate(base_color.unsqueeze(0), size=(half, half), mode="bilinear", align_corners=False)[0]

    # Single-channel maps → expand to 3-channel for visualization
    def _gray_to_rgb_half(t):
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [1, H, H]
        if t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        return F.interpolate(t.unsqueeze(0), size=(half, half), mode="bilinear", align_corners=False)[0]

    metallic = _gray_to_rgb_half(res.get("metallic", torch.zeros(H, H, device="cuda")))
    roughness = _gray_to_rgb_half(res.get("roughness", torch.zeros(H, H, device="cuda")))
    alpha = _gray_to_rgb_half(res.get("alpha", torch.zeros(H, H, device="cuda")))

    # Row 1: shaded + normal → [3, H, 2H]
    row1 = torch.cat([shaded, normal], dim=-1)
    # Row 2: base_color + metallic + roughness + alpha → [3, H/2, 2H]
    row2 = torch.cat([base_color, metallic, roughness, alpha], dim=-1)

    # Full panel: [3, 1.5H, 2H]
    panel = torch.cat([row1, row2], dim=-2)
    return panel.clamp(0, 1)


def render_diagnostic_panels(
    mesh_with_voxel,
    render_size: int = 512,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Render a MeshWithVoxel into (views_panel, pbr_panel).

    - ``views_panel``: 2×2 four-view shaded grid [3, 2H, 2H]
    - ``pbr_panel``: front-view PBR diagnostic [3, 1.5H, 2H]

    For Shape-only meshes (no PBR attrs), construct a clay MeshWithVoxel first,
    then call this function.
    """
    renderer, envmap = _get_pbr_renderer_and_envmap(render_size)
    if renderer is None:
        return None, None

    H = render_size
    exts, ints = get_camera_views(render_size)

    # 4-view shaded panel
    views_panel = torch.zeros(3, 2 * H, 2 * H, device="cuda")
    front_res = None
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(mesh_with_voxel, ext, intr, envmap=envmap)
        row, col = j // 2, j % 2
        views_panel[:, row * H : (row + 1) * H, col * H : (col + 1) * H] = res["shaded"]
        if j == 0:
            front_res = res  # keep front view for PBR diagnostic

    # Front-view PBR diagnostic panel
    pbr_panel = _compose_pbr_diagnostic(front_res, render_size) if front_res else None

    return views_panel.clamp(0, 1), pbr_panel


def mesh_to_clay_mesh_with_voxel(mesh, decode_resolution: int = 1024):
    """Wrap a bare Mesh into a MeshWithVoxel with uniform clay material.

    This allows Shape-only outputs (no PBR) to go through the same PBR
    renderer path — producing shaded + normal + uniform material channels.
    """

    V = mesh.vertices.shape[0]
    device = mesh.vertices.device

    # PBR attrs: base_color(3)=0.8, metallic(1)=0, roughness(1)=0.6, alpha(1)=1
    # Use .repeat() instead of .expand() — nvdiffrast requires contiguous tensors
    clay_attrs = torch.tensor([[0.8, 0.8, 0.8, 0.0, 0.6, 1.0]], device=device, dtype=torch.float32)
    clay_attrs = clay_attrs.repeat(V, 1)  # [V, 6] contiguous

    # Use vertex positions as voxel coords (quantized to grid)
    verts_norm = (mesh.vertices.float().contiguous() + 0.5) * decode_resolution
    coords_int = verts_norm.clamp(0, decode_resolution - 1).int().contiguous()

    return MeshWithVoxel(
        mesh.vertices.float().contiguous(),
        mesh.faces.contiguous(),
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / decode_resolution,
        coords=coords_int,
        attrs=clay_attrs,
        voxel_shape=torch.Size([1, 6, decode_resolution, decode_resolution, decode_resolution]),
        layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
    )


def _make_trellis_mesh(vertices_np, faces_np, fit_half_extent: float = 0.5):
    """Wrap marching-cubes output into a TRELLIS ``Mesh``, fit to [-h, h] cube.

    Hunyuan ``latents2mesh`` returns vertices in roughly ``[-bounds, bounds]``
    (bounds ~ 1.01). The TRELLIS nvdiffrast camera expects vertices in
    ``[-0.5, 0.5]``, so we recenter and uniformly rescale to fill that cube.
    """
    from smd.networks.Trellis2.representations.mesh import Mesh

    v = torch.as_tensor(vertices_np, dtype=torch.float32, device="cuda")
    f = torch.as_tensor(faces_np.copy(), dtype=torch.int32, device="cuda")
    vmin = v.amin(dim=0)
    vmax = v.amax(dim=0)
    center = 0.5 * (vmin + vmax)
    scale = (vmax - vmin).max().clamp(min=1e-6)
    v = (v - center) / scale * (2 * fit_half_extent)
    return Mesh(v, f)


def _render_clay_panel(mesh, render_size: int = 512) -> torch.Tensor:
    """4-view white-clay Lambertian shaded panel ``[3, 2H, 2H]`` in [0, 1].

    Shading is synthesized from the renderer's per-view normal + mask: a white
    material lit by a fixed head-light gives a clean "white model" look for
    side-by-side comparison with the normal panel.
    """
    from smd.utils.rendering.render_utils import get_renderer

    NVDIFFRAST_FACE_LIMIT = 16_777_216
    if mesh.faces.shape[0] > NVDIFFRAST_FACE_LIMIT:
        mesh.simplify(NVDIFFRAST_FACE_LIMIT)

    exts, ints = get_camera_views(render_size)
    renderer = get_renderer(mesh, resolution=render_size)
    H = render_size
    panel = torch.zeros(3, 2 * H, 2 * H, device="cuda")
    # Head-light roughly along the camera/view direction (world +z-ish, up tilt).
    light = torch.tensor([0.3, 0.4, 0.85], device="cuda")
    light = light / light.norm()
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(mesh, ext, intr, return_types=["normal", "mask"])
        normal = res["normal"]  # [3, H, H] in [0, 1] (0.5 = no surface)
        mask = res["mask"]  # [H, H]
        n = normal * 2.0 - 1.0  # back to [-1, 1]
        n = n / n.norm(dim=0, keepdim=True).clamp(min=1e-6)
        lambert = (n * light.view(3, 1, 1)).sum(0).clamp(0, 1)  # [H, H]
        shade = (0.2 + 0.8 * lambert) * mask  # ambient + diffuse, masked
        rgb = shade.unsqueeze(0).repeat(3, 1, 1)  # white material
        row, col = j // 2, j % 2
        panel[:, row * H : (row + 1) * H, col * H : (col + 1) * H] = rgb
    return panel
