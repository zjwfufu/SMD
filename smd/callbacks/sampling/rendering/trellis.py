# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate TRELLIS decoding and preview panels."""
from __future__ import annotations
import torch
from smd.networks.Trellis2.decoders import _load_decoder
from smd.networks.Trellis2.sparse import SparseTensor
from smd.networks.Trellis2.representations.mesh import Mesh, MeshWithVoxel
from smd.utils.rendering.panels import render_mesh_normal_panel, render_diagnostic_panels
from smd.utils import logging_utils as logger

def slice_first_sample(sp):
    """Extract the first sample (batch_idx == 0) from a batched SparseTensor."""
    mask = sp.coords[:, 0] == 0
    feats = sp.feats[mask]
    coords = sp.coords[mask].clone()
    coords[:, 0] = 0  # ensure batch idx is 0
    return sp.replace(feats).__class__(
        {"feats": feats, "coords": coords},
        shape=torch.Size([1, feats.shape[-1]]) if hasattr(sp, "_shape") and sp._shape is not None else None,
        scale=getattr(sp, "_scale", None),
    )


@torch.no_grad()
def decode_and_render_shape(
    shape_slat,
    decoder_path: str,
    decode_resolution: int = 512,
    render_size: int = 512,
    denormalize: bool = True,
):
    """Decode shape SparseTensor → Mesh → (views_panel, pbr_panel, meshes, subs).

    Parameters
    ----------
    denormalize : bool
        If True, reverse the dataloader's per-channel normalization before
        feeding to the decoder. The decoder expects raw (un-normalized) latents.

    Returns:
        views_panel: 4-view shaded grid [3, 2H, 2H] or None
        pbr_panel: front-view PBR diagnostic [3, 1.5H, 2H] or None
        meshes: list of decoded Mesh objects
        subs: subdivision structures (needed for texture decode)
    """
    from smd.networks.Trellis2.normalization import SHAPE_NORM_MEAN, SHAPE_NORM_STD

    decoder = _load_decoder(decoder_path)
    decoder.set_resolution(decode_resolution)

    try:
        # Build a clean SparseTensor without stale caches from training forward pass.
        # Cast to float32 to match decoder's from_latent layer.
        slat_cuda = shape_slat.cuda()
        feats = slat_cuda.feats.float()
        if denormalize:
            feats = feats * SHAPE_NORM_STD.to(feats.device) + SHAPE_NORM_MEAN.to(feats.device)
        clean_sp = SparseTensor(
            {"feats": feats, "coords": slat_cuda.coords.clone()},
            shape=slat_cuda.shape,
        )
        meshes, subs = decoder(clean_sp, return_subs=True)

        # Wrap external Mesh objects into our vendored Mesh class
        # (decoder returns trellis2.representations.mesh.Mesh from the external repo,
        # but our renderers expect smd.networks.Trellis2.representations.mesh.Mesh)
        meshes = [Mesh(m.vertices, m.faces) for m in meshes]
    except Exception as e:
        logger.warning(f"[TrellisSample] shape decode failed: {e}")
        return None, None, None, None
    finally:
        del decoder
        import gc

        gc.collect()  # Break traceback reference cycles to free GPU tensors
        torch.cuda.empty_cache()

    # Fill holes for cleaner rendering
    try:
        for m in meshes:
            m.fill_holes()
    except Exception:
        pass  # fill_holes requires cumesh; skip if unavailable

    mesh = meshes[0]
    NVDIFFRAST_FACE_LIMIT = 16_777_216
    if mesh.faces.shape[0] > NVDIFFRAST_FACE_LIMIT:
        mesh.simplify(NVDIFFRAST_FACE_LIMIT)

    # Shape has no PBR attributes — render normal map panel only
    views_panel = render_mesh_normal_panel(mesh, render_size)

    return views_panel, None, meshes, subs


@torch.no_grad()
def decode_and_render_texture(
    tex_slat,
    shape_slat,
    shape_decoder_path: str,
    tex_decoder_path: str,
    decode_resolution: int = 512,
    render_size: int = 512,
    denormalize: bool = True,
):
    """Decode texture SparseTensor → MeshWithVoxel → (views_panel, pbr_panel).

    Parameters
    ----------
    denormalize : bool
        If True, reverse the dataloader's per-channel normalization on both
        shape and texture latents before feeding to decoders.

    Loads both shape decoder (for mesh + subs) and texture decoder.
    Returns (views_panel, pbr_panel) or (None, None) on failure.
    """
    from smd.networks.Trellis2.normalization import SHAPE_NORM_MEAN, SHAPE_NORM_STD
    from smd.networks.Trellis2.normalization import PBR_NORM_MEAN, PBR_NORM_STD

    # Step 1: Decode shape → mesh + subs
    shape_dec = _load_decoder(shape_decoder_path)
    shape_dec.set_resolution(decode_resolution)
    try:
        slat_cuda = shape_slat.cuda()
        shape_feats = slat_cuda.feats.float()
        if denormalize:
            shape_feats = shape_feats * SHAPE_NORM_STD.to(shape_feats.device) + SHAPE_NORM_MEAN.to(shape_feats.device)
        clean_sp = SparseTensor(
            {"feats": shape_feats, "coords": slat_cuda.coords.clone()},
            shape=slat_cuda.shape,
        )
        meshes, subs = shape_dec(clean_sp, return_subs=True)
    except Exception as e:
        logger.warning(f"[TrellisSample] shape decode (for texture) failed: {e}")
        return None
    finally:
        del shape_dec
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # Step 2: Decode texture → PBR voxels
    tex_dec = _load_decoder(tex_decoder_path)
    try:
        tex_cuda = tex_slat.cuda()
        tex_feats = tex_cuda.feats.float()
        if denormalize:
            tex_feats = tex_feats * PBR_NORM_STD.to(tex_feats.device) + PBR_NORM_MEAN.to(tex_feats.device)
        clean_tex = SparseTensor(
            {"feats": tex_feats, "coords": tex_cuda.coords.clone()},
            shape=tex_cuda.shape,
        )
        pbr_voxels = tex_dec(clean_tex, guide_subs=subs) * 0.5 + 0.5
    except Exception as e:
        logger.warning(f"[TrellisSample] texture decode failed: {e}")
        return None
    finally:
        del tex_dec
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # Step 3: Build MeshWithVoxel
    mesh = meshes[0]
    try:
        mesh.fill_holes()
    except Exception:
        pass
    # Stay within nvdiffrast rasterization limits (same as app.py)
    NVDIFFRAST_FACE_LIMIT = 16_777_216
    if mesh.faces.shape[0] > NVDIFFRAST_FACE_LIMIT:
        mesh.simplify(NVDIFFRAST_FACE_LIMIT)

    # pbr_voxels is a SparseTensor with feats [N, 6]: base_color(3), metallic(1), roughness(1), alpha(1)
    vox = pbr_voxels[0] if isinstance(pbr_voxels, list) else pbr_voxels
    # For single-sample, coords[:, 0] == 0; drop batch index
    vox_coords = vox.coords[:, 1:] if vox.coords.shape[1] == 4 else vox.coords

    # When decode_resolution matches the decoder's internal grid (1024),
    # vertices are already in [-0.5, 0.5] world space.  Use the standard
    # origin/voxel_size from the official pipeline (decode_latent).
    mesh_with_voxel = MeshWithVoxel(
        mesh.vertices.float(),
        mesh.faces,  # nvdiffrast requires float32
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / decode_resolution,
        coords=vox_coords,
        attrs=vox.feats.float(),  # nvdiffrast texture sampling requires float32
        voxel_shape=torch.Size([*vox.shape, *vox.spatial_shape])
        if hasattr(vox, "spatial_shape")
        else torch.Size([1, vox.feats.shape[-1], decode_resolution, decode_resolution, decode_resolution]),
        layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
    )

    # Step 4: PBR render — both views panel and front PBR diagnostic
    views_panel, pbr_panel = render_diagnostic_panels(mesh_with_voxel, render_size)

    return views_panel, pbr_panel
