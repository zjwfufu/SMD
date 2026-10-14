# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored & adapted (DECODE-ONLY) from Tencent Hunyuan3D-2.1
#   hy3dshape/hy3dshape/models/autoencoders/{model,attention_blocks,
#                                            volume_decoders,surface_extractors}.py
# Original: Copyright (C) 2024 THL A29 Limited (Tencent), TENCENT HUNYUAN
# NON-COMMERCIAL LICENSE AGREEMENT.
#
# Only the decode path is vendored: post_kl -> transformer -> geo_decoder ->
# volume decode (dense grid) -> marching cubes. The point-cloud ENCODER,
# pre_kl, FlashVDM/Hierarchical decoders, and DMC extractor are intentionally
# omitted (we never encode shapes — distillation is data-free in latent space,
# and the encoder pulls in torch_cluster.fps). Loading uses strict=False so the
# unused encoder/pre_kl checkpoint keys are simply ignored.
#
# `scaled_dot_product_attention` and `sageattention` indirection from the
# original `attention_processors.py` are collapsed to a plain F.sdpa call.

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ======================================================================
# Positional embedding
# ======================================================================
class FourierEmbedder(nn.Module):
    """Sin/cos positional embedding of 3D query coordinates."""

    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


# ======================================================================
# Transformer blocks (decoder transformer + geo cross-attention)
# ======================================================================
class MLP(nn.Module):
    def __init__(self, *, width: int, expand_ratio: int = 4, output_width: int = None, drop_path_rate: float = 0.0):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        # drop_path_rate is always 0 in the released ShapeVAE config -> Identity
        self.drop_path = nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads=heads, width=width, norm_layer=norm_layer, qk_norm=qk_norm)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        if data_width is None:
            data_width = width
        self.attn = MultiheadCrossAttention(
            width=width, heads=heads, data_width=data_width, qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, *, width: int, heads: int, qkv_bias: bool, norm_layer=nn.LayerNorm, qk_norm: bool = False):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, width=width, norm_layer=norm_layer, qk_norm=qk_norm)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, *, width: int, heads: int, qkv_bias: bool = True, norm_layer=nn.LayerNorm, qk_norm: bool = False):
        super().__init__()
        self.attn = MultiheadAttention(width=width, heads=heads, qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm)
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, *, n_ctx: int, width: int, layers: int, heads: int, qkv_bias: bool = True,
        norm_layer=nn.LayerNorm, qk_norm: bool = False, drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width=width, heads=heads, qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):
    """Queries the latent set at 3D points to predict occupancy logits."""

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
    ):
        super().__init__()
        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post is False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width, mlp_expand_ratio=mlp_expand_ratio, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm
        )
        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ


# ======================================================================
# Volume decoding (dense grid) + marching cubes
# ======================================================================
def generate_dense_grid_points(bbox_min, bbox_max, octree_resolution, indexing="ij"):
    length = bbox_max - bbox_min
    num_cells = octree_resolution
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]
    return xyz, grid_size, length


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = False,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min=bbox_min, bbox_max=bbox_max, octree_resolution=octree_resolution, indexing="ij"
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            chunk_queries = xyz_samples[start : start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()
        return grid_logits


def _extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """Mask of grid cells near the iso-surface (sign change among 6 neighbors).

    Vendored from Tencent ``volume_decoders.extract_near_surface_volume_fn``.
    Used by the hierarchical (sparse-octree) decoder to pick which coarse cells
    to refine. ``-9000`` sentinels mark invalid (already-pruned) cells.
    """
    val = input_tensor + alpha
    valid_mask = val > -9000

    def get_neighbor(t, shift, axis):
        if shift == 0:
            return t.clone()
        pad_dims = [0, 0, 0, 0, 0, 0]  # [x-, x+, y-, y+, z-, z+]
        if axis == 0:
            pad_dims[0 if shift > 0 else 1] = abs(shift)
        elif axis == 1:
            pad_dims[2 if shift > 0 else 3] = abs(shift)
        elif axis == 2:
            pad_dims[4 if shift > 0 else 5] = abs(shift)
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode="replicate")
        slice_dims = [slice(None)] * 3
        sl = slice(shift, None) if shift > 0 else slice(None, shift)
        slice_dims[axis] = sl
        padded = padded.squeeze(0).squeeze(0)
        return padded[tuple(slice_dims)]

    left = get_neighbor(val, 1, axis=0)
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)
    up = get_neighbor(val, -1, axis=2)

    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left, right = safe_where(left), safe_where(right)
    back, front = safe_where(back), safe_where(front)
    down, up = safe_where(down), safe_where(up)

    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack(
        [
            torch.sign(left.to(torch.float32)),
            torch.sign(right.to(torch.float32)),
            torch.sign(back.to(torch.float32)),
            torch.sign(front.to(torch.float32)),
            torch.sign(down.to(torch.float32)),
            torch.sign(up.to(torch.float32)),
        ],
        dim=0,
    )
    same_sign = torch.all(neighbors_sign == sign, dim=0)
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


class HierarchicalVolumeDecoding:
    """Sparse-octree volume decoding: query ``geo_decoder`` densely at a COARSE
    grid, then only refine cells near the surface at each finer level.

    Vendored from Tencent ``volume_decoders.HierarchicalVolumeDecoding``. Gives
    the same fidelity as ``VanillaVolumeDecoder`` at a high ``octree_resolution``
    but at a fraction of the geo_decoder queries (which dominate runtime: dense
    256³ ≈ 17M points ≈ 37s; hierarchical reaches 256 in a few seconds).

    Does NOT depend on FlashVDM / sageattention — only standard torch ops + the
    same ``geo_decoder(queries=, latents=)`` interface. Empty regions are filled
    with NaN so marching cubes skips them (matching the official MC extractor,
    which feeds the NaN grid straight to ``skimage.measure.marching_cubes``).
    """

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        enable_pbar: bool = False,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min=bbox_min, bbox_max=bbox_max, octree_resolution=resolutions[0], indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # Level 0: dense query of the coarse grid.
        batch_size = latents.shape[0]
        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start : start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=batch_queries, latents=latents)
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2]))

        # Finer levels: refine only near-surface cells.
        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000.0, dtype=dtype, device=device)
            curr_points = _extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            expand_num = 0 if octree_depth_now == resolutions[-1] else 1
            for _ in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for _ in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = next_points * torch.tensor(
                resolution, dtype=next_points.dtype, device=device
            ) + torch.tensor(bbox_min, dtype=next_points.dtype, device=device)

            # Empty-guard: if no near-surface cells were found at this level (can
            # happen on degenerate / low-detail shapes where the coarse grid has
            # no sign change), fall back to a DENSE decode at the final target
            # resolution. Without this, ``torch.cat([])`` below crashes.
            if next_points.shape[0] == 0:
                return VanillaVolumeDecoder()(
                    latents, geo_decoder, bounds=bounds, num_chunks=num_chunks,
                    octree_resolution=resolutions[-1], enable_pbar=enable_pbar,
                )

            batch_logits = []
            for start in range(0, next_points.shape[0], num_chunks):
                queries = next_points[start : start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                logits = geo_decoder(queries=batch_queries.to(latents.dtype), latents=latents)
                batch_logits.append(logits)
            grid_logits = torch.cat(batch_logits, dim=1)
            next_logits[nidx] = grid_logits[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)

        grid_logits[grid_logits == -10000.0] = float("nan")
        return grid_logits


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


class MCSurfaceExtractor:
    """Marching cubes via scikit-image (lazy-imported; only needed for viz)."""

    def _box_stat(self, bounds, octree_resolution):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1] * 3
        return grid_size, bbox_min, bbox_size

    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        from skimage import measure

        vertices, faces, _normals, _ = measure.marching_cubes(
            grid_logit.cpu().numpy(), mc_level, method="lewiner"
        )
        grid_size, bbox_min, bbox_size = self._box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces

    def __call__(self, grid_logits, **kwargs):
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                if len(vertices) == 0:
                    raise ValueError("Empty mesh from marching cubes")
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))
            except Exception as e:
                # mc_level=0 has no zero-crossing (common for early-training noise).
                # Fallback: normalize grid to [-1, 1] then cut at 0. This guarantees
                # a surface regardless of the original value range or skimage version.
                try:
                    g = grid_logits[i]
                    g_min, g_max = float(g.min()), float(g.max())
                    if g_max - g_min < 1e-6:
                        print(f"[MC fallback] flat grid (range={g_max - g_min:.2e}), skipping", flush=True)
                        outputs.append(None)
                        continue
                    g_np = g.cpu().numpy().astype(np.float64)
                    g_normalized = 2.0 * (g_np - g_min) / (g_max - g_min) - 1.0
                    print(
                        f"[MC fallback] primary failed ({e}), trying normalized: "
                        f"range=[{g_min:.4f},{g_max:.4f}] norm_range=[{g_normalized.min():.4f},{g_normalized.max():.4f}]",
                        flush=True,
                    )
                    from skimage import measure as _measure

                    vertices, faces, _, _ = _measure.marching_cubes(g_normalized, 0.0, method="lewiner")
                    print(f"[MC fallback] normalized MC: {len(vertices)} verts, {len(faces)} faces", flush=True)
                    if len(vertices) == 0:
                        outputs.append(None)
                        continue
                    # Map vertices back to original grid coordinates
                    grid_size, bbox_min, bbox_size = self._box_stat(
                        kwargs.get("bounds", 1.01), kwargs.get("octree_resolution", 256)
                    )
                    vertices = vertices / grid_size * bbox_size + bbox_min
                    vertices = vertices.astype(np.float32)
                    faces = np.ascontiguousarray(faces)
                    outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))
                except Exception as e2:
                    print(f"[MC fallback] also failed: {e2}", flush=True)
                    outputs.append(None)
        return outputs


# ======================================================================
# ShapeVAE (decode-only)
# ======================================================================
class ShapeVAE(nn.Module):
    """Decode-only Hunyuan3D vec2set VAE: latent set [B, num_latents, embed_dim]
    -> occupancy grid -> mesh. Encoder/pre_kl omitted (load with strict=False)."""

    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,  # unused (encoder omitted); kept for config compat
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        use_ln_post: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

        # Vanilla = dense full-grid query (simple, slow at high res).
        # Hierarchical = sparse octree (same fidelity, far fewer geo_decoder
        # queries) — the default for viz so high octree stays fast.
        self.volume_decoder = VanillaVolumeDecoder()
        self.hierarchical_decoder = HierarchicalVolumeDecoding()
        self.surface_extractor = MCSurfaceExtractor()

    def decode(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    forward = decode

    def latents2mesh(
        self,
        latents: torch.FloatTensor,
        bounds: Union[float, list] = 1.01,
        mc_level: float = 0.0,
        num_chunks: int = 10000,
        octree_resolution: int = 256,
        use_hierarchical: bool = False,
        enable_pbar: bool = False,
        **kwargs,
    ):
        """Decode latents → occupancy grid → mesh.

        ``use_hierarchical=False`` (default) uses the dense decoder — robust on
        any shape, matching the official ``minimal_demo`` default. Set
        ``octree_resolution`` to trade quality for speed (192 ≈ a few s; 256 ≈
        37 s).

        ``use_hierarchical=True`` uses the sparse-octree decoder: much faster at
        high resolutions on CLEAN shapes, but **unreliable on blurry / degenerate
        shapes** — its near-surface refinement can drop the surface entirely
        (empty or all-negative grid → "no surface"). Since distillation routinely
        produces blurry intermediate shapes, the callback keeps the dense default.
        The hierarchical path is retained for clean-shape / final-export use.
        """
        if use_hierarchical:
            grid_logits = self.hierarchical_decoder(
                latents,
                self.geo_decoder,
                bounds=bounds,
                num_chunks=num_chunks,
                mc_level=mc_level,
                octree_resolution=octree_resolution,
                enable_pbar=enable_pbar,
            )
        else:
            grid_logits = self.volume_decoder(
                latents,
                self.geo_decoder,
                bounds=bounds,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                enable_pbar=enable_pbar,
            )
        outputs = self.surface_extractor(
            grid_logits, mc_level=mc_level, bounds=bounds, octree_resolution=octree_resolution
        )
        return outputs
