# SPDX-License-Identifier: Apache-2.0
"""One sampling/collection path for batch and fixed-case previews.

All ranks execute the same number of student forwards. CP peers share noise;
only one peer contributes media. Gather only the requested samples, on CPU, to
rank zero. Model-specific code here describes latent layouts, not schedules.
"""

from __future__ import annotations

import io
import math
from dataclasses import fields, is_dataclass, replace

import numpy as np
import torch
import torch.distributed as dist

from smd.callbacks.sampling.base import sanitize_stem
from smd.callbacks.sampling.rendering.decoder import DecoderSpec, describe_decoder
from smd.callbacks.sampling.rendering.protocol import VisualizationItem
from smd.networks.io_contract import NetworkIOAdapter
from smd.utils.distributed import get_rank, is_rank0, world_size


def copy_tree(value, device=None):
    """Detach preview inputs without sharing mutable training tensors."""
    if torch.is_tensor(value):
        return value.detach().to(device=device or value.device, copy=True)
    if hasattr(value, "feats") and hasattr(value, "coords"):
        # SparseTensor.replace shares spatial/attention caches, including
        # device-specific indices. A preview owns a fresh sparse structure.
        return type(value)(
            feats=copy_tree(value.feats, device),
            coords=copy_tree(value.coords, device),
            shape=value.shape,
            scale=value._scale,
        )
    if is_dataclass(value):
        return replace(value, **{f.name: copy_tree(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: copy_tree(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(copy_tree(v, device) for v in value)
    return value


def topology(net):
    cp = getattr(net, "context_parallel_state", None)
    return (cp.data_parallel_rank, cp.data_parallel_size, cp.rank == 0) if cp else (get_rank(), world_size(), True)


def sample_student(model, net, noise, condition, generator):
    sampler = getattr(model, "sample_student_with_generator", None)
    if sampler is not None:
        return sampler(noise=noise, condition=condition, net=net, generator=generator)
    return model.sample_student(noise=noise, condition=condition, net=net)


def _pixels(encoded):
    from PIL import Image

    with Image.open(io.BytesIO(encoded)) as image:
        return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0)


def _caption(data, i):
    for key in ("condition_raw", "fname"):
        values = data.get(key)
        if isinstance(values, (list, tuple)) and len(values) > i and isinstance(values[i], str):
            return values[i]
    return None


def _gather(records, *, device, limit):
    if not dist.is_initialized() or world_size() == 1:
        return copy_tree(records[:limit], "cpu")
    count = torch.tensor(len(records), device=device, dtype=torch.long)
    counts = [torch.zeros_like(count) for _ in range(world_size())]
    dist.all_gather(counts, count)
    offset = sum(int(c.item()) for c in counts[: get_rank()])
    local = copy_tree(records[: max(0, limit - offset)], "cpu")
    buckets = [None] * world_size() if is_rank0() else None
    dist.gather_object(local, object_gather_list=buckets, dst=0)
    return [r for bucket in buckets for r in bucket][:limit] if buckets is not None else []


def sample_records(model, net, data, *, grid_size, seed, include_real):
    """Sample already-preprocessed data; never call the training lifecycle."""
    adapter = NetworkIOAdapter.from_model(net)
    family = net.network_family
    rank, size, contributes = topology(net)
    available = adapter.batch_size(data["condition"])
    count = min(available, math.ceil(grid_size / size))
    if dist.is_initialized():
        agreed = torch.tensor(count, device=model.device, dtype=torch.long)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        count = int(agreed.item())
    if count < 1:
        raise ValueError("A preview rank has no conditions.")
    generator = torch.Generator(device=model.device).manual_seed(seed + rank * 100_003)
    condition = copy_tree(adapter.trim_padding(adapter.slice(data["condition"], count)))
    real = data.get("real")
    source = data.get("source_latents")
    concat = data.get("concat_cond")
    records = []
    if family == "minimax_h3":
        from smd.networks.MiniMaxH3.network import MiniMaxH3Latents

        shapes = [tuple(int(v) for v in data[k][0].tolist()) for k in ("video_latent_shape", "audio_latent_shape")]
        noise = MiniMaxH3Latents(
            *(torch.randn((1, *s), device=model.device, dtype=torch.float32, generator=generator) for s in shapes)
        )
        # Match native training/inference reference corruption, without changing
        # the cached clean reference or borrowing the training RNG stream.
        t = net.condition_video_timestep
        condition = replace(
            condition,
            video_latents=tuple(
                (
                    t * v.float()
                    + (1 - t) * torch.randn(v.shape, device=v.device, dtype=torch.float32, generator=generator)
                ).to(v.dtype)
                for v in condition.video_latents
            ),
        )
        result = sample_student(model, net, noise, condition, generator)
        records = [
            {
                "student": {"video": result.video, "audio": result.audio},
                "caption": _caption(data, 0),
                "fps": int(data["fps"].item()),
            }
        ]
        clean = data["condition"]
        if clean.video_latents and clean.audio_latents:
            records[0]["reference"] = {"video": clean.video_latents[0], "audio": clean.audio_latents[0]}
    else:
        extras = {}
        if source is not None:
            extras["source_latents"] = copy_tree(source[:count])
        if concat is not None:
            extras["concat_cond"] = copy_tree(concat[:count])
        condition = adapter.pack_condition(condition, extras)
        sparse = hasattr(real, "feats") and hasattr(real, "coords")
        if sparse:
            structure = copy_tree(real[:count])
            noise = structure.replace(
                torch.randn(structure.feats.shape, device=model.device, dtype=model.precision, generator=generator)
            )
        else:
            shape = (
                list(source.shape[1:])
                if source is not None
                else list(real.shape[1:])
                if torch.is_tensor(real)
                else list(model.input_shape)
            )
            noise = torch.randn((count, *shape), device=model.device, dtype=model.precision, generator=generator)
        result = sample_student(model, net, noise, condition, generator)
        for i in range(count):
            record = {"caption": _caption(data, i)}
            if sparse:
                sample = result[i : i + 1]
                record["student"] = {"student_feats": sample.feats, "student_coords": sample.coords}
                if concat is not None:
                    shape = concat[i : i + 1]
                    record["student"].update(shape_feats=shape.feats, shape_coords=shape.coords)
                if include_real:
                    target = real[i : i + 1]
                    record["student"].update(gt_feats=target.feats, gt_coords=target.coords)
            else:
                record["student"] = result[i : i + 1]
                if source is not None:
                    record["source"] = source[i : i + 1]
                raw = data.get("real_raw")
                if include_real and torch.is_tensor(raw) and raw.ndim == 4 and raw.shape[1] in (1, 3):
                    record["real_pixels"] = raw[i : i + 1].detach().add(1).mul(127.5).clamp(0, 255).byte()
                elif include_real and torch.is_tensor(real):
                    record["real"] = real[i : i + 1]
            images = data.get("condition_images_raw")
            if images:
                record["input_pixels"] = _pixels(images[i])
            references = data.get("reference_image_raw")
            if include_real and references:
                record["reference_pixels"] = _pixels(references[i])
            records.append(record)
    return _gather(records if contributes else [], device=model.device, limit=grid_size)


def decoder_spec(net):
    family = net.network_family
    if family == "hunyuan3d_shape":
        return DecoderSpec(
            "smd.callbacks.sampling.rendering.three_d.Hunyuan3DPreviewRenderer",
            {"model_id": net.model_id, "octree_resolution": 192, "render_size": 512},
            "render",
        )
    if family == "trellis2":
        from pathlib import Path

        shape, _ = net._resolve_from_hf(net.model_id, "shape_dec_next_dc_f16c32_fp16")
        kwargs = {"shape_decoder_path": str(Path(shape).with_suffix("")), "decode_resolution": 1024, "render_size": 512}
        name = "TrellisShapePreviewRenderer"
        if net.ckpt_name != net.SHAPE_CKPT:
            texture, _ = net._resolve_from_hf(net.model_id, "tex_dec_next_dc_f16c32_fp16")
            kwargs["tex_decoder_path"] = str(Path(texture).with_suffix(""))
            name = "TrellisTexturePreviewRenderer"
        return DecoderSpec("smd.callbacks.sampling.rendering.three_d." + name, kwargs, "render")
    return describe_decoder(net)


def render_items(records, *, family, prefix, fps, omit_single_index=False):
    """Build media requests; a single fixed result can use its case key directly."""

    def item(name, **kwargs):
        key = "/".join(part for part in (prefix, name.strip("/")) if part)
        return VisualizationItem(key=key, stem=sanitize_stem(key), fps=fps, **kwargs)

    if not records:
        return []
    if family not in ("minimax_h3", "hunyuan3d_shape", "trellis2") and "source" not in records[0]:
        caption = "\n".join(r["caption"] or "" for r in records)
        items = [item("student", caption=caption, latents_list=[r["student"] for r in records])]
        for key in ("real", "real_pixels"):
            if all(key in r for r in records):
                kw = (
                    {"pixels_list": [r[key] for r in records], "normalized": True}
                    if key == "real_pixels"
                    else {"latents_list": [r[key] for r in records]}
                )
                items.append(item("real", caption=caption, **kw))
        return items
    items = []
    for i, record in enumerate(records):
        name = "" if omit_single_index and len(records) == 1 else f"{i:02d}"
        caption = record["caption"]
        if family == "minimax_h3":
            for kind in ("student", "reference"):
                if kind in record:
                    media = item(f"{name}/{kind}", caption=caption, joint_latents=record[kind], normalized=True)
                    media.fps = record["fps"]
                    items.append(media)
        elif family in ("hunyuan3d_shape", "trellis2"):
            data = {"student_latents": record["student"]} if family == "hunyuan3d_shape" else record["student"]
            items.append(item(name, caption=caption, render_data=data, normalized=True))
            if "input_pixels" in record:
                items.append(item(f"{name}/input", caption=caption, pixels=record["input_pixels"], normalized=True))
            if "reference_pixels" in record:
                items.append(
                    item(
                        f"{name}/reference",
                        caption=f"Cached reference: {caption}",
                        pixels=record["reference_pixels"],
                        normalized=True,
                    )
                )
        else:
            panels = [{"latents": record["source"]}, {"latents": record["student"]}]
            if "real_pixels" in record:
                panels.append({"pixels": record["real_pixels"].float().div(127.5).sub(1)})
            elif "real" in record:
                panels.append({"latents": record["real"]})
            items.append(item(name, caption=caption, panels=panels))
    return items
