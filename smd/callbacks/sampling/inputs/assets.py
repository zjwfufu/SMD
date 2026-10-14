# SPDX-License-Identifier: Apache-2.0
"""Read fixed preview assets; never start a training iterator or an encoder."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def asset_cases(manifest_path, *, family):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("family") != family:
        raise ValueError(f"Incompatible fixed preview manifest: {path} ({family})")
    records = manifest["records"]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError(f"Fixed preview records must have unique IDs: {path}")
    cases = []
    for record in records:
        files = record["files"]

        def read(key, files=files, case_id=record["id"]):
            entry = files[key]
            payload = (path.parent / entry["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Fixed preview checksum mismatch: {case_id}/{key}")
            return payload

        def array(key):
            return torch.from_numpy(np.load(io.BytesIO(read(key)), allow_pickle=False).copy())

        if family == "qwen_image_edit":
            from smd.networks.QwenImageEdit.preprocessing import source_pixels

            with Image.open(io.BytesIO(read("source"))) as image:
                source = image.convert("RGB").copy()
            pixels = source_pixels(source)
            with Image.open(io.BytesIO(read("target"))) as image:
                target = image.convert("RGB").resize((pixels.shape[2], pixels.shape[1]), Image.Resampling.BICUBIC)
                target = torch.from_numpy(np.array(target)).permute(2, 0, 1).float().div(127.5).sub(1)
            prompt = read("prompt").decode("utf-8").strip()
            if not prompt:
                raise ValueError(f"Empty edit instruction: {record['id']}")
            data = {
                "source": pixels.unsqueeze(0),
                "source_pil": [source],
                "real": target.unsqueeze(0),
                "condition": [prompt],
            }
        elif family == "hunyuan3d_shape":
            reference = torch.load(io.BytesIO(read("reference_latents")), map_location="cpu", weights_only=True)
            if not torch.is_tensor(reference) or reference.shape != (1, 4096, 64) or not reference.isfinite().all():
                raise ValueError(f"Invalid cached Hunyuan3D reference: {record['id']}")
            data = {"real": reference, "condition_images": [read("image")], "condition": [torch.zeros(1, 1024)]}
        elif family == "trellis2":
            from smd.networks.Trellis2.normalization import (
                PBR_NORM_MEAN,
                PBR_NORM_STD,
                SHAPE_NORM_MEAN,
                SHAPE_NORM_STD,
            )
            from smd.networks.Trellis2.sparse import SparseTensor

            for key, mean, std in (
                ("shape_normalization", SHAPE_NORM_MEAN, SHAPE_NORM_STD),
                ("texture_normalization", PBR_NORM_MEAN, PBR_NORM_STD),
            ):
                if not torch.equal(torch.tensor(manifest[key]["mean"]), mean) or not torch.equal(
                    torch.tensor(manifest[key]["std"]), std
                ):
                    raise ValueError(f"Cached TRELLIS normalization differs: {key}")
            condition, xyz, shape = array("condition").float(), array("coords"), array("shape_latents").float()
            if condition.ndim != 2 or condition.shape[1] != 1024 or not condition.isfinite().all():
                raise ValueError(f"Invalid cached DINOv3 condition: {record['id']}")
            if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.is_floating_point() or shape.shape != (len(xyz), 32):
                raise ValueError(f"Mismatched cached shape/coordinates: {record['id']}")
            if not shape.isfinite().all():
                raise ValueError(f"Nonfinite cached shape: {record['id']}")
            coords = torch.cat([torch.zeros(len(xyz), 1, dtype=torch.int32), xyz.int()], dim=1)
            shape = SparseTensor(feats=shape, coords=coords)
            data = {
                "real": shape,
                "condition": [condition],
                "neg_condition": [torch.zeros_like(condition)],
                "condition_images_raw": [read("image")],
            }
            if manifest["mode"] == "texture":
                texture = array("texture_latents").float()
                if texture.shape != shape.feats.shape or not texture.isfinite().all():
                    raise ValueError(f"Mismatched cached texture/shape: {record['id']}")
                data.update(
                    real=SparseTensor(feats=(texture - PBR_NORM_MEAN) / PBR_NORM_STD, coords=coords),
                    concat_cond=shape,
                )
            elif manifest["mode"] != "shape":
                raise ValueError(f"Unknown cached TRELLIS mode: {manifest['mode']}")
        else:
            raise ValueError(f"No fixed-asset reader for {family}")
        data["fname"] = [record["id"]]
        if "reference_render" in files:
            data["reference_image_raw"] = [read("reference_render")]
        cases.append(data)
    return cases
