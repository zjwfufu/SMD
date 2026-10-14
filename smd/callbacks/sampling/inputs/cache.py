# SPDX-License-Identifier: Apache-2.0
"""Versioned, encoder-free fixed preview inputs, with checked tensor payloads."""

import hashlib
import io
import json
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]


def cache_key(config):
    net = config.model.net
    target = str(net._target_)
    for token, key in (
        ("SD35Medium", "sd35"),
        ("Flux2", "flux2_4b"),
        ("Flux", "flux1"),
        ("Lens", "lens"),
        ("QwenImageEdit", "qwen_image_edit"),
        ("QwenImage", "qwen_image"),
        ("Hunyuan3DShape", "hunyuan3d_shape"),
    ):
        if f".{token}." in target:
            return key
    if ".MiniMaxH3." in target:
        return "h3_" + config.model.h3_condition_mode
    if ".Trellis2." in target:
        return "trellis2_texture" if "tex" in net.ckpt_name else "trellis2_shape"
    if target.endswith(".CausalWan"):
        return "wan13_causal"
    if ".Wan." in target:
        return "wan14" if "14B" in str(net.get("model_id_or_local_path", "")) else "wan13"
    raise ValueError(f"No fixed preview cache for {target}")


def cache_path(config):
    return f"assets/prompt_samples/cache/{cache_key(config)}/manifest.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(config, source):
    """Bind caches to inputs, student preprocessing and its numerical policy.

    Training teachers, adapters, NFE and distributed topology do not change
    the student's fixed conditions. Local aliases of a model need an explicit
    rebuild, as do changes to preprocessing code or paired source manifests.
    """
    net = config.model.net
    model = {
        k: net.get(k)
        for k in ("model_id", "model_id_or_local_path", "model_revision", "ckpt_name", "disable_mxfp4")
        if k in net
    }
    value = dict(
        key=cache_key(config),
        model=model,
        precision=str(config.model.precision),
        precision_amp_enc=str(config.model.precision_amp_enc),
        neg_prompt=config.model.neg_prompt,
        source=source,
    )
    if "assets" in source:
        value["asset_manifest"] = digest(ROOT / source["assets"])
    elif "dataset" in source:
        folder = ROOT / source["dataset"]["dataset_path"]
        value["dataset_files"] = {
            str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()
        }
    # These implement the preparation contract, not sampling or training losses.
    paths = [
        ROOT / "smd/trainer.py",
        ROOT / "tools/cache_prompt_samples.py",
        ROOT / "smd/methods/model.py",
        ROOT / "smd/callbacks/sampling/inputs/assets.py",
        ROOT / "smd/callbacks/sampling/inputs/sources.py",
    ]
    if "dataset" in source:
        loader_module = source["dataset"]["loader"].rsplit(".", 1)[0]
        # Released source manifests retain this original loader identifier.
        if loader_module == "smd.datasets.minimax_h3_ref2va_onpolicy_cache":
            loader_module = "smd.datasets.minimax_h3_ref2va_cache"
        paths.append(ROOT / (loader_module.replace(".", "/") + ".py"))
    module = str(net._target_).rsplit(".", 1)[0]
    if ".CausalWan." in module:
        module = "smd.networks.Wan.network"
    paths += sorted((ROOT / module.replace(".", "/")).parent.glob("*.py"))
    value["preprocessing_code"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    current = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    # A checked, behavior-preserving file relocation need not re-encode the
    # published inputs. Match the complete new signature (code, inputs, model
    # and precision) to its pinned predecessor; any later change still fails.
    manifest = json.loads((ROOT / "assets/resource_manifest.json").read_text())
    family = manifest["families"].get(value["key"], {})
    if current == family.get("signature_after_relocation"):
        return family["signature"]
    return current


def pack(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        if value.is_floating_point() and not value.isfinite().all():
            raise ValueError("Nonfinite fixed preview tensor")
        return value
    if hasattr(value, "feats") and hasattr(value, "coords"):
        return {
            "__preview_type__": "sparse",
            "feats": pack(value.feats),
            "coords": pack(value.coords),
            "shape": list(value.shape),
            "scale": [[x.numerator, x.denominator] for x in value._scale],
        }
    if type(value).__name__ == "MiniMaxH3Condition":
        return {"__preview_type__": "h3", "fields": {f.name: pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(pack(v) for v in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise TypeError(f"Unsupported fixed preview value: {type(value)}")


def unpack(value):
    if isinstance(value, dict):
        kind = value.get("__preview_type__")
        if kind == "sparse":
            from smd.networks.Trellis2.sparse import SparseTensor
            from fractions import Fraction

            return SparseTensor(
                feats=value["feats"],
                coords=value["coords"],
                shape=torch.Size(value["shape"]),
                scale=tuple(Fraction(*x) for x in value["scale"]),
            )
        if kind == "h3":
            from smd.networks.MiniMaxH3.condition import MiniMaxH3Condition

            result = MiniMaxH3Condition(**value["fields"])
            result.validate()
            return result
        if kind is not None:
            raise ValueError(f"Unknown preview payload type: {kind}")
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unpack(v) for v in value)
    return value


def read_cache(path, *, expected_signature=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "preprocessed_preview":
        raise ValueError(f"Invalid fixed preview cache: {path}")
    if expected_signature is not None and manifest["signature"] != expected_signature:
        raise ValueError(f"Stale fixed preview cache: {path}. Rebuild with python tools/cache_prompt_samples.py.")
    records = manifest["records"]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Fixed preview cache requires unique ordered records")
    cases = []
    for record in records:
        payload_path = (path.parent / record["path"]).resolve()
        if not payload_path.is_relative_to(path.parent.resolve()):
            raise ValueError("Preview payload must stay inside its cache directory")
        payload = payload_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Fixed preview checksum mismatch: {record['id']}")
        case = unpack(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True))
        if "condition" not in case or any(k in case for k in ("source", "source_pil", "condition_images")):
            raise ValueError(f"Unprepared fixed preview record: {record['id']}")
        cases.append(case)
    return manifest, cases


def write_cache(path, cases, *, cache_signature, source, model_source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        if "condition" not in case or any(k in case for k in ("source", "source_pil", "condition_images")):
            raise ValueError("Unprepared fixed preview record")
    for index, case in enumerate(cases):
        # Content-addressed payloads let an existing manifest remain valid until
        # the complete new manifest is atomically installed.
        stream = io.BytesIO()
        torch.save(pack(case), stream)
        payload = stream.getvalue()
        sha = hashlib.sha256(payload).hexdigest()
        name = f"{index:02d}_{sha[:16]}.pt"
        target = path.parent / name
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        records.append(dict(id=f"{index:02d}", path=name, sha256=sha))
    manifest = dict(
        schema_version=1,
        kind="preprocessed_preview",
        signature=cache_signature,
        source=source,
        model_source=model_source,
        repeat_text="prompts" in source,
        records=records,
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return read_cache(path, expected_signature=cache_signature)
