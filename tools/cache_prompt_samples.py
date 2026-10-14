# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed preview inputs once, without constructing a student or teacher.

python tools/cache_prompt_samples.py --experiments sd35_multi --device cuda:0
"""

import sys
from pathlib import Path
import argparse
import importlib
import inspect
import json
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from smd.callbacks.sampling.inputs.cache import ROOT, cache_key, cache_path, read_cache, signature, write_cache
from smd.configs.callbacks import fixed_preview_source
from smd.configs.reference import load_experiment


def preprocessing_network(config, device):
    """Construct only the frozen encoders required by these fixed inputs."""
    key = cache_key(config)
    net_config = config.model.net
    model_id = net_config.get("model_id", net_config.get("model_id_or_local_path"))
    if model_id is None:
        module_name, class_name = str(net_config._target_).rsplit(".", 1)
        network_class = getattr(importlib.import_module(module_name), class_name)
        model_id = inspect.signature(network_class).parameters["model_id"].default
        if not isinstance(model_id, str):
            raise ValueError("Preview encoder requires a configured model source")
    net = torch.nn.Module()
    encoders = {
        "sd35": ("SD35Medium", "SD35TextEncoder", "sd35medium"),
        "flux1": ("Flux", "FluxTextEncoder", "flux"),
        "flux2_4b": ("Flux2", "Flux2TextEncoder", "flux2"),
        "lens": ("Lens", "LensTextEncoder", "lens"),
        "qwen_image": ("QwenImage", "QwenImageTextEncoder", "qwen_image"),
        "qwen_image_edit": ("QwenImageEdit", "QwenImageEditTextEncoder", "qwen_image_edit"),
        "wan13": ("Wan", "WanTextEncoder", "wan"),
        "wan14": ("Wan", "WanTextEncoder", "wan"),
        "wan13_causal": ("Wan", "WanTextEncoder", "wan"),
    }
    if key in encoders:
        module, name, net.network_family = encoders[key]
        cls = getattr(importlib.import_module(f"smd.networks.{module}.network"), name)
        net.text_encoder = cls(model_id)
        net.text_encoder.to(device=device, dtype=getattr(torch, config.model.precision))
        if key == "qwen_image_edit":
            from smd.networks.QwenImage.network import QwenImageImageEncoder

            net.is_edit = True
            net.vae = QwenImageImageEncoder(model_id)
            net.vae.to(device=device, dtype=getattr(torch, config.model.precision))
    elif key == "hunyuan3d_shape":
        from smd.networks.Hunyuan3DShape.network import Hunyuan3DShape

        # Use the production image processor and encode_image, without a DiT.
        net = Hunyuan3DShape.__new__(Hunyuan3DShape)
        torch.nn.Module.__init__(net)
        net._init_image_cond_model()
        net.image_cond_model.to(device=device, dtype=getattr(torch, config.model.precision))
    elif key.startswith("h3_"):
        net.network_family = "minimax_h3"
    elif key.startswith("trellis2_"):
        net.network_family = "trellis2"
    else:
        raise ValueError(key)
    net.eval()
    return net, model_id


def raw_cases(config, source, family):
    if "assets" in source:
        from smd.callbacks.sampling.inputs.assets import asset_cases

        return asset_cases(source["assets"], family=family)
    if "dataset" in source:
        from smd.callbacks.sampling.inputs.sources import dataset_cases

        return dataset_cases(source["dataset"], source["case_indices"])
    return [{"condition": [p]} for p in source["prompts"]]


def build(config, device="cuda:0", force=False):
    from smd.trainer import Trainer
    from smd.callbacks.sampling.base import preview_mode
    from smd.callbacks.sampling.engine import copy_tree

    source = fixed_preview_source(config)
    expected = signature(config, source)
    path = ROOT / cache_path(config)
    if path.exists() and not force:
        manifest, _ = read_cache(path, expected_signature=expected)
        print(f"Reused {path}: {len(manifest['records'])} fixed cases", flush=True)
        return path
    net, model_source = preprocessing_network(config, torch.device(device))
    model = SimpleNamespace(
        net=net,
        device=torch.device(device),
        config=config.model,
        input_shape=config.model.input_shape,
        precision=getattr(torch, config.model.precision),
        precision_amp_enc=getattr(torch, config.model.precision_amp_enc)
        if config.model.precision_amp_enc and config.model.precision_amp_enc != config.model.precision
        else None,
        precision_amp_infer=getattr(torch, config.model.precision_amp_infer)
        if config.model.precision_amp_infer and config.model.precision_amp_infer != config.model.precision
        else None,
    )
    trainer = Trainer.__new__(Trainer)
    cases = []
    for index, raw in enumerate(raw_cases(config, source, net.network_family)):
        with preview_mode(model, net, index * 1000):
            data = trainer.preprocess_data(model, copy_tree(raw))
            cases.append(copy_tree(data, "cpu"))
        print(f"Encoded {cache_key(config)} case {index + 1}", flush=True)
    write_cache(path, cases, cache_signature=expected, source=source, model_source=model_source)
    print(f"Prepared {path}: {len(cases)} fixed cases", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True, help="Catalog experiment names, or all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Explicitly rebuild a stale cache")
    args = parser.parse_args()
    import os

    os.chdir(ROOT)
    names = json.loads((ROOT / "smd/configs/models.json").read_text()) if args.experiments == ["all"] else args.experiments
    seen = set()
    for name in names:
        config = load_experiment(name)
        identity = cache_key(config), signature(config, fixed_preview_source(config))
        if identity in seen:
            continue
        build(config, device=args.device, force=args.force)
        seen.add(identity)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
