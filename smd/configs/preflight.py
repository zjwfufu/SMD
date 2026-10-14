# SPDX-License-Identifier: Apache-2.0
"""Check configured preview inputs before distributed or model initialization."""

from pathlib import Path
from shlex import join

ROOT = Path(__file__).resolve().parents[2]


def resource_hint(config_path, *, model=None, config=None):
    path = Path(config_path).resolve()
    if model is None and path.parent == ROOT / "smd/configs/experiments" and path.suffix == ".py":
        model = path.stem
    if model is None and config is not None:
        from smd.callbacks.sampling.inputs.cache import cache_key

        try:
            model = cache_key(config)
        except (AttributeError, ValueError):
            pass
    if model is None:
        return "Download the matching experiment's resources with python tools/download_resources.py --model EXPERIMENT."
    command = join(["python", "tools/download_resources.py", "--model", model])
    return f"Restore the bundled resources with:\n  {command}\nOr add --download-resources before the standalone '-' in the training command (custom configs also need --resource-model EXPERIMENT)."


def import_training_config(config_path, *, model=None):
    from smd.configs.config_utils import import_config_from_python_file

    try:
        return import_config_from_python_file(config_path)
    except FileNotFoundError as error:
        # Signature construction reads source manifests before CLI overrides.
        # Leave unrelated configuration/data errors unchanged.
        if not error.filename or not Path(error.filename).resolve().is_relative_to(ROOT / "assets"):
            raise
        raise FileNotFoundError(
            f"Training configuration needs a missing asset: {error.filename}\n"
            + resource_hint(config_path, model=model)
        ) from error


def validate_training_weights(config, *, config_path, model=None):
    """Check configured extra weights without requiring bundled paths or values."""
    if ".causal." not in str(config.model_class.get("_target_", "")):
        return
    keys = ["student_init_checkpoint"]
    if config.model.causal_teacher_type == "causal":
        keys.append("causal_teacher_checkpoint")
    missing = []
    for key in keys:
        value = getattr(config.model, key, "")
        if not value:
            if key == "causal_teacher_checkpoint":
                missing.append(f"model.{key}: no checkpoint configured")
            continue
        if not Path(value).is_file() or Path(value).stat().st_size == 0:
            missing.append(f"model.{key}: {value}")
    if missing:
        raise FileNotFoundError(
            "Missing causal training checkpoints (checked before model initialization):\n  "
            + "\n  ".join(missing) + "\n"
            + resource_hint(config_path, model=model, config=config)
            + "\nThe bundled weights install under SMD_MODEL_ROOT/causal_forcing "
            "(default: models/causal_forcing). For custom checkpoint paths, prepare those exact files."
        )


def validate_preview_resources(config, *, config_path, model=None):
    from smd.callbacks.sampling.inputs.cache import read_cache
    from smd.callbacks.sampling.samplers import BatchSampleCallback, PromptSampleCallback
    from smd.configs.config_validation import _resolve_target

    callbacks = getattr(getattr(config, "trainer", None), "callbacks", {})
    items = callbacks.items() if hasattr(callbacks, "items") else enumerate(callbacks)
    needs_lighting = False
    for name, node in items:
        target = _resolve_target(node.get("_target_"))
        if not isinstance(target, type) or not issubclass(target, (BatchSampleCallback, PromptSampleCallback)):
            continue
        if node.get("every_n", 500) <= 0:
            continue
        needs_lighting = True
        if not issubclass(target, PromptSampleCallback):
            continue
        path = node.get("cache")
        if not path:
            raise ValueError(f"trainer.callbacks.{name}.cache must name a prepared fixed preview manifest.")
        try:
            # Use the normal reader: signature, payload hashes, CPU decoding and
            # condition validation must agree with the actual callback.
            read_cache(path, expected_signature=node.get("cache_signature"))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            hint = resource_hint(config_path, model=model, config=config)
            raise RuntimeError(
                f"Fixed preview preflight failed for trainer.callbacks.{name} ({path}): {error}\n"
                f"{hint}\nFor a custom cache, restore its files or rebuild it with "
                "python tools/cache_prompt_samples.py. For changed inputs/preprocessing, "
                "use --force when rebuilding; do not bypass the signature or hash checks."
            ) from error

    net = config.model.net
    if needs_lighting and ".Trellis2." in str(net.get("_target_", "")) and "tex" in net.get("ckpt_name", ""):
        lighting = ROOT / "assets/hdri/forest.exr"
        if not lighting.is_file():
            raise FileNotFoundError(
                f"Texture previews require {lighting}. Restore it from Git or run python assets/hdri/generate.py."
            )
