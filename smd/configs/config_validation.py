# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate LazyCall keys against constructor signatures before model loading."""

from __future__ import annotations

import inspect
from pydoc import locate
from typing import Any, Iterable

from omegaconf import DictConfig

#: Keys that describe the call itself rather than a constructor argument.
_RESERVED_KEYS = {"_target_"}


class ConfigValidationError(TypeError):
    """A config node names arguments its target cannot accept."""


def validate_h3_launch_config(config: Any) -> None:
    """Reject mismatched cache/model parallel groups before initializing CUDA."""
    target = str(config.model_class.get("_target_", ""))
    if ".minimax_h3." not in target:
        return
    from smd.methods.simplified_distill.minimax_h3 import validate_h3_config

    validate_h3_config(config.model)
    loader = config.dataloader_train
    cp = int(config.model.net.get("context_parallel_size", 1))
    if cp < 1 or int(loader.get("context_parallel_size", 1)) != cp:
        raise ValueError("H3 model.net and dataloader_train context_parallel_size must match and be positive.")
    if loader.batch_size != 1 or int(loader.get("condition_group_size", 1)) != 1:
        raise ValueError("H3 requires batch_size=1 and condition_group_size=1; CP already groups conditions.")
    if not config.trainer.fsdp or config.trainer.ddp:
        raise ValueError("H3 training requires FSDP2 and trainer.ddp=false.")
    if config.trainer.batch_size_global is not None:
        raise ValueError("Set H3 grad_accum_rounds explicitly; batch_size_global counts physical ranks, including CP.")
    mode = config.model.h3_condition_mode
    expected = ("minimax_h3_prompt_cache.MiniMaxH3PromptCacheLoader" if mode == "t2va" else
                "minimax_h3_ref2va_cache.MiniMaxH3Ref2VACacheLoader")
    if not str(loader.get("_target_", "")).endswith(expected):
        raise ValueError("H3 condition mode must match its production cache loader.")


def _resolve_target(target: Any) -> type | None:
    """Best-effort resolution of a ``_target_`` entry to a callable."""
    if callable(target):
        return target
    if isinstance(target, str):
        resolved = locate(target)
        return resolved if callable(resolved) else None
    return None


def _acceptable_parameters(target: Any) -> set[str] | None:
    """Parameter names ``target`` accepts, or ``None`` if it takes ``**kwargs``."""
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for name, parameter in parameters.items():
        if parameter.kind is parameter.VAR_KEYWORD:
            return None  # accepts anything; nothing to validate
        if parameter.kind is parameter.VAR_POSITIONAL:
            continue
        names.add(name)
    return names


def _suggest(unknown: str, known: Iterable[str]) -> str:
    """Point at a plausible replacement, so the error is actionable."""
    import difflib

    matches = difflib.get_close_matches(unknown, sorted(known), n=1, cutoff=0.6)
    if matches:
        return f" Did you mean {matches[0]!r}?"
    # Renames that changed wording entirely are worth spelling out.
    if unknown in {"num_images_per_prompt", "num_videos_per_prompt"}:
        return " Use 'grid_size' instead (tiles per preview grid; 4 renders as 2x2)."
    if unknown in {
        "async_decode",
        "decode_device",
        "decode_cuda_memory_fraction",
        "decode_timeout_s",
        "decode_max_inflight",
        "publish_every_n",
    }:
        return " Preview rendering is now a job-wide isolated synchronous service; remove this argument."
    return ""


def validate_lazy_call(node: Any, *, path: str) -> None:
    """Raise if ``node`` passes arguments its ``_target_`` cannot accept."""
    if not isinstance(node, (dict, DictConfig)) or "_target_" not in node:
        return

    target = _resolve_target(node["_target_"])
    if target is None:
        return  # unresolvable target; instantiate() reports this far better

    accepted = _acceptable_parameters(target)
    if accepted is None:
        return

    supplied = {str(key) for key in node.keys()} - _RESERVED_KEYS
    unknown = sorted(supplied - accepted)
    if not unknown:
        return

    target_name = getattr(target, "__name__", str(target))
    details = "".join(f"\n  - {path}.{name}:{_suggest(name, accepted)}" for name in unknown)
    raise ConfigValidationError(
        f"{target_name} does not accept {len(unknown)} configured argument(s):{details}\n"
        f"Accepted arguments: {', '.join(sorted(accepted))}.\n"
        "This is a stale config key — it would raise TypeError during Trainer "
        "initialization, after the job has already started."
    )


def validate_callback_configs(config: Any) -> None:
    """Validate every ``trainer.callbacks`` entry against its target signature."""
    callbacks = getattr(getattr(config, "trainer", None), "callbacks", None)
    if not callbacks:
        return
    items = callbacks.items() if hasattr(callbacks, "items") else enumerate(callbacks)
    for name, node in items:
        path = f"trainer.callbacks.{name}"
        if node is None:
            raise ConfigValidationError(
                f"{path} is null. Callback entries must be LazyCall mappings with a '_target_'. "
                "To disable a callback, remove it in the Python config or use the callback's supported "
                "inert/scheduling mechanism; do not set a callback entry to null."
            )
        if not isinstance(node, (dict, DictConfig)):
            raise ConfigValidationError(
                f"{path} must be a LazyCall mapping with a '_target_', got {type(node).__name__}."
            )
        if "_target_" not in node:
            raise ConfigValidationError(
                f"{path} is missing '_target_'. Remove the entry or configure a valid callback LazyCall."
            )
        validate_lazy_call(node, path=path)
