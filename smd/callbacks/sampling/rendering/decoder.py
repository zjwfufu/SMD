# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable recipes for rebuilding a network's pixel decoder out-of-process.

Sample callbacks run inside the training process, but the *latent → pixel* step
(VAE decode) is what historically hung distributed jobs: a heavyweight rank-0
GPU op sitting between two collectives. We therefore ship latents to a detached
worker, which has to rebuild the decoder on its own.

Every ``SMDNetwork.init_vae`` in this repo builds a small wrapper class from
a single model id, e.g. ``QwenImageImageEncoder(model_id=self.model_id)``, so a
decoder is fully described by ``(class import path, constructor kwargs)``.
A network may override the inference by implementing
``visualization_decoder_spec() -> dict | DecoderSpec | None``.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any

from smd.utils import logging_utils as logger

#: Constructor parameter names that carry a HuggingFace repo id / local path.
MODEL_ID_PARAMS = ("model_id", "model_id_or_local_path", "model_id_or_path", "pretrained_model_name_or_path")


class DecoderUnavailable(RuntimeError):
    """Raised when a network's decoder cannot be rebuilt in a worker process."""


@dataclass(frozen=True)
class DecoderSpec:
    """How to rebuild a latent → pixel decoder in a fresh process."""

    target: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: Attribute on the constructed object that maps latents to pixels.
    method: str = "decode"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecoderSpec":
        return cls(
            target=payload["target"],
            kwargs=dict(payload.get("kwargs") or {}),
            method=payload.get("method", "decode"),
        )


def _qualified_name(obj: type) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def describe_decoder(net: Any) -> DecoderSpec | None:
    """Best-effort recipe for ``net``'s pixel decoder, or ``None`` if unknown.

    ``None`` is not an error — the preview is skipped rather than decoded in the
    training process.
    """
    hook = getattr(net, "visualization_decoder_spec", None)
    if callable(hook):
        try:
            spec = hook()
        except Exception as exc:  # noqa: BLE001 - visualization must never break training
            logger.warning(f"[VisDecoder] {type(net).__name__}.visualization_decoder_spec() failed: {exc}")
            spec = None
        if isinstance(spec, DecoderSpec):
            return _validated(spec)
        if isinstance(spec, dict):
            return _validated(DecoderSpec.from_dict(spec))
        if spec is not None:
            logger.warning(f"[VisDecoder] ignoring unsupported decoder spec of type {type(spec).__name__}")

    decoder_cls = _decoder_class(net)
    if decoder_cls is None:
        return None

    try:
        parameters = inspect.signature(decoder_cls.__init__).parameters
    except (TypeError, ValueError):
        return None

    vae = getattr(net, "vae", None)
    model_id = getattr(vae, "model_id", None) or getattr(net, "model_id", None)
    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in MODEL_ID_PARAMS:
            if not isinstance(model_id, str):
                logger.warning(f"[VisDecoder] {decoder_cls.__name__} needs '{name}' but no model id was found")
                return None
            kwargs[name] = model_id
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = getattr(vae, name, None)
        if not isinstance(value, (str, int, float, bool)):
            logger.warning(f"[VisDecoder] cannot reconstruct required argument '{name}' of {decoder_cls.__name__}")
            return None
        kwargs[name] = value

    return _validated(DecoderSpec(target=_qualified_name(decoder_cls), kwargs=kwargs))


def _validated(spec: DecoderSpec) -> DecoderSpec | None:
    """Reject recipes a worker process could never import.

    Catching this here turns a silent "no previews appeared" into one loud
    warning in the training log, instead of a stack trace buried in a worker log.
    """
    module_path, _, class_name = spec.target.rpartition(".")
    if not module_path or module_path == "__main__":
        logger.warning(
            f"[VisDecoder] decoder {spec.target!r} is not importable from a worker process; "
            "previews are disabled. Define the decoder in an importable module."
        )
        return None
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning(f"[VisDecoder] cannot import {module_path!r} ({exc}); previews are disabled.")
        return None
    if not hasattr(module, class_name):
        logger.warning(f"[VisDecoder] {module_path!r} has no attribute {class_name!r}; previews are disabled.")
        return None
    return spec


def _decoder_class(net: Any) -> type | None:
    """Find the decoder wrapper class for ``net`` without requiring a live VAE.

    Prefers the instantiated ``net.vae``. When the VAE has not been built —
    which is the common case now that decoding happens out-of-process, so the
    trainer never needs one — fall back to reading the class referenced by
    ``init_vae``. Every implementation in this repo has the shape::

        def init_vae(self):
            self.vae = SomeImageEncoder(model_id=self.model_id)

    so the class name appears in the method's code object and resolves against
    its defining module.
    """
    vae = getattr(net, "vae", None)
    if vae is not None:
        return type(vae)

    init_vae = getattr(net, "init_vae", None)
    code = getattr(init_vae, "__code__", None)
    if code is None:
        return None
    module = inspect.getmodule(init_vae)
    if module is None:
        return None
    candidates = [
        getattr(module, name)
        for name in code.co_names
        if isinstance(getattr(module, name, None), type) and hasattr(getattr(module, name), "decode")
    ]
    if len(candidates) != 1:
        # Zero means we found nothing; more than one means the body is not the
        # simple single-assignment shape this fallback can reason about.
        return None
    return candidates[0]


def build_decoder(spec: DecoderSpec | dict[str, Any]) -> Any:
    """Instantiate the decoder described by ``spec`` (worker-side)."""
    if isinstance(spec, dict):
        spec = DecoderSpec.from_dict(spec)
    module_path, _, class_name = spec.target.rpartition(".")
    if not module_path:
        raise DecoderUnavailable(f"Malformed decoder target: {spec.target!r}")
    try:
        module = importlib.import_module(module_path)
        decoder_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise DecoderUnavailable(f"Cannot import decoder {spec.target!r}: {exc}") from exc
    return decoder_cls(**spec.kwargs)


class IdentityDecoder:
    """Pass-through decoder that exercises the pipeline without loading a VAE.

    Keeps the first three channels and rescales to ``[-1, 1]``, so a payload of
    random latents still renders as a valid image/video grid. Useful to verify
    that worker spawning, media writing and W&B publication work on a new
    cluster before spending time on real weights.
    """

    def __init__(self, model_id: str = "identity") -> None:
        self.model_id = model_id

    def to(self, *args: Any, **kwargs: Any) -> "IdentityDecoder":
        del args, kwargs
        return self

    def decode(self, latents: Any) -> Any:
        import torch

        pixels = latents.float()[:, :3]
        if pixels.shape[1] < 3:
            pixels = pixels.repeat(1, -(-3 // max(1, pixels.shape[1])), *([1] * (pixels.ndim - 2)))[:, :3]
        return torch.tanh(pixels)
