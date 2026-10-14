# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network input/output contracts and condition adapters.

The contracts are grounded in the concrete ``smd.networks`` implementations.
They describe the condition trees, latent layouts, and auxiliary forward inputs
used by each network family so any training method can share the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from smd.utils.condition_utils import (
    cond_batch_size,
    cond_repeat,
    cond_repeat_for_vis,
    cond_slice,
    cond_trim_padding,
)

BatchSemantics = Literal["tensor_batch_dim0", "per_sample_list"]

# Key holding the primary (network-native) condition inside a "condition
# container" dict.  A container follows the same dict-condition convention as
# the I2V/video2world families (``text_embeds`` + shared auxiliary inputs), but
# uses an architecture-neutral primary key because some families (e.g. TRELLIS)
# carry image features, not text embeddings.  Extra network inputs
# (``source_latents``, ``concat_cond``) ride alongside the primary condition so
# they slice/repeat/trim together with it and never travel as a separate
# ``fwd_kwargs`` payload.
PRIMARY_CONDITION_KEY = "cond"


def is_condition_container(condition: Any) -> bool:
    """True when *condition* is a container dict carrying the primary key."""
    return isinstance(condition, dict) and PRIMARY_CONDITION_KEY in condition


def unpack_condition_container(condition: Any) -> tuple[Any, dict[str, Any]]:
    """Split a condition container into ``(primary_condition, extra_inputs)``.

    Plain (non-container) conditions pass through unchanged with an empty extras
    dict, so network forwards can call this unconditionally.
    """
    if is_condition_container(condition):
        primary = condition[PRIMARY_CONDITION_KEY]
        extras = {k: v for k, v in condition.items() if k != PRIMARY_CONDITION_KEY}
        return primary, extras
    return condition, {}


@dataclass(frozen=True)
class TensorSpec:
    """A tensor leaf in a network input/output tree."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str
    meaning: str


@dataclass(frozen=True)
class ExtraInputSpec:
    """A non-condition forward kwarg that affects network input shape."""

    name: str
    shape: tuple[int | str, ...]
    required: bool
    meaning: str


@dataclass(frozen=True)
class NetworkIOContract:
    """Concrete input/output contract for one network family."""

    network_family: str
    source: str
    condition_container: str
    batch_semantics: BatchSemantics
    condition_leaves: tuple[TensorSpec, ...]
    latent: TensorSpec
    extra_inputs: tuple[ExtraInputSpec, ...] = ()
    accepts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


NETWORK_IO_CONTRACTS: dict[str, NetworkIOContract] = {
    "flux": NetworkIOContract(
        network_family="flux",
        source="FluxTextEncoder.encode -> Flux.forward",
        condition_container="tuple(pooled_prompt_embeds, prompt_embeds)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("pooled_prompt_embeds", ("B", 768), "floating", "CLIP pooled projections"),
            TensorSpec("prompt_embeds", ("B", "S<=512", 4096), "floating", "T5 token embeddings"),
        ),
        latent=TensorSpec("x_t", ("B", "C", "H", "W"), "floating", "dense image latent"),
        accepts=("tuple[Tensor, Tensor]",),
        notes=("Embedded guidance passes an optional guidance tensor [B].",),
    ),
    "wan": NetworkIOContract(
        network_family="wan",
        source="WanTextEncoder.encode -> Wan.forward",
        condition_container="Tensor",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(TensorSpec("prompt_embeds", ("B", 512, "D_text"), "floating", "padded T5 text embeddings"),),
        latent=TensorSpec("x_t", ("B", "C", "T", "H", "W"), "floating", "dense video latent"),
        accepts=("Tensor[B,512,D]", "list[Tensor[512,D]] stacked in forward"),
    ),
    "sana_wm": NetworkIOContract(
        network_family="sana_wm",
        source="SanaWMTextEncoder.encode -> SanaWM.forward",
        condition_container="tuple(text_embeds, attention_mask) + world-condition extras",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("text_embeds", ("B", 1, 300, 2304), "floating", "Gemma-2 decoder hidden states"),
            TensorSpec("attention_mask", ("B", 300), "integer", "valid caption tokens"),
        ),
        latent=TensorSpec("x_t", ("B", 128, "T", "H", "W"), "floating", "LTX-2 video latent"),
        extra_inputs=(
            ExtraInputSpec(
                "reference_latents",
                ("B", 128, 1, "H", "W"),
                True,
                "clean first-frame LTX-2 latent used to anchor autoregressive rollout",
            ),
            ExtraInputSpec(
                "camera_conditions",
                ("B", "T", 20),
                True,
                "relative 4x4 camera pose plus latent-pixel fx/fy/cx/cy per latent frame",
            ),
            ExtraInputSpec(
                "chunk_plucker",
                ("B", 48, "T_chunk", "H", "W"),
                False,
                "optional official post-attention Plucker ray-map conditioning",
            ),
        ),
        accepts=("tuple[Tensor[B,1,300,2304], Tensor[B,1,1,300]]",),
        notes=(
            "Frame zero is kept clean and excluded from the generated-frame objective.",
            "Streaming mode additionally consumes start_f/end_f/frame_index and per-block KV caches.",
        ),
    ),
    "ltx2": NetworkIOContract(
        network_family="ltx2",
        source="precomputed Gemma contexts -> LTX2.forward",
        condition_container="LTX2Condition(video_context, audio_context, video_mask, audio_mask, fps)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("video_context", ("B", "S_video", "D_video"), "floating", "projected Gemma video context"),
            TensorSpec("audio_context", ("B", "S_audio", "D_audio"), "floating", "projected Gemma audio context"),
        ),
        latent=TensorSpec(
            "LTX2Latents(video,audio)",
            ("video:B,128,F,H,W", "audio:B,8,T,16"),
            "floating",
            "coupled LTX-2.3 AV latent pair",
        ),
        accepts=("LTX2Condition", "dict with video_context/audio_context", "tuple(video_context,audio_context,...)"),
        notes=(
            "The initial integration is latent/context-only; it does not fabricate Gemma or VAE prompt/decode paths.",
            "Video/audio must remain paired through every transformer and Euler step.",
            "fps is part of the condition because official video RoPE positions are expressed in seconds.",
            "Paired SMD training requires data keys video_latents, audio_latents, and both text contexts.",
        ),
    ),
    "minimax_h3": NetworkIOContract(
        network_family="minimax_h3",
        source="Qwen3-VL hidden_states[50] -> MiniMaxH3.forward",
        condition_container="MiniMaxH3Condition(prompt_embeds, text_token_tags)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("prompt_embeds", (1, "S", 5120), "floating", "Qwen3-VL layer-50 hidden states"),
            TensorSpec("text_token_tags", ("S",), "integer", "H3 modality tags; T2VA rows are text tag 1"),
        ),
        latent=TensorSpec(
            "MiniMaxH3Latents(video,audio)",
            ("video:B,24,F,H,W", "audio:B,2,32,T"),
            "floating",
            "coupled MiniMax H3 video/audio latent pair",
        ),
        accepts=("MiniMaxH3Condition", "dict with prompt_embeds/text_token_tags", "tuple(prompt_embeds,tags)"),
        notes=(
            "The released checkpoint is guidance-distilled and has no negative-condition or CFG branch.",
            "Video and audio share one packed transformer sequence but follow shift-12 and shift-3 schedules.",
            "One condition per rank; T2VA or Ref2VA with paired reference video/audio latents.",
            "text_token_tags has a sequence axis only; batch slicing/repetition must leave it unchanged.",
        ),
    ),
    "qwen_image": NetworkIOContract(
        network_family="qwen_image",
        source="QwenImageTextEncoder.encode -> QwenImage.forward",
        condition_container="tuple(prompt_embeds, prompt_mask)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("prompt_embeds", ("B", "S<=512", 3584), "floating", "Qwen text embeddings"),
            TensorSpec("prompt_mask", ("B", "S<=512"), "integer", "1 for valid text tokens"),
        ),
        latent=TensorSpec("x_t", ("B", "C", "H", "W"), "floating", "dense image latent"),
        accepts=("tuple[Tensor, Tensor]",),
        notes=("Padding must be trimmed with the mask before single-sample visualization.",),
    ),
    "zimage": NetworkIOContract(
        network_family="zimage",
        source="ZImageTextEncoder.encode -> ZImage.forward",
        condition_container="list[Tensor[L_i,2560]]",
        batch_semantics="per_sample_list",
        condition_leaves=(
            TensorSpec("condition[i]", ("L_i", 2560), "floating", "unpadded Qwen3 penultimate hidden states"),
        ),
        latent=TensorSpec("x_t", ("B", 16, "H", "W"), "floating", "dense Z-Image VAE latent"),
        accepts=("list[Tensor[L_i,2560]]",),
        notes=("The native transformer consumes one variable-length caption tensor per sample.",),
    ),
    "qwen_image_edit": NetworkIOContract(
        network_family="qwen_image_edit",
        source="QwenImageEditTextEncoder.encode -> QwenImageEdit.forward",
        condition_container="tuple(prompt_embeds, prompt_mask)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("prompt_embeds", ("B", "S<=512", 3584), "floating", "Qwen-VL text/image embeddings"),
            TensorSpec("prompt_mask", ("B", "S<=512"), "integer", "1 for valid multimodal tokens"),
        ),
        latent=TensorSpec("x_t", ("B", "C", "H", "W"), "floating", "target/noise image latent"),
        extra_inputs=(
            ExtraInputSpec(
                "source_latents",
                ("B", "C", "Hs", "Ws"),
                True,
                "source image latents concatenated to noise tokens",
            ),
        ),
        accepts=("tuple[Tensor, Tensor]",),
        notes=("Data-free noise shape must follow source_latents spatial shape when present.",),
    ),
    "lens": NetworkIOContract(
        network_family="lens",
        source="LensTextEncoder.encode -> Lens.forward",
        condition_container="tuple(features_list, mask)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("features_list[0..3]", ("B", "S", 2880), "floating", "selected GPT-OSS layer features"),
            TensorSpec("mask", ("B", "S"), "bool", "valid token mask after chat-template offset"),
        ),
        latent=TensorSpec("x_t", ("B", 128, "H", "W"), "floating", "BN-normalized Lens latent"),
        accepts=("tuple[list[Tensor], Tensor]", "tuple[Tensor, Tensor] for single-layer mode"),
        notes=("Mask is index 1 for padding trim.",),
    ),
    "sd35medium": NetworkIOContract(
        network_family="sd35medium",
        source="SD35TextEncoder.encode -> SD35Medium.forward",
        condition_container="tuple(pooled_projections, encoder_hidden_states)",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("pooled_projections", ("B", 2048), "floating", "CLIP-L/G pooled projections"),
            TensorSpec("encoder_hidden_states", ("B", "77+T5", 4096), "floating", "CLIP tokens plus T5 tokens"),
        ),
        latent=TensorSpec("x_t", ("B", 16, "H", "W"), "floating", "SD3.5 latent"),
        accepts=("tuple[Tensor, Tensor]", "Tensor encoder_hidden_states with zero pooled fallback"),
    ),
    "flux2": NetworkIOContract(
        network_family="flux2",
        source="Flux2TextEncoder.encode -> Flux2.forward",
        condition_container="Tensor",
        batch_semantics="tensor_batch_dim0",
        condition_leaves=(
            TensorSpec("prompt_embeds", ("B", "S<=512", "D_flux2_text"), "floating", "Qwen3 hidden-layer concat"),
        ),
        latent=TensorSpec("x_t", ("B", 128, "H", "W"), "floating", "Flux2 patchified latent"),
        accepts=("Tensor[B,S,D]", "tuple/list where forward selects the text tensor"),
    ),
    "trellis2": NetworkIOContract(
        network_family="trellis2",
        source="Trellis2.encode_image -> Trellis2.forward",
        condition_container="list[Tensor[L_i,1024]]",
        batch_semantics="per_sample_list",
        condition_leaves=(
            TensorSpec("condition[i]", ("L_i", 1024), "floating", "DINOv3 image features for one sample"),
        ),
        latent=TensorSpec("x_t.feats", ("N_total", 32), "floating", "SparseTensor voxel features"),
        extra_inputs=(ExtraInputSpec("concat_cond", ("N_total", 32), False, "texture path sparse feature concat"),),
        accepts=("list[Tensor[L_i,1024]]", "Tensor[B,L,1024]"),
        notes=("Batch size is len(condition), not condition[0].shape[0].",),
    ),
    "hunyuan3d_shape": NetworkIOContract(
        network_family="hunyuan3d_shape",
        source="Hunyuan3DShape.encode_image -> Hunyuan3DShape.forward",
        condition_container="list[Tensor[1370,1024]]",
        batch_semantics="per_sample_list",
        condition_leaves=(
            TensorSpec("condition[i]", (1370, 1024), "floating", "DINOv2-L image features for one sample"),
        ),
        latent=TensorSpec("x_t", ("B", 4096, 64), "floating", "dense vec2set shape latent"),
        accepts=("list[Tensor[1370,1024]]", "Tensor[B,1370,1024]", 'dict{"main": Tensor[B,1370,1024]}'),
        notes=("Batch size is len(condition); base tensor-tree slicing would slice the token dimension.",),
    ),
}

_ALIASES = {
    "qwenimage": "qwen_image",
    "qwenimageedit": "qwen_image_edit",
    "sd35": "sd35medium",
    "sd3.5": "sd35medium",
    "hunyuan3d": "hunyuan3d_shape",
}


def normalize_network_family(network_family: str) -> str:
    key = str(network_family).lower().replace("-", "_")
    return _ALIASES.get(key, key)


def get_network_io_contract(network_family: str) -> NetworkIOContract:
    key = normalize_network_family(network_family)
    try:
        return NETWORK_IO_CONTRACTS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(NETWORK_IO_CONTRACTS))
        raise KeyError(f"No network I/O contract for {network_family!r}. Supported: {supported}") from exc


def uses_per_sample_list_condition(network_family: str) -> bool:
    return get_network_io_contract(network_family).batch_semantics == "per_sample_list"


class NetworkIOAdapter:
    """Runtime condition/input adapter shared by training methods and callbacks."""

    def __init__(self, network_family: str):
        self.network_family = normalize_network_family(network_family)
        self.contract = get_network_io_contract(self.network_family)

    @classmethod
    def from_model(cls, model: Any) -> "NetworkIOAdapter":
        family = getattr(model, "network_family", None)
        if family is None:
            raise AttributeError(f"Cannot build NetworkIOAdapter for {type(model).__name__}: missing network_family")
        return cls(str(family))

    @property
    def uses_per_sample_list(self) -> bool:
        return self.contract.batch_semantics == "per_sample_list"

    def batch_size(self, condition: Any) -> int:
        if is_condition_container(condition):
            return self.batch_size(condition[PRIMARY_CONDITION_KEY])
        if self.uses_per_sample_list and isinstance(condition, (tuple, list)):
            return len(condition)
        return cond_batch_size(condition)

    def slice(self, condition: Any, n: int) -> Any:
        if is_condition_container(condition):
            primary, extras = unpack_condition_container(condition)
            sliced = {PRIMARY_CONDITION_KEY: self.slice(primary, n)}
            sliced.update({k: cond_slice(v, n) for k, v in extras.items()})
            return sliced
        if self.uses_per_sample_list and isinstance(condition, list):
            return condition[:n]
        if self.uses_per_sample_list and isinstance(condition, tuple):
            return condition[:n]
        return cond_slice(condition, n)

    def repeat(self, condition: Any, repeats: int) -> Any:
        if repeats <= 1:
            return condition
        if is_condition_container(condition):
            primary, extras = unpack_condition_container(condition)
            repeated = {PRIMARY_CONDITION_KEY: self.repeat(primary, repeats)}
            repeated.update({k: cond_repeat(v, repeats) for k, v in extras.items()})
            return repeated
        if self.uses_per_sample_list and isinstance(condition, list):
            if not condition:
                return condition
            return [condition[0] for _ in range(repeats)]
        if self.uses_per_sample_list and isinstance(condition, tuple):
            if not condition:
                return condition
            return tuple(condition[0] for _ in range(repeats))
        return cond_repeat(condition, repeats)

    def concat(self, first: Any, second: Any) -> Any:
        """Concatenate two condition batches according to the network contract."""
        if is_condition_container(first) or is_condition_container(second):
            if not (is_condition_container(first) and is_condition_container(second)):
                raise TypeError("Cannot concatenate a packed condition container with an unpacked condition.")
            if first.keys() != second.keys():
                raise ValueError(
                    f"Condition container keys must match for concatenation, got {sorted(first)} and {sorted(second)}."
                )
            primary = self.concat(first[PRIMARY_CONDITION_KEY], second[PRIMARY_CONDITION_KEY])
            extras = {
                key: self._concat_tensor_tree(first[key], second[key]) for key in first if key != PRIMARY_CONDITION_KEY
            }
            return {PRIMARY_CONDITION_KEY: primary, **extras}
        if self.uses_per_sample_list and isinstance(first, list) and isinstance(second, list):
            return [*first, *second]
        if self.uses_per_sample_list and isinstance(first, tuple) and isinstance(second, tuple):
            return (*first, *second)
        return self._concat_tensor_tree(first, second)

    @classmethod
    def _concat_tensor_tree(cls, first: Any, second: Any) -> Any:
        if isinstance(first, Tensor) and isinstance(second, Tensor):
            if first.ndim != second.ndim or first.shape[1:] != second.shape[1:]:
                raise ValueError(
                    f"Condition tensor shapes must match outside batch dim, got {tuple(first.shape)} and "
                    f"{tuple(second.shape)}."
                )
            return torch.cat([first, second], dim=0)
        if isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                raise ValueError(f"Condition list lengths must match, got {len(first)} and {len(second)}.")
            return [cls._concat_tensor_tree(a, b) for a, b in zip(first, second)]
        if isinstance(first, tuple) and isinstance(second, tuple):
            if len(first) != len(second):
                raise ValueError(f"Condition tuple lengths must match, got {len(first)} and {len(second)}.")
            return tuple(cls._concat_tensor_tree(a, b) for a, b in zip(first, second))
        if isinstance(first, dict) and isinstance(second, dict):
            if first.keys() != second.keys():
                raise ValueError(f"Condition dict keys must match, got {sorted(first)} and {sorted(second)}.")
            return {key: cls._concat_tensor_tree(first[key], second[key]) for key in first}
        if type(first) is not type(second) or first != second:
            raise TypeError(
                f"Non-tensor condition leaves must match exactly, got {type(first).__name__}={first!r} and "
                f"{type(second).__name__}={second!r}."
            )
        return first

    def trim_padding(self, condition: Any) -> Any:
        if is_condition_container(condition):
            primary, extras = unpack_condition_container(condition)
            trimmed = {PRIMARY_CONDITION_KEY: self.trim_padding(primary)}
            trimmed.update(extras)
            return trimmed
        return cond_trim_padding(condition, self.network_family)

    def pack_condition(self, primary: Any, data: dict[str, Any]) -> Any:
        """Wrap *primary* into a container dict when the network has extra inputs.

        Extra network inputs (``source_latents``, ``concat_cond``) are pulled
        from *data* according to the network contract and ride alongside the
        primary condition so downstream slice/repeat/trim operate on them
        together.  A required extra that is missing raises; a text-only family
        (no extras present) returns *primary* unchanged.
        """
        extras: dict[str, Any] = {}
        for spec in self.contract.extra_inputs:
            value = data.get(spec.name, None)
            if value is None:
                if spec.required:
                    raise KeyError(
                        f"Network family {self.network_family!r} requires forward input "
                        f"{spec.name!r} ({spec.meaning}), but it is missing from the training batch."
                    )
                continue
            extras[spec.name] = value
        if not extras:
            return primary
        return {PRIMARY_CONDITION_KEY: primary, **extras}

    def forward_kwargs_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for spec in self.contract.extra_inputs:
            value = data.get(spec.name, None)
            if value is None:
                continue
            kwargs[spec.name] = value
        return kwargs

    def input_shape_from_data(
        self,
        data: dict[str, Any],
        *,
        default_input_shape: list[int],
    ) -> list[int]:
        if self.network_family == "qwen_image_edit":
            source_latents = data.get("source_latents", None)
            if isinstance(source_latents, Tensor):
                return list(source_latents.shape[1:])
        return list(default_input_shape)


def make_dummy_condition(
    network_family: str,
    *,
    batch_size: int = 2,
    seq_len: int = 8,
    device: str | torch.device = "meta",
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Build a no-data dummy condition with the real container semantics."""
    key = normalize_network_family(network_family)
    B, S = batch_size, seq_len
    if key == "flux":
        return torch.empty(B, 768, device=device, dtype=dtype), torch.empty(B, S, 4096, device=device, dtype=dtype)
    if key == "wan":
        return torch.empty(B, 512, 4096, device=device, dtype=dtype)
    if key == "sana_wm":
        return (
            torch.empty(B, 1, 300, 2304, device=device, dtype=dtype),
            torch.empty(B, 300, device=device, dtype=torch.long),
        )
    if key in {"qwen_image", "qwen_image_edit"}:
        return (
            torch.empty(B, S, 3584, device=device, dtype=dtype),
            torch.empty(B, S, device=device, dtype=torch.long),
        )
    if key == "zimage":
        return [torch.empty(S, 2560, device=device, dtype=dtype) for _ in range(B)]
    if key == "lens":
        return [torch.empty(B, S, 2880, device=device, dtype=dtype) for _ in range(4)], torch.empty(
            B, S, device=device, dtype=torch.bool
        )
    if key == "sd35medium":
        return torch.empty(B, 2048, device=device, dtype=dtype), torch.empty(
            B, 77 + S, 4096, device=device, dtype=dtype
        )
    if key == "flux2":
        return torch.empty(B, S, 10240, device=device, dtype=dtype)
    if key == "trellis2":
        return [torch.empty(S + i, 1024, device=device, dtype=dtype) for i in range(B)]
    if key == "hunyuan3d_shape":
        return [torch.empty(1370, 1024, device=device, dtype=dtype) for _ in range(B)]
    raise KeyError(f"No dummy condition builder for {network_family!r}")


def make_dummy_extra_inputs(
    network_family: str,
    *,
    batch_size: int = 2,
    device: str | torch.device = "meta",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Build dummy non-condition forward kwargs for shape-sensitive networks."""
    key = normalize_network_family(network_family)
    if key == "qwen_image_edit":
        return {"source_latents": torch.empty(batch_size, 16, 64, 64, device=device, dtype=dtype)}
    if key == "trellis2":
        return {"concat_cond": "SparseTensor[N_total, 32]"}
    if key == "sana_wm":
        return {
            "reference_latents": torch.empty(batch_size, 128, 1, 22, 40, device=device, dtype=dtype),
            "camera_conditions": torch.empty(batch_size, 121, 20, device=device, dtype=dtype),
        }
    return {}


def encode_prompt_condition_for_vis(
    net: Any,
    prompt: str,
    *,
    precision: torch.dtype,
    repeats: int,
    max_length: int | None = None,
) -> Any:
    """Encode one prompt and tile it for visualization/sample callbacks.

    This consolidates the text-encoder API differences that were previously
    duplicated across prompt-sampling callbacks.
    """
    if not hasattr(net, "text_encoder"):
        raise RuntimeError(f"{type(net).__name__} has no text_encoder for prompt sampling.")

    text_encoder = net.text_encoder

    if hasattr(text_encoder, "encode_prompts"):
        prompt_embeds, prompt_mask = text_encoder.encode_prompts([prompt], max_length=max_length)
        return cond_repeat_for_vis({"prompt_embeds": prompt_embeds, "prompt_mask": prompt_mask}, repeats)

    if not hasattr(text_encoder, "encode"):
        raise RuntimeError("Unsupported text encoder API: expected `encode_prompts` or `encode`.")

    family = normalize_network_family(getattr(net, "network_family", ""))
    if family in {"wan", "sana_wm"}:
        encoded = text_encoder.encode([prompt], precision=precision)
    else:
        encoded = text_encoder.encode(
            [prompt],
            precision=precision,
            max_sequence_length=max_length or 512,
        )
    adapter = NetworkIOAdapter.from_model(net)
    if adapter.uses_per_sample_list:
        return adapter.repeat(encoded, repeats)
    return cond_repeat_for_vis(encoded, repeats)
