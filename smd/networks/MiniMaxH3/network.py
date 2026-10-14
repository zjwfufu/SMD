# SPDX-License-Identifier: Apache-2.0

"""MiniMax H3 joint audio/video network bridge."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import nn

import smd.utils.logging_utils as logger
from smd.networks.network import SMDNetwork
from smd.networks.noise_schedule import BaseNoiseSchedule
from .condition import MiniMaxH3Condition
from smd.networks.MiniMaxH3.context_parallel import get_context_parallel_state
from smd.utils.basic_utils import str2bool
from smd.utils.hf_utils import get_hf_cache_dir, safe_load_from_pretrained

_TEXT_EMBED_DIM = 5120
_VIDEO_CHANNELS = 24
_AUDIO_CHANNELS = 2
_AUDIO_BANDS = 32
_AUDIO_VAE_SUBFOLDER = "audio_vae"
_T2VA_ALLOW_PATTERNS = (
    "model_index.json",
    "modular_model_index.json",
    "transformer/**",
    "text_encoder/**",
    "vae/**",
    "audio_vae/**",
    "tokenizer/**",
    "processor/**",
    "scheduler/**",
    "audio_scheduler/**",
)


def _pad_h3_sparse_text_slots(layout: Any, text_capacity: int) -> Any:
    num_text_tokens = int(layout.text_indices.numel())
    if num_text_tokens > text_capacity:
        raise ValueError(
            f"MiniMax H3 sparse attention received {num_text_tokens} text tokens, "
            f"exceeding max_text_tokens={text_capacity}."
        )
    pad_rows = text_capacity - num_text_tokens
    if pad_rows == 0:
        return layout
    if num_text_tokens == 0 or not torch.equal(
        layout.text_indices,
        torch.arange(num_text_tokens, device=layout.text_indices.device),
    ):
        raise ValueError("MiniMax H3 sparse text padding requires a non-empty contiguous text prefix.")

    position_padding = layout.position_ids.new_zeros((pad_rows, layout.position_ids.shape[-1]))
    tag_padding = layout.token_tags.new_full((pad_rows,), int(layout.token_tags[0]))
    return replace(
        layout,
        position_ids=torch.cat(
            (layout.position_ids[:num_text_tokens], position_padding, layout.position_ids[num_text_tokens:])
        ),
        token_tags=torch.cat((layout.token_tags[:num_text_tokens], tag_padding, layout.token_tags[num_text_tokens:])),
        audio_indices=layout.audio_indices + pad_rows,
        video_indices=layout.video_indices + pad_rows,
    )


def _default_minimax_h3_lora_targets() -> list[str]:
    return [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ]


def _latent_num_frames_to_video_frames(num_latent_frames: int) -> int:
    if num_latent_frames < 2 or (num_latent_frames - 2) % 5:
        raise ValueError(
            "MiniMaxH3 latent video frames must satisfy the 2 + 5k contract implied by 5 + 17k source frames; "
            f"got {num_latent_frames}."
        )
    return 5 + 17 * ((num_latent_frames - 2) // 5)


def _resolve_h3_cache_dir(model_id: str) -> str:
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    return get_hf_cache_dir(model_id)


def _resolve_snapshot_root(model_id: str, revision: str | None = None) -> Path:
    if os.path.isdir(model_id):
        return Path(model_id).expanduser().resolve()
    return Path(
        snapshot_download(
            model_id,
            revision=revision,
            cache_dir=get_hf_cache_dir(model_id),
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            allow_patterns=_T2VA_ALLOW_PATTERNS,
        )
    )


def _load_minimax_runtime() -> Any:
    return importlib.import_module("smd.networks.MiniMaxH3")


def _load_qwen3_vl_types() -> tuple[type[Any], type[Any], type[Any]]:
    transformers = importlib.import_module("transformers")
    try:
        return (
            transformers.Qwen3VLForConditionalGeneration,
            transformers.Qwen2TokenizerFast,
            transformers.Qwen3VLProcessor,
        )
    except AttributeError as exc:
        raise ImportError(
            "MiniMaxH3 prompt encoding requires transformers with Qwen3VLForConditionalGeneration, "
            "Qwen2TokenizerFast, and Qwen3VLProcessor."
        ) from exc


@dataclass(frozen=True)
class MiniMaxH3Latents:
    """MiniMax H3 latent state: video ``[B,24,F,H,W]`` and audio ``[B,2,32,T]``."""

    video: torch.Tensor
    audio: torch.Tensor

    def batch_size(self) -> int:
        return int(self.video.shape[0])

    @property
    def device(self) -> torch.device:
        return self.video.device

    def validate(self) -> None:
        if self.video.ndim != 5:
            raise ValueError(f"MiniMaxH3 video latents must be [B,24,F,H,W], got {tuple(self.video.shape)}.")
        if self.audio.ndim != 4:
            raise ValueError(f"MiniMaxH3 audio latents must be [B,2,32,T], got {tuple(self.audio.shape)}.")
        if self.video.shape[0] != self.audio.shape[0]:
            raise ValueError("MiniMaxH3 video/audio batches must match.")
        if self.video.shape[1] != _VIDEO_CHANNELS:
            raise ValueError(f"MiniMaxH3 video latent channels must be {_VIDEO_CHANNELS}, got {self.video.shape[1]}.")
        if self.audio.shape[1] != _AUDIO_CHANNELS or self.audio.shape[2] != _AUDIO_BANDS:
            raise ValueError(
                f"MiniMaxH3 audio latents must be [B,{_AUDIO_CHANNELS},{_AUDIO_BANDS},T], "
                f"got {tuple(self.audio.shape)}."
            )
        if self.video.device != self.audio.device:
            raise ValueError("MiniMaxH3 video/audio latents must be on the same device.")




@dataclass(frozen=True)
class MiniMaxH3Timesteps:
    """MiniMax H3 clean-time values ``t=1-sigma`` for video and audio."""

    video: torch.Tensor
    audio: torch.Tensor

    def validate(self, batch_size: int) -> None:
        if tuple(self.video.shape) != (batch_size,):
            raise ValueError(
                f"MiniMaxH3 video timesteps must be [B], got {tuple(self.video.shape)} for B={batch_size}."
            )
        if tuple(self.audio.shape) != (batch_size,):
            raise ValueError(
                f"MiniMaxH3 audio timesteps must be [B], got {tuple(self.audio.shape)} for B={batch_size}."
            )


@dataclass(frozen=True)
class MiniMaxH3Decoded:
    video: torch.Tensor
    audio: torch.Tensor
    sampling_rate: int


class MiniMaxH3TextEncoder(nn.Module):
    """Frozen Qwen3-VL prompt encoder for MiniMax H3 text-to-audio-video prompts."""

    HIDDEN_STATE_INDEX = 50

    def __init__(
        self,
        model_id: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 4096,
        cpu_offload: bool = True,
        model_revision: str = "939557dc319dd91227e30195a763f272ba7f8765",
    ) -> None:
        super().__init__()
        text_encoder_cls, tokenizer_cls, processor_cls = _load_qwen3_vl_types()
        cache_dir = _resolve_h3_cache_dir(model_id)
        self.tokenizer = safe_load_from_pretrained(
            tokenizer_cls.from_pretrained,
            model_id,
            cache_dir=cache_dir,
            subfolder="tokenizer",
            revision=model_revision,
        )
        self.text_encoder = safe_load_from_pretrained(
            text_encoder_cls.from_pretrained,
            model_id,
            cache_dir=cache_dir,
            subfolder="text_encoder",
            revision=model_revision,
            torch_dtype=torch_dtype,
        )
        self.text_encoder.eval().requires_grad_(False)
        snapshot_root = _resolve_snapshot_root(model_id, revision=model_revision)
        processor_dir = snapshot_root / "processor"
        if not processor_dir.is_dir():
            raise FileNotFoundError(f"MiniMaxH3 processor subfolder does not exist: {processor_dir}")
        self.processor = processor_cls.from_pretrained(os.fspath(processor_dir), local_files_only=True)
        self.max_sequence_length = int(max_sequence_length)
        self.keep_on_device = not bool(cpu_offload)
        self._encode_device = torch.device("cpu")
        self._smd_keep_cpu_after_fsdp_init = bool(cpu_offload)

    def to(self, *args: Any, **kwargs: Any) -> MiniMaxH3TextEncoder:
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if args:
            first = args[0]
            if isinstance(first, torch.dtype):
                dtype = first
            elif isinstance(first, torch.Tensor):
                device, dtype = first.device, first.dtype
            else:
                device = torch.device(first)
        if device is not None:
            self._encode_device = torch.device(device)
        target_device = self._encode_device if self.keep_on_device else torch.device("cpu")
        move_kwargs: dict[str, Any] = {"device": target_device}
        if dtype is not None:
            move_kwargs["dtype"] = dtype
        self.text_encoder.to(**move_kwargs)
        return self

    def train(self, mode: bool = True) -> MiniMaxH3TextEncoder:
        super().train(False)
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def encode(
        self,
        conditioning: str | list[str] | None = None,
        precision: torch.dtype = torch.bfloat16,
        max_sequence_length: int | None = None,
    ) -> MiniMaxH3Condition:
        if conditioning is None:
            raise ValueError("MiniMaxH3TextEncoder.encode requires a prompt or prompt list.")
        prompts = [conditioning] if isinstance(conditioning, str) else list(conditioning)
        if len(prompts) != 1:
            raise ValueError(
                f"MiniMaxH3TextEncoder currently supports batch_size=1 per rank only; got {len(prompts)} prompts."
            )

        max_length = int(max_sequence_length or self.max_sequence_length)
        device = self._encode_device
        model_device = device if device.type == "cuda" else torch.device("cpu")
        inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device)
        token_ids = input_ids[0].tolist()
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]),
            dtype=torch.long,
            device=model_device,
        )

        if not hasattr(self.text_encoder, "model"):
            raise AttributeError("MiniMaxH3 Qwen3-VL text encoder must expose a `.model` attribute.")

        self.text_encoder.to(model_device)
        self.text_encoder.eval()
        try:
            outputs = self.text_encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[self.HIDDEN_STATE_INDEX].to(device=model_device, dtype=precision)
        finally:
            if model_device.type == "cuda" and not self.keep_on_device:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

        seq_len = int(hidden_states.shape[1])
        tags = torch.ones(seq_len, dtype=torch.long, device=model_device)
        condition = MiniMaxH3Condition(prompt_embeds=hidden_states.clone(), text_token_tags=tags)
        condition.validate()
        return condition




class MiniMaxH3(SMDNetwork):
    """MiniMax H3 joint audio/video velocity transformer."""

    network_family = "minimax_h3"
    supports_joint_av = True
    _OFFICIAL_FP32_MODULES = (
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
    )
    MODEL_ID = "MiniMaxAI/MiniMax-H3"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        model_revision: str = "939557dc319dd91227e30195a763f272ba7f8765",
        transformer_model_id: str | None = None,
        load_pretrained: bool = True,
        pretrained_device: str | torch.device | None = None,
        transformer_config: dict[str, Any] | None = None,
        gradient_checkpointing: bool = True,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,
        text_encoder_cpu_offload: bool = True,
        transformer_subfolder: str = "transformer",
        condition_video_timestep: float = 0.999,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        ff_chunk_size: int = 0,
        qk_norm_chunk_size: int = 0,
        sparse_attention_temporal_radius: int = 0,
        sparse_attention_max_text_tokens: int = 0,
        context_parallel_size: int = 1,
        context_parallel_attention_head_chunk_size: int = 0,
        context_parallel_attention_query_chunk_size: int = 0,
        debug_active_block_indices: list[int] | None = None,
        debug_trainable_block_indices: list[int] | None = None,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        **model_kwargs: Any,
    ) -> None:
        if net_pred_type != "flow":
            raise ValueError("MiniMaxH3 natively predicts flow velocity; net_pred_type must be 'flow'.")
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)
        self.model_id = model_id
        self.model_revision = model_revision
        self.transformer_model_id = transformer_model_id or model_id
        self.pretrained_device = pretrained_device
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = list(lora_target_modules or [])
        self.text_encoder_cpu_offload = bool(text_encoder_cpu_offload)
        if transformer_subfolder not in {"transformer", "transformer_ref"}:
            raise ValueError(f"Unsupported MiniMax H3 transformer subfolder {transformer_subfolder!r}.")
        self.transformer_subfolder = transformer_subfolder
        self.condition_video_timestep = float(condition_video_timestep)
        if not 0.0 <= self.condition_video_timestep <= 1.0:
            raise ValueError("MiniMax H3 condition_video_timestep must be in [0, 1].")
        self.context_parallel_state = get_context_parallel_state(int(context_parallel_size))
        if self.context_parallel_state.enabled:
            logger.info(
                "MiniMax H3 context parallel enabled: "
                f"degree={self.context_parallel_state.degree} "
                f"cp_rank={self.context_parallel_state.rank} "
                f"group={self.context_parallel_state.group_ranks} "
                f"data_rank={self.context_parallel_state.data_parallel_rank}/"
                f"{self.context_parallel_state.data_parallel_size}"
            )
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        if self.video_shift <= 0 or self.audio_shift <= 0:
            raise ValueError("MiniMaxH3 video_shift and audio_shift must be positive.")
        self._lora_attached = False
        self.debug_trainable_block_indices = (
            None if debug_trainable_block_indices is None else [int(index) for index in debug_trainable_block_indices]
        )
        self.transformer = self._build_transformer(load_pretrained=load_pretrained, explicit_config=transformer_config)
        self._set_gradient_checkpointing(bool(gradient_checkpointing))
        if hasattr(self.transformer, "set_ff_chunk_size"):
            self.transformer.set_ff_chunk_size(int(ff_chunk_size))
        if hasattr(self.transformer, "set_qk_norm_chunk_size"):
            self.transformer.set_qk_norm_chunk_size(int(qk_norm_chunk_size))
        if hasattr(self.transformer, "set_sparse_attention"):
            self.transformer.set_sparse_attention(
                int(sparse_attention_temporal_radius),
                int(sparse_attention_max_text_tokens),
            )
        if hasattr(self.transformer, "set_context_parallel"):
            self.transformer.set_context_parallel(self.context_parallel_state)
        if hasattr(self.transformer, "set_context_parallel_head_chunk_size"):
            self.transformer.set_context_parallel_head_chunk_size(int(context_parallel_attention_head_chunk_size))
        if hasattr(self.transformer, "set_context_parallel_query_chunk_size"):
            self.transformer.set_context_parallel_query_chunk_size(int(context_parallel_attention_query_chunk_size))
        if hasattr(self.transformer, "set_active_block_indices"):
            self.transformer.set_active_block_indices(debug_active_block_indices)

    @property
    def effective_lora_target_modules(self) -> str | list[str]:
        if self.debug_trainable_block_indices is not None:
            blocks = "|".join(str(index) for index in self.debug_trainable_block_indices)
            return (
                rf"transformer_blocks\.(?:{blocks})\."
                r"(?:attn\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))"
            )
        return list(self.lora_target_modules or _default_minimax_h3_lora_targets())

    def _set_gradient_checkpointing(self, enabled: bool) -> None:
        if hasattr(self.transformer, "set_gradient_checkpointing"):
            self.transformer.set_gradient_checkpointing(enabled)
        elif enabled and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        elif not enabled and hasattr(self.transformer, "disable_gradient_checkpointing"):
            self.transformer.disable_gradient_checkpointing()

    def _build_transformer(
        self,
        *,
        load_pretrained: bool,
        explicit_config: dict[str, Any] | None,
    ) -> nn.Module:
        runtime = _load_minimax_runtime()
        transformer_cls = runtime.MiniMaxH3Transformer3DModel
        cache_dir = _resolve_h3_cache_dir(self.transformer_model_id)
        in_meta_context = self._is_in_meta_context()

        if load_pretrained and not in_meta_context:
            placement = {} if self.pretrained_device is None else {
                "device_map": torch.device(self.pretrained_device),
                "low_cpu_mem_usage": True,
            }
            return safe_load_from_pretrained(
                transformer_cls.from_pretrained,
                self.transformer_model_id,
                cache_dir=cache_dir,
                subfolder=self.transformer_subfolder,
                revision=self.model_revision,
                torch_dtype=torch.bfloat16,
                **placement,
            )

        config = explicit_config
        if config is None:
            config = safe_load_from_pretrained(
                transformer_cls.load_config,
                self.transformer_model_id,
                cache_dir=cache_dir,
                subfolder=self.transformer_subfolder,
                revision=self.model_revision,
            )
        constructor_config = {key: value for key, value in dict(config).items() if not str(key).startswith("_")}
        return transformer_cls(**constructor_config)

    def get_inference_module(self) -> nn.Module:
        return self.transformer

    def get_noise_scheduler_snapshot(self) -> dict[str, Any]:
        return {
            "type": "minimax_h3_joint_rf",
            "video_shift": self.video_shift,
            "audio_shift": self.audio_shift,
            "supports_joint_av": True,
        }

    def visualization_decoder_spec(self) -> dict[str, object]:
        return {
            "target": "smd.networks.MiniMaxH3.decoder.MiniMaxH3AVDecoder",
            "kwargs": {"model_id": self.model_id, "model_revision": self.model_revision},
            "method": "decode_joint",
        }

    def init_preprocessors(self) -> None:
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()

    def init_text_encoder(self) -> None:
        self.text_encoder = MiniMaxH3TextEncoder(
            self.model_id,
            cpu_offload=self.text_encoder_cpu_offload,
            model_revision=self.model_revision,
        )

    def _load_native_vae(self, wrapper_cls: type[Any], subfolder: str) -> Any:
        vae = safe_load_from_pretrained(
            wrapper_cls.from_pretrained,
            self.model_id,
            cache_dir=_resolve_h3_cache_dir(self.model_id),
            subfolder=subfolder,
            revision=self.model_revision,
        )
        vae.eval().requires_grad_(False)
        return vae

    def init_video_vae(self) -> None:
        runtime = _load_minimax_runtime()
        self.video_vae = self._load_native_vae(runtime.AutoencoderKLMiniMaxH3, "vae")

    def init_audio_vae(self) -> None:
        runtime = _load_minimax_runtime()
        self.audio_vae = self._load_native_vae(
            runtime.AutoencoderKLMiniMaxH3Audio,
            _AUDIO_VAE_SUBFOLDER,
        )

    def init_vae(self) -> None:
        if not hasattr(self, "video_vae"):
            self.init_video_vae()
        if not hasattr(self, "audio_vae"):
            self.init_audio_vae()

    @torch.no_grad()
    def decode_native(self, latents: MiniMaxH3Latents) -> MiniMaxH3Decoded:
        from .decoder import decode_h3_latents

        latents.validate()
        self.init_vae()
        self.video_vae.to(latents.device)
        self.audio_vae.to(latents.device)
        return decode_h3_latents(self.video_vae, self.audio_vae, latents)

    @torch.no_grad()
    def decode_joint(
        self,
        latents: MiniMaxH3Latents | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | int]:
        if isinstance(latents, dict):
            latents = MiniMaxH3Latents(video=latents["video"], audio=latents["audio"])
        decoded = self.decode_native(latents)
        return {
            "video": decoded.video,
            "waveform": decoded.audio,
            "sample_rate": decoded.sampling_rate,
        }

    def attach_lora(self) -> None:
        if not self.use_lora:
            return
        if self._lora_attached:
            raise RuntimeError("attach_lora() called twice on the same MiniMaxH3 network.")
        self.transformer = self.attach_lora_to_module(
            self.transformer,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self._lora_attached = True
        logger.info(
            "Attached LoRA to MiniMaxH3 transformer: "
            f"r={self.lora_rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout}, "
            f"targets={self.effective_lora_target_modules}"
        )

    def _validate_inputs(
        self,
        latents: MiniMaxH3Latents,
        timesteps: MiniMaxH3Timesteps,
        condition: MiniMaxH3Condition,
    ) -> None:
        latents.validate()
        condition.validate()
        if latents.batch_size() != condition.batch_size():
            raise ValueError(
                f"MiniMaxH3 latent batch {latents.batch_size()} does not match condition batch {condition.batch_size()}."
            )
        timesteps.validate(latents.batch_size())
        if latents.batch_size() != 1:
            raise ValueError(
                "MiniMaxH3 currently supports batch_size=1 per rank only for the joint packed text/audio/video layout."
            )

    def _build_runtime_layout(
        self, x_t: MiniMaxH3Latents, condition: MiniMaxH3Condition
    ) -> tuple[Any, torch.Tensor, torch.Tensor]:
        runtime = _load_minimax_runtime()
        patch_size = tuple(getattr(self.transformer.config, "patch_size", (1, 2, 2)))
        patchify = runtime.patchify_video_latents
        pack_audio = runtime.pack_stereo_audio_latents
        target_video_rows = patchify(x_t.video, patch_size=patch_size)
        target_audio_rows = pack_audio(x_t.audio, audio_channels=_AUDIO_CHANNELS)
        video_rows = target_video_rows
        audio_rows = target_audio_rows
        layout_builder = runtime.build_t2va_packed_sequence
        layout_kwargs: dict[str, Any] = {}
        if condition.mode == "fl2va":
            condition_rows = [patchify(latent, patch_size=patch_size) for latent in condition.video_latents]
            video_rows = torch.cat([*condition_rows, target_video_rows], dim=1)
            layout_builder = runtime.build_fl2va_packed_sequence
            layout_kwargs["keyframe_anchors"] = condition.keyframe_anchors
        elif condition.mode == "ref2va":
            reference_video = condition.video_latents[0]
            reference_audio = condition.audio_latents[0]
            video_rows = torch.cat([patchify(reference_video, patch_size=patch_size), target_video_rows], dim=1)
            audio_rows = torch.cat(
                [pack_audio(reference_audio, audio_channels=_AUDIO_CHANNELS), target_audio_rows],
                dim=1,
            )
            layout_builder = runtime.build_ref2va_packed_sequence
            layout_kwargs.update(
                reference_video_shape=tuple(int(value) for value in reference_video.shape[2:]),
                num_reference_audio_latents=int(reference_audio.shape[-1]),
            )
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
        ) = layout_builder(
            condition.text_token_tags,
            num_latent_frames=x_t.video.shape[2],
            latent_height=x_t.video.shape[3],
            latent_width=x_t.video.shape[4],
            num_audio_latents=x_t.audio.shape[-1],
            patch_size=patch_size,
            audio_channels=_AUDIO_CHANNELS,
            **layout_kwargs,
        )
        layout = runtime.MiniMaxH3PackedLayout(
            position_ids=position_ids.to(device=x_t.device),
            token_tags=token_tags.to(device=x_t.device),
            video_indices=video_indices.to(device=x_t.device),
            audio_indices=audio_indices.to(device=x_t.device),
            text_indices=text_indices.to(device=x_t.device),
            num_condition_video_rows=int(num_condition_video_rows),
            num_condition_audio_rows=int(num_condition_audio_rows),
            height=x_t.video.shape[3] * 16,
            width=x_t.video.shape[4] * 16,
            num_frames=_latent_num_frames_to_video_frames(x_t.video.shape[2]),
            num_latent_frames=x_t.video.shape[2],
            latent_height=x_t.video.shape[3],
            latent_width=x_t.video.shape[4],
            num_audio_latents=x_t.audio.shape[-1],
            patch_size=patch_size,
            audio_channels=_AUDIO_CHANNELS,
            batch_size=x_t.batch_size(),
        )
        if int(getattr(self.transformer, "sparse_attention_temporal_radius", 0)) > 0:
            layout = _pad_h3_sparse_text_slots(
                layout,
                int(getattr(self.transformer, "sparse_attention_max_text_tokens", 0)),
            )
        return layout, video_rows, audio_rows

    def forward(
        self,
        x_t: MiniMaxH3Latents,
        t: MiniMaxH3Timesteps,
        condition: MiniMaxH3Condition,
        fwd_pred_type: str = "flow",
        return_logvar: bool = False,
        skip_layers: list[int] | tuple[int, ...] | None = None,
        **_kwargs: Any,
    ) -> MiniMaxH3Latents:
        if fwd_pred_type != "flow":
            raise ValueError("MiniMaxH3 only exposes its native flow velocity output.")
        if return_logvar:
            raise NotImplementedError("MiniMaxH3 does not expose an adaptive log-variance head.")
        self._validate_inputs(x_t, t, condition)

        runtime = _load_minimax_runtime()
        layout, video_rows, audio_rows = self._build_runtime_layout(x_t, condition)
        text_capacity = layout.sequence_length - layout.video_indices.numel() - layout.audio_indices.numel()
        timestep, timestep_indices = runtime.build_row_timesteps(
            layout.video_indices,
            layout.audio_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
            text_capacity,
            video_timestep=float(t.video[0].to(torch.float32).item()),
            audio_timestep=float(t.audio[0].to(torch.float32).item()),
            condition_video_timestep=max(
                float(t.video[0].to(torch.float32).item()),
                self.condition_video_timestep,
            ),
            condition_audio_timestep=1.0,
        )
        self._active_condition_row_counts = (
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
        )
        try:
            output = self.transformer(
                hidden_states=video_rows,
                audio_hidden_states=audio_rows,
                encoder_hidden_states=condition.prompt_embeds.to(device=x_t.device, dtype=x_t.video.dtype),
                timestep=timestep.to(device=x_t.device, dtype=torch.float32),
                timestep_indices=timestep_indices.to(device=x_t.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                skip_layers=skip_layers,
                attention_kwargs={
                    "video_rows_per_frame": layout.rows_per_frame,
                    "text_padding_start": layout.text_indices.numel(),
                    "text_padding_end": text_capacity,
                },
                return_dict=False,
            )
        finally:
            del self._active_condition_row_counts
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("MiniMaxH3 transformer forward must return a (video_rows, audio_rows) tuple.")
        patch_size = tuple(getattr(self.transformer.config, "patch_size", (1, 2, 2)))
        generated_video_rows = output[0][:, layout.num_condition_video_rows :]
        generated_audio_rows = output[1][:, layout.num_condition_audio_rows :]
        video = runtime.unpatchify_video_latents(
            generated_video_rows,
            patch_size=patch_size,
            num_frames=x_t.video.shape[2],
            height=x_t.video.shape[3],
            width=x_t.video.shape[4],
        )
        audio = runtime.unpack_stereo_audio_rows(
            generated_audio_rows,
            num_audio_latents=x_t.audio.shape[-1],
            audio_channels=_AUDIO_CHANNELS,
        )
        return MiniMaxH3Latents(video=video, audio=audio)

    @staticmethod
    def _warp_raw_schedule(raw_sigmas: torch.Tensor, shift: float) -> torch.Tensor:
        shift_value = torch.tensor(float(shift), dtype=torch.float32, device=raw_sigmas.device)
        return raw_sigmas * shift_value / (1.0 + (shift_value - 1.0) * raw_sigmas)

    def build_paired_schedule(
        self,
        num_steps: int,
        final_step_size_scale: float,
        device: torch.device | None = None,
        *,
        video_shift: float | None = None,
        audio_shift: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_shift = self.video_shift if video_shift is None else float(video_shift)
        audio_shift = self.audio_shift if audio_shift is None else float(audio_shift)
        raw_sigmas = torch.tensor(
            BaseNoiseSchedule.compute_raw_grid_knots(
                int(num_steps),
                float(final_step_size_scale),
            ),
            dtype=torch.float32,
            device=device,
        )
        return (
            self._warp_raw_schedule(raw_sigmas, shift=video_shift),
            self._warp_raw_schedule(raw_sigmas, shift=audio_shift),
        )

    @staticmethod
    def official_euler(
        state: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma.to(dtype=torch.float32, device=state.device)
        sigma_next = sigma_next.to(dtype=torch.float32, device=state.device)
        if sigma.ndim == 1:
            sigma = sigma.view(state.shape[0], *([1] * (state.ndim - 1)))
        elif sigma.ndim != 0:
            raise ValueError(f"MiniMaxH3 sigma must be scalar or [B], got {tuple(sigma.shape)}.")
        if sigma_next.ndim == 1:
            sigma_next = sigma_next.view(state.shape[0], *([1] * (state.ndim - 1)))
        elif sigma_next.ndim != 0:
            raise ValueError(f"MiniMaxH3 sigma_next must be scalar or [B], got {tuple(sigma_next.shape)}.")
        return (state.float() + (sigma - sigma_next) * velocity.float()).to(state.dtype)

    @torch.no_grad()
    def sample(
        self,
        noise: MiniMaxH3Latents,
        condition: MiniMaxH3Condition,
        neg_condition: MiniMaxH3Condition | None = None,
        guidance_scale: float | None = None,
        num_steps: int = 4,
        final_step_size_scale: float = 0.5,
        sigmas: tuple[torch.Tensor, torch.Tensor] | None = None,
        **_kwargs: Any,
    ) -> MiniMaxH3Latents:
        if neg_condition is not None:
            raise ValueError("MiniMaxH3 is guidance-distilled and does not accept a negative condition.")
        if guidance_scale not in (None, 1.0):
            raise ValueError("MiniMaxH3 is guidance-distilled and does not support external CFG scaling.")
        state = MiniMaxH3Latents(video=noise.video.clone(), audio=noise.audio.clone())
        state.validate()
        condition.validate()
        if condition.batch_size() != state.batch_size():
            raise ValueError("MiniMaxH3 sampling condition batch must match latent batch.")
        video_sigmas, audio_sigmas = sigmas or self.build_paired_schedule(
            num_steps,
            final_step_size_scale,
            device=state.device,
        )
        if video_sigmas.shape != audio_sigmas.shape:
            raise ValueError("MiniMaxH3 video/audio schedules must have the same length.")
        for index in range(video_sigmas.numel() - 1):
            timesteps = MiniMaxH3Timesteps(
                video=(1.0 - video_sigmas[index]).expand(state.batch_size()),
                audio=(1.0 - audio_sigmas[index]).expand(state.batch_size()),
            )
            velocity = self(state, timesteps, condition)
            state = MiniMaxH3Latents(
                video=self.official_euler(state.video, velocity.video, video_sigmas[index], video_sigmas[index + 1]),
                audio=self.official_euler(state.audio, velocity.audio, audio_sigmas[index], audio_sigmas[index + 1]),
            )
        return state

    def fully_shard(self, **kwargs: Any) -> None:
        from torch.distributed._composable import checkpoint
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        self.restore_official_mixed_precision()
        base_policy = kwargs.get("mp_policy")
        fp32_kwargs = dict(kwargs)
        fp32_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=getattr(base_policy, "reduce_dtype", None),
            output_dtype=None,
            cast_forward_inputs=False,
        )
        for name in self._OFFICIAL_FP32_MODULES:
            fully_shard(getattr(self.transformer, name), **fp32_kwargs)

        use_checkpointing = bool(
            getattr(self.transformer, "gradient_checkpointing", False)
            or getattr(self.transformer, "_gradient_checkpointing", False)
            or getattr(self.transformer, "_enable_gradient_checkpointing", False)
        )
        token_refiner = getattr(self.transformer, "token_refiner", None)
        refiner_blocks = list(getattr(token_refiner, "refiner_blocks", [])) if token_refiner is not None else []
        transformer_blocks = list(
            getattr(self.transformer, "transformer_blocks", getattr(self.transformer, "blocks", []))
        )
        context_parallel_enabled = self.context_parallel_state.enabled
        for block in refiner_blocks:
            if use_checkpointing:
                checkpoint(block)
            fully_shard(block, **kwargs)
        for block in transformer_blocks:
            if use_checkpointing and not context_parallel_enabled:
                checkpoint(block)
            elif use_checkpointing:
                block.gradient_checkpointing = True
            fully_shard(block, **kwargs)
        if use_checkpointing:
            if context_parallel_enabled:
                self.transformer.token_refiner.gradient_checkpointing = False
                for block in refiner_blocks:
                    block.gradient_checkpointing = False
            else:
                self.transformer.use_external_gradient_checkpointing()
        if token_refiner is not None:
            fully_shard(token_refiner, **kwargs)
        fully_shard(self.transformer, **kwargs)

    def restore_official_mixed_precision(self) -> None:
        for name in self._OFFICIAL_FP32_MODULES:
            module = getattr(self.transformer, name, None)
            if module is not None:
                module.to(dtype=torch.float32)
        for name in ("time_proj", "rope"):
            module = getattr(self.transformer, name, None)
            if module is None:
                continue
            for buffer_name, buffer in module.named_buffers(recurse=True):
                owner = module
                parts = buffer_name.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                owner._buffers[parts[-1]] = buffer.to(dtype=torch.float32)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        rope = self.transformer.rope
        rope_freq_dim = int(rope.rope_freq_dim)
        rope_theta = float(self.transformer.config.rope_theta)
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(
                    0,
                    2 * rope_freq_dim,
                    2,
                    dtype=torch.float32,
                    device=rope.inv_freq.device,
                )
                / (2 * rope_freq_dim)
            )
        )
        rope._buffers["inv_freq"] = inv_freq

    def validate_official_runtime_contract(self) -> None:
        for name in self._OFFICIAL_FP32_MODULES:
            module = getattr(self.transformer, name)
            dtypes = {parameter.dtype for parameter in module.parameters()}
            if dtypes != {torch.float32}:
                raise RuntimeError(f"MiniMax H3 module {name!r} must remain FP32, got {dtypes}.")
        block_dtype = self.transformer.transformer_blocks[0].attn.to_q.weight.dtype
        if block_dtype != torch.bfloat16:
            raise RuntimeError(f"MiniMax H3 block stack must remain BF16, got {block_dtype}.")
        rope = self.transformer.rope
        expected = 1.0 / (
            float(self.transformer.config.rope_theta)
            ** (
                torch.arange(
                    0,
                    2 * int(rope.rope_freq_dim),
                    2,
                    dtype=torch.float32,
                    device=rope.inv_freq.device,
                )
                / (2 * int(rope.rope_freq_dim))
            )
        )
        torch.testing.assert_close(rope.inv_freq, expected)

    def restore_unsharded_transformer_tensors(self, device: torch.device | str) -> None:
        from torch.distributed.tensor import DTensor

        for module in self.transformer.modules():
            for parameter in module.parameters(recurse=False):
                if not isinstance(parameter, DTensor) and parameter.device.type == "cpu":
                    parameter.data = parameter.data.to(device=device)
            for name, buffer in module.named_buffers(recurse=False):
                if buffer is not None and not isinstance(buffer, DTensor) and buffer.device.type == "cpu":
                    module._buffers[name] = buffer.to(device=device)

    restore_unsharded_tensors = restore_unsharded_transformer_tensors

    def to(self, *args: Any, **kwargs: Any) -> MiniMaxH3:
        # Do not round native FP32 heads/buffers through BF16 before restoring
        # their dtype: that permanently loses checkpoint precision.
        from .precision import to_with_fp32_children

        names = (*self._OFFICIAL_FP32_MODULES, "time_proj", "rope")
        to_with_fp32_children(self, self.transformer, names, *args, **kwargs)
        for attr_name in ("text_encoder", "video_vae", "audio_vae"):
            attr = getattr(self, attr_name, None)
            if attr is not None and hasattr(attr, "to"):
                attr.to(*args, **kwargs)
        return self
