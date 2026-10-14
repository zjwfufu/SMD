# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SD3.5-Medium network family for text-to-image generation.

This module integrates Stable Diffusion 3.5 Medium into SMD using
the ``SD3Transformer2DModel`` from HuggingFace diffusers.

Key architectural characteristics:
- MMDiT-X: 24 JointTransformerBlocks with dual attention on layers 0-12
- Rectified flow with shift=3.0 (no dynamic shifting)
- Three text encoders: CLIP-L + CLIP-G + T5-XXL
- 16-channel VAE with 8x spatial downscale
- True classifier-free guidance (CFG scale=5.0 default)
- No RoPE — uses 2D sin-cos positional embeddings via PatchEmbed
- Patch size 2: latents [B,16,H,W] → patches [B, (H/2)*(W/2), 1536]
"""

import types
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import AutoencoderKL
from diffusers.models import SD3Transformer2DModel
from diffusers.models.attention import JointTransformerBlock
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    T5Config,
    T5EncoderModel,
    T5TokenizerFast,
)

from smd.networks.network import SMDNetwork
from smd.networks.noise_schedule import NET_PRED_TYPES
from smd.utils.distributed.fsdp import apply_fsdp_checkpointing
from smd.utils.hf_utils import get_hf_cache_dir, safe_load_from_pretrained
import smd.utils.logging_utils as logger


# ---------------------------------------------------------------------------
# Built-in SD3.5-Medium transformer config (avoids HF download for unit tests)
# ---------------------------------------------------------------------------


def _sd35_medium_default_config() -> dict:
    """Return the SD3.5-Medium transformer config dict.

    These values match ``stabilityai/stable-diffusion-3.5-medium/transformer/config.json``.
    Used as a fallback when HF authentication is not available.
    """
    return {
        "_class_name": "SD3Transformer2DModel",
        "sample_size": 128,
        "patch_size": 2,
        "in_channels": 16,
        "num_layers": 24,
        "attention_head_dim": 64,
        "num_attention_heads": 24,
        "joint_attention_dim": 4096,
        "caption_projection_dim": 1536,
        "pooled_projection_dim": 2048,
        "out_channels": 16,
        "pos_embed_max_size": 192,
        "dual_attention_layers": list(range(13)),  # [0..12]
        "qk_norm": "rms_norm",
    }


# ---------------------------------------------------------------------------
# classify_forward — monkey-patched onto SD3Transformer2DModel
# ---------------------------------------------------------------------------


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    joint_attention_kwargs: Optional[dict] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
    raw_features: bool = False,
    skip_layers: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Modified SD3.5 forward with feature extraction and logvar support.

    Mirrors the upstream ``SD3Transformer2DModel.forward`` but adds:
    - Feature extraction at arbitrary block indices
    - Logvar head for adaptive loss
    - Early return when all requested features are collected

    SD3 is simpler than Flux — no RoPE, no packing, no separate text/image streams.
    The patch embedding, unpatchifying, and position embedding are handled by the
    model's own ``pos_embed``, ``norm_out``, and ``proj_out``.
    """
    if feature_indices is None:
        feature_indices = set()
    if return_features_early and len(feature_indices) == 0:
        return []

    idx, features = 0, []

    # Store spatial dims for feature reshaping
    height, width = hidden_states.shape[-2:]
    patch_size = self.config.patch_size
    patched_h, patched_w = height // patch_size, width // patch_size

    # --- SD3 preprocessing (mirrors upstream forward) ---
    hidden_states = self.pos_embed(hidden_states)  # PatchEmbed: [B,16,H,W] → [B, seq, 1536] + pos
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # --- Transformer blocks ---
    for index_block, block in enumerate(self.transformer_blocks):
        is_skip = skip_layers is not None and index_block in skip_layers

        if not is_skip:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # Feature extraction
        if idx in feature_indices:
            feat = hidden_states.clone()
            if raw_features:
                features.append(feat)
            else:
                B, S, C = feat.shape
                feat = feat.permute(0, 2, 1).reshape(B, C, patched_h, patched_w)
                features.append(feat)

        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1

    # --- Output head ---
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # Unpatchify: [B, seq, patch_size*patch_size*out_channels] → [B, out_ch, H, W]
    output = output.reshape(output.shape[0], patched_h, patched_w, patch_size, patch_size, self.config.out_channels)
    output = torch.einsum("nhwpqc->nchpwq", output)
    output = output.reshape(output.shape[0], self.config.out_channels, height, width)

    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # Prepare output
    if len(feature_indices) == 0:
        out = output
    else:
        out = [output, features]

    if return_logvar:
        logvar = self.logvar_linear(temb)
        return out, logvar

    return out


# ---------------------------------------------------------------------------
# LoRA targets
# ---------------------------------------------------------------------------


def _default_sd35_lora_targets() -> List[str]:
    """Default LoRA target modules for SD3.5-Medium.

    Matches the DiffusionNFT adapter_config.json target_modules, which targets
    all attention Q/K/V/O projections in both the main and dual attention paths.
    """
    return [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    ]


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

_SD35_T5_EXPECTED = {
    "d_model": 4096,
    "num_layers": 24,
    "d_ff": 10240,
    "num_heads": 64,
}


def _validate_sd35_t5_config(config: Any) -> Any:
    """Reject Transformers' silent fallback to default T5-small config.

    Transformers 5 returns an empty config dict when ``text_encoder_3/config.json``
    is briefly not resolved; ``T5Config.from_dict({})`` then becomes T5-small
    (512 dim, 6 layers). Loading SD3.5-M T5-XXL shards into that default model
    produces the misleading ``ignore_mismatched_sizes=False`` report. Raising a
    ValueError here lets ``safe_load_from_pretrained`` retry before weights load.
    """
    mismatches = []
    for key, expected in _SD35_T5_EXPECTED.items():
        actual = getattr(config, key, None)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError(
            "Resolved SD3.5-M text_encoder_3 config is not T5-XXL: "
            + ", ".join(mismatches)
            + ". This usually means text_encoder_3/config.json was not resolved from the HF cache; retrying may fix it."
        )
    return config


def _load_sd35_t5_encoder(model_id: str, *, local_files_only: bool, **kwargs) -> T5EncoderModel:
    config = T5Config.from_pretrained(model_id, local_files_only=local_files_only, **kwargs)
    config = _validate_sd35_t5_config(config)
    return T5EncoderModel.from_pretrained(model_id, config=config, local_files_only=local_files_only, **kwargs)


class SD35TextEncoder:
    """Triple text encoder for SD3.5: CLIP-L + CLIP-G + T5-XXL.

    Produces:
    - ``encoder_hidden_states``: ``[B, seq_len, 4096]`` — concatenation of
      zero-padded CLIP-L||CLIP-G embeddings and T5-XXL embeddings along seq dim.
    - ``pooled_projections``: ``[B, 2048]`` — concatenation of CLIP-L and CLIP-G
      pooled embeddings.

    Reference: sd3.5 repo ``sd3_infer.py:319-329``.
    """

    # Max token lengths
    CLIP_MAX_LENGTH = 77
    T5_MAX_LENGTH = 256  # SD3.5-M uses up to 256 T5 tokens

    def __init__(self, model_id: str):
        cache_dir = get_hf_cache_dir(model_id)

        # CLIP-L
        logger.info("Loading CLIP-L text encoder")
        self.clip_l_tokenizer: CLIPTokenizer = safe_load_from_pretrained(
            CLIPTokenizer.from_pretrained, model_id, cache_dir=cache_dir, subfolder="tokenizer"
        )
        self.clip_l: CLIPTextModel = safe_load_from_pretrained(
            CLIPTextModel.from_pretrained, model_id, cache_dir=cache_dir, subfolder="text_encoder"
        )
        self.clip_l.eval().requires_grad_(False)

        # CLIP-G (with projection)
        logger.info("Loading CLIP-G text encoder")
        self.clip_g_tokenizer: CLIPTokenizer = safe_load_from_pretrained(
            CLIPTokenizer.from_pretrained, model_id, cache_dir=cache_dir, subfolder="tokenizer_2"
        )
        self.clip_g: CLIPTextModelWithProjection = safe_load_from_pretrained(
            CLIPTextModelWithProjection.from_pretrained, model_id, cache_dir=cache_dir, subfolder="text_encoder_2"
        )
        self.clip_g.eval().requires_grad_(False)

        # T5-XXL
        logger.info("Loading T5-XXL text encoder")
        self.t5_tokenizer: T5TokenizerFast = safe_load_from_pretrained(
            T5TokenizerFast.from_pretrained, model_id, cache_dir=cache_dir, subfolder="tokenizer_3"
        )
        self.t5: T5EncoderModel = safe_load_from_pretrained(
            _load_sd35_t5_encoder, model_id, cache_dir=cache_dir, subfolder="text_encoder_3"
        )
        self.t5.eval().requires_grad_(False)

    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            precision: Data type for outputs.
            max_sequence_length: Max T5 sequence length.

        Returns:
            Tuple of (encoder_hidden_states, pooled_projections):
                - encoder_hidden_states: ``[B, clip_len + t5_len, 4096]``
                - pooled_projections: ``[B, 2048]``
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        device = next(self.clip_l.parameters()).device

        # --- CLIP-L ---
        clip_l_inputs = self.clip_l_tokenizer(
            conditioning,
            padding="max_length",
            max_length=self.CLIP_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            clip_l_out = self.clip_l(**clip_l_inputs, output_hidden_states=True)
            # Use penultimate hidden state (layer -2), no final LN
            l_hidden = clip_l_out.hidden_states[-2]  # [B, 77, 768]
            l_pooled = clip_l_out.pooler_output  # [B, 768]

        # --- CLIP-G ---
        clip_g_inputs = self.clip_g_tokenizer(
            conditioning,
            padding="max_length",
            max_length=self.CLIP_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            clip_g_out = self.clip_g(**clip_g_inputs, output_hidden_states=True)
            g_hidden = clip_g_out.hidden_states[-2]  # [B, 77, 1280]
            g_pooled = clip_g_out.text_embeds  # [B, 1280]

        # --- T5-XXL ---
        t5_inputs = self.t5_tokenizer(
            conditioning,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            t5_out = self.t5(**t5_inputs)
            t5_hidden = t5_out.last_hidden_state  # [B, t5_len, 4096]

        # --- Combine ---
        # CLIP-L (768) || CLIP-G (1280) → [B, 77, 2048] → zero-pad to [B, 77, 4096]
        lg_hidden = torch.cat([l_hidden, g_hidden], dim=-1)  # [B, 77, 2048]
        lg_hidden = torch.nn.functional.pad(lg_hidden, (0, 4096 - 2048))  # [B, 77, 4096]

        # Concatenate along sequence dimension
        encoder_hidden_states = torch.cat([lg_hidden, t5_hidden], dim=1)  # [B, 77+t5_len, 4096]

        # Pooled: CLIP-L (768) || CLIP-G (1280) = [B, 2048]
        pooled_projections = torch.cat([l_pooled, g_pooled], dim=-1)  # [B, 2048]

        return pooled_projections.to(precision), encoder_hidden_states.to(precision)

    def to(self, *args, **kwargs):
        self.clip_l.to(*args, **kwargs)
        self.clip_g.to(*args, **kwargs)
        self.t5.to(*args, **kwargs)
        return self


# ---------------------------------------------------------------------------
# VAE encoder/decoder
# ---------------------------------------------------------------------------


class SD35ImageEncoder:
    """VAE encoder/decoder for SD3.5.

    Uses ``AutoencoderKL`` with 16 latent channels, 8x spatial downscale.
    Applies SD3LatentFormat scaling: scale_factor=1.5305, shift_factor=0.0609.
    """

    SCALE_FACTOR = 1.5305
    SHIFT_FACTOR = 0.0609

    def __init__(self, model_id: str):
        self.vae: AutoencoderKL = safe_load_from_pretrained(
            AutoencoderKL.from_pretrained,
            model_id,
            cache_dir=get_hf_cache_dir(model_id),
            subfolder="vae",
        )
        self.vae.eval().requires_grad_(False)

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space with SD3 scaling.

        Args:
            real_images: Input images in [-1, 1] range, shape ``[B, 3, H, W]``.

        Returns:
            Scaled latent representations ``[B, 16, H/8, W/8]``.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        real_images = real_images.to(device=vae_device, dtype=vae_dtype)
        latent_dist = self.vae.encode(real_images, return_dict=False)[0]
        latent_images = latent_dist.sample()
        # Apply SD3 latent scaling: process_in
        latent_images = (latent_images - self.SHIFT_FACTOR) * self.SCALE_FACTOR
        return latent_images

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        """Decode latents to images with SD3 scaling.

        Args:
            latent_images: Scaled latent representations.

        Returns:
            Decoded images in [-1, 1] range.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        # Reverse SD3 latent scaling: process_out
        latents = (latent_images / self.SCALE_FACTOR) + self.SHIFT_FACTOR
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        images = self.vae.decode(latents, return_dict=False)[0].clip(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        return self


# ---------------------------------------------------------------------------
# Main network class
# ---------------------------------------------------------------------------


class SD35Medium(SMDNetwork):
    """Stable Diffusion 3.5 Medium network for text-to-image generation.

    Wraps diffusers' ``SD3Transformer2DModel`` (~2.24B params) as a
    SMDNetwork for use with SMD training methods.

    SD3.5-Medium uses:
    - 24 JointTransformerBlocks (0-12 dual_attention, 13-22 single, 23 context_pre_only)
    - Rectified flow with fixed shift=3.0
    - True CFG (not embedded guidance)
    - 2D sin-cos positional embedding (no RoPE)
    - CLIP-L + CLIP-G + T5-XXL text encoding → [B, seq, 4096] context + [B, 2048] pooled
    """

    network_family: str = "sd35medium"

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        shift: float = 3.0,
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        use_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: Optional[int] = 64,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        enable_logvar_linear: bool = True,
        **model_kwargs,
    ):
        """SD3.5-Medium constructor.

        Args:
            model_id: HuggingFace model ID. Defaults to SD3.5-Medium.
            net_pred_type: Prediction type. Defaults to "flow".
            schedule_type: Schedule type. Defaults to "rf" (rectified flow).
            shift: Time shift for rectified flow. SD3.5-M uses 3.0.
            disable_grad_ckpt: Disable gradient checkpointing.
            load_pretrained: Whether to load pretrained weights.
            use_lora: If True, ``attach_lora()`` may be called later.
            lora_rank: LoRA rank. Defaults to 32 (matches DiffusionNFT).
            lora_alpha: LoRA alpha. Defaults to 64 (matches DiffusionNFT).
            lora_dropout: LoRA dropout.
            lora_target_modules: LoRA target modules. If None, uses defaults.
            enable_logvar_linear: Add logvar linear head.
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, shift=shift, **model_kwargs)

        self.model_id = model_id
        self._shift = shift
        self._disable_grad_ckpt = disable_grad_ckpt
        logger.debug(f"SD35Medium model_id: {model_id}, shift: {shift}")

        # LoRA config (injection deferred to attach_lora())
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else []
        self._lora_attached = False

        # Initialize transformer
        self._enable_logvar_linear = enable_logvar_linear
        self._initialize_network(model_id, load_pretrained)

        # Override forward with classify_forward
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Gradient checkpointing
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the SD3.5 transformer network."""
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info(f"Loading SD3.5 transformer from pretrained: {model_id}")
            self.transformer: SD3Transformer2DModel = safe_load_from_pretrained(
                SD3Transformer2DModel.from_pretrained,
                model_id,
                cache_dir=get_hf_cache_dir(model_id),
                subfolder="transformer",
            )
        else:
            # Try to load config from HF; fall back to known SD3.5-Medium config
            try:
                config = safe_load_from_pretrained(
                    SD3Transformer2DModel.load_config,
                    model_id,
                    cache_dir=get_hf_cache_dir(model_id),
                    subfolder="transformer",
                )
            except (OSError, EnvironmentError):
                logger.info("Cannot fetch config from HF; using built-in SD3.5-Medium config")
                config = _sd35_medium_default_config()

            if in_meta_context:
                logger.info("Initializing SD3.5 transformer on meta device (FSDP)")
            else:
                logger.info("Initializing SD3.5 transformer from config (no pretrained weights)")
                logger.warning("SD3.5 transformer initialized without weights!")
            self.transformer: SD3Transformer2DModel = SD3Transformer2DModel.from_config(config)

        # Logvar head — SD3.5-M hidden_size = num_heads * head_dim = 24 * 64 = 1536
        if self._enable_logvar_linear:
            hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
            self.transformer.logvar_linear = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Bundle export contract
    # ------------------------------------------------------------------

    def get_inference_module(self) -> torch.nn.Module:
        return self.transformer

    @property
    def bundle_skip_prefixes(self) -> Tuple[str, ...]:
        if hasattr(self.transformer, "logvar_linear"):
            return ("logvar_linear",)
        return ()

    def get_noise_scheduler_snapshot(self) -> Dict[str, Any]:
        return {
            "type": "flow_match_euler",
            "schedule_type": "rf",
            "network_family": "sd35medium",
            "shift": self._shift,
        }

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def attach_lora(self) -> None:
        """Inject LoRA adapters into self.transformer."""
        if not self.use_lora:
            return
        if self._lora_attached:
            raise RuntimeError("attach_lora() called twice on the same SD35Medium network")

        from peft import LoraConfig

        targets = self.effective_lora_target_modules
        lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=targets,
            lora_dropout=self.lora_dropout,
        )
        self.transformer.add_adapter(lora_cfg)
        self._lora_attached = True
        logger.info(
            f"Attached LoRA to SD3.5 transformer: r={self.lora_rank}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, targets={targets}"
        )

    @property
    def effective_lora_target_modules(self) -> list[str]:
        return list(self.lora_target_modules or _default_sd35_lora_targets())

    # ------------------------------------------------------------------
    # FSDP
    # ------------------------------------------------------------------

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # Reinitialize noise scheduler, preserving shift
        if hasattr(self, "noise_scheduler") and self.noise_scheduler is not None:
            self.set_noise_schedule(shift=self._shift)
            logger.debug(f"Reinitialized noise scheduler with shift={self._shift}")

        logger.debug("Reinitialized SD3.5 parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the SD3.5 network for FSDP2."""
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda block: isinstance(block, JointTransformerBlock),
            )
            logger.info("Applied FSDP activation checkpointing to SD3.5 transformer blocks")

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    # ------------------------------------------------------------------
    # Preprocessors (text encoder + VAE)
    # ------------------------------------------------------------------

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        self.text_encoder = SD35TextEncoder(model_id=self.model_id)

    def init_vae(self):
        self.vae = SD35ImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        raw_features: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of SD3.5 diffusion model.

        Args:
            x_t: Diffused data ``[B, 16, H, W]`` (16 latent channels).
            t: Current sigma in ``[0, 1]`` (rectified flow).
            condition: Tuple of (pooled, encoder_hidden_states) or single tensor.
                pooled: ``[B, 2048]`` from CLIP-L+G pooled.
                encoder_hidden_states: ``[B, seq_len, 4096]``.
            r: Unused, for API compatibility.
            return_features_early: Return features once collected.
            feature_indices: Block indices for feature extraction.
            return_logvar: Return logvar alongside output.
            fwd_pred_type: Override prediction type.
            raw_features: Keep features as 3D token sequences.

        Returns:
            Model output (velocity), optionally with features or logvar.
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        # Unpack condition
        # Convention: condition is (pooled, encoder_hidden_states) or just encoder_hidden_states
        if isinstance(condition, (tuple, list)):
            pooled_projections = condition[0]
            encoder_hidden_states = condition[1]
        else:
            # If single tensor, assume it's encoder_hidden_states with zero pooled
            encoder_hidden_states = condition
            pooled_projections = torch.zeros(x_t.shape[0], 2048, device=x_t.device, dtype=x_t.dtype)

        # SD3 expects timestep = sigma * 1000 (integer-like but can be float)
        timestep = t * 1000.0

        model_outputs = self.transformer(
            hidden_states=x_t,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            raw_features=raw_features,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Convert prediction type if needed
        if isinstance(out, torch.Tensor):
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        guidance_scale: Optional[float] = None,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching with fixed shift.

        SD3.5-Medium uses true CFG: two forward passes (cond + uncond)
        combined as ``v = v_uncond + scale * (v_cond - v_uncond)``.

        Args:
            noise: Initial noise ``[B, 16, H, W]``.
            condition: Tuple of (pooled, encoder_hidden_states).
            neg_condition: Negative prompt embeddings for CFG.
            guidance_scale: CFG scale. Default 5.0 for SD3.5-M.
            num_steps: Number of sampling steps.

        Returns:
            Generated latent samples ``[B, 16, H, W]``.
        """
        if guidance_scale is None:
            guidance_scale = 5.0

        batch_size = noise.shape[0]

        # Compute sigma schedule with shift
        # sigma(t) = shift * t / (1 + (shift - 1) * t)
        timesteps_linear = torch.linspace(1.0, 0.0, num_steps + 1, device=noise.device)
        shift = self._shift
        sigmas = shift * timesteps_linear / (1 + (shift - 1) * timesteps_linear)

        latents = noise  # At sigma=1, x_t = noise

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            t = sigma.expand(batch_size).to(latents.dtype)
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t)

            # Conditional forward
            noise_pred = self(latents, t, condition, fwd_pred_type="flow")

            # Apply CFG
            if guidance_scale > 1.0 and neg_condition is not None:
                neg_pred = self(latents, t, neg_condition, fwd_pred_type="flow")
                noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

            # Euler step: x_{t-dt} = x_t + v * (sigma_next - sigma)
            # Note: sigmas decrease, so (sigma_next - sigma) < 0
            dt = sigma_next - sigma
            latents = latents + noise_pred * dt

        return latents
