# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, List, Set, Union, Tuple
import types

import torch
import torch.utils.checkpoint
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from smd.networks.network import SMDNetwork
from smd.networks.noise_schedule import NET_PRED_TYPES
from smd.utils.distributed.fsdp import apply_fsdp_checkpointing
from smd.utils.hf_utils import get_hf_cache_dir, safe_load_from_pretrained
import smd.utils.logging_utils as logger
from smd.utils.lora_utils import freeze_for_lora as freeze_for_lora  # noqa: F401


def _default_flux_lora_targets() -> List[str]:
    """Default LoRA target module substrings for Flux.

    Covers MLP/FFN projections in both double and single transformer blocks plus
    the time-embedding MLP. Attention q/k/v/o are deliberately left full-rank
    (frozen when use_lora=True).
    """
    return [
        "proj_mlp",  # single_transformer_blocks MLP
        "proj_out",  # output projection (root + per-block)
        "ff.net.0.proj",  # double-block image FFN in
        "ff.net.2",  # double-block image FFN out
        "ff_context.net.0.proj",  # double-block text FFN in
        "ff_context.net.2",  # double-block text FFN out
        "timestep_embedder.linear_1",
        "timestep_embedder.linear_2",
    ]


class FluxTextEncoder:
    """Text encoder for Flux using CLIP and T5 models."""

    def __init__(self, model_id: str):
        cache_dir = get_hf_cache_dir(model_id)

        # CLIP text encoder
        self.tokenizer = safe_load_from_pretrained(
            CLIPTokenizer.from_pretrained,
            model_id,
            cache_dir=cache_dir,
            subfolder="tokenizer",
        )
        self.text_encoder = safe_load_from_pretrained(
            CLIPTextModel.from_pretrained,
            model_id,
            cache_dir=cache_dir,
            subfolder="text_encoder",
        )
        self.text_encoder.eval().requires_grad_(False)

        # T5 text encoder
        self.tokenizer_2 = safe_load_from_pretrained(
            T5TokenizerFast.from_pretrained,
            model_id,
            cache_dir=cache_dir,
            subfolder="tokenizer_2",
        )
        self.text_encoder_2 = safe_load_from_pretrained(
            T5EncoderModel.from_pretrained,
            model_id,
            cache_dir=cache_dir,
            subfolder="text_encoder_2",
        )
        self.text_encoder_2.eval().requires_grad_(False)

    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            precision: Data type for the output embeddings.
            max_sequence_length: Maximum sequence length for T5 tokenization.

        Returns:
            Tuple of (pooled_prompt_embeds, prompt_embeds) tensors.
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        # CLIP encoding for pooled embeddings
        text_inputs = self.tokenizer(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            prompt_embeds = self.text_encoder(
                text_input_ids,
                output_hidden_states=False,
            )
            pooled_prompt_embeds = prompt_embeds.pooler_output.to(precision)

        # T5 encoding for text embeddings
        text_inputs_2 = self.tokenizer_2(
            conditioning,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_input_ids_2 = text_inputs_2.input_ids.to(self.text_encoder_2.device)
            prompt_embeds_2 = self.text_encoder_2(
                text_input_ids_2,
                output_hidden_states=False,
            )[0].to(precision)

        return pooled_prompt_embeds, prompt_embeds_2

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.text_encoder.to(*args, **kwargs)
        self.text_encoder_2.to(*args, **kwargs)
        return self


class FluxImageEncoder:
    """VAE encoder/decoder for Flux.

    Flux VAE uses both scaling_factor and shift_factor for latent normalization.
    """

    def __init__(self, model_id: str):
        self.vae: AutoencoderKL = safe_load_from_pretrained(
            AutoencoderKL.from_pretrained,
            model_id,
            cache_dir=get_hf_cache_dir(model_id),
            subfolder="vae",
        )
        self.vae.eval().requires_grad_(False)

        # Flux VAE uses shift_factor in addition to scaling_factor
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.3611)
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.1159)

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            real_images: Input images in [-1, 1] range.

        Returns:
            torch.Tensor: Latent representations (shifted and scaled).
        """
        latent_images = self.vae.encode(real_images, return_dict=False)[0].sample()
        # Apply Flux-specific shift and scale
        latent_images = (latent_images - self.shift_factor) * self.scaling_factor
        return latent_images

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latent_images: Latent representations (shifted and scaled).

        Returns:
            torch.Tensor: Decoded images in [-1, 1] range.
        """
        # Reverse Flux-specific shift and scale
        latents = (latent_images / self.scaling_factor) + self.shift_factor
        # Match VAE dtype/device for decode
        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        images = self.vae.decode(latents, return_dict=False)[0].clip(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.vae.to(*args, **kwargs)
        return self


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[dict] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
    raw_features: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Modified forward pass for FluxTransformer2DModel with feature extraction support.

    Args:
        hidden_states: Input latent states.
        encoder_hidden_states: T5 text encoder hidden states.
        pooled_projections: CLIP pooled text embeddings.
        timestep: Current timestep.
        img_ids: Image position IDs.
        txt_ids: Text position IDs.
        guidance: Guidance scale embedding.
        joint_attention_kwargs: Additional attention kwargs.
        return_features_early: If True, return features as soon as collected.
        feature_indices: Set of block indices to extract features from.
        return_logvar: If True, return log variance estimate.
        raw_features: If True, return features as raw 3-D token sequences
            ``[B, seq_len, hidden_dim]`` instead of reshaping to 4-D
            ``[B, hidden_dim, H, W]``.  Useful for non-square latents
            where the square reshape would fail.

    Returns:
        Model output, optionally with features or logvar.
    """
    if feature_indices is None:
        feature_indices = set()

    if return_features_early and len(feature_indices) == 0:
        return []

    idx, features = 0, []

    # Store original sequence length to compute spatial dims for feature reshaping
    # hidden_states: [B, seq_len, C*4] where seq_len = (H//2) * (W//2)
    seq_len = hidden_states.shape[1]
    if not raw_features:
        spatial_size = int(seq_len**0.5)  # Assuming square spatial dimensions

    # 1. Patch embedding
    hidden_states = self.x_embedder(hidden_states)

    # 2. Time embedding
    timestep_scaled = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance_scaled = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep_scaled, guidance_scaled, pooled_projections)
    else:
        temb = self.time_text_embed(timestep_scaled, pooled_projections)

    # 3. Text embedding
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 4. Prepare positional embeddings
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    # 5. Joint transformer blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Check if we should extract features at this index
        if idx in feature_indices:
            feat = hidden_states.clone()
            if raw_features:
                # Keep as [B, seq_len, hidden_dim] — works for any spatial layout
                features.append(feat)
            else:
                # Reshape to [B, hidden_dim, H, W] for discriminator (assumes square)
                B, S, C = feat.shape
                feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
                features.append(feat)

        # Early return if we have all features
        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1

    # 6. Single transformer blocks
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Check if we should extract features at this index
        if idx in feature_indices:
            feat = hidden_states.clone()
            if raw_features:
                features.append(feat)
            else:
                B, S, C = feat.shape
                feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
                features.append(feat)

        # Early return if we have all features
        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1

    # 7. Final projection - hidden_states is already image-only after single blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # If we have all the features, we can exit early
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


class Flux(SMDNetwork):
    """Flux.1 network for text-to-image generation.

    Reference: https://huggingface.co/black-forest-labs/FLUX.1-dev
    """

    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    # Used by the inference-bundle exporter / loader to dispatch to the
    # right per-architecture inference script. Keep in sync with
    # ``tools/infer_<family>.py``.
    network_family: str = "flux"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        guidance_scale: Optional[float] = 3.5,
        load_pretrained: bool = True,
        use_lora: bool = False,
        lora_rank: int = 256,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        enable_logvar_linear: bool = True,
        **model_kwargs,
    ):
        """Flux.1 constructor.

        Args:
            model_id: The HuggingFace model ID to load.
                Defaults to "black-forest-labs/FLUX.1-dev".
            net_pred_type: Prediction type. Defaults to "flow" for flow matching.
            schedule_type: Schedule type. Defaults to "rf" (rectified flow).
            disable_grad_ckpt: Whether to disable gradient checkpointing during training.
                Defaults to False. Set to True when using FSDP to avoid memory access errors.
            guidance_scale: Default guidance scale for Flux.1-dev guidance distillation.
                None means no guidance. Defaults to 3.5 (recommended for Flux.1-dev).
            use_lora: If True, ``attach_lora()`` may be called later (typically
                from ``SimplifiedDistillation.build_model``) to inject LoRA adapters.
                ``__init__`` itself does NOT call ``add_adapter`` so that any
                deepcopy of the student transformer (e.g. Pivot) sees a
                clean module first. Defaults to False.
            lora_rank: LoRA rank ``r``. Defaults to 256.
            lora_alpha: LoRA ``alpha``. If None, set to ``lora_rank``
                (effective scale = 1). Defaults to None.
            lora_dropout: LoRA dropout. Defaults to 0.0.
            lora_target_modules: List of target-module substrings. If None,
                ``_default_flux_lora_targets()`` is used.
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self._disable_grad_ckpt = disable_grad_ckpt
        logger.debug(f"Embedded guidance scale: {guidance_scale}")

        # LoRA configuration (adapter injection deferred to attach_lora()).
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else []
        self._lora_attached = False

        # Initialize the network (handles meta device and pretrained loading)
        self._enable_logvar_linear = enable_logvar_linear
        self._initialize_network(model_id, load_pretrained)

        # Override forward with classify_forward
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Disable cuDNN SDPA backend to avoid mha_graph->execute errors during backward.
        # This is a known issue with Flux transformer and cuDNN attention.
        # Flash and mem_efficient backends still work; only cuDNN is problematic.
        if torch.backends.cuda.is_built():
            torch.backends.cuda.enable_cudnn_sdp(False)
            logger.info("Disabled cuDNN SDPA backend for Flux compatibility")

        # Gradient checkpointing configuration
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading Flux transformer from pretrained")
            self.transformer: FluxTransformer2DModel = safe_load_from_pretrained(
                FluxTransformer2DModel.from_pretrained,
                model_id,
                cache_dir=get_hf_cache_dir(model_id),
                subfolder="transformer",
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = safe_load_from_pretrained(
                FluxTransformer2DModel.load_config,
                model_id,
                cache_dir=get_hf_cache_dir(model_id),
                subfolder="transformer",
            )
            if in_meta_context:
                logger.info(
                    "Initializing Flux transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Flux transformer from config (no pretrained weights)")
                logger.warning("Flux transformer being initialized from config. No weights are loaded!")
            self.transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_config(config)

        # Add logvar linear layer for variance estimation - Flux uses 3072-dim time embeddings
        if self._enable_logvar_linear:
            self.transformer.logvar_linear = torch.nn.Linear(3072, 1)

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
        ns = self.noise_scheduler
        return {
            "type": "flow_match_euler",
            "shift": float(ns.shift),
            "base_shift": float(ns._base_shift),
            "max_shift": float(ns._max_shift),
            "base_seq_len": int(ns._base_seq_len),
            "max_seq_len": int(ns._max_seq_len),
            "time_shift_type": str(ns._time_shift_type),
        }

    def attach_lora(self) -> None:
        """Inject LoRA adapters into ``self.transformer``.

        Idempotent **only in the sense that calling twice raises**: this method
        is meant to be called exactly once, AFTER any deepcopy of student tail
        blocks (e.g. Pivot construction) and BEFORE FSDP wrap. The Flux
        ``__init__`` itself never calls ``add_adapter``.
        """
        if not self.use_lora:
            return
        if self._lora_attached:
            raise RuntimeError("attach_lora() called twice on the same Flux network")

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
            f"Attached LoRA to Flux transformer: r={self.lora_rank}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, targets={targets}"
        )

    @property
    def effective_lora_target_modules(self) -> list[str]:
        return list(self.lora_target_modules or _default_flux_lora_targets())

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2.
        Reinitializes all linear layers and embeddings.
        """
        import torch.nn as nn

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()

        logger.debug("Reinitialized Flux parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the Flux network for FSDP.

        Note: Flux has two types of transformer blocks:
        - transformer_blocks: Joint attention blocks for text-image interaction
        - single_transformer_blocks: Single stream blocks for image processing

        We shard `self.transformer` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.
        """
        # Note: Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing:
            # Disable the built-in gradient checkpointing (which uses torch.utils.checkpoint)
            self.transformer.disable_gradient_checkpointing()
            # Apply FSDP-compatible activation checkpointing to both block types
            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda block: isinstance(block, (FluxTransformerBlock, FluxSingleTransformerBlock)),
            )
            logger.info("Applied FSDP activation checkpointing to Flux transformer blocks")

        # Apply FSDP sharding to joint transformer blocks
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)

        # Apply FSDP sharding to single transformer blocks
        for block in self.transformer.single_transformer_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Flux."""
        self.text_encoder = FluxTextEncoder(model_id=self.model_id)

    def init_vae(self):
        """Initialize only the VAE for visualization."""
        self.vae = FluxImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def _prepare_latent_image_ids(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare image position IDs for the transformer.

        Args:
            height: Latent height (before packing, will be divided by 2).
            width: Latent width (before packing, will be divided by 2).
            device: Target device.
            dtype: Target dtype.

        Returns:
            torch.Tensor: Image position IDs [(H//2)*(W//2), 3] (2D, no batch dim).
        """
        # Use packed dimensions
        packed_height = height // 2
        packed_width = width // 2
        latent_image_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(packed_height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(packed_width, device=device, dtype=dtype)[None, :]
        latent_image_ids = latent_image_ids.reshape(packed_height * packed_width, 3)
        return latent_image_ids

    def _prepare_text_ids(
        self,
        seq_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare text position IDs.

        Args:
            seq_length: Text sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            torch.Tensor: Text position IDs [seq_length, 3] (2D, no batch dim).
        """
        text_ids = torch.zeros(seq_length, 3, device=device, dtype=dtype)
        return text_ids

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from [B, C, H, W] to [B, (H//2)*(W//2), C*4] for Flux transformer.

        Flux uses 2x2 patch packing where each 2x2 spatial block is flattened into channels.

        Args:
            latents: Input latents [B, C, H, W].

        Returns:
            Packed latents [B, (H//2)*(W//2), C*4].
        """
        batch_size, channels, height, width = latents.shape
        # Reshape to [B, C, H//2, 2, W//2, 2]
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        # Permute to [B, H//2, W//2, C, 2, 2]
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        # Reshape to [B, (H//2)*(W//2), C*4]
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        return latents

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack latents from [B, (H//2)*(W//2), C*4] to [B, C, H, W].

        Reverses the 2x2 patch packing used by Flux.

        Args:
            latents: Packed latents [B, (H//2)*(W//2), C*4].
            height: Target height (original H before packing).
            width: Target width (original W before packing).

        Returns:
            Unpacked latents [B, C, H, W].
        """
        batch_size = latents.shape[0]
        channels = latents.shape[2] // 4  # C*4 -> C
        # Reshape to [B, H//2, W//2, C, 2, 2]
        latents = latents.reshape(batch_size, height // 2, width // 2, channels, 2, 2)
        # Permute to [B, C, H//2, 2, W//2, 2]
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # Reshape to [B, C, H, W]
        latents = latents.reshape(batch_size, channels, height, width)
        return latents

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,  # unused, kept for API compatibility
        guidance: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        raw_features: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of Flux diffusion model.

        Args:
            x_t: The diffused data sample [B, C, H, W].
            t: The current timestep.
            condition: Tuple of (pooled_prompt_embeds, prompt_embeds) from text encoder.
            r: Another timestep (for mean flow methods).
            return_features_early: If True, return features once collected.
            feature_indices: Set of block indices for feature extraction.
            return_logvar: If True, return the logvar.
            fwd_pred_type: Override network prediction type.
            guidance: Optional guidance scale embedding.

        Returns:
            Model output tensor or tuple with logvar/features.
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        batch_size = x_t.shape[0]
        height, width = x_t.shape[2], x_t.shape[3]

        # Unpack condition: (pooled_prompt_embeds, prompt_embeds)
        pooled_prompt_embeds, prompt_embeds = condition

        # Prepare position IDs (2D tensors, no batch dimension)
        img_ids = self._prepare_latent_image_ids(height, width, x_t.device, x_t.dtype)
        txt_ids = self._prepare_text_ids(prompt_embeds.shape[1], x_t.device, x_t.dtype)

        # Pack latents for transformer: [B, C, H, W] -> [B, (H//2)*(W//2), C*4]
        hidden_states = self._pack_latents(x_t)

        # Note: Flux.1-dev (w/ guidance distillation) uses embedded guidance, so the default guidance is not None
        if guidance is None:
            guidance = torch.full(
                (batch_size,), self.guidance_scale, device=hidden_states.device, dtype=hidden_states.dtype
            )

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=t,  # Flux expects timestep in [0, 1]
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
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

        # Unpack output: [B, H*W, C] -> [B, C, H, W]
        if isinstance(out, torch.Tensor):
            out = self._unpack_latents(out, height, width)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            out[0] = self._unpack_latents(out[0], height, width)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    def _calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ) -> float:
        """Calculate the shift value for the scheduler based on image resolution.

        This implements the resolution-dependent shift from the Flux paper.
        """

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 3.5,
        num_steps: int = 28,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching.

        Args:
            noise: Initial noise tensor [B, C, H, W].
            condition: Tuple of (pooled_prompt_embeds, prompt_embeds).
            neg_condition: Optional negative condition tuple for CFG.
            guidance_scale: Guidance scale (if not None, enables guidance via distillation).
            num_steps: Number of sampling steps (default 28 for good quality/speed balance).
            **kwargs: Additional keyword arguments

        Returns:
            Generated latent samples.
        """
        batch_size, channels, height, width = noise.shape

        # Calculate image sequence length for shift calculation
        # After 2x2 packing: seq_len = (H // 2) * (W // 2)
        image_seq_len = (height // 2) * (width // 2)

        # Calculate resolution-dependent shift (mu)
        mu = self._calculate_shift(image_seq_len)

        # Initialize scheduler with proper shift
        scheduler = FlowMatchEulerDiscreteScheduler(shift=mu)
        scheduler.set_timesteps(num_steps, device=noise.device)
        timesteps = scheduler.timesteps

        # Initialize latents with proper scaling based on the initial timestep
        t_init = self.noise_scheduler.safe_clamp(
            timesteps[0] / 1000.0, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t
        )
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        pooled_prompt_embeds, prompt_embeds = condition

        # Prepare guidance embedding for guidance distillation (Flux.1-dev mode)
        # Note: Flux.1-dev uses embedded guidance, not traditional CFG
        guidance_tensor = None
        if guidance_scale is not None:
            guidance_tensor = torch.full((batch_size,), guidance_scale, device=latents.device, dtype=latents.dtype)

        # Sampling loop
        for timestep in timesteps:
            # Scheduler timesteps are in [0, 1000], transformer expects [0, 1]
            t = (timestep / 1000.0).expand(batch_size)
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            # Two guidance modes:
            # 1. CFG mode: when neg_condition is provided (doubles batch, uses uncond/cond difference)
            # 2. Guidance distillation mode: when neg_condition is None (single forward, guidance embedded)
            if neg_condition is not None:
                # Traditional CFG mode
                neg_pooled, neg_prompt = neg_condition
                latent_model_input = torch.cat([latents, latents], dim=0)
                pooled_input = torch.cat([neg_pooled, pooled_prompt_embeds], dim=0)
                prompt_input = torch.cat([neg_prompt, prompt_embeds], dim=0)
                t_input = torch.cat([t, t], dim=0)

                noise_pred = self(
                    latent_model_input,
                    t_input,
                    (pooled_input, prompt_input),
                    fwd_pred_type="flow",
                    guidance=None,  # No guidance embedding for CFG mode
                )

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Guidance distillation mode (recommended for Flux.1-dev)
                noise_pred = self(
                    latents,
                    t,
                    condition,
                    fwd_pred_type="flow",
                    guidance=guidance_tensor,
                )

            # Euler step
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents
