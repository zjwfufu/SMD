# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lens network family (3.8B MMDiT) for text-to-image generation.

Lens uses a 48-block double-stream DiT with:
- GPT-OSS-20B-A3B text encoder (MXFP4 quantized, multi-layer features)
- Flux2 semantic VAE (shared with Flux2 Klein)
- Same patchified + BN-normalized latent space as Flux2 Klein

IMPORTANT: Latents operate in patchified BN-normalized space [B, 128, H/16, W/16].
The VAE wrapper applies patchification and batch normalization.

Model variants:
- zjwfufu/SMD-Lens/base  : Original Lens model and its frozen components
"""

import os
import types
from typing import Any, Dict

import torch
import torch.utils.checkpoint
from torch.distributed.fsdp import fully_shard

from diffusers import AutoencoderKLFlux2

from smd.networks.network import SMDNetwork
from smd.utils.distributed.fsdp import apply_fsdp_checkpointing
from smd.utils.hf_utils import get_hf_cache_dir, safe_load_from_pretrained
import smd.utils.logging_utils as logger
from smd.utils.lora_utils import freeze_for_lora as freeze_for_lora  # noqa: F401


# ---------------------------------------------------------------------------
# Text encoder: GPT-OSS with multi-layer features
# ---------------------------------------------------------------------------

# Chat template constants from lens/pipeline.py
_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97

LENS_MODEL_ID = "zjwfufu/SMD-Lens"
LENS_BASE_REVISION = "d37d28a343d577fa8bcb3a3f84c65ee9ac04855d"


def _component_options(model_id, component):
    if model_id == LENS_MODEL_ID:
        return {"subfolder": f"base/{component}", "revision": LENS_BASE_REVISION}
    return {"subfolder": component}


class LensTextEncoder:
    """GPT-OSS text encoder for Lens, with multi-layer feature extraction.

    Produces a list of per-layer hidden states at ``selected_layer_index``,
    then concatenates them along the channel dimension after per-layer RMSNorm
    (done inside the transformer, not here). We return per-layer tensors as a
    list — the transformer's ``txt_norm`` + ``txt_in`` handles the concat.

    MXFP4 handling:
        By default, GPT-OSS weights are in MXFP4. We try to load with
        ``Mxfp4Config(dequantize=False)`` to keep native MXFP4 (requires
        ``kernels`` package + capable GPU). If ``disable_mxfp4=True``,
        weights are dequantized to the pipeline dtype (bf16).
    """

    SELECTED_LAYERS = (5, 11, 17, 23)

    @staticmethod
    def _resolve_local_snapshot(model_id: str, cache_dir: str) -> str:
        """Return the local snapshot dir for *model_id* — NO collectives.

        Uses ``snapshot_download`` so the returned path is a complete local
        directory (``.../snapshots/<rev>/``) that can be indexed by subfolder
        without triggering any online metadata lookups.  Honors
        ``LOCAL_FILES_ONLY`` for air-gapped clusters (the repo must be
        pre-cached there).

        **Must not issue any collective (barrier/broadcast/all-*).** This method
        runs inside the network ``__init__``, which is itself serialized by the
        outer :func:`smd.utils.hf_utils.rank0_first_load` coordinator in
        ``SMDModel._coordinated_instantiate``: rank 0 runs the whole
        ``__init__`` first (warming the cache), then the other ranks run it from
        the warm cache, with exactly one barrier *outside* this call.  An earlier
        version added its own rank-0-first ``torch.distributed.barrier()`` here —
        that inner barrier deadlocked (AB-BA) against the outer one: rank 0 blocked
        on the inner barrier waiting for peers that were stuck on the outer barrier
        waiting for rank 0.  Cache warmth / download ordering is the OUTER
        coordinator's job; here every rank just resolves its own local path.
        """
        from huggingface_hub import snapshot_download

        from smd.utils.basic_utils import str2bool

        offline = str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
        # local_files_only when offline; otherwise let HF serve from the warm
        # cache (the outer rank0_first_load already ensured rank 0 populated it).
        if os.path.isdir(model_id):
            return model_id
        prefix = "base/" if model_id == LENS_MODEL_ID else ""
        root = snapshot_download(
            model_id, cache_dir=cache_dir, local_files_only=offline,
            revision=LENS_BASE_REVISION if model_id == LENS_MODEL_ID else None,
            allow_patterns=[prefix + "tokenizer/*", prefix + "text_encoder/*"],
        )
        return os.path.join(root, "base") if prefix else root

    def __init__(self, model_id: str, disable_mxfp4: bool = False):
        cache_dir = get_hf_cache_dir(model_id)

        # Load tokenizer.
        #
        # NOTE: AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer",
        # local_files_only=True) FAILS for Lens because the tokenizer/ subfolder
        # has no config.json, and AutoTokenizer insists on resolving an
        # AutoConfig first — which raises LocalEntryNotFoundError offline.
        # Instead we resolve the local snapshot dir explicitly and load the
        # tokenizer/ subdirectory as a plain local path (no repo_id, no
        # subfolder, no AutoConfig lookup).
        from transformers import AutoTokenizer

        snapshot_root = self._resolve_local_snapshot(model_id, cache_dir)
        tokenizer_dir = os.path.join(snapshot_root, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load text encoder with MXFP4 config
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

        # Try to use LensGptOssEncoder subclass for early-exit optimization
        te_kwargs: Dict[str, Any] = {"cache_dir": cache_dir, "subfolder": "text_encoder"}
        try:
            from transformers import Mxfp4Config

            te_kwargs["quantization_config"] = Mxfp4Config(dequantize=disable_mxfp4)
        except ImportError:
            logger.info("Mxfp4Config not available, loading GPT-OSS as-is")

        try:
            from smd.networks.Lens.text_encoder import LensGptOssEncoder

            self.text_encoder = safe_load_from_pretrained(LensGptOssEncoder.from_pretrained, snapshot_root, **te_kwargs)
            self.text_encoder.set_selected_layers(self.SELECTED_LAYERS)
            self._use_lens_encoder = True
        except (ImportError, Exception) as e:
            logger.info(f"LensGptOssEncoder not available ({e}), falling back to vanilla GptOss")
            self.text_encoder = safe_load_from_pretrained(GptOssForCausalLM.from_pretrained, snapshot_root, **te_kwargs)
            self._use_lens_encoder = False

        self.text_encoder.eval().requires_grad_(False)
        self._txt_offset = DEFAULT_TXT_OFFSET

    def _build_chat_inputs(self, prompts, max_sequence_length=512):
        """Build chat-template inputs following Lens convention."""
        rendered = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
            ]
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            text = text.split("<|return|>")[0]
            rendered.append(text)

        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    @torch.no_grad()
    def encode(self, conditioning=None, precision=torch.bfloat16, max_sequence_length=512):
        """Encode prompts → (features_list, attention_mask).

        Returns:
            features: list of 4 tensors, each [B, S, enc_hidden_dim=2880] for layers (5,11,17,23)
            mask: bool tensor [B, S] (True = valid token)

        The offset (first 97 chat-template tokens) is stripped.
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        device = next(self.text_encoder.parameters()).device
        input_ids, attn_mask = self._build_chat_inputs(conditioning, max_sequence_length)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        if self._use_lens_encoder:
            layer_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        else:
            output = self.text_encoder(
                input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False
            )
            layer_outputs = [output.hidden_states[k] for k in self.SELECTED_LAYERS]

        # Strip chat template prefix
        offset = self._txt_offset
        if input_ids.shape[1] > offset:
            features = [feat[:, offset:, :].contiguous().to(precision) for feat in layer_outputs]
            mask = attn_mask[:, offset:].bool()
        else:
            zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
            features = [layer_outputs[0].new_zeros(zero_shape).to(precision) for _ in layer_outputs]
            mask = torch.zeros((input_ids.shape[0], 0), dtype=torch.bool, device=device)

        return features, mask

    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        return self


class LensImageEncoder:
    """VAE encoder/decoder for Lens (shared Flux2 semantic VAE).

    CRITICAL: Uses the same patchified + BN-normalized latent space as Flux2 Klein.
    - Encode: vae.encode -> patchify (2x2) -> BN normalize -> latents [B, 128, H/16, W/16]
    - Decode: BN inv_normalize -> unpatchify -> vae.decode -> images
    """

    def __init__(self, model_id: str):
        cache_dir = get_hf_cache_dir(model_id)
        self.vae = safe_load_from_pretrained(
            AutoencoderKLFlux2.from_pretrained, model_id, cache_dir=cache_dir, **_component_options(model_id, "vae")
        )
        self.vae.eval().requires_grad_(False)

    def encode(self, real_images):
        """Encode images to patchified + BN-normalized latent space [B, 128, H/16, W/16]."""
        from einops import rearrange

        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        real_images = real_images.to(device=vae_device, dtype=vae_dtype)
        latent_dist = self.vae.encode(real_images, return_dict=False)[0]
        z = latent_dist.sample()

        # Patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]
        z_patch = rearrange(z, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)

        # BN normalize
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(z_patch.device, z_patch.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1).to(z_patch.device, z_patch.dtype) + self.vae.config.batch_norm_eps
        )
        z_norm = (z_patch - bn_mean) / bn_std
        return z_norm

    def decode(self, latent_images):
        """Decode from patchified + BN-normalized latents to images."""
        from einops import rearrange

        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        z = latent_images.to(device=vae_device, dtype=vae_dtype)

        # BN inv_normalize
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1).to(z.device, z.dtype) + self.vae.config.batch_norm_eps
        )
        z_denorm = z * bn_std + bn_mean

        # Unpatchify: [B, 128, H/2, W/2] -> [B, 32, H, W]
        z_unpatch = rearrange(z_denorm, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)

        images = self.vae.decode(z_unpatch, return_dict=False)[0].clip(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        return self


# ---------------------------------------------------------------------------
# classify_forward: monkey-patched onto the LensTransformer2DModel
# ---------------------------------------------------------------------------


def classify_forward(
    self,
    hidden_states,
    encoder_hidden_states=None,
    encoder_hidden_states_mask=None,
    timestep=None,
    img_shapes=None,
    return_features_early=False,
    feature_indices=None,
    return_logvar=False,
    raw_features=False,
    **kwargs,
):
    """Drop-in replacement forward for LensTransformer2DModel.

    Adds feature extraction hooks (``feature_indices``) and optional
    ``logvar_linear`` support, matching the Flux2 classify_forward API.
    """
    if feature_indices is None:
        feature_indices = set()
    if return_features_early and len(feature_indices) == 0:
        return []

    bsz, img_len, _ = hidden_states.shape
    seq_len = img_len
    if not raw_features:
        spatial_size = int(seq_len**0.5)

    idx, features = 0, []

    # Text feature processing (multi-layer norm + concat + project)
    if self.multi_layer_encoder_feature:
        if not isinstance(encoder_hidden_states, (list, tuple)):
            raise ValueError("multi_layer_encoder_feature=True expects a list of per-layer text tensors.")
        text_seq_len = encoder_hidden_states[0].shape[1]
        normed = [self.txt_norm[i](encoder_hidden_states[i]) for i in range(len(self.selected_layer_index))]
        encoder_hidden_states_cat = torch.cat(normed, dim=-1)
    else:
        text_seq_len = encoder_hidden_states.shape[1]
        encoder_hidden_states_cat = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states_proj = self.txt_in(encoder_hidden_states_cat)

    # Build attention mask
    attention_mask = self._build_joint_attention_mask(encoder_hidden_states_mask, img_len)

    # Image embedding
    hidden_states = self.img_in(hidden_states)
    timestep_inp = timestep.to(hidden_states.dtype)
    temb = self.time_text_embed(timestep_inp, hidden_states)

    # RoPE
    image_rotary_emb = self.pos_embed(img_shapes, [text_seq_len], device=hidden_states.device)

    # Transformer blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states_proj, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states_proj,
                temb,
                image_rotary_emb,
                attention_mask,
            )
        else:
            encoder_hidden_states_proj, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states_proj,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )
        if idx in feature_indices:
            feat = hidden_states.clone()
            if not raw_features:
                B, S, C = feat.shape
                feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)
        if return_features_early and len(features) == len(feature_indices):
            return features
        idx += 1

    # Final norm + projection
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if return_features_early:
        assert len(features) == len(feature_indices)
        return features

    out = output if len(feature_indices) == 0 else [output, features]
    if return_logvar:
        return out, self.logvar_linear(temb)
    return out


# ---------------------------------------------------------------------------
# LoRA targets for LensTransformer2DModel
# ---------------------------------------------------------------------------


def _default_lens_lora_targets():
    """Default LoRA target modules for LensTransformerBlock architecture."""
    return [
        # Joint attention
        "attn.img_qkv",
        "attn.txt_qkv",
        "attn.to_out.0",
        "attn.to_add_out",
        # SwiGLU MLP (image stream)
        "img_mlp.w1",
        "img_mlp.w2",
        "img_mlp.w3",
        # SwiGLU MLP (text stream)
        "txt_mlp.w1",
        "txt_mlp.w2",
        "txt_mlp.w3",
        # Modulation
        "img_mod.1",
        "txt_mod.1",
        # Final projection
        "proj_out",
    ]


# ---------------------------------------------------------------------------
# Main network class
# ---------------------------------------------------------------------------


class Lens(SMDNetwork):
    """Lens 3.8B MMDiT network for SMD distillation.

    Architecture: 48 double-stream blocks, GPT-OSS multi-layer text features,
    Flux2 semantic VAE, patchified BN-normalized latent space [B, 128, H/16, W/16].
    """

    network_family = "lens"

    def __init__(
        self,
        model_id=LENS_MODEL_ID,
        net_pred_type="flow",
        schedule_type="rf",
        disable_grad_ckpt=False,
        guidance_scale=None,
        load_pretrained=True,
        use_lora=False,
        lora_rank=256,
        lora_alpha=None,
        lora_dropout=0.0,
        lora_target_modules=None,
        enable_logvar_linear=True,
        disable_mxfp4=False,
        **model_kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        # Replace base noise schedule with Lens-specific one
        from smd.networks.Lens.noise_schedule import LensNoiseSchedule

        base_ns = self.noise_scheduler
        self.noise_scheduler = LensNoiseSchedule(
            min_t=base_ns.min_t,
            max_t=base_ns.max_t,
            num_steps=base_ns.num_steps,
            shift=base_ns.shift,
            base_shift=base_ns._base_shift,
            max_shift=base_ns._max_shift,
            base_seq_len=base_ns._base_seq_len,
            max_seq_len=base_ns._max_seq_len,
            time_shift_type=base_ns._time_shift_type,
        )

        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self._disable_grad_ckpt = disable_grad_ckpt
        self._disable_mxfp4 = disable_mxfp4
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else []
        self._lora_attached = False
        self._enable_logvar_linear = enable_logvar_linear

        self._initialize_network(model_id, load_pretrained)

        # Monkey-patch forward for feature extraction + logvar support
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        if torch.backends.cuda.is_built():
            torch.backends.cuda.enable_cudnn_sdp(False)

        # Ensure gradient_checkpointing attribute exists (some diffusers versions need it)
        if not hasattr(self.transformer, "gradient_checkpointing"):
            self.transformer.gradient_checkpointing = False

        if disable_grad_ckpt:
            self.transformer.gradient_checkpointing = False
        else:
            try:
                self.transformer.enable_gradient_checkpointing()
            except (ValueError, AttributeError):
                # Manual fallback: set the flag directly
                self.transformer.gradient_checkpointing = True
                logger.info("Enabled gradient checkpointing via manual flag")
        torch.cuda.empty_cache()

    def _initialize_network(self, model_id, load_pretrained):
        from smd.networks.Lens.transformer import LensTransformer2DModel

        in_meta = self._is_in_meta_context()
        if load_pretrained and not in_meta:
            logger.info(f"Loading Lens transformer from pretrained: {model_id}")
            cache_dir = get_hf_cache_dir(model_id)
            self.transformer = safe_load_from_pretrained(
                LensTransformer2DModel.from_pretrained,
                model_id,
                cache_dir=cache_dir,
                **_component_options(model_id, "transformer"),
            )
        else:
            config = safe_load_from_pretrained(
                LensTransformer2DModel.load_config,
                model_id,
                cache_dir=get_hf_cache_dir(model_id),
                **_component_options(model_id, "transformer"),
            )
            self.transformer = LensTransformer2DModel.from_config(config)

        if self._enable_logvar_linear:
            hs = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
            self.transformer.logvar_linear = torch.nn.Linear(hs, 1)

    def get_inference_module(self):
        return self.transformer

    @property
    def bundle_skip_prefixes(self):
        return ("logvar_linear",) if hasattr(self.transformer, "logvar_linear") else ()

    def get_noise_scheduler_snapshot(self):
        return {"type": "flow_match_euler", "schedule_type": "rf", "network_family": "lens"}

    def attach_lora(self):
        if not self.use_lora:
            return
        if self._lora_attached:
            raise RuntimeError("attach_lora() called twice")
        from peft import LoraConfig

        targets = self.effective_lora_target_modules
        self.transformer.add_adapter(
            LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=targets,
                lora_dropout=self.lora_dropout,
            )
        )
        self._lora_attached = True

    @property
    def effective_lora_target_modules(self) -> list[str]:
        return list(self.lora_target_modules or _default_lens_lora_targets())

    def reset_parameters(self):
        import torch.nn as nn

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        if hasattr(self, "noise_scheduler") and self.noise_scheduler is not None:
            from smd.networks.Lens.noise_schedule import LensNoiseSchedule

            old_ns = self.noise_scheduler
            if isinstance(old_ns, LensNoiseSchedule):
                emp = old_ns._empirical_steps
                self.set_noise_schedule()
                base_ns = self.noise_scheduler
                self.noise_scheduler = LensNoiseSchedule(
                    empirical_steps=emp,
                    min_t=base_ns.min_t,
                    max_t=base_ns.max_t,
                    num_steps=base_ns.num_steps,
                    shift=base_ns.shift,
                    base_shift=base_ns._base_shift,
                    max_shift=base_ns._max_shift,
                    base_seq_len=base_ns._base_seq_len,
                    max_seq_len=base_ns._max_seq_len,
                    time_shift_type=base_ns._time_shift_type,
                )
            else:
                self.set_noise_schedule()

    def fully_shard(self, **kwargs):
        from smd.networks.Lens.transformer import LensTransformerBlock

        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, check_fn=lambda b: isinstance(b, LensTransformerBlock))
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    # ---- Preprocessor init ----

    def init_preprocessors(self):
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        self.text_encoder = LensTextEncoder(model_id=self.model_id, disable_mxfp4=self._disable_mxfp4)

    def init_vae(self):
        self.vae = LensImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    # ---- Latent packing (patchified space, same as Flux2) ----

    def _pack_latents(self, latents):
        """Pack patchified latents [B, 128, H, W] -> [B, H*W, 128] for transformer."""
        B, C, H, W = latents.shape
        return latents.view(B, C, H * W).permute(0, 2, 1)

    def _unpack_latents(self, latents, height, width):
        """Unpack [B, H*W, 128] -> [B, 128, H, W]."""
        B = latents.shape[0]
        C = latents.shape[2]
        return latents.permute(0, 2, 1).reshape(B, C, height, width)

    def rescale_cfg_velocity(self, conditional, combined):
        """Match Lens's official per-token CFG norm rescaling."""
        if conditional.ndim != 4 or combined.shape != conditional.shape:
            raise ValueError(
                "Lens CFG rescale expects matched dense [B,C,H,W] velocities, "
                f"got {tuple(conditional.shape)} and {tuple(combined.shape)}."
            )
        height, width = conditional.shape[-2:]
        conditional_packed = self._pack_latents(conditional)
        combined_packed = self._pack_latents(combined)
        conditional_norm = conditional_packed.norm(dim=-1, keepdim=True)
        combined_norm = combined_packed.norm(dim=-1, keepdim=True)
        scaled = combined_packed * (conditional_norm / combined_norm.clamp_min(1e-12))
        return self._unpack_latents(scaled, height, width)

    # ---- Forward ----

    def forward(
        self,
        x_t,
        t,
        condition=None,
        r=None,
        guidance=None,
        return_features_early=False,
        feature_indices=None,
        return_logvar=False,
        fwd_pred_type=None,
        raw_features=False,
        **fwd_kwargs,
    ):
        """Forward pass through Lens transformer.

        Args:
            x_t: [B, 128, H, W] patchified BN-normalized latents
            t: [B] timestep in [0, 1]
            condition: tuple of (features_list, mask) from LensTextEncoder.encode()
                       or (prompt_embeds_list, mask) pre-encoded
            guidance: unused for Lens (no guidance embedding)
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type

        height, width = x_t.shape[2], x_t.shape[3]

        # Parse condition: (features_list, mask)
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            encoder_features, encoder_mask = condition[0], condition[1]
            if isinstance(encoder_features, torch.Tensor):
                # Single-layer mode: wrap in list
                encoder_features = [encoder_features]
        else:
            raise ValueError(
                "Lens.forward expects condition=(features_list, mask) from LensTextEncoder.encode(). "
                f"Got type: {type(condition)}"
            )

        hidden_states = self._pack_latents(x_t)
        img_shapes = [(1, height, width)]

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            timestep=t,
            img_shapes=img_shapes,
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

        return (out, logvar) if return_logvar else out

    @torch.no_grad()
    def sample(self, noise, condition=None, neg_condition=None, guidance_scale=None, num_steps=50, **kwargs):
        """Euler sampling matching the official LensPipeline exactly.

        Lens schedule (from scheduler_config.json):
          use_dynamic_shifting=True, time_shift_type=exponential
          1) raw_sigmas = linspace(1, 1/N, N)
          2) mu = compute_empirical_mu(seq_len, N)
          3) shifted_sigmas = time_shift(mu, 1.0, raw_sigmas)
                            = exp(mu) * s / (1 + (exp(mu) - 1) * s)
          4) Euler step with shifted sigmas

        CFG uses Lens-style per-token norm rescaling in packed [B, S, C] space.
        """
        import math
        import numpy as np

        if guidance_scale is None:
            guidance_scale = 4.0

        batch_size, channels, height, width = noise.shape
        image_seq_len = height * width

        # Step 1: raw sigmas
        raw_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)

        # Step 2: compute mu and apply exponential time shift (matching scheduler)
        from smd.networks.Lens.noise_schedule import compute_empirical_mu

        mu = compute_empirical_mu(image_seq_len, num_steps)
        shift = math.exp(mu)
        shifted_sigmas = shift * raw_sigmas / (1.0 + (shift - 1.0) * raw_sigmas)

        # Append terminal 0. Use float64 for sigma arithmetic to avoid bf16 truncation.
        sigmas = np.concatenate([shifted_sigmas, [0.0]])
        sigmas = torch.tensor(sigmas, device=noise.device, dtype=torch.float64)

        latents = noise
        for i in range(num_steps):
            t = sigmas[i].to(latents.dtype).expand(batch_size)
            noise_pred = self(latents, t, condition, fwd_pred_type="flow", guidance=None)

            if guidance_scale > 1.0 and neg_condition is not None:
                neg_pred = self(latents, t, neg_condition, fwd_pred_type="flow", guidance=None)
                # Lens-style CFG with per-token norm rescaling
                # Must operate in packed [B, S, C] space for correct per-token norms
                cond_packed = self._pack_latents(noise_pred)  # [B, H*W, 128]
                neg_packed = self._pack_latents(neg_pred)  # [B, H*W, 128]
                comb_packed = neg_packed + guidance_scale * (cond_packed - neg_packed)
                cond_norm = torch.norm(cond_packed, dim=-1, keepdim=True)  # [B, S, 1]
                comb_norm = torch.norm(comb_packed, dim=-1, keepdim=True)  # [B, S, 1]
                scale = torch.where(
                    comb_norm > 0,
                    cond_norm / comb_norm.clamp_min(1e-12),
                    torch.ones_like(comb_norm),
                )
                noise_pred = self._unpack_latents(comb_packed * scale, height, width)

            # Euler step with float64 dt to avoid bf16 truncation
            dt = (sigmas[i + 1] - sigmas[i]).to(latents.dtype)
            latents = latents + dt * noise_pred

        return latents
