# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux2 Klein network family (9B/4B) for text-to-image generation.

VAE decode uses BFL's native AutoEncoder with BatchNorm inverse for correct
color reproduction.
"""

import types
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast

from smd.networks.network import SMDNetwork
from smd.networks.noise_schedule import NET_PRED_TYPES
from smd.utils.distributed.fsdp import apply_fsdp_checkpointing
from smd.utils.hf_utils import get_hf_cache_dir, safe_load_from_pretrained
import smd.utils.logging_utils as logger
from smd.utils.lora_utils import freeze_for_lora as freeze_for_lora  # noqa: F401


class Flux2TextEncoder:
    HIDDEN_LAYERS = (9, 18, 27)

    def __init__(self, model_id):
        cache_dir = get_hf_cache_dir(model_id)
        self.tokenizer = safe_load_from_pretrained(Qwen2TokenizerFast.from_pretrained, model_id, cache_dir=cache_dir, subfolder="tokenizer")
        self.text_encoder = safe_load_from_pretrained(Qwen3ForCausalLM.from_pretrained, model_id, cache_dir=cache_dir, subfolder="text_encoder")
        self.text_encoder.eval().requires_grad_(False)

    def encode(self, conditioning=None, precision=torch.float32, max_sequence_length=512):
        if isinstance(conditioning, str):
            conditioning = [conditioning]
        all_texts = []
        for prompt in conditioning:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            all_texts.append(text)
        inputs = self.tokenizer(all_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_sequence_length)
        input_ids = inputs["input_ids"].to(self.text_encoder.device)
        attention_mask = inputs["attention_mask"].to(self.text_encoder.device)
        with torch.no_grad():
            output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            stacked = torch.stack([output.hidden_states[k] for k in self.HIDDEN_LAYERS], dim=1)
            B, nl, sl, hd = stacked.shape
            prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, sl, nl * hd)
        return prompt_embeds.to(precision)

    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        return self


class Flux2ImageEncoder:
    """VAE encoder/decoder for Flux2 Klein.

    Follows the same latent space convention as ``Flux2KleinPipeline``:
    - Encode: vae.encode → patchify (2×2) → BN normalize → latents [B, 128, H/16, W/16]
    - Decode: BN inv_normalize → unpatchify → vae.decode → images

    The transformer operates in this patchified + BN-normalized space.
    """

    def __init__(self, model_id):
        cache_dir = get_hf_cache_dir(model_id)
        self.vae = safe_load_from_pretrained(AutoencoderKLFlux2.from_pretrained, model_id, cache_dir=cache_dir, subfolder="vae")
        self.vae.eval().requires_grad_(False)

    def encode(self, real_images):
        """Encode images to patchified + BN-normalized latent space [B, 128, H/16, W/16].

        Matches Flux2KleinPipeline: vae.encode → patchify → BN normalize.
        """
        from einops import rearrange

        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        real_images = real_images.to(device=vae_device, dtype=vae_dtype)
        latent_dist = self.vae.encode(real_images, return_dict=False)[0]
        z = latent_dist.sample()

        # Patchify: [B, 32, H, W] → [B, 128, H/2, W/2]
        z_patch = rearrange(z, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)

        # BN normalize
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(z_patch.device, z_patch.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1).to(z_patch.device, z_patch.dtype) + self.vae.config.batch_norm_eps)
        z_norm = (z_patch - bn_mean) / bn_std

        return z_norm

    def decode(self, latent_images):
        """Decode from patchified + BN-normalized latents [B, 128, H/16, W/16] to images.

        Matches Flux2KleinPipeline: BN inv_normalize → unpatchify → vae.decode.
        """
        from einops import rearrange

        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        z = latent_images.to(device=vae_device, dtype=vae_dtype)

        # BN inv_normalize
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1).to(z.device, z.dtype) + self.vae.config.batch_norm_eps)
        z_denorm = z * bn_std + bn_mean

        # Unpatchify: [B, 128, H/2, W/2] → [B, 32, H, W]
        z_unpatch = rearrange(z_denorm, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)

        # VAE decode
        images = self.vae.decode(z_unpatch, return_dict=False)[0].clip(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        return self


def classify_forward(self, hidden_states, encoder_hidden_states=None, timestep=None,
                     img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None,
                     return_features_early=False, feature_indices=None, return_logvar=False, raw_features=False):
    if feature_indices is None:
        feature_indices = set()
    if return_features_early and len(feature_indices) == 0:
        return []
    idx, features = 0, []
    num_txt_tokens = encoder_hidden_states.shape[1]
    seq_len = hidden_states.shape[1]
    if not raw_features:
        spatial_size = int(seq_len**0.5)
    timestep_scaled = timestep.to(hidden_states.dtype) * 1000
    guidance_scaled = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None
    temb = self.time_guidance_embed(timestep_scaled, guidance_scaled)
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    if img_ids.ndim == 3: img_ids = img_ids[0]
    if txt_ids.ndim == 3: txt_ids = txt_ids[0]
    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0), torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0))
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, double_stream_mod_img, double_stream_mod_txt, concat_rotary_emb, joint_attention_kwargs)
        else:
            encoder_hidden_states, hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb_mod_img=double_stream_mod_img, temb_mod_txt=double_stream_mod_txt, image_rotary_emb=concat_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
        if idx in feature_indices:
            feat = hidden_states.clone()
            if not raw_features:
                B, S, C = feat.shape; feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)
        if return_features_early and len(features) == len(feature_indices): return features
        idx += 1
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(block, hidden_states, None, single_stream_mod, concat_rotary_emb, joint_attention_kwargs)
        else:
            hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=None, temb_mod=single_stream_mod, image_rotary_emb=concat_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
        if idx in feature_indices:
            feat = hidden_states[:, num_txt_tokens:, :].clone()
            if not raw_features:
                B, S, C = feat.shape; feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)
        if return_features_early and len(features) == len(feature_indices): return features
        idx += 1
    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if return_features_early:
        assert len(features) == len(feature_indices)
        return features
    out = output if len(feature_indices) == 0 else [output, features]
    if return_logvar:
        return out, self.logvar_linear(temb)
    return out


def _default_flux2_lora_targets():
    return ["ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out",
            "attn.to_qkv_mlp_proj", "attn.to_out.0", "proj_out",
            "double_stream_modulation_img.linear", "double_stream_modulation_txt.linear",
            "single_stream_modulation.linear",
            "time_guidance_embed.timestep_embedder.linear_1", "time_guidance_embed.timestep_embedder.linear_2"]


class Flux2(SMDNetwork):
    network_family = "flux2"

    def __init__(self, model_id="black-forest-labs/FLUX.2-klein-base-4B", net_pred_type="flow",
                 schedule_type="rf", disable_grad_ckpt=False, guidance_scale=None,
                 load_pretrained=True, use_lora=False, lora_rank=256, lora_alpha=None,
                 lora_dropout=0.0, lora_target_modules=None, enable_logvar_linear=True, **model_kwargs):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)
        from smd.networks.Flux2.noise_schedule import Flux2NoiseSchedule
        base_ns = self.noise_scheduler
        self.noise_scheduler = Flux2NoiseSchedule(
            min_t=base_ns.min_t, max_t=base_ns.max_t, num_steps=base_ns.num_steps,
            shift=base_ns.shift, base_shift=base_ns._base_shift, max_shift=base_ns._max_shift,
            base_seq_len=base_ns._base_seq_len, max_seq_len=base_ns._max_seq_len,
            time_shift_type=base_ns._time_shift_type)
        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self._disable_grad_ckpt = disable_grad_ckpt
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else []
        self._lora_attached = False
        self._enable_logvar_linear = enable_logvar_linear
        self._initialize_network(model_id, load_pretrained)
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)
        if torch.backends.cuda.is_built():
            torch.backends.cuda.enable_cudnn_sdp(False)
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()
        torch.cuda.empty_cache()

    def _initialize_network(self, model_id, load_pretrained):
        in_meta = self._is_in_meta_context()
        if load_pretrained and not in_meta:
            logger.info(f"Loading Flux2 transformer from pretrained: {model_id}")
            self.transformer = safe_load_from_pretrained(Flux2Transformer2DModel.from_pretrained, model_id, cache_dir=get_hf_cache_dir(model_id), subfolder="transformer")
        else:
            config = safe_load_from_pretrained(Flux2Transformer2DModel.load_config, model_id, cache_dir=get_hf_cache_dir(model_id), subfolder="transformer")
            self.transformer = Flux2Transformer2DModel.from_config(config)
        if self._enable_logvar_linear:
            hs = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
            self.transformer.logvar_linear = torch.nn.Linear(hs, 1)

    def get_inference_module(self): return self.transformer
    @property
    def bundle_skip_prefixes(self): return ("logvar_linear",) if hasattr(self.transformer, "logvar_linear") else ()
    def get_noise_scheduler_snapshot(self): return {"type": "flow_match_euler", "schedule_type": "rf", "network_family": "flux2"}

    def attach_lora(self):
        if not self.use_lora: return
        if self._lora_attached: raise RuntimeError("attach_lora() called twice")
        from peft import LoraConfig
        targets = self.effective_lora_target_modules
        self.transformer.add_adapter(LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha, init_lora_weights="gaussian", target_modules=targets, lora_dropout=self.lora_dropout))
        self._lora_attached = True

    @property
    def effective_lora_target_modules(self) -> list[str]:
        return list(self.lora_target_modules or _default_flux2_lora_targets())

    def reset_parameters(self):
        import torch.nn as nn
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        if hasattr(self, "noise_scheduler") and self.noise_scheduler is not None:
            from smd.networks.Flux2.noise_schedule import Flux2NoiseSchedule
            old_ns = self.noise_scheduler
            if isinstance(old_ns, Flux2NoiseSchedule):
                emp = old_ns._empirical_steps; self.set_noise_schedule(); base_ns = self.noise_scheduler
                self.noise_scheduler = Flux2NoiseSchedule(empirical_steps=emp, min_t=base_ns.min_t, max_t=base_ns.max_t, num_steps=base_ns.num_steps, shift=base_ns.shift, base_shift=base_ns._base_shift, max_shift=base_ns._max_shift, base_seq_len=base_ns._base_seq_len, max_seq_len=base_ns._max_seq_len, time_shift_type=base_ns._time_shift_type)
            else:
                self.set_noise_schedule()

    def fully_shard(self, **kwargs):
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, check_fn=lambda b: isinstance(b, (Flux2TransformerBlock, Flux2SingleTransformerBlock)))
        for block in self.transformer.transformer_blocks: fully_shard(block, **kwargs)
        for block in self.transformer.single_transformer_blocks: fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        if not hasattr(self, "text_encoder"): self.init_text_encoder()
        if not hasattr(self, "vae"): self.init_vae()
    def init_text_encoder(self): self.text_encoder = Flux2TextEncoder(model_id=self.model_id)
    def init_vae(self): self.vae = Flux2ImageEncoder(model_id=self.model_id)
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"): self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"): self.vae.to(*args, **kwargs)
        return self

    def _prepare_latent_image_ids(self, height, width, device, dtype):
        return torch.cartesian_prod(torch.arange(1, device=device), torch.arange(height, device=device), torch.arange(width, device=device), torch.arange(1, device=device)).to(dtype)
    def _prepare_text_ids(self, seq_length, device, dtype):
        t = torch.zeros(seq_length, 4, device=device, dtype=dtype); t[:, 3] = torch.arange(seq_length, device=device, dtype=dtype); return t
    def _pack_latents(self, latents):
        """Pack patchified latents [B, 128, H, W] → [B, H*W, 128] for transformer."""
        B, C, H, W = latents.shape
        return latents.view(B, C, H * W).permute(0, 2, 1)
    def _unpack_latents(self, latents, height, width):
        """Unpack [B, H*W, 128] → [B, 128, H, W]."""
        B = latents.shape[0]; C = latents.shape[2]
        return latents.permute(0, 2, 1).reshape(B, C, height, width)

    def forward(self, x_t, t, condition=None, r=None, guidance=None, return_features_early=False,
                feature_indices=None, return_logvar=False, fwd_pred_type=None, raw_features=False, **fwd_kwargs):
        if feature_indices is None: feature_indices = set()
        if return_features_early and len(feature_indices) == 0: return []
        if fwd_pred_type is None: fwd_pred_type = self.net_pred_type
        height, width = x_t.shape[2], x_t.shape[3]
        prompt_embeds = condition[1] if isinstance(condition, (tuple, list)) and len(condition) > 1 else (condition[0] if isinstance(condition, (tuple, list)) else condition)
        # x_t is already patchified [B, 128, H, W], position IDs use H, W directly
        img_ids = self._prepare_latent_image_ids(height, width, x_t.device, x_t.dtype)
        txt_ids = self._prepare_text_ids(prompt_embeds.shape[1], x_t.device, x_t.dtype)
        hidden_states = self._pack_latents(x_t)
        model_outputs = self.transformer(hidden_states=hidden_states, encoder_hidden_states=prompt_embeds, timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, return_features_early=return_features_early, feature_indices=feature_indices, return_logvar=return_logvar, raw_features=raw_features)
        if return_features_early: return model_outputs
        if return_logvar: out, logvar = model_outputs[0], model_outputs[1]
        else: out = model_outputs
        if isinstance(out, torch.Tensor):
            out = self._unpack_latents(out, height, width)
            out = self.noise_scheduler.convert_model_output(x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type)
        else:
            out[0] = self._unpack_latents(out[0], height, width)
            out[0] = self.noise_scheduler.convert_model_output(x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type)
        return (out, logvar) if return_logvar else out

    @torch.no_grad()
    def sample(self, noise, condition=None, neg_condition=None, guidance_scale=None, num_steps=50, **kwargs):
        import math
        if guidance_scale is None: guidance_scale = 4.0
        batch_size, channels, height, width = noise.shape
        # noise is already in patchified space [B, 128, H, W], seq_len = H * W
        image_seq_len = height * width
        from smd.networks.Flux2.noise_schedule import compute_empirical_mu
        mu = compute_empirical_mu(image_seq_len, num_steps); shift = math.exp(mu)
        raw_t = torch.linspace(1, 0, num_steps + 1, device=noise.device, dtype=noise.dtype)
        sigmas = shift * raw_t / (1.0 + (shift - 1.0) * raw_t)
        latents = noise
        for i in range(num_steps):
            t = sigmas[i].expand(batch_size)
            noise_pred = self(latents, t, condition, fwd_pred_type="flow", guidance=None)
            if guidance_scale > 1.0 and neg_condition is not None:
                neg_pred = self(latents, t, neg_condition, fwd_pred_type="flow", guidance=None)
                noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)
            latents = latents + (sigmas[i + 1] - sigmas[i]) * noise_pred
        return latents
