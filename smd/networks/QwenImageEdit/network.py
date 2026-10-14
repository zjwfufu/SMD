# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QwenImageEdit — image editing variant of QwenImage.

Key architectural differences from QwenImage (text-to-image generation):

1. **VLM text encoder processes source images** — The Qwen2.5-VL text encoder
   receives both source image pixels (via Qwen2VLProcessor) and the edit
   instruction, producing joint image-aware condition embeddings.

2. **Source image latents concatenated in token space** — The source image is
   VAE-encoded, 2x2-packed, and appended to the noise latent tokens along the
   sequence dimension. This is NOT channel concatenation — it's sequence-level.

3. **Output sliced** — Only the noise token predictions are used:
   ``output[:, :noise_seq_len, :]``.

4. **Multi-image img_shapes** — RoPE receives shapes for both the noise image
   and the source image(s).

5. **Different prompt template** — Editing-focused system prompt with 64 token
   preamble (vs 34 for generation).

Reference: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
"""

import os
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import dtype

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2VLProcessor

from smd.networks.QwenImage.network import (
    QwenImage,
    QwenImageTextEncoder,
)
from smd.networks.noise_schedule import NET_PRED_TYPES
from smd.networks.io_contract import unpack_condition_container
from smd.utils.basic_utils import str2bool
from smd.utils.hf_utils import get_hf_cache_dir
from smd.networks.QwenImageEdit.preprocessing import (
    CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE, calculate_dimensions as _calculate_dimensions, resize_image,
)


class QwenImageEditTextEncoder(QwenImageTextEncoder):
    """Text encoder for Qwen-Image-Edit using Qwen2.5-VL-7B with image understanding.

    Unlike the generation variant (QwenImageTextEncoder) which only processes text,
    this encoder feeds source images through the VLM's vision branch alongside
    the edit instruction. The VLM produces joint image+text embeddings that tell
    the diffusion transformer what to change.
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture, "
        "objects, background), then explain how the user's text instruction should "
        "alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where appropriate."
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    PROMPT_TEMPLATE_DROP_IDX = 64  # Longer preamble than generation (34)

    def __init__(self, model_id: str, torch_dtype: Optional[dtype] = torch.bfloat16):
        super().__init__(model_id=model_id, torch_dtype=torch_dtype)

        # Load the VL processor for image preprocessing
        local_files_only = str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
        cache_dir = get_hf_cache_dir(model_id)
        snapshot_root = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.processor = Qwen2VLProcessor.from_pretrained(
            os.path.join(snapshot_root, "processor"),
            local_files_only=True,
        )

    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 512,
        images: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts + optional source images to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            precision: Data type for the output embeddings.
            max_sequence_length: Maximum sequence length for tokenization.
            images: Optional list of PIL Images (source images for editing).
                When provided, they are processed through the VLM's vision branch.

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask) tensors.
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        drop_idx = self.PROMPT_TEMPLATE_DROP_IDX

        # Build prompt with image tokens if images are provided
        if images is not None:
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            if isinstance(images, list) and not isinstance(images[0], list):
                # Single image per sample: images = [pil1, pil2, ...]
                # For batch encoding, each sample gets one image
                base_img_prompts = [img_prompt_template.format(1)] * len(conditioning)
            else:
                # Multi-image per sample: images = [[pil1a, pil1b], [pil2a, ...], ...]
                base_img_prompts = []
                for img_list in images:
                    prompt = ""
                    for i, _ in enumerate(img_list):
                        prompt += img_prompt_template.format(i + 1)
                    base_img_prompts.append(prompt)

            txt = [self.PROMPT_TEMPLATE.format(base_img_prompts[i] + conditioning[i]) for i in range(len(conditioning))]
        else:
            txt = [self.PROMPT_TEMPLATE.format(p) for p in conditioning]

        # When images are provided, use the VL processor for full multimodal encoding
        if images is not None:
            # For now, encode one sample at a time since processor handles batching differently
            all_embeds = []
            all_masks = []
            for i in range(len(conditioning)):
                img = images[i] if images is not None else None
                # Processor expects images as a list
                if img is not None and not isinstance(img, list):
                    img = [img]
                img = [resize_image(image, CONDITION_IMAGE_SIZE) for image in img]

                model_inputs = self.processor(
                    text=[txt[i]],
                    images=img,
                    padding=True,
                    return_tensors="pt",
                ).to(self.text_encoder.device)

                with torch.no_grad():
                    outputs = self.text_encoder(
                        input_ids=model_inputs["input_ids"],
                        attention_mask=model_inputs["attention_mask"],
                        pixel_values=model_inputs.get("pixel_values"),
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        output_hidden_states=True,
                    )

                hidden_states = outputs.hidden_states[-1]
                split_hs = self._extract_masked_hidden(hidden_states, model_inputs["attention_mask"])
                split_hs = [e[drop_idx:] for e in split_hs]
                all_embeds.extend(split_hs)
                all_masks.extend([torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hs])

            # Pad and stack
            max_seq_len = min(max(e.size(0) for e in all_embeds), max_sequence_length)
            prompt_embeds = torch.stack(
                [
                    torch.cat([u[:max_seq_len], u.new_zeros(max(0, max_seq_len - u.size(0)), u.size(1))])
                    for u in all_embeds
                ]
            )
            encoder_attention_mask = torch.stack(
                [torch.cat([u[:max_seq_len], u.new_zeros(max(0, max_seq_len - u.size(0)))]) for u in all_masks]
            )

            return prompt_embeds.to(dtype=precision), encoder_attention_mask
        else:
            # No images: fall back to text-only encoding (parent class logic)
            return super().encode(conditioning, precision, max_sequence_length)


class QwenImageEdit(QwenImage):
    """Qwen-Image-Edit network for instruction-based image editing.

    Extends QwenImage with source image conditioning: source image latents are
    concatenated in the token (sequence) dimension, and the VLM text encoder
    processes source images alongside edit instructions.
    """

    network_family: str = "qwen_image_edit"
    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        condition_image_size: int = CONDITION_IMAGE_SIZE,
        **kwargs,
    ):
        self.condition_image_size = condition_image_size
        self.is_edit = True
        # Track noise sequence length for output slicing
        self._noise_seq_len: Optional[int] = None
        super().__init__(model_id=model_id, net_pred_type=net_pred_type, schedule_type=schedule_type, **kwargs)

        # Keep Qwen-Image-Edit's official zero_cond_t semantics: noise tokens are
        # modulated at timestep t while source-image condition tokens are modulated
        # at timestep 0 via classify_forward's modulate_index path.

    def init_preprocessors(self):
        """Initialize text and image encoders for editing."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self, torch_dtype: Optional[dtype] = torch.bfloat16):
        """Initialize the multimodal text encoder for QwenImageEdit."""
        self.text_encoder = QwenImageEditTextEncoder(model_id=self.model_id, torch_dtype=torch_dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        source_latents: Optional[torch.Tensor] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with source image latent concatenation.

        Args:
            x_t: Noise latent [B, C, H, W].
            t: Timestep in [0, 1].
            condition: (encoder_hidden_states, mask) from VLM text encoder.
            source_latents: Source image latents [B, C, Hs, Ws] (pre-VAE-encoded).
                Will be 2x2-packed and concatenated to noise tokens in sequence dim.
            Other args: same as QwenImage.forward().

        Returns:
            Model output (same shape as x_t).
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        # Unpack an optional condition container: the primary (embeds, mask)
        # rides under the primary key with ``source_latents`` alongside it.
        condition, _extra_inputs = unpack_condition_container(condition)
        if source_latents is None:
            source_latents = _extra_inputs.pop("source_latents", None)
        if _extra_inputs:
            fwd_kwargs = {**_extra_inputs, **fwd_kwargs}

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        batch_size = x_t.shape[0]
        height, width = x_t.shape[2], x_t.shape[3]

        # Unpack condition
        encoder_hidden_states, encoder_hidden_states_mask = condition

        # Pack noise latents: [B, C, H, W] -> [B, noise_seq_len, C*4]
        noise_packed = self._pack_latents(x_t)
        noise_seq_len = noise_packed.shape[1]
        self._noise_seq_len = noise_seq_len

        # Build img_shapes: starts with noise image shape
        noise_h_packed = height // 2
        noise_w_packed = width // 2
        img_shapes_per_sample = [(1, noise_h_packed, noise_w_packed)]

        # Concatenate source image latents if provided
        if source_latents is not None:
            src_h, src_w = source_latents.shape[2], source_latents.shape[3]
            source_packed = self._pack_latents(source_latents)
            hidden_states = torch.cat([noise_packed, source_packed], dim=1)
            # Add source image shape(s) for RoPE
            img_shapes_per_sample.append((1, src_h // 2, src_w // 2))
        else:
            hidden_states = noise_packed

        img_shapes = [img_shapes_per_sample] * batch_size

        # Prepare txt_seq_lens
        txt_seq_lens = encoder_hidden_states_mask.sum(dim=1).int().tolist()

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=t,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Slice to noise tokens only and unpack
        if isinstance(out, torch.Tensor):
            out = out[:, :noise_seq_len, :]
            out = self._unpack_latents(out, height, width)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            out[0] = out[0][:, :noise_seq_len, :]
            out[0] = self._unpack_latents(out[0], height, width)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_steps: int = 50,
        source_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate edited samples using Euler flow matching with true CFG.

        Args:
            noise: Initial noise tensor [B, C, H, W].
            condition: (encoder_hidden_states, mask) from VLM text encoder.
            neg_condition: Optional negative condition for CFG.
            guidance_scale: CFG scale. Default 4.0.
            num_steps: Sampling steps. Default 50.
            source_latents: Source image latents [B, C, Hs, Ws] for editing.

        Returns:
            Edited latent samples [B, C, H, W].
        """
        batch_size, _, height, width = noise.shape

        if not hasattr(self, "scheduler"):
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_id,
                subfolder="scheduler",
                cache_dir=get_hf_cache_dir(self.model_id),
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        scheduler = self.scheduler

        # Calculate resolution-dependent shift
        image_seq_len = (height // 2) * (width // 2)
        cfg = scheduler.config
        mu = cfg.base_shift + (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len) * (
            image_seq_len - cfg.base_image_seq_len
        )

        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(sigmas=sigmas, device=noise.device, mu=mu)
        timesteps = scheduler.timesteps

        t_init = self.noise_scheduler.safe_clamp(
            timesteps[0] / 1000.0, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t
        )
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        do_cfg = guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None

        scheduler.set_begin_index(0)
        for timestep in timesteps:
            t = (timestep / 1000.0).expand(batch_size).to(latents.dtype)

            pred = self(latents, t, condition, fwd_pred_type="flow", source_latents=source_latents)

            if do_cfg:
                neg_pred = self(latents, t, neg_condition, fwd_pred_type="flow", source_latents=source_latents)
                comb_pred = neg_pred + guidance_scale * (pred - neg_pred)
                # Norm-preserving rescaling in packed token space
                pred_packed = self._pack_latents(pred)
                comb_packed = self._pack_latents(comb_pred)
                cond_norm = torch.norm(pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb_packed, dim=-1, keepdim=True)
                pred = self._unpack_latents(comb_packed * (cond_norm / (comb_norm + 1e-8)), height, width)

            latents = scheduler.step(pred, timestep, latents, return_dict=False)[0]

        return latents
