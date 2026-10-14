# SPDX-License-Identifier: Apache-2.0

"""Detached native MiniMax H3 audio/video decoder."""

from __future__ import annotations

import torch

from smd.networks.MiniMaxH3 import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from smd.utils.hf_utils import safe_load_from_pretrained

from .network import MiniMaxH3Decoded, MiniMaxH3Latents, _resolve_h3_cache_dir


@torch.no_grad()
def decode_h3_latents(
    video_vae: torch.nn.Module,
    audio_vae: torch.nn.Module,
    latents: MiniMaxH3Latents,
) -> MiniMaxH3Decoded:
    latents.validate()
    device = latents.device
    video_mean = torch.tensor(video_vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    video_std = torch.tensor(video_vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    video_latents = latents.video.float() * video_std + video_mean
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        video = video_vae.decode(video_latents, return_dict=False)[0]
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

    batch, channels, latent_channels, frames = latents.audio.shape
    audio_latents = latents.audio.reshape(batch * channels, latent_channels, frames)
    audio_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    audio_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    audio_latents = audio_latents.float() * audio_std + audio_mean
    audio = audio_vae.decode(audio_latents, return_dict=False)[0]
    audio = audio.float().reshape(batch, channels, -1)
    return MiniMaxH3Decoded(
        video=video,
        audio=audio,
        sampling_rate=int(audio_vae.config.sampling_rate),
    )


class MiniMaxH3AVDecoder:
    def __init__(self, model_id: str = "MiniMaxAI/MiniMax-H3",
                 model_revision: str = "939557dc319dd91227e30195a763f272ba7f8765") -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.video_vae = None
        self.audio_vae = None
        self.vae = None

    def to(self, device: torch.device | str) -> MiniMaxH3AVDecoder:
        cache_dir = _resolve_h3_cache_dir(self.model_id)
        self.video_vae = safe_load_from_pretrained(
            AutoencoderKLMiniMaxH3.from_pretrained,
            self.model_id,
            cache_dir=cache_dir,
            subfolder="vae",
            revision=self.model_revision,
            device_map=torch.device(device),
            low_cpu_mem_usage=True,
        ).to(device)
        self.audio_vae = safe_load_from_pretrained(
            AutoencoderKLMiniMaxH3Audio.from_pretrained,
            self.model_id,
            cache_dir=cache_dir,
            subfolder="audio_vae",
            revision=self.model_revision,
            device_map=torch.device(device),
            low_cpu_mem_usage=True,
        ).to(device)
        self.video_vae.eval().requires_grad_(False)
        self.audio_vae.eval().requires_grad_(False)
        self.vae = self.video_vae
        return self

    @torch.no_grad()
    def decode_native(self, latents: MiniMaxH3Latents) -> MiniMaxH3Decoded:
        if self.video_vae is None or self.audio_vae is None:
            raise RuntimeError("MiniMaxH3AVDecoder must be moved to a device before decoding.")
        return decode_h3_latents(self.video_vae, self.audio_vae, latents)

    @torch.no_grad()
    def decode_joint(self, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | int]:
        decoded = self.decode_native(MiniMaxH3Latents(video=latents["video"], audio=latents["audio"]))
        return {
            "video": decoded.video,
            "waveform": decoded.audio,
            "sample_rate": decoded.sampling_rate,
        }
