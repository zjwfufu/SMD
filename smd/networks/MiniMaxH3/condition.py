# SPDX-License-Identifier: Apache-2.0

"""Lightweight MiniMax H3 conditioning shared by dataset and network code."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

_TEXT_EMBED_DIM = 5120
_VIDEO_CHANNELS = 24
_AUDIO_CHANNELS = 2
_AUDIO_BANDS = 32


@dataclass(frozen=True)
class MiniMaxH3Condition:
    """MiniMax H3 text and optional packed-media conditioning."""

    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    mode: str = "t2va"
    video_latents: tuple[torch.Tensor, ...] = ()
    audio_latents: tuple[torch.Tensor, ...] = ()
    keyframe_anchors: tuple[str, ...] = ()

    def batch_size(self) -> int:
        return int(self.prompt_embeds.shape[0])

    def to(self, device=None, dtype=None) -> MiniMaxH3Condition:
        # Modality tags index a shared packed sequence, never a batch axis.
        return replace(
            self,
            prompt_embeds=self.prompt_embeds.to(device=device, dtype=dtype),
            text_token_tags=self.text_token_tags.to(device=device),
            video_latents=tuple(t.to(device=device, dtype=dtype) for t in self.video_latents),
            audio_latents=tuple(t.to(device=device, dtype=dtype) for t in self.audio_latents),
        )

    def slice_batch(self, n: int) -> MiniMaxH3Condition:
        return replace(
            self, prompt_embeds=self.prompt_embeds[:n],
            video_latents=tuple(t[:n] for t in self.video_latents),
            audio_latents=tuple(t[:n] for t in self.audio_latents),
        )

    def repeat_batch(self, n: int) -> MiniMaxH3Condition:
        return replace(
            self, prompt_embeds=self.prompt_embeds.repeat_interleave(n, dim=0),
            video_latents=tuple(t.repeat_interleave(n, dim=0) for t in self.video_latents),
            audio_latents=tuple(t.repeat_interleave(n, dim=0) for t in self.audio_latents),
        )

    def validate(self) -> None:
        if self.prompt_embeds.ndim != 3:
            raise ValueError(
                f"MiniMaxH3 prompt_embeds must be [B,S,{_TEXT_EMBED_DIM}], got {tuple(self.prompt_embeds.shape)}."
            )
        if self.prompt_embeds.shape[-1] != _TEXT_EMBED_DIM:
            raise ValueError(
                f"MiniMaxH3 prompt embeddings must have dim {_TEXT_EMBED_DIM}, got {self.prompt_embeds.shape[-1]}."
            )
        if self.text_token_tags.ndim != 1:
            raise ValueError(f"MiniMaxH3 text_token_tags must be 1D [S], got {tuple(self.text_token_tags.shape)}.")
        if self.prompt_embeds.shape[1] != self.text_token_tags.shape[0]:
            raise ValueError(
                "MiniMaxH3 text_token_tags length must match prompt_embeds sequence length: "
                f"{self.text_token_tags.shape[0]} vs {self.prompt_embeds.shape[1]}."
            )
        if self.mode not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError(f"Unsupported MiniMax H3 condition mode {self.mode!r}.")
        if self.mode == "t2va":
            if self.video_latents or self.audio_latents or self.keyframe_anchors:
                raise ValueError("MiniMax H3 T2VA condition cannot contain media latents.")
        elif self.mode == "fl2va":
            if (
                len(self.video_latents) not in {1, 2}
                or self.audio_latents
                or len(self.keyframe_anchors) != len(self.video_latents)
            ):
                raise ValueError("MiniMax H3 FL2VA requires one or two keyframes with matching anchors.")
        elif len(self.video_latents) != 1 or len(self.audio_latents) != 1 or self.keyframe_anchors:
            raise ValueError("MiniMax H3 Ref2VA requires exactly one reference video and audio.")
        for latent in self.video_latents:
            if latent.ndim != 5 or latent.shape[0] != self.batch_size() or latent.shape[1] != _VIDEO_CHANNELS:
                raise ValueError(f"Invalid MiniMax H3 condition video shape {tuple(latent.shape)}.")
            if latent.device != self.prompt_embeds.device:
                raise ValueError("MiniMax H3 condition video and prompt embeddings must share a device.")
        for latent in self.audio_latents:
            if latent.ndim != 4 or latent.shape[:3] != (self.batch_size(), _AUDIO_CHANNELS, _AUDIO_BANDS):
                raise ValueError(f"Invalid MiniMax H3 condition audio shape {tuple(latent.shape)}.")
            if latent.device != self.prompt_embeds.device:
                raise ValueError("MiniMax H3 condition audio and prompt embeddings must share a device.")
