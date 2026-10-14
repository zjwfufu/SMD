# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

MINIMAX_H3_FPS = 24
MINIMAX_H3_SPATIAL_COMPRESSION_RATIO = 16
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_VIDEO_SHIFT = 12.0
MINIMAX_H3_AUDIO_SHIFT = 3.0

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class MiniMaxH3PackedLayout:
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    patch_size: tuple[int, int, int]
    audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS
    batch_size: int = 1

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def rows_per_frame(self) -> int:
        _, patch_h, patch_w = self.patch_size
        return (self.latent_height // patch_h) * (self.latent_width // patch_w)

    def to(self, device: torch.device | str | None = None) -> MiniMaxH3PackedLayout:
        if device is None:
            return self
        return replace(
            self,
            position_ids=self.position_ids.to(device),
            token_tags=self.token_tags.to(device),
            video_indices=self.video_indices.to(device),
            audio_indices=self.audio_indices.to(device),
            text_indices=self.text_indices.to(device),
        )

    def expand_batch(self, batch_size: int) -> MiniMaxH3PackedLayout:
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        return replace(self, batch_size=int(batch_size))


def align_num_frames(
    num_frames: int,
    frames_per_chunk: int = MINIMAX_H3_FRAMES_PER_CHUNK,
    latents_per_chunk: int = MINIMAX_H3_LATENTS_PER_CHUNK,
) -> int:
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


def video_latent_num_frames(
    num_frames: int,
    frames_per_chunk: int = MINIMAX_H3_FRAMES_PER_CHUNK,
    latents_per_chunk: int = MINIMAX_H3_LATENTS_PER_CHUNK,
) -> int:
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"`num_frames` must be of the form {frames_per_chunk} * n + {latents_per_chunk}, got {num_frames}."
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int,
    fps: float = MINIMAX_H3_FPS,
    latents_per_second: int = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
) -> int:
    return round(num_frames / fps * latents_per_second)


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    if latents.ndim != 5:
        raise ValueError(f"`latents` must have shape `(batch, channels, frames, height, width)`, got {latents.shape}.")
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(batch_size, -1, channels * patch_t * patch_h * patch_w).contiguous()


def inverse_patchify_video_latents(
    rows: torch.Tensor,
    patch_size: tuple[int, int, int],
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    squeeze_batch = rows.ndim == 2
    if squeeze_batch:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3:
        raise ValueError(f"`rows` must have shape `(batch, tokens, dim)` or `(tokens, dim)`, got {rows.shape}.")
    batch_size, _, patch_dim = rows.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"`num_frames`, `height`, and `width` must be divisible by patch {patch_size}; got {num_frames}, {height}, {width}."
        )
    channels, remainder = divmod(patch_dim, patch_t * patch_h * patch_w)
    if remainder:
        raise ValueError(f"Patch dimension {patch_dim} is not divisible by patch volume {patch_t * patch_h * patch_w}.")

    rows = rows.reshape(
        batch_size,
        num_frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = rows.reshape(batch_size, channels, num_frames, height, width).contiguous()
    return latents[0] if squeeze_batch else latents


unpatchify_video_latents = inverse_patchify_video_latents


def pack_stereo_audio_latents(latents: torch.Tensor, audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS) -> torch.Tensor:
    squeeze_batch = latents.ndim == 3
    if squeeze_batch:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4:
        raise ValueError(
            f"`latents` must have shape `(channels, dim, frames)` or `(batch, channels, dim, frames)`, got {latents.shape}."
        )
    if latents.shape[1] != audio_channels:
        raise ValueError(f"Expected {audio_channels} audio channels, got {latents.shape[1]}.")
    rows = latents.permute(0, 1, 3, 2).reshape(latents.shape[0], -1, latents.shape[2]).contiguous()
    return rows[0] if squeeze_batch else rows


def unpack_stereo_audio_rows(
    rows: torch.Tensor,
    num_audio_latents: int,
    audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS,
) -> torch.Tensor:
    squeeze_batch = rows.ndim == 2
    if squeeze_batch:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3:
        raise ValueError(f"`rows` must have shape `(tokens, dim)` or `(batch, tokens, dim)`, got {rows.shape}.")
    batch_size, num_rows, latent_dim = rows.shape
    expected_rows = num_audio_latents * audio_channels
    if num_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} audio rows, got {num_rows}.")
    latents = rows.reshape(batch_size, audio_channels, num_audio_latents, latent_dim).permute(0, 1, 3, 2).contiguous()
    return latents[0] if squeeze_batch else latents


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int,
) -> None:
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def _coerce_text_token_tags(text_token_tags: int | torch.Tensor) -> torch.Tensor:
    if isinstance(text_token_tags, int):
        if text_token_tags < 0:
            raise ValueError(f"`text_token_tags` must be non-negative, got {text_token_tags}.")
        return torch.full((text_token_tags,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    text_token_tags = torch.as_tensor(text_token_tags, dtype=torch.long).detach().cpu()
    if text_token_tags.ndim != 1:
        raise ValueError(f"`text_token_tags` must be 1-D, got shape {tuple(text_token_tags.shape)}.")
    return text_token_tags


def build_t2va_packed_sequence(
    text_token_tags: int | torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS,
    audio_tag: int = MINIMAX_H3_AUDIO_TAG,
    video_tag: int = MINIMAX_H3_VIDEO_TAG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    text_token_tags = _coerce_text_token_tags(text_token_tags)
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = int(text_token_tags.shape[0])
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_audio_rows + num_video_rows

    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    text_indices = torch.arange(num_text_tokens, dtype=torch.long)
    audio_indices = torch.arange(audio_start, video_start, dtype=torch.long)
    video_indices = torch.arange(video_start, sequence_length, dtype=torch.long)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return position_ids, token_tags, video_indices, audio_indices, text_indices, 0, 0


def build_fl2va_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    keyframe_anchors: tuple[str, ...],
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS,
    audio_tag: int = MINIMAX_H3_AUDIO_TAG,
    video_tag: int = MINIMAX_H3_VIDEO_TAG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the official ``[text | keyframes | target audio | target video]`` layout."""
    text_token_tags = _coerce_text_token_tags(text_token_tags)
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = int(text_token_tags.shape[0])
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset, multiplier in enumerate(_ROPE_FRAMES_PER_LATENT):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= multiplier
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        float(num_text_tokens),
        width_grid,
        audio_channels,
    )
    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_rows,
        0,
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    *,
    reference_video_shape: tuple[int, int, int],
    num_reference_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS,
    audio_tag: int = MINIMAX_H3_AUDIO_TAG,
    video_tag: int = MINIMAX_H3_VIDEO_TAG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the official single-video ``ref2va`` packed layout."""
    text_token_tags = _coerce_text_token_tags(text_token_tags)
    _, patch_h, patch_w = patch_size
    reference_frames, reference_height, reference_width = reference_video_shape
    num_text_tokens = int(text_token_tags.shape[0])
    num_reference_video_rows = reference_frames * (reference_height // patch_h) * (reference_width // patch_w)
    num_reference_audio_rows = num_reference_audio_latents * audio_channels
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * audio_channels
    sequence_length = (
        num_text_tokens
        + num_reference_audio_rows
        + num_reference_video_rows
        + num_target_audio_rows
        + num_target_video_rows
    )
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    reference_frame_grid, reference_width_grid = _frame_position_grid(
        reference_height, reference_width, patch_h, patch_w
    )

    reference_audio_start = num_text_tokens
    reference_video_start = reference_audio_start + num_reference_audio_rows
    target_audio_start = reference_video_start + num_reference_video_rows
    target_video_start = target_audio_start + num_target_audio_rows
    rotary_time = float(num_text_tokens)
    _fill_audio_positions(
        position_ids,
        slice(reference_audio_start, reference_video_start),
        num_reference_audio_latents,
        rotary_time,
        reference_width_grid,
        audio_channels,
    )
    reference_frame_time = _temporal_position_grid(reference_frames, rotary_time)
    position_ids[reference_video_start:target_audio_start, 0] = reference_frame_time.repeat_interleave(
        reference_frame_grid.shape[0]
    )
    position_ids[reference_video_start:target_audio_start, 1:] = reference_frame_grid.repeat(reference_frames, 1)
    reference_video_span = sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(reference_frames)
    )
    rotary_time += max(float(num_reference_audio_latents), reference_video_span)

    _fill_audio_positions(
        position_ids,
        slice(target_audio_start, target_video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
        audio_channels,
    )
    target_frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[target_video_start:, 0] = target_frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[target_video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(
        [
            torch.arange(reference_video_start, target_audio_start),
            torch.arange(target_video_start, sequence_length),
        ]
    )
    audio_indices = torch.cat(
        [
            torch.arange(reference_audio_start, reference_video_start),
            torch.arange(target_audio_start, target_video_start),
        ]
    )
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_reference_video_rows,
        num_reference_audio_rows,
    )


def build_t2va_packed_layout(
    text_token_tags: int | torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    spatial_compression_ratio: int = MINIMAX_H3_SPATIAL_COMPRESSION_RATIO,
    frames_per_chunk: int = MINIMAX_H3_FRAMES_PER_CHUNK,
    latents_per_chunk: int = MINIMAX_H3_LATENTS_PER_CHUNK,
    fps: int = MINIMAX_H3_FPS,
    audio_latents_per_second: int = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    audio_channels: int = MINIMAX_H3_AUDIO_CHANNELS,
    batch_size: int = 1,
) -> MiniMaxH3PackedLayout:
    if height % spatial_compression_ratio or width % spatial_compression_ratio:
        raise ValueError(
            f"`height` and `width` must be divisible by {spatial_compression_ratio}, got {height}x{width}."
        )
    aligned_num_frames = align_num_frames(num_frames, frames_per_chunk, latents_per_chunk)
    num_latent_frames = video_latent_num_frames(aligned_num_frames, frames_per_chunk, latents_per_chunk)
    latent_height = height // spatial_compression_ratio
    latent_width = width // spatial_compression_ratio
    num_audio_latents = audio_latent_num_frames(aligned_num_frames, fps, audio_latents_per_second)
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = build_t2va_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
    )
    return MiniMaxH3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
        height=height,
        width=width,
        num_frames=aligned_num_frames,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        patch_size=tuple(int(value) for value in patch_size),
        audio_channels=audio_channels,
        batch_size=int(batch_size),
    )


def build_row_timesteps(
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)
