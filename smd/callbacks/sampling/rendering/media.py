# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor → media-file helpers shared by callbacks and decode workers.

Everything here is pure CPU work on plain tensors, so it runs identically
inside the training process (debug) or inside a detached decode worker.

Conventions (identical to :func:`smd.callbacks.wandb.to_wandb`):
  * images are ``[B, C, H, W]``
  * videos are ``[B, C, T, H, W]`` (channel before time)
  * pixel values are in ``[-1, 1]`` unless ``normalized=True``
"""

from __future__ import annotations

import math
import os
import tempfile
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Literal

import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import functional as tv_F

MediaKind = Literal["image", "video"]

IMAGE_SUFFIX = ".jpg"
VIDEO_SUFFIX = ".mp4"

GRID_PAD_VALUE = 255


def infer_media_kind(tensor: torch.Tensor) -> MediaKind:
    """Classify a decoded pixel tensor as an image or a video batch."""
    if tensor.ndim == 4:
        return "image"
    if tensor.ndim == 5:
        return "video"
    raise ValueError(f"Expected a 4D image or 5D video tensor, got shape {tuple(tensor.shape)}")


def grid_columns(num_samples: int) -> int:
    """Columns for a square-ish tile layout — 4 samples give the canonical 2x2."""
    if num_samples <= 1:
        return 1
    return int(math.ceil(math.sqrt(num_samples)))


def to_uint8(tensor: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
    """Map a float pixel tensor to ``uint8`` in ``[0, 255]``."""
    if tensor.dtype == torch.uint8:
        return tensor.detach().to(device="cpu")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    factor, offset = (255.0, 0.0) if normalized else (127.5, 127.5)
    return tensor.mul(factor).add(offset).clamp_(0, 255).to(torch.uint8)


def stack_variable_samples(samples: list[torch.Tensor], *, pad_value: float | None = None) -> torch.Tensor:
    """Pad variable-resolution ``[1,C,(T),H,W]`` samples into one batch."""
    if not samples:
        raise ValueError("Expected at least one sample")
    ndim = samples[0].ndim
    if ndim not in (4, 5):
        raise ValueError(f"Expected image/video samples, got ndim={ndim}")
    leading = samples[0].shape[1:-2]
    if any(sample.ndim != ndim or sample.shape[0] != 1 or sample.shape[1:-2] != leading for sample in samples):
        shapes = [tuple(sample.shape) for sample in samples]
        raise ValueError(f"Variable preview samples disagree outside H/W: {shapes}")

    max_h = max(sample.shape[-2] for sample in samples)
    max_w = max(sample.shape[-1] for sample in samples)
    if pad_value is None:
        pad_value = 255 if samples[0].dtype == torch.uint8 else 1.0
    padded = []
    for sample in samples:
        pad_h = max_h - sample.shape[-2]
        pad_w = max_w - sample.shape[-1]
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        padded.append(F.pad(sample, (left, right, top, bottom), value=pad_value))
    return torch.cat(padded, dim=0)


def _require_rgb(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[-3]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.expand(*tensor.shape[:-3], 3, *tensor.shape[-2:])
    raise ValueError(f"Expected 1 or 3 channels in [..., C, H, W] layout, got {channels}")


def _pad_to_even(frames: torch.Tensor) -> torch.Tensor:
    """Pad ``[T, H, W, C]`` so H and W are even (required by yuv420p/libx264)."""
    pad_h = frames.shape[1] % 2
    pad_w = frames.shape[2] % 2
    if not (pad_h or pad_w):
        return frames
    padded = torch.nn.functional.pad(frames.permute(0, 3, 1, 2).float(), (0, pad_w, 0, pad_h), mode="replicate")
    return padded.permute(0, 2, 3, 1).to(torch.uint8)


def image_grid(tensor: torch.Tensor, *, normalized: bool = False, columns: int | None = None):
    """Tile ``[B, C, H, W]`` into a single ``PIL.Image``."""
    tensor = _require_rgb(to_uint8(tensor, normalized=normalized))
    columns = grid_columns(tensor.shape[0]) if columns is None else columns
    grid = torchvision.utils.make_grid(tensor, nrow=columns, pad_value=GRID_PAD_VALUE)
    return tv_F.to_pil_image(grid).convert("RGB")


def video_grid(tensor: torch.Tensor, *, normalized: bool = False, columns: int | None = None) -> torch.Tensor:
    """Tile ``[B, C, T, H, W]`` into ``[T, H, W, C]`` uint8 frames."""
    tensor = to_uint8(tensor, normalized=normalized)
    columns = grid_columns(tensor.shape[0]) if columns is None else columns
    frames = _require_rgb(tensor.permute(2, 0, 1, 3, 4))  # [T, B, C, H, W]
    tiled = torch.stack(
        [torchvision.utils.make_grid(frame, nrow=columns, pad_value=GRID_PAD_VALUE) for frame in frames]
    )
    return _pad_to_even(tiled.permute(0, 2, 3, 1).contiguous())


def _write_image(tensor: torch.Tensor, path: Path, *, normalized: bool = False, columns: int | None = None) -> Path:
    path = Path(path).with_suffix(IMAGE_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_grid(tensor, normalized=normalized, columns=columns).save(path, quality=92)
    return path


def _write_video(
    tensor: torch.Tensor,
    path: Path,
    *,
    fps: int = 16,
    normalized: bool = False,
    columns: int | None = None,
) -> Path:
    path = Path(path).with_suffix(VIDEO_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    _encode_video(video_grid(tensor, normalized=normalized, columns=columns), path, fps=max(1, int(fps)))
    return path


def _encode_video(frames: torch.Tensor, path: Path, *, fps: int) -> None:
    """Encode ``[T, H, W, C]`` uint8 frames to H.264, preferring imageio/ffmpeg."""
    try:
        import imageio.v2 as imageio

        # imageio-ffmpeg already selects yuv420p for RGB input, which is what
        # browsers and W&B need; macro_block_size=1 keeps our exact grid size.
        imageio.mimwrite(path, list(frames.numpy()), fps=fps, codec="libx264", macro_block_size=1)
    except Exception:  # noqa: BLE001 - fall back to torchvision/PyAV
        torchvision.io.write_video(str(path), frames, fps=fps, video_codec="h264")


def write_media(
    tensor: torch.Tensor,
    path_stem: Path,
    *,
    fps: int = 16,
    normalized: bool = False,
    columns: int | None = None,
) -> tuple[Path, MediaKind]:
    """Write ``tensor`` as a tiled ``.jpg`` (image) or ``.mp4`` (video)."""
    kind = infer_media_kind(tensor)
    if kind == "image":
        return write_image(tensor, path_stem, normalized=normalized, columns=columns), kind
    return write_video(tensor, path_stem, fps=fps, normalized=normalized, columns=columns), kind


def _write_av_media(
    video: torch.Tensor,
    waveform: torch.Tensor,
    path_stem: Path,
    *,
    fps: int,
    sample_rate: int,
    normalized: bool = False,
    columns: int | None = None,
) -> Path:
    """Mux audio, averaging across samples while preserving audio channels."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Wheels provide an executable even when no system ffmpeg is on PATH.
        # Use the same resolver as imageio's video writer (including its override).
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    silent_path = _write_video(
        video,
        path_stem.with_name(f"{path_stem.name}_silent"),
        fps=fps,
        normalized=normalized,
        columns=columns,
    )
    wav_path = path_stem.with_suffix(".wav")
    output_path = path_stem.with_suffix(VIDEO_SUFFIX)
    audio = waveform.detach().float().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    if audio.ndim != 3 or audio.shape[1] not in (1, 2):
        raise ValueError(f"Expected waveform [B,T] or [B,C,T] with C=1 or 2, got {tuple(audio.shape)}")
    channels = audio.shape[1]
    pcm = (
        (audio.mean(dim=0).transpose(0, 1).clamp(-1, 1) * 32767.0)
        .round().to(torch.int16).contiguous().numpy().tobytes()
    )
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent_path),
                "-i",
                str(wav_path),
                "-c:v",
                "copy",
                "-af",
                "apad",
                "-c:a",
                "aac",
                # Bound apad explicitly. Some ffmpeg builds keep generating
                # silence indefinitely with stream-copy video and -shortest.
                "-t",
                str(video.shape[2] / max(1, int(fps))),
                str(output_path),
            ],
            check=True,
        )
    finally:
        silent_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
    return output_path


def _publish_media(writer, tensor, path: Path, *, suffix: str, **kwargs) -> Path:
    """Encode on local scratch; publish only a complete file to the output mount."""
    destination = Path(path).with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smd-media-") as directory:
        local = writer(tensor, Path(directory) / destination.stem, **kwargs)
        descriptor, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".part", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as out, local.open("rb") as inp:
                shutil.copyfileobj(inp, out, length=1024**2)
            if Path(temporary).stat().st_size != local.stat().st_size:
                raise OSError(f"Incomplete preview copy to {destination}")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination


def write_image(tensor: torch.Tensor, path: Path, *, normalized: bool = False, columns: int | None = None) -> Path:
    return _publish_media(_write_image, tensor, path, suffix=IMAGE_SUFFIX, normalized=normalized, columns=columns)


def write_video(tensor: torch.Tensor, path: Path, *, fps: int = 16,
                normalized: bool = False, columns: int | None = None) -> Path:
    return _publish_media(_write_video, tensor, path, suffix=VIDEO_SUFFIX,
                          fps=fps, normalized=normalized, columns=columns)


def write_av_media(video: torch.Tensor, waveform: torch.Tensor, path_stem: Path, *, fps: int,
                   sample_rate: int, normalized: bool = False, columns: int | None = None) -> Path:
    return _publish_media(lambda tensor, path, **kwargs: _write_av_media(tensor, waveform, path, **kwargs),
                          video, path_stem, suffix=VIDEO_SUFFIX, fps=fps, sample_rate=sample_rate,
                          normalized=normalized, columns=columns)
