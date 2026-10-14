# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire format shared by the trainer and the isolated synchronous render worker."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

PROTOCOL_VERSION = 3

PAYLOAD_SUFFIX = ".payload.pt"
MANIFEST_SUFFIX = ".manifest.json"


@dataclass
class VisualizationItem:
    """One image/video grid or multi-panel strip."""

    key: str
    stem: str
    caption: str | None = None
    latents: torch.Tensor | None = None
    pixels: torch.Tensor | None = None
    latents_list: list[torch.Tensor] | None = None
    pixels_list: list[torch.Tensor] | None = None
    panels: list[dict[str, torch.Tensor]] | None = None
    joint_latents: dict[str, torch.Tensor] | None = None
    render_data: dict[str, torch.Tensor] | None = None
    render_keys: dict[str, str] | None = None
    fps: int = 16
    columns: int | None = None
    normalized: bool = False

    def has_content(self) -> bool:
        return (
            self.latents is not None
            or self.pixels is not None
            or bool(self.latents_list)
            or bool(self.pixels_list)
            or bool(self.panels)
            or bool(self.joint_latents)
            or bool(self.render_data)
        )

    def to_dict(self) -> dict[str, Any]:
        panels = (
            [{key: tensor.detach().to(device="cpu") for key, tensor in panel.items()} for panel in self.panels]
            if self.panels
            else None
        )
        return {
            "key": self.key,
            "stem": self.stem,
            "caption": self.caption,
            "latents": self.latents.detach().to(device="cpu") if self.latents is not None else None,
            "pixels": self.pixels.detach().to(device="cpu") if self.pixels is not None else None,
            "latents_list": (
                [tensor.detach().to(device="cpu") for tensor in self.latents_list] if self.latents_list else None
            ),
            "pixels_list": (
                [tensor.detach().to(device="cpu") for tensor in self.pixels_list] if self.pixels_list else None
            ),
            "panels": panels,
            "joint_latents": (
                {key: tensor.detach().to(device="cpu") for key, tensor in self.joint_latents.items()}
                if self.joint_latents
                else None
            ),
            "render_data": (
                {key: tensor.detach().to(device="cpu") for key, tensor in self.render_data.items()}
                if self.render_data
                else None
            ),
            "render_keys": dict(self.render_keys) if self.render_keys else None,
            "fps": self.fps,
            "columns": self.columns,
            "normalized": self.normalized,
        }


@dataclass
class VisualizationGroup:
    """Items sharing an output directory and decoder recipe."""

    label: str
    media_dir: str
    items: list[VisualizationItem] = field(default_factory=list)
    decoder: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "media_dir": self.media_dir,
            "items": [item.to_dict() for item in self.items],
            "decoder": self.decoder,
        }


@dataclass
class VisualizationPayload:
    """Every preview request produced by one callback lifecycle event."""

    manifest_path: str
    iteration: int
    groups: list[VisualizationGroup] = field(default_factory=list)
    version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "iteration": int(self.iteration),
            "groups": [group.to_dict() for group in self.groups],
            "version": self.version,
        }


def atomic_write(path: Path, writer: Callable[[Path], Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def save_payload(path: Path, payload: VisualizationPayload) -> Path:
    return atomic_write(path, lambda tmp: torch.save(payload.to_dict(), tmp))


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported visualization payload version {version!r} (expected {PROTOCOL_VERSION})")
    return payload


def save_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    manifest = {**manifest, "version": PROTOCOL_VERSION}
    return atomic_write(path, lambda tmp: Path(tmp).write_text(json.dumps(manifest, indent=2), encoding="utf-8"))


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported manifest version {manifest.get('version')!r}")
    return manifest
