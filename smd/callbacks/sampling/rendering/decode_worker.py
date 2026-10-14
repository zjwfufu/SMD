# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot isolated worker that renders every preview requested by one hook."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from smd.callbacks.sampling.rendering import media as media_utils
from smd.callbacks.sampling.rendering.decoder import DecoderUnavailable, build_decoder
from smd.callbacks.sampling.rendering.protocol import load_payload, save_manifest


def _resolve_devices() -> list[torch.device]:
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(index)
        except Exception as exc:  # noqa: BLE001 - an unusable device is skipped
            print(f"[decode_worker] skipping cuda:{index}: {exc}", flush=True)
            continue
        devices.append(torch.device("cuda", index))
        print(
            f"[decode_worker] available cuda:{index}, free={free / 2**30:.1f}GiB / total={total / 2**30:.1f}GiB",
            flush=True,
        )
    if not devices:
        raise RuntimeError("No usable CUDA device is visible to the preview worker")
    return devices


def _build_decoder(spec: dict[str, Any] | None, device: torch.device):
    if not spec:
        return None
    decoder = build_decoder(spec)
    mover = getattr(decoder, "to", None)
    if callable(mover):
        mover(device=device)
    return decoder


def _release_decoder(decoder: Any) -> None:
    if decoder is None:
        return
    del decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DecoderPool:
    """One decoder per participating GPU, with rank-ordered parallel decode."""

    def __init__(self, spec: dict[str, Any], devices: list[torch.device]):
        self.spec = spec
        self.devices = devices
        self.decoders: dict[torch.device, Any] = {}
        self.next_device = 0

    def decode(self, method: str, latents: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.decode_samples(method, list(latents.split(1, dim=0))), dim=0)

    def decode_samples(self, method: str, samples: list[torch.Tensor]) -> list[torch.Tensor]:
        assignments = [self.devices[(self.next_device + index) % len(self.devices)] for index in range(len(samples))]
        self.next_device = (self.next_device + len(samples)) % len(self.devices)
        for device in dict.fromkeys(assignments):
            self._decoder(device)

        outputs: list[torch.Tensor | None] = [None] * len(samples)
        indices_by_device: dict[torch.device, list[int]] = {}
        for index, device in enumerate(assignments):
            indices_by_device.setdefault(device, []).append(index)
        cancelled = threading.Event()

        def decode_device(device: torch.device, indices: list[int]) -> None:
            for index in indices:
                if cancelled.is_set():
                    return
                try:
                    outputs[index] = _decode(self.decoders[device], method, samples[index], device)
                except BaseException:
                    cancelled.set()
                    raise

        if len(indices_by_device) == 1:
            device, indices = next(iter(indices_by_device.items()))
            decode_device(device, indices)
        else:
            with ThreadPoolExecutor(
                max_workers=len(indices_by_device),
                thread_name_prefix="preview-decode",
            ) as executor:
                futures = [
                    executor.submit(decode_device, device, indices) for device, indices in indices_by_device.items()
                ]
                for future in futures:
                    future.result()
        return [output for output in outputs if output is not None]

    def decode_joint(self, method: str, latents: dict[str, torch.Tensor]) -> dict[str, Any]:
        batch_sizes = {value.shape[0] for value in latents.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"Joint latent batch dimensions disagree: {sorted(batch_sizes)}")
        batch = batch_sizes.pop()
        samples = [{key: value[index : index + 1] for key, value in latents.items()} for index in range(batch)]
        # Joint AV items are commonly single-sample requests processed serially.
        # Reuse loaded decoders; rotating these requests builds a fresh VAE per GPU
        # without adding concurrency. Multi-sample requests still use several GPUs.
        assignments = [self.devices[index % len(self.devices)] for index in range(batch)]
        for device in dict.fromkeys(assignments):
            self._decoder(device)

        outputs: list[dict[str, Any] | None] = [None] * batch
        indices_by_device: dict[torch.device, list[int]] = {}
        for index, device in enumerate(assignments):
            indices_by_device.setdefault(device, []).append(index)

        def decode_device(device: torch.device, indices: list[int]) -> None:
            decoder = self.decoders[device]
            decode_fn = getattr(decoder, method, None)
            if not callable(decode_fn):
                raise DecoderUnavailable(f"Decoder {type(decoder).__name__} has no callable {method!r}")
            dtype = _decoder_dtype(decoder)
            for index in indices:
                inputs = {key: value.to(device=device, dtype=dtype) for key, value in samples[index].items()}
                with torch.no_grad():
                    decoded = decode_fn(inputs)
                if not isinstance(decoded, dict):
                    raise TypeError(f"{type(decoder).__name__}.{method} must return a dictionary")
                outputs[index] = {
                    key: value.detach().float().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in decoded.items()
                }

        with ThreadPoolExecutor(
            max_workers=len(indices_by_device),
            thread_name_prefix="preview-joint-decode",
        ) as executor:
            futures = [executor.submit(decode_device, device, indices) for device, indices in indices_by_device.items()]
            for future in futures:
                future.result()

        resolved = [output for output in outputs if output is not None]
        keys = resolved[0]
        return {
            key: torch.cat([output[key] for output in resolved], dim=0)
            if isinstance(keys[key], torch.Tensor)
            else keys[key]
            for key in keys
        }

    def render_data(self, method: str, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reuse one decoder for serial structured rendering requests."""
        # These requests are processed serially. Rotating GPUs only loads another
        # copy of the VAE for each item without adding rendering concurrency.
        device = self.devices[0]
        decoder = self._decoder(device)
        render_fn = getattr(decoder, method, None)
        if not callable(render_fn):
            raise DecoderUnavailable(f"Decoder {type(decoder).__name__} has no callable {method!r}")
        dtype = _decoder_dtype(decoder)
        inputs = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in data.items()
        }
        with torch.no_grad():
            outputs = render_fn(inputs)
        if not isinstance(outputs, dict):
            raise TypeError(f"{type(decoder).__name__}.{method} must return a dictionary")
        return {key: value.detach().float().cpu() for key, value in outputs.items() if isinstance(value, torch.Tensor)}

    def close(self) -> None:
        for decoder in self.decoders.values():
            _release_decoder(decoder)
        self.decoders.clear()

    def _decoder(self, device: torch.device):
        if device not in self.decoders:
            self.decoders[device] = _build_decoder(self.spec, device)
        return self.decoders[device]


def _decoder_dtype(decoder: Any) -> torch.dtype:
    module = getattr(decoder, "vae", decoder)
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).dtype
        except StopIteration:
            pass
    return torch.float32


def _decode(decoder: Any, method: str, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    decode_fn = getattr(decoder, method, None)
    if not callable(decode_fn):
        raise DecoderUnavailable(f"Decoder {type(decoder).__name__} has no callable {method!r}")
    with torch.no_grad():
        return decode_fn(latents.to(device=device, dtype=_decoder_dtype(decoder))).detach().float().cpu()


def _tile_to(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch or tensor.shape[0] != 1:
        return tensor
    return tensor.repeat(batch, *([1] * (tensor.ndim - 1)))


def _resolve_pixels(item: dict[str, Any], pool: DecoderPool | None, method: str) -> torch.Tensor | None:
    panels = item.get("panels")
    if panels:
        rendered = [
            panel["pixels"].float() if "pixels" in panel else pool.decode(method, panel["latents"]) for panel in panels
        ]
        batch = max(part.shape[0] for part in rendered)
        return torch.cat([_tile_to(part, batch) for part in rendered], dim=-1)

    pixels = item.get("pixels")
    if pixels is not None:
        return pixels
    pixels_list = item.get("pixels_list")
    if pixels_list:
        return media_utils.stack_variable_samples(pixels_list)
    latents_list = item.get("latents_list")
    if latents_list:
        decoded = pool.decode_samples(method, latents_list)
        return media_utils.stack_variable_samples(decoded)
    latents = item.get("latents")
    if latents is None:
        return None
    return pool.decode(method, latents)


def _needs_decoder(item: dict[str, Any]) -> bool:
    panels = item.get("panels")
    if panels:
        return any("pixels" not in panel for panel in panels)
    if item.get("pixels") is not None or item.get("pixels_list"):
        return False
    return (
        item.get("latents") is not None
        or bool(item.get("latents_list"))
        or bool(item.get("joint_latents"))
        or bool(item.get("render_data"))
    )


def render_payload(
    payload: dict[str, Any],
    *,
    device: torch.device | None = None,
    devices: list[torch.device] | None = None,
    used_devices: set[str] | None = None,
    progress_callback: Callable[[list[dict[str, Any]], set[str]], None] | None = None,
) -> list[dict[str, Any]]:
    if device is not None and devices is not None:
        raise ValueError("Pass either device or devices, not both")
    devices = devices or ([device] if device is not None else _resolve_devices())
    entries: list[dict[str, Any]] = []
    pool: DecoderPool | None = None
    active_spec: dict[str, Any] | None = None

    try:
        for group in payload["groups"]:
            spec = group.get("decoder")
            if spec != active_spec:
                if pool is not None:
                    if used_devices is not None:
                        used_devices.update(str(device) for device in pool.decoders)
                    pool.close()
                pool = None
                active_spec = spec
                if spec:
                    pool = DecoderPool(spec, devices)

            decode_method = (spec or {}).get("method", "decode")
            media_dir = Path(group["media_dir"])
            media_dir.mkdir(parents=True, exist_ok=True)
            for item in group["items"]:
                try:
                    if pool is None and _needs_decoder(item):
                        raise DecoderUnavailable("no usable decoder recipe")
                    joint_latents = item.get("joint_latents")
                    if joint_latents:
                        if pool is None:
                            raise DecoderUnavailable("no usable decoder recipe")
                        decoded = pool.decode_joint(decode_method, joint_latents)
                        path = media_utils.write_av_media(
                            decoded["video"],
                            decoded["waveform"],
                            media_dir / item["stem"],
                            fps=int(item.get("fps", 24)),
                            sample_rate=int(decoded["sample_rate"]),
                            normalized=bool(item.get("normalized", False)),
                            columns=item.get("columns"),
                        )
                        entries.append(
                            {"key": item["key"], "path": str(path), "caption": item.get("caption"), "kind": "video"}
                        )
                        if progress_callback is not None:
                            progress_callback(entries, {str(device) for device in pool.decoders})
                        continue
                    render_data = item.get("render_data")
                    if render_data:
                        if pool is None:
                            raise DecoderUnavailable("no usable renderer recipe")
                        rendered = pool.render_data(decode_method, render_data)
                        for name, pixels in rendered.items():
                            if pixels.ndim == 3:
                                pixels = pixels.unsqueeze(0)
                            output_key = (item.get("render_keys") or {}).get(name, f"{item['key']}/{name}")
                            output_stem = f"{item['stem']}_{name}"
                            path, kind = media_utils.write_media(
                                pixels,
                                media_dir / output_stem,
                                fps=int(item.get("fps", 16)),
                                normalized=bool(item.get("normalized", True)),
                                columns=item.get("columns"),
                            )
                            entries.append(
                                {
                                    "key": output_key,
                                    "path": str(path),
                                    "caption": item.get("caption"),
                                    "kind": kind,
                                }
                            )
                            if progress_callback is not None:
                                progress_callback(entries, {str(device) for device in pool.decoders})
                        continue
                    pixels = _resolve_pixels(item, pool, decode_method)
                    if pixels is None:
                        continue
                    path, kind = media_utils.write_media(
                        pixels,
                        media_dir / item["stem"],
                        fps=int(item.get("fps", 16)),
                        normalized=bool(item.get("normalized", False)),
                        columns=item.get("columns"),
                    )
                    entries.append(
                        {"key": item["key"], "path": str(path), "caption": item.get("caption"), "kind": kind}
                    )
                    if progress_callback is not None:
                        progress_callback(
                            entries,
                            {str(device) for device in pool.decoders} if pool is not None else set(),
                        )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[decode_worker] item {item.get('key')!r} exhausted CUDA memory; aborting the payload",
                        flush=True,
                    )
                    raise
                except Exception as exc:  # noqa: BLE001 - one preview never owns the worker
                    print(f"[decode_worker] item {item.get('key')!r} failed: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                finally:
                    for current_device in devices:
                        if current_device.type == "cuda":
                            with torch.cuda.device(current_device):
                                torch.cuda.empty_cache()
    finally:
        if pool is not None:
            if used_devices is not None:
                used_devices.update(str(device) for device in pool.decoders)
            pool.close()
    return entries


def write_manifest(payload: dict[str, Any], entries: list[dict[str, Any]], *, device: str) -> Path:
    return save_manifest(
        Path(payload["manifest_path"]),
        {
            "iteration": int(payload.get("iteration", 0)),
            "device": device,
            "entries": entries,
        },
    )


def run_payload(payload_path: str | Path) -> int:
    payload = load_payload(Path(payload_path))
    devices = _resolve_devices()
    used_devices: set[str] = set()
    entries = render_payload(
        payload,
        devices=devices,
        used_devices=used_devices,
        progress_callback=lambda current_entries, current_devices: write_manifest(
            payload,
            current_entries,
            device=",".join(sorted(used_devices | current_devices)),
        ),
    )
    write_manifest(payload, entries, device=",".join(sorted(used_devices)))
    print(f"[decode_worker] wrote {len(entries)} media file(s)", flush=True)
    return 0 if entries else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="SMD isolated preview render worker")
    parser.add_argument("--payload", required=True, help="Path to a *.payload.pt file")
    args = parser.parse_args()
    sys.exit(run_payload(args.payload))


if __name__ == "__main__":
    main()
