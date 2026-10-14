# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-wide isolated synchronous preview rendering."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable

import torch
import wandb

from smd.callbacks.sampling.rendering.decoder import DecoderSpec
from smd.callbacks.sampling.rendering.protocol import (
    MANIFEST_SUFFIX,
    PAYLOAD_SUFFIX,
    VisualizationGroup,
    VisualizationItem,
    VisualizationPayload,
    load_manifest,
    save_payload,
)
from smd.callbacks.wandb_utils import log_at_iteration
from smd.utils import logging_utils as logger
from smd.utils.distributed import is_rank0, synchronize
from smd.utils.distributed.ddp import distributed_timeout_seconds

WORKER_MODULE = "smd.callbacks.sampling.rendering.decode_worker"
MAX_PREVIEW_TIMEOUT_S = 540.0
PREVIEW_TIMEOUT_FRACTION = 0.9
MAX_FAILURE_LOGS = 20


def preview_timeout_seconds(
    distributed_timeout_s: float | None = None,
    *,
    max_timeout_s: float = MAX_PREVIEW_TIMEOUT_S,
) -> float:
    """Derive a hard preview deadline strictly below the process-group timeout."""
    distributed_timeout_s = float(
        distributed_timeout_seconds() if distributed_timeout_s is None else distributed_timeout_s
    )
    if distributed_timeout_s <= 0:
        raise ValueError(f"Distributed timeout must be positive, got {distributed_timeout_s}")
    if max_timeout_s <= 0:
        raise ValueError(f"Maximum preview timeout must be positive, got {max_timeout_s}")
    timeout_s = min(float(max_timeout_s), distributed_timeout_s * PREVIEW_TIMEOUT_FRACTION)
    if not 0 < timeout_s < distributed_timeout_s:
        raise ValueError(
            f"Preview timeout must be strictly below distributed timeout, got "
            f"preview={timeout_s:g}s distributed={distributed_timeout_s:g}s"
        )
    return timeout_s


class PreviewService:
    """Aggregate every callback's preview request and render once per hook."""

    def __init__(
        self,
        *,
        save_path: str | os.PathLike[str] | None,
        distributed_timeout_s: float | None = None,
        worker_module: str = WORKER_MODULE,
    ) -> None:
        root = Path(save_path) if save_path else Path(os.environ.get("SMD_OUTPUT_ROOT", "SMD_OUTPUT")) / "debug"
        self.visualization_root = root / "visualizations"
        self.work_dir = Path(tempfile.gettempdir()) / "smd-preview" / str(os.getpid())
        self.payload_dir = self.work_dir / "payloads"
        self.manifest_dir = self.work_dir / "manifests"
        self.log_dir = self.work_dir / "logs"
        self.distributed_timeout_s = float(
            distributed_timeout_seconds() if distributed_timeout_s is None else distributed_timeout_s
        )
        self.default_timeout_s = preview_timeout_seconds(self.distributed_timeout_s)
        self.timeout_s = self.default_timeout_s
        self.recovery_timeout_s = min(60.0, max(1.0, (self.distributed_timeout_s - self.timeout_s) * 0.5))
        self.worker_module = worker_module
        self._active = False
        self._iteration = 0
        self._groups: list[VisualizationGroup] = []
        self._publisher_thread: threading.Thread | None = None

    def begin_event(self, *, iteration: int) -> None:
        if self._active or self._groups:
            logger.warning("[PreviewService] discarding unfinished requests from the previous callback event")
            self._reset_timeout()
        self._active = False
        self._iteration = int(iteration)
        self._groups.clear()

    def enqueue(
        self,
        *,
        label: str,
        items: Iterable[VisualizationItem],
        iteration: int,
        decoder: DecoderSpec | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Mark this event active on every rank; retain payload data only on rank 0."""
        if int(iteration) != self._iteration:
            raise RuntimeError(
                f"Preview request iteration {iteration} does not match active event iteration {self._iteration}"
            )
        self._active = True
        if timeout_s is not None:
            self.timeout_s = max(
                self.timeout_s,
                preview_timeout_seconds(self.distributed_timeout_s, max_timeout_s=timeout_s),
            )
            self.recovery_timeout_s = min(
                60.0,
                max(1.0, (self.distributed_timeout_s - self.timeout_s) * 0.5),
            )
        if not is_rank0():
            return

        items = [item for item in items if item.has_content()]
        if not items:
            return
        media_dir = self.visualization_root / label / "media" / f"iter_{iteration:08d}"
        decoder_dict = decoder.to_dict() if decoder is not None else None
        for group in self._groups:
            if group.label == label and group.decoder == decoder_dict:
                group.items.extend(items)
                return
        self._groups.append(
            VisualizationGroup(
                label=label,
                media_dir=str(media_dir),
                items=items,
                decoder=decoder_dict,
            )
        )

    def finish_event(self, *, iteration: int) -> int:
        """Render pending requests on rank 0, then release every rank together."""
        if not self._active:
            return 0
        published = 0
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            # Every rank must release its unused allocator blocks before the
            # worker uses their GPUs. It runs only while training is quiescent.
            synchronize()
            if is_rank0() and self._groups:
                published = self._render(iteration=int(iteration))
        except Exception as exc:  # noqa: BLE001 - preview failures never own training
            logger.warning(f"[PreviewService] iter={iteration}: preview failed: {type(exc).__name__}: {exc}")
        finally:
            self._active = False
            self._groups.clear()
            synchronize()
            self._reset_timeout()
        return published

    def abort_event(self) -> None:
        """Discard requests when callback sampling itself raises."""
        self._active = False
        self._groups.clear()
        self._reset_timeout()

    def _reset_timeout(self) -> None:
        self.timeout_s = self.default_timeout_s
        self.recovery_timeout_s = min(60.0, max(1.0, (self.distributed_timeout_s - self.timeout_s) * 0.5))

    def _render(self, *, iteration: int) -> int:
        started = time.monotonic()
        deadline = started + self.timeout_s
        self._ensure_dirs()
        timestamp = int(time.time() * 1000)
        stamp = f"iter_{iteration:08d}_{timestamp}"
        payload_path = self.payload_dir / f"{stamp}{PAYLOAD_SUFFIX}"
        manifest_path = self.manifest_dir / f"{stamp}{MANIFEST_SUFFIX}"
        log_path = self.log_dir / f"{stamp}.log"
        payload = VisualizationPayload(
            manifest_path=str(manifest_path),
            iteration=iteration,
            groups=self._ordered_groups(),
        )
        process: subprocess.Popen | None = None
        try:
            save_payload(payload_path, payload)
            if time.monotonic() >= deadline:
                logger.warning(f"[PreviewService] iter={iteration}: payload serialization exhausted the preview budget")
                return 0

            command = [sys.executable, "-m", self.worker_module, "--payload", str(payload_path)]
            env = os.environ.copy()
            env.setdefault("WANDB_SILENT", "true")
            with log_path.open("ab") as log_file:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            rc, timed_out = self._wait(process, deadline=deadline)
            if timed_out:
                logger.warning(
                    f"[PreviewService] iter={iteration}: worker exceeded {self.timeout_s:g}s and was killed; "
                    f"training will continue (log: {log_path})"
                )
                self._prune_failure_logs()
                return self._publish_partial_manifest(manifest_path, iteration=iteration)
            if rc != 0:
                logger.warning(
                    f"[PreviewService] iter={iteration}: worker failed rc={rc}; "
                    f"training will continue (log: {log_path})"
                )
                self._prune_failure_logs()
                return self._publish_partial_manifest(manifest_path, iteration=iteration)

            log_path.unlink(missing_ok=True)
            if not manifest_path.exists():
                logger.warning(f"[PreviewService] iter={iteration}: worker succeeded without a manifest")
                return 0
            manifest = load_manifest(manifest_path)
            logger.info(
                f"[PreviewService] iter={iteration}: worker decode devices={manifest.get('device') or 'cpu-only'}"
            )
            return self._publish_bounded(manifest, iteration=iteration, deadline=deadline)
        finally:
            if process is not None and process.poll() is None:
                self._terminate(process, deadline=deadline)
            diagnostics = self.visualization_root / "events"
            diagnostics.mkdir(parents=True, exist_ok=True)
            for artifact in (manifest_path, log_path):
                if artifact.exists():
                    shutil.copy2(artifact, diagnostics / artifact.name)
            payload_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

    def _publish_partial_manifest(self, manifest_path: Path, *, iteration: int) -> int:
        """Publish items checkpointed before a worker timeout or failure."""
        if not manifest_path.exists():
            return 0
        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001 - partial recovery is best effort
            logger.warning(f"[PreviewService] iter={iteration}: cannot read partial manifest: {exc}")
            return 0
        entries = manifest.get("entries") or []
        if not entries:
            return 0
        logger.warning(
            f"[PreviewService] iter={iteration}: recovering {len(entries)} completed preview(s) from partial render"
        )
        return self._publish_bounded(
            manifest,
            iteration=iteration,
            deadline=time.monotonic() + self.recovery_timeout_s,
        )

    def _ordered_groups(self) -> list[VisualizationGroup]:
        """Place equal decoder recipes together so the worker constructs each once."""

        def key(group: VisualizationGroup) -> tuple[str, str]:
            decoder_key = json.dumps(group.decoder, sort_keys=True) if group.decoder is not None else ""
            return decoder_key, group.label

        return sorted(self._groups, key=key)

    def _wait(self, process: subprocess.Popen, *, deadline: float) -> tuple[int, bool]:
        kill_grace_s = min(5.0, max(0.1, self.timeout_s * 0.1))
        normal_wait_s = max(0.01, deadline - time.monotonic() - kill_grace_s)
        try:
            return process.wait(timeout=normal_wait_s), False
        except subprocess.TimeoutExpired:
            self._terminate(process, deadline=deadline)
            return process.poll() if process.poll() is not None else -signal.SIGKILL, True

    @staticmethod
    def _terminate(process: subprocess.Popen, *, deadline: float) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=min(2.0, remaining))
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning(f"[PreviewService] worker pid={process.pid} did not exit before the hard deadline")

    def _publish(self, manifest: dict, *, iteration: int) -> int:
        entries = manifest.get("entries") or []
        if not entries or not wandb.run:
            return 0
        source_iter = int(manifest.get("iteration", iteration))
        media: dict[str, object] = {}
        for entry in entries:
            path = Path(entry["path"])
            if not path.exists():
                logger.warning(f"[PreviewService] manifest references missing file {path}")
                continue
            caption = entry.get("caption") or ""
            caption = f"iter {source_iter} | {caption}" if caption else f"iter {source_iter}"
            media[entry["key"]] = (
                wandb.Video(str(path), format="mp4", caption=caption)
                if entry.get("kind") == "video"
                else wandb.Image(str(path), caption=caption)
            )
        if not media:
            return 0
        log_at_iteration(media, iteration=source_iter)
        logger.info(f"[PreviewService] published {len(media)} preview(s) from iter={source_iter}")
        return len(media)

    def _publish_bounded(self, manifest: dict, *, iteration: int, deadline: float) -> int:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(f"[PreviewService] iter={iteration}: no budget remains for W&B publication")
            return 0
        if self._publisher_thread is not None and self._publisher_thread.is_alive():
            logger.warning("[PreviewService] previous W&B publication is still blocked; keeping media on disk")
            return 0

        result: list[int] = []
        failure: list[BaseException] = []

        def publish() -> None:
            try:
                result.append(self._publish(manifest, iteration=iteration))
            except BaseException as exc:  # noqa: BLE001 - surfaced in the trainer thread
                failure.append(exc)

        thread = threading.Thread(target=publish, name="smd-preview-wandb", daemon=True)
        self._publisher_thread = thread
        thread.start()
        thread.join(timeout=max(0.0, remaining))
        if thread.is_alive():
            logger.warning(
                f"[PreviewService] iter={iteration}: W&B publication exceeded the preview deadline; "
                "media remain on disk"
            )
            return 0
        if failure:
            raise failure[0]
        return result[0] if result else 0

    def _ensure_dirs(self) -> None:
        for directory in (self.payload_dir, self.manifest_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _prune_failure_logs(self) -> None:
        logs = sorted(self.log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
        for stale in logs[: max(0, len(logs) - MAX_FAILURE_LOGS)]:
            stale.unlink(missing_ok=True)
