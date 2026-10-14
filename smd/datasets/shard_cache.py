# SPDX-License-Identifier: Apache-2.0
"""Optional node-local staging for seek-heavy readers on remote mounts."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from smd.datasets.streaming_runtime import get_data_worker_context


@contextmanager
def local_shard(path: str, cache_dir: str | None, *, worker_context=None):
    """Keep one complete shard per rank/worker slot, preserving the source order.

    Cache storage must be node-local. Each slot owns at most one shard; the old entry is removed before
    staging its replacement. Locks serialize preflight/training readers
    using the same slot; killed copies are never accepted as complete files.
    """
    if not cache_dir:
        yield path
        return
    source = Path(path)
    stat = source.stat()
    context = worker_context or get_data_worker_context()
    namespace = hashlib.sha256(str(source.parent.resolve()).encode()).hexdigest()[:16]
    slot = Path(cache_dir) / namespace / f"rank{context.rank}_worker{context.worker_id}"
    slot.mkdir(parents=True, exist_ok=True)
    version = hashlib.sha256(f"{source.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    target = slot / f"{version}.tar"
    with (slot / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists() or target.stat().st_size != stat.st_size:
            for old in slot.glob("*.tar"):
                old.unlink()
            for old in slot.glob("*.part"):
                old.unlink()
            if shutil.disk_usage(slot).free < stat.st_size + 1024**3:
                raise OSError(f"Insufficient local space for shard {source} ({stat.st_size} bytes)")
            descriptor, temporary = tempfile.mkstemp(suffix=".part", dir=slot)
            try:
                with os.fdopen(descriptor, "wb") as out, source.open("rb") as inp:
                    shutil.copyfileobj(inp, out, length=16 * 1024**2)
                after = source.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise OSError(f"Source shard changed while copying: {source}")
                if Path(temporary).stat().st_size != stat.st_size:
                    raise OSError(f"Incomplete local shard copy: {source}")
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        yield str(target)


class ShardPrefetcher:
    """Stage the next ordered shard while reading the current one.

    Two alternating slots bound disk usage and keep the current file locked
    until its reader exits. Capture worker identity before starting a thread.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.context = get_data_worker_context()
        self.executor = ThreadPoolExecutor(max_workers=1) if cache_dir else None
        self.pending = None
        self.slot = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def _directory(self, slot):
        return self.cache_dir if slot == 0 else str(Path(self.cache_dir) / "lookahead")

    def _copy(self, path, slot):
        with local_shard(path, self._directory(slot), worker_context=self.context):
            pass

    @contextmanager
    def open(self, path, next_path):
        if not self.cache_dir:
            yield path
            return
        if self.pending is not None:
            self.pending.result()
        with local_shard(path, self._directory(self.slot), worker_context=self.context) as staged:
            self.pending = self.executor.submit(self._copy, next_path, 1 - self.slot)
            yield staged
        self.slot = 1 - self.slot
