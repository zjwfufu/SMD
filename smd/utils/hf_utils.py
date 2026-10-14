# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face cache resolution and distributed loading.

Runtime loaders search both direct cache roots and ``hub/`` layouts. Each rank
loads independently; model construction owns coarse rank-zero coordination.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from smd.utils.basic_utils import str2bool
from smd.utils.distributed import get_rank, is_rank0, synchronize
import smd.utils.logging_utils as logger

# ---------------------------------------------------------------------------
# Cache directory resolution
# ---------------------------------------------------------------------------

_DEFAULT_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _hub_variants(path: str) -> list[str]:
    """Return ``path`` plus its ``hub/`` sibling variant for redundant probing.

    - ``X``       → ``[X, X/hub]``       (covers callers that pass a bare root)
    - ``X/hub``   → ``[X/hub, X]``       (covers callers that already added hub)

    This makes cache lookup agnostic to whether a ``hub/`` sub-directory was
    used when the model was downloaded.
    """
    path = os.path.normpath(path)
    if os.path.basename(path) == "hub":
        return [path, os.path.dirname(path)]
    return [path, os.path.join(path, "hub")]


def _official_hub_cache() -> str:
    """Return the canonical Hugging Face hub directory for runtime downloads."""
    explicit_hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_hub:
        return os.path.normpath(explicit_hub)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_home = os.path.normpath(hf_home)
        if os.path.basename(hf_home) == "hub":
            return hf_home
        return os.path.join(hf_home, "hub")

    return _DEFAULT_HF_CACHE


def get_hf_search_dirs() -> list[str]:
    """Return ordered Hugging Face cache directories.

    The canonical hub directory is searched first.  Raw ``HF_HOME`` and
    project-local layouts are retained as read-only compatibility fallbacks for
    existing precache jobs and developer caches created before hub semantics
    were unified.
    """
    dirs: list[str] = []
    seen: set[str] = set()

    def _add(p: str | None) -> None:
        if not p:
            return
        norm = os.path.normpath(p)
        if norm not in seen:
            dirs.append(p)
            seen.add(norm)

    _add(_official_hub_cache())

    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        _add(os.environ.get(key))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        for variant in _hub_variants(hf_home):
            _add(variant)

    _add(_DEFAULT_HF_CACHE)
    _add(os.path.dirname(_DEFAULT_HF_CACHE))

    project_cache = os.path.join(os.environ.get("SMD_OUTPUT_ROOT", "SMD_OUTPUT"), ".cache")
    _add(os.path.join(project_cache, "hub"))
    _add(project_cache)
    return dirs


def get_hf_download_target() -> str:
    """Return the raw cache root for cache preparation utilities."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return hf_home
    return os.path.join(os.environ.get("SMD_OUTPUT_ROOT", "SMD_OUTPUT"), ".cache")


def get_hf_runtime_download_target() -> str:
    """Return the canonical hub directory for runtime model downloads."""
    return _official_hub_cache()


def get_hf_cache_dir(model_id: str | None = None) -> str:
    """Resolve the best HF cache directory for *model_id*.

    If *model_id* is given, returns the **first** search directory that already
    contains a cached snapshot of the model.  If no cached copy exists (or
    *model_id* is ``None``), returns :func:`get_hf_runtime_download_target`.
    """
    if model_id:
        # A caller may already have resolved a repo ID to a node-local snapshot
        # and pass that directory to ``from_pretrained``. Do not reinterpret
        # the absolute path as a Hub repo ID and scan legacy shared caches for
        # entries such as ``models----tmp--...``. Besides being unnecessary,
        # probing a stale BlobFuse mount can raise ENOTCONN and abort startup.
        candidate = Path(model_id).expanduser()
        try:
            if candidate.is_absolute() and candidate.is_dir():
                return get_hf_runtime_download_target()
        except OSError:
            pass
        for d in get_hf_search_dirs():
            if is_cached_locally(model_id, d):
                return d

    return get_hf_runtime_download_target()


# ---------------------------------------------------------------------------
# Local cache probing
# ---------------------------------------------------------------------------


def is_cached_locally(model_id: str, cache_dir: str) -> bool:
    """Return ``True`` if *model_id* has a non-empty snapshot inside *cache_dir*."""
    # Fast path: use huggingface_hub's own scanner when available.
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir(cache_dir)
        for repo_info in cache_info.repos:
            if repo_info.repo_id == model_id:
                return True
        return False
    except Exception:
        pass

    # Fallback: walk the on-disk layout manually.
    safe_name = model_id.replace("/", "--")
    snapshot_dir = Path(cache_dir) / f"models--{safe_name}" / "snapshots"
    try:
        if snapshot_dir.is_dir():
            for child in snapshot_dir.iterdir():
                if child.is_dir() and any(child.iterdir()):
                    return True
    except OSError:
        # Cache discovery is best-effort. A disconnected optional/shared mount
        # must not prevent falling back to the configured node-local cache.
        return False
    return False


# ---------------------------------------------------------------------------
# Distributed-aware model loading
# ---------------------------------------------------------------------------


def safe_load_from_pretrained(
    load_fn,
    model_id: str,
    *,
    cache_dir: str | None = None,
    subfolder: str,
    max_retries: int = 5,
    **extra_kwargs,
):
    """Load a HuggingFace component from the (pre-cached) local HF cache, with retry.

    (Formerly ``hf_load_with_rank0_download`` — renamed because the function does
    NOT special-case rank 0 and does NOT actively download; every rank loads
    independently and the only added behaviour is transient-failure *retry*. The
    rank-0-first serialization lives in the separate :func:`rank0_first_load`. A
    backward-compatible alias under the old name is kept below.)

    **No collectives.** Every rank loads independently, then retries on transient
    failures with jittered backoff. This is deadlock-proof (no barrier) and robust
    to the intermittent shard-race that surfaces under 8-rank concurrent
    ``from_pretrained`` on some clusters.

    Models may be cached before training. Even
    with a warm cache, ``from_pretrained(local_files_only=False)`` still does an
    online re-resolution (etag check + an internal ``snapshot_download`` that
    re-lists shards). When 8 ranks do this against the *same* cache dir
    concurrently, one rank can observe another's half-written shard file and raise::

        OSError: ... does not appear to have a file named
                 text_encoder/model-0000X-of-00005.safetensors

    This is a benign timing race — the cache *is* complete; the rank just caught a
    transient mid-write state. We retry with backoff + per-rank jitter, which both
    rides out the transient and de-synchronizes the ranks so the race window
    closes. ``OSError`` also covers ``LocalEntryNotFoundError`` (a subclass), so an
    offline run whose precache hasn't fully landed yet also gets retried.

    **Historical note (why no barrier here):** an earlier version did a
    *rank-0-first + barrier* dance inside this function. That barrier deadlocked
    because a network whose ``__init__`` issues a *different number* of these calls
    per rank desyncs the barrier count, and the next rank waits the full NCCL/gloo
    timeout (600 s) for a peer that never arrives. So per-call barriers are
    forbidden here. For coarse-grained serialization of a *whole* preprocessor
    block (where the call count IS symmetric across ranks), use
    :func:`rank0_first_load` instead — it barriers exactly once per rank.

    Two ``local_files_only`` modes, neither issues a collective:

    * **Offline** (``LOCAL_FILES_ONLY=true``): ``local_files_only=True``.
    * **Online** (default): ``local_files_only=False`` — HF serves from the warm
      cache and gracefully handles a subfolder lacking a ``config.json``.

    Args:
        max_retries: total attempts before giving up (default 5). The final
            failure is re-raised unchanged so callers see the real error.

    If *cache_dir* is ``None``, it is resolved via :func:`get_hf_cache_dir`.
    """
    if cache_dir is None:
        cache_dir = get_hf_cache_dir(model_id)

    local_only = str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
    kwargs = dict(cache_dir=cache_dir, subfolder=subfolder, **extra_kwargs)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return load_fn(model_id, local_files_only=local_only, **kwargs)
        except (OSError, ValueError) as e:
            # OSError covers the shard-race "does not appear to have a file" and
            # LocalEntryNotFoundError; ValueError covers some sharded-metadata
            # races. Both are transient under concurrent load — retry.
            last_err = e
            if attempt == max_retries - 1:
                break
            # Deterministic backoff (scripts have no RNG): grows with attempt and
            # is offset by a per-rank phase so the 8 ranks fan out in time and the
            # race window closes. No collectives — deadlock-proof.
            backoff = 1.5 * (attempt + 1)
            jitter = (get_rank() % 8) * 0.4 + attempt * 0.3
            wait = backoff + jitter
            logger.warning(
                f"hf_load {model_id}/{subfolder} attempt {attempt + 1}/{max_retries} failed "
                f"({type(e).__name__}: {e}); retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    assert last_err is not None
    logger.error(f"hf_load {model_id}/{subfolder} failed after {max_retries} attempts: {last_err}")
    raise last_err


# Backward-compatible alias for the historical name. New code should import
# ``safe_load_from_pretrained``; this keeps any out-of-tree / not-yet-migrated
# callers working.
hf_load_with_rank0_download = safe_load_from_pretrained


@contextlib.contextmanager
def rank0_first_load():
    """Serialize a *symmetric* model-loading block so rank 0 warms the cache first.

    Use this around a block that EVERY rank executes identically (e.g. a Wan
    network's ``init_preprocessors()``, which loads text_encoder + vae). Rank 0
    runs the block first and fully populates the cache; the other ranks wait, then
    run the same block reading from the now-warm cache — eliminating the 8-rank
    concurrent re-resolution that triggers the shard race at its source.

    **Deadlock-safety.** Each rank calls :func:`synchronize` (a barrier) EXACTLY
    ONCE, so the barrier count is symmetric and cannot desync:

    * rank 0      : ``yield`` (run body) → ``synchronize()``  → return
    * non-rank-0  : ``synchronize()``    → ``yield`` (run body) → return

    Both paths hit exactly one barrier. This is the safe counterpart to the
    per-call barrier that was (correctly) banned from
    :func:`hf_load_with_rank0_download`: there the per-rank call count was
    variable; here the block is executed once per rank by construction.

    If the body raises on rank 0, the ``finally`` still releases the barrier so the
    other ranks don't hang for the full NCCL timeout — they then run the body
    themselves (and, with L1 retry underneath, usually still succeed).
    """
    if is_rank0():
        try:
            yield
        finally:
            synchronize()
    else:
        synchronize()
        yield
