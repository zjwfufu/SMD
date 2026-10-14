# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local tar shard discovery and bounded data errors."""
import glob
import os
from pathlib import Path
from typing import List
import torch
import smd.utils.logging_utils as logger

class DataErrorLimitExceeded(RuntimeError):
    """Raised when one data worker exceeds its consecutive-error budget."""

class BoundedDataErrorHandler:
    """Skip isolated WDS errors, but fail after repeated errors without a valid sample."""

    def __init__(self, *, max_consecutive_errors: int, datatags: List[str]):
        if max_consecutive_errors < 1:
            raise ValueError(f"max_consecutive_data_errors must be >= 1, got {max_consecutive_errors}")
        self.max_consecutive_errors = max_consecutive_errors
        self.datatags = tuple(datatags)
        self.consecutive_errors = 0
        self.total_errors = 0

    @staticmethod
    def _worker_context() -> tuple[int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def record_failure(self, exn: Exception, *, context: str = "") -> bool:
        self.consecutive_errors += 1
        self.total_errors += 1
        worker_id, num_workers = self._worker_context()
        rank = int(os.environ.get("RANK", "0"))
        context_suffix = f" {context}" if context else ""
        message = (
            "[data-error] "
            f"rank={rank} worker={worker_id}/{num_workers} "
            f"consecutive={self.consecutive_errors}/{self.max_consecutive_errors} "
            f"total={self.total_errors} datatags={self.datatags!r}{context_suffix} error={exn!r}"
        )
        logger.error(message)
        if self.consecutive_errors >= self.max_consecutive_errors:
            raise DataErrorLimitExceeded(
                f"{message}; refusing to continue after repeated data failures in this worker"
            ) from exn
        return True

    def __call__(self, exn: Exception) -> bool:
        return self.record_failure(exn)

    def record_success(self) -> None:
        if self.consecutive_errors:
            worker_id, num_workers = self._worker_context()
            logger.info(
                "[data-recovered] "
                f"rank={int(os.environ.get('RANK', '0'))} worker={worker_id}/{num_workers} "
                f"after_consecutive={self.consecutive_errors} total_errors={self.total_errors} "
                f"datatags={self.datatags!r}"
            )
        self.consecutive_errors = 0

def get_shards(datatags):
    """Resolve local tar files, directories, or glob patterns; no cloud credentials."""
    shards = []
    for tag in datatags:
        path = str(tag).removeprefix('WDS:')
        if '://' in path or path.startswith('pipe:'):
            raise ValueError('Use a local shard directory, tar file or glob pattern')
        path = os.path.expanduser(path)
        candidates = sorted(Path(path).glob('*.tar')) if Path(path).is_dir() else [Path(p) for p in sorted(glob.glob(path))]
        candidates = [p.resolve() for p in candidates if p.is_file() and p.suffix == '.tar']
        if not candidates:
            raise FileNotFoundError(f'No tar shards match {tag!r}')
        shards.extend(map(str, candidates))
    return sorted(set(shards))
