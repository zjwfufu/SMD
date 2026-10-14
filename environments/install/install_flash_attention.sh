#!/usr/bin/env bash
# Called after the pinned PyTorch and core Python dependencies are installed.
set -euo pipefail
SMD_ATTN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SMD_ATTN_PYTHON="${1:-python}"
SMD_ATTN_PINS="$SMD_ATTN_ROOT/environments/requirements/core-cu126-py312.constraints.txt"
"$SMD_ATTN_PYTHON" -m pip --isolated install --index-url https://pypi.org/simple -c "$SMD_ATTN_PINS" ninja
if [[ -n "${SMD_FLASH_ATTN_WHEEL:-}" ]]; then
  : "${SMD_FLASH_ATTN_SHA256:?Provide the trusted SHA-256 of the cached wheel}"
  "$SMD_ATTN_PYTHON" - <<'PY'
import hashlib, os
from pathlib import Path
p = Path(os.environ['SMD_FLASH_ATTN_WHEEL']).resolve(strict=True)
assert p.name.startswith('flash_attn-2.7.4.post1-') and p.suffix == '.whl', p.name
with p.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
assert digest == os.environ['SMD_FLASH_ATTN_SHA256'], 'Cached wheel SHA-256 mismatch'
print('Verified cached FlashAttention wheel:', digest)
PY
  "$SMD_ATTN_PYTHON" -m pip --isolated install --no-deps -c "$SMD_ATTN_PINS" "$SMD_FLASH_ATTN_WHEEL"
else
  # Build against the installed Torch ABI when an upstream wheel is unavailable.
  export MAX_JOBS="${MAX_JOBS:-2}"
  "$SMD_ATTN_PYTHON" -m pip --isolated install --index-url https://pypi.org/simple \
    --no-build-isolation --no-deps -c "$SMD_ATTN_PINS" \
    -r "$SMD_ATTN_ROOT/environments/requirements/attention-cu126-py312.requirements.txt"
fi
"$SMD_ATTN_PYTHON" - <<'PY'
from importlib.metadata import version
import flash_attn, flash_attn_2_cuda
assert version('flash-attn') == '2.7.4.post1'
print('FlashAttention CUDA extension loaded:', flash_attn_2_cuda.__file__)
PY
