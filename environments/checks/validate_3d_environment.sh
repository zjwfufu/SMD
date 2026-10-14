#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == --operators-only ]]; then
  shift
fi
SMD_CONDA="${CONDA_EXE:-conda}"
SMD_BASE="$("$SMD_CONDA" info --base)"
SMD_ENV_ROOT="${SMD_ENV_ROOT:-$SMD_BASE/envs}"
export SMD_CORE_PYTHON="$SMD_ENV_ROOT/smd/bin/python"
export SMD_3D_PYTHON="$SMD_ENV_ROOT/smd_3d_extend/bin/python"
[[ -x "$SMD_3D_PYTHON" ]] || { echo 'Run bash environments/setup_3d_environment.sh first.'; exit 2; }
# Use the base interpreter to read setup metadata and the extension interpreter for CUDA checks.
export CONDA_PREFIX="$SMD_ENV_ROOT/smd" PATH="$SMD_ENV_ROOT/smd/bin:$PATH"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 SPARSE_CONV_BACKEND=flex_gemm SPARSE_ATTN_BACKEND=flash_attn
output=LOCAL_DEV/clean_environment_3d
# Some render utilities compile kernels lazily; retain the build toolchain.
"$SMD_CORE_PYTHON" - <<'PYCODE'
import json
from pathlib import Path
s=json.loads(Path('LOCAL_DEV/clean_environment_3d/summary.json').read_text())
assert s['status']=='installed_pending_runtime', 'Finish 3D setup successfully before validation'
PYCODE
mapfile -t toolchain < <("$SMD_CORE_PYTHON" - <<'PYCODE'
import json
from pathlib import Path
v=json.loads(Path('LOCAL_DEV/clean_environment_3d/summary.json').read_text())['toolchain']
for k in ('cuda_home','arch','cxx_command','cc_command','max_jobs'): print(v[k])
PYCODE
)
export CUDA_HOME="${toolchain[0]}" TORCH_CUDA_ARCH_LIST="${toolchain[1]}"
export CXX="${toolchain[2]}" CC="${toolchain[3]}" MAX_JOBS="${toolchain[4]}"
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$output"
exec 8>LOCAL_DEV/.manual_validation.lock
flock -n 8 || { echo 'Another validation is active.' >&2; exit 2; }
echo '3D CUDA operators: running'
if ! "$SMD_3D_PYTHON" -m environments.checks.check_3d_runtime --output "$output/runtime.json" >"$output/runtime.log" 2>&1; then
  echo "3D CUDA operators: failed; send $output/runtime.json"
  exit 1
fi
echo '3D CUDA operators: passed'
flock -u 8
exec 8>&-
echo "summary: $output/runtime.json"
