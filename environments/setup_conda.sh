#!/usr/bin/env bash
# Build the pinned core environment without changing existing research environments.
set -euo pipefail
SMD_SETUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SMD_CONDA="${CONDA_EXE:-conda}"
SMD_CONDA_BASE="$("$SMD_CONDA" info --base)"
SMD_ENV_ROOT="${SMD_ENV_ROOT:-$SMD_CONDA_BASE/envs}"
SMD_PREFIX="$SMD_ENV_ROOT/smd"
SMD_REPORT="$SMD_SETUP_ROOT/LOCAL_DEV/clean_environment_core"
SMD_PINS="$SMD_SETUP_ROOT/environments/requirements/core-cu126-py312.constraints.txt"
mkdir -p "$SMD_REPORT"
exec 9>"$SMD_REPORT/.lock"
flock -n 9 || { echo 'Another environment setup is running.' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
SMD_STAGE=create
on_exit() {
  local code=$?
  if ((code)); then
    if [[ "$SMD_STAGE" == probe && -f "$SMD_REPORT/probe.json" ]]; then
      cp "$SMD_REPORT/probe.json" "$SMD_REPORT/summary.json"
    else
      printf '{"status":"failed","stage":"%s","exit_code":%s}\n' \
        "$SMD_STAGE" "$code" > "$SMD_REPORT/summary.json"
    fi
    printf 'Failed: %s; details: %s/%s.log\n' "$SMD_STAGE" "$SMD_REPORT" "$SMD_STAGE" >&2
  fi
}
trap on_exit EXIT
if [[ -e "$SMD_PREFIX" ]]; then
  [[ -f "$SMD_PREFIX/.smd-managed-environment" ]] || { echo 'Refusing to change an existing unmanaged smd environment.' >&2; exit 2; }
else
  echo 'create: running'
  "$SMD_CONDA" create --yes --prefix "$SMD_PREFIX" --no-default-packages --override-channels -c conda-forge python=3.12.14 pip > "$SMD_REPORT/create.log" 2>&1
  touch "$SMD_PREFIX/.smd-managed-environment"
  echo 'create: passed'
fi
SMD_PYTHON="$SMD_PREFIX/bin/python"
run_pip() {
  SMD_STAGE="$1"; shift
  printf '%s: running\n' "$SMD_STAGE"
  "$SMD_PYTHON" -m pip --isolated "$@" > "$SMD_REPORT/$SMD_STAGE.log" 2>&1
  printf '%s: passed\n' "$SMD_STAGE"
}
run_pip bootstrap install --index-url https://pypi.org/simple -c "$SMD_PINS" pip setuptools wheel
run_pip pytorch install --no-deps --index-url https://download.pytorch.org/whl/cu126 -c "$SMD_PINS" 'torch==2.7.1+cu126' 'torchvision==0.22.1+cu126'
run_pip project install --index-url https://pypi.org/simple --no-build-isolation -c "$SMD_PINS" -e "$SMD_SETUP_ROOT" --report "$SMD_REPORT/install-report.json"
SMD_STAGE=flash_attention
echo 'flash_attention: running'
bash "$SMD_SETUP_ROOT/environments/install/install_flash_attention.sh" "$SMD_PYTHON" > "$SMD_REPORT/flash_attention.log" 2>&1
echo 'flash_attention: passed'
SMD_STAGE=probe
echo 'probe: running'
rm -f "$SMD_REPORT/probe.json"
cd "$SMD_SETUP_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
"$SMD_PYTHON" -m environments.checks.check_environment --expected-prefix "$SMD_PREFIX" --output "$SMD_REPORT/probe.json" > "$SMD_REPORT/probe.log" 2>&1
"$SMD_PYTHON" -m pip freeze --all > "$SMD_REPORT/installed.txt"
"$SMD_CONDA" list --prefix "$SMD_PREFIX" --explicit > "$SMD_REPORT/conda-explicit.txt"
cp "$SMD_REPORT/probe.json" "$SMD_REPORT/summary.json"
printf 'smd: passed (installation, pinned versions, CPU and imports)\nActivate: conda activate "%s"\nSummary: %s/summary.json\n' "$SMD_PREFIX" "$SMD_REPORT"
