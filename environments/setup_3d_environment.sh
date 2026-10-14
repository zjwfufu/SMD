#!/usr/bin/env bash
# Clone the validated base environment; build optional 3D extensions serially.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SMD_CONDA="${CONDA_EXE:-conda}"
SMD_BASE="$("$SMD_CONDA" info --base)"
SMD_ENV_ROOT="${SMD_ENV_ROOT:-$SMD_BASE/envs}"
mkdir -p LOCAL_DEV/clean_environment_3d
exec 9>LOCAL_DEV/clean_environment_3d/.lock
flock -n 9 || { echo 'Another 3D setup is running.' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null MAX_JOBS="${MAX_JOBS:-2}"
"$SMD_ENV_ROOT/smd/bin/python" environments/install/setup_3d_environment.py --conda "$SMD_CONDA" --base "$SMD_BASE" --env-root "$SMD_ENV_ROOT" "$@"
