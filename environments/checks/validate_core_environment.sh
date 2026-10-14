#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != smd ]]; then
    echo 'Activate the clean environment first: conda activate smd' >&2
    exit 2
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
exec "$CONDA_PREFIX/bin/python" -m environments.checks.validate_core_environment "$@"
