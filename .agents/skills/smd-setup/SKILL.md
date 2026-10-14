---
name: smd-setup
description: Install or troubleshoot SMD's pinned core and optional 3D environments, including FlashAttention and CUDA runtime checks. Use when preparing a machine for SMD or diagnosing dependency and extension failures.
---

# Set up SMD

Run commands from the repository root. Read [core setup](references/environment.md#core-environment) for installation or dependency failures; read [3D setup](references/environment.md#3d-environment) when mesh decoding or sparse extensions are needed.

- Use the pinned recipes in `environments/` and `requirements.txt`. Check the selected interpreter, driver and build toolkit before changing dependencies.
- `bash environments/setup_conda.sh` creates `smd`; `bash environments/setup_3d_environment.sh` clones it into `smd_3d_extend`. Preserve installer ownership checks and existing user environments. Keep a custom `SMD_ENV_ROOT` consistent across setup and validation.
- Separate installation checks from model execution. The core runtime check requires two GPUs; the reference also provides a single-GPU subset. The 3D operator check does not load model weights.
- Diagnose the failed stage from its summary and exception. Do not silently substitute attention or sparse backends to obtain a pass.

For a real-model check after installation, use the [training](../smd-distill/references/training.md) or [inference](../smd-distill/references/inference.md#student-inference) recipe for the selected family. Imports and kernel checks do not establish model behavior.

Honor user-run GPU workflows. Keep full logs and validation receipts in ignored `LOCAL_DEV/`; report the interpreter, checks actually completed and any remaining command.
