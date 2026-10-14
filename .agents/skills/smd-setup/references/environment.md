# Environment setup

Read the core section for installation or dependency failures; use the 3D section only for mesh decoding, rendering or sparse extensions. Run commands from the repository root.

## Core environment

The two setup entry points stay in `environments/`. Dependency pins and source revisions live in `environments/requirements/`; build helpers are in `environments/install/`, and runtime checks are in `environments/checks/`.

SMD's core environment supports image, video and audio-video training. It targets Linux x86_64, Python 3.12, PyTorch 2.7.1 and CUDA 12.6. The optional [3D environment](#3d-environment) adds mesh decoding, rendering and sparse operators.

### Install the core environment

Prerequisites are Conda and an NVIDIA driver compatible with CUDA 12.6. Building FlashAttention from source additionally requires a CUDA toolkit and C++ compiler. Dependency versions are pinned by the setup script.

From the repository root:

```bash
bash environments/setup_conda.sh
conda activate smd
```

The script installs Torch, the pinned Python dependencies and FlashAttention in that order. It then checks package versions, imports and CPU operations. Dependency versions live in [requirements.txt](../../../../requirements.txt) and the [environment constraints](../../../../environments/requirements/core-cu126-py312.constraints.txt).

By default, the environment is created under Conda's base `envs` directory. If that directory is read-only, choose a writable location:

```bash
export SMD_ENV_ROOT="$HOME/.conda/envs"
bash environments/setup_conda.sh
conda activate "$SMD_ENV_ROOT/smd"
```

Keep `SMD_ENV_ROOT` set when installing or checking the 3D environment. The installer can reuse an environment it created, but refuses to overwrite an unrelated environment with the same name.

Installation logs and a summary are saved in `LOCAL_DEV/clean_environment_core/`. The installer does not download model weights.

### Check GPU support

With two GPUs visible:

```bash
bash environments/checks/validate_core_environment.sh
```

This checks FlashAttention outputs and gradients, CUDA operators, media encoding and two-GPU communication using small inputs. Results are written to `LOCAL_DEV/manual_environment/summary.json`, with full logs in the corresponding run directory. The default timeout is 600 seconds per stage; use `--timeout` to change it.

For one GPU, run the subset without distributed communication:

```bash
python -m environments.checks.check_environment \
  --expected-prefix "$CONDA_PREFIX" --cuda \
  --output LOCAL_DEV/environment_cuda.json
```

These checks are useful for diagnosing installation problems. To try a model, continue with the [training guide](../../smd-distill/references/training.md#training-and-resume).

### FlashAttention

FlashAttention is installed after Torch so it can build against the installed Torch version. Builds default to `MAX_JOBS=2`; increase this if the machine has enough CPU memory.

You can also supply a compatible wheel and its checksum:

```bash
export SMD_FLASH_ATTN_WHEEL=/path/to/flash_attn.whl
export SMD_FLASH_ATTN_SHA256="<sha256>"
bash environments/setup_conda.sh
```

The installer verifies the checksum before using the wheel. Torch's CUDA wheels contain a runtime, but not the CUDA compiler required for a source build.

### Model caches and extra weights

Base models use the Hugging Face cache. Set `HF_HOME` before the first download to choose its location:

```bash
export HF_HOME=/path/to/huggingface_cache
```

Some base models require access approval and authentication. Follow the relevant model repository's access and license terms.

SD3.5 reward specialists and causal Wan also need files under `SMD_MODEL_ROOT`:

```text
/path/to/models/
├── sd35_teachers/
│   ├── clipscore/
│   ├── geneval/
│   ├── ocr/
│   └── pickscore/
└── causal_forcing/
    ├── causal_ode.pt
    └── ar_diffusion.pt
```

Set `export SMD_MODEL_ROOT=/path/to/models` when using those experiments. This directory holds the additional teacher and initialization weights; base models still use the Hugging Face cache.

`python tools/download_resources.py --model sd35_multi` installs the four SD3.5
teacher adapters under this root and prepares the fixed preview cache. Use a
single specialist experiment name to download only its teacher.

`python tools/download_resources.py --model wan13_causal` installs the
`chunkwise/causal_ode.pt` and `chunkwise/ar_diffusion.pt` checkpoints from
[Causal Forcing](https://huggingface.co/zhuhz22/Causal-Forcing) under
`$SMD_MODEL_ROOT/causal_forcing/`, plus the fixed preview cache. The two weights
total about 10.6 GiB; the Hub cache and installed copies each need that space.
Revision and SHA-256 hashes are pinned in `assets/resource_lock.json`.
See [training resources](../../smd-distill/references/training.md#training-resources).

Lens loads its original transformer, text encoder, tokenizer and VAE from
`base/` in `zjwfufu/SMD-Lens`, at the revision pinned by its network wrapper.
The repository is public and does not require authentication.

Lens's GPT-OSS text encoder runs in BF16 in this environment, which increases its memory use compared with native MXFP4. The pinned environment does not include the optional MXFP4 `kernels` package.

H3 uses the core environment. To check its vendored runtime imports without loading weights:

```bash
python -m environments.checks.check_h3_environment
```

See the [H3 data guide](../../smd-distill/references/data.md#h3-cached-conditions) for cached inputs and [H3 training](../../smd-distill/references/training.md#h3-training) for parallel settings.

## 3D environment

Hunyuan3D mesh previews and TRELLIS.2 need extra CUDA extensions. Install the [core environment](#core-environment) first, then run:

```bash
bash environments/setup_3d_environment.sh
bash environments/checks/validate_3d_environment.sh --operators-only
conda activate smd_3d_extend
```

If you set `SMD_ENV_ROOT` during core setup, keep it set and activate `"$SMD_ENV_ROOT/smd_3d_extend"` instead.

### Build requirements

You need Git, GCC/G++ and a CUDA 12.x development toolkit. Set `CUDA_HOME` if the toolkit is not installed at `/usr/local/cuda-12.6` or `/usr/local/cuda-12.8`. Source downloads include recursive Git submodules.

The installer clones `smd`, adds the [3D dependencies](../../../../environments/requirements/3d-cu126-py312.requirements.txt) and builds the revisions listed in [3d-sources.json](../../../../environments/requirements/3d-sources.json). Builds use `MAX_JOBS=2` by default. GPU architecture is inferred from visible GPUs; set `TORCH_CUDA_ARCH_LIST` when building for another machine.

TRELLIS sparse operations use FlexGEMM. Mesh decoding and rendering use `o_voxel`, `nvdiffrast`, `nvdiffrec_render`, `cumesh` and `utils3d`. The installer reuses completed extension builds when their versions match and keeps build logs under `LOCAL_DEV/`.

The validation command checks attention, sparse convolution, rasterization and mesh operators without model weights.

### Train and sample

With `smd_3d_extend` activated, point the loader at your [3D shards](../../smd-distill/references/data.md#image-conditioned-3d):

```bash
python tools/download_resources.py --model trellis2_shape
export SMD_OUTPUT_ROOT=/path/to/training_runs
torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/trellis2_shape.py \
  - dataloader_train.shard_dir=/path/to/3d_shards \
  log_config.name=shape-my-data dataloader_train.batch_size=1 \
  trainer.max_iter=10000 trainer.save_ckpt_iter=500 trainer.resume=false
```

Use `trellis2_texture` or `hunyuan3d_shape` in both the download and training commands for the other tasks. Hunyuan training without mesh previews can use the core environment. The bundled [3D dataset](../../smd-distill/references/data.md#image-conditioned-3d) has two objects; replace it with your own shards for a longer run.

After [exporting the student](../../smd-distill/references/inference.md#weight-export):

```bash
python infer.py trellis2_shape --weights /path/to/student \
  --dataset-case-index 0 --output /path/to/samples
```

See [training and resume](../../smd-distill/references/training.md#training-and-resume) to continue from a checkpoint.

### Decoders and lighting

Hunyuan3D uses its base ShapeVAE and shares the mesh renderer used by TRELLIS. TRELLIS loads `shape_dec_next_dc_f16c32_fp16` and `tex_dec_next_dc_f16c32_fp16` from its base repository. Keep the base `ckpts` configuration and weights in the Hugging Face cache.

TRELLIS Texture uses [assets/hdri/forest.exr](../../../../assets/hdri/forest.exr) for shaded views and PBR diagnostic panels. This procedural studio environment has neutral ambient light and two soft light sources; it contains no photographs or downloaded artwork. The generated lighting asset is dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

To regenerate it, use an environment with NumPy and OpenCV EXR support:

```bash
python assets/hdri/generate.py
```
