# Training and previews

Use this reference for resource preparation, Python configuration, launching or resuming a run, and inspecting its callbacks. Run commands from the repository root. Prepare resources before constructing a configuration.

## Training resources

Fixed preview conditions and SD3.5 specialist teachers are downloaded separately
from the source repository. Prepare them before loading an experiment configuration:

```bash
python tools/download_resources.py --model qwen_image
```

Use the experiment name from `smd/configs/experiments/`. Multiple names select
the union of their files. Related experiments share downloads: H3's 4- and
8-step variants use the same cache for each mode, and all SD3.5 experiments share
one preview cache.

```bash
python tools/download_resources.py --model h3_t2va_4step h3_ref2va_4step
python tools/download_resources.py --model sd35_clipscore
python tools/download_resources.py --model sd35_multi
python tools/download_resources.py --model wan13_causal
```

SD3.5 specialist experiments also download their teacher adapters; `sd35_multi`
downloads all four. Teachers are installed under
`$SMD_MODEL_ROOT/sd35_teachers/`, or `models/sd35_teachers/` by default. CLIPScore
comes from `zjwfufu/SMD-SD3.5-Medium/teachers/clipscore/`; GenEval, OCR and
PickScore come from their pinned Flow-GRPO upstream repositories. Base models
are loaded separately by the model wrappers.

`wan13_causal` also downloads the two Causal Forcing **chunkwise** checkpoints
to `$SMD_MODEL_ROOT/causal_forcing/` (default: `models/causal_forcing/`). Both
files are pinned and hash-checked. Startup checks the configured student and
causal teacher checkpoint paths before loading models, including custom paths.
Set `SMD_MODEL_ROOT` consistently when downloading and training; `--model-root`
only changes the downloader's destination.

If a selected resource or model repository is private or gated, authenticate
with a Hugging Face account that has access before downloading:

```bash
hf auth login
```

The revisions and hashes are committed in
[`assets/resource_lock.json`](../../../../assets/resource_lock.json) and
[`assets/resource_manifest.json`](../../../../assets/resource_manifest.json). Downloads verify
SHA-256 and reuse matching local files. A conflicting file raises an error;
move or preserve it before retrying. The downloader does not overwrite custom
assets or alter preview signatures.

### Download during launch

The training entry point can perform the same preparation before calling
`create_config()`. Preparation is serialized across local distributed ranks.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/sd35_clipscore.py --download-resources
```

For a custom Python configuration, also pass `--resource-model` with the
corresponding built-in experiment name. If the custom configuration changes
teacher paths, download its teachers to those paths separately.
Without `--download-resources`, startup checks the configured fixed preview caches locally before initializing distributed training or loading model weights. Missing files report a download command; stale signatures and corrupt payloads stop startup. Checks also run with `--print-config` and `--dryrun`. Disabled fixed previews (`every_n=0`) skip this cache check. Configuration construction can still require source manifests and H3 fixtures to calculate signatures.

### Cache rebuilding and offline use

The default `runtime` scope includes encoded fixed conditions and their
manifests. H3 also includes the complete default training fixtures: T2VA's
condition shard and Ref2VA's text/multimodal conditions and reference latents.
Relative paths are preserved. The shared TRELLIS HDRI remains in Git.

To rebuild fixed conditions, download the original input images and features:

```bash
python tools/download_resources.py --model trellis2_texture --scope rebuild
```

Use `--list` to inspect a selection without network access, or `--check-only`
to verify existing files without downloading:

```bash
python tools/download_resources.py --model h3_ref2va_4step --check-only
```

The Hub cache is separate from the installed assets. Set `--cache-dir` to
choose another cache location; it must be outside `assets/`. Download metadata,
locks and temporary files must not enter H3's signature-covered data directories.
The existing cache reader still checks payload hashes and preprocessing signatures.
For the source-layout migration, `assets/resource_manifest.json` binds each complete
relocated signature to its original cache signature. This only covers the verified
file moves: a changed input, model, precision or preprocessing file still requires
a rebuild. Do not extend this mapping to hide a preprocessing change.

Training datasets for other tasks are separate from these preview resources;
see the [data guide](data.md).

## Training and resume

Start with a configuration under [smd/configs/experiments/](../../../../smd/configs/experiments). Each Python file defines `create_config()`, which returns the model, data loader, trainer and callback settings. Run from the repository root with the [SMD environment](../../smd-setup/references/environment.md#core-environment) activated.

### Data and configuration

First [download the experiment's resources](#training-resources). This must precede
configuration construction, including `--print-config`, because fixed preview
signatures read their input manifests and H3 fixture files. Alternatively,
add `--download-resources` to the training command to prepare them before loading
the configuration.

For text-to-image training, use a Parquet file with a `prompt` column. Set `dataloader_train.dataset_path` to that file or a directory of Parquet shards. Ordinary Wan1.3B and Wan14B training uses cached video WebDataset data; see [the Wan configuration below](#wan-with-cached-video-data). The default Wan configs contain only a small prompt-only example. Image editing, H3 and 3D have their own data layouts; see [data formats](data.md). The bundled datasets are small examples, except for the SD3.5 prompt tables, which contain the full supplied corpus.

Edit the Python configuration or pass overrides after a standalone `-`. To inspect the resulting configuration before loading weights:

```bash
python train.py --config smd/configs/experiments/qwen_image.py --print-config \
  - dataloader_train.dataset_path=/path/to/prompts.parquet
```

To derive an experiment, import an existing configuration and modify the returned object:

```python
from smd.configs.experiments.qwen_image import create_config as create_qwen_config


def create_config():
    config = create_qwen_config()
    config.dataloader_train.dataset_path = "/path/to/prompts.parquet"
    config.model.net_optimizer.lr = 5e-5
    config.trainer.callbacks["prompt_samples"].every_n = 100
    config.log_config.name = "qwen-image-my-data"
    return config
```

Save this as a Python file and pass its relative or absolute path to `--config`.
External configuration packages can use relative imports for shared settings.
Common defaults live in
[`smd/configs/defaults.py`](../../../../smd/configs/defaults.py). Each call returns independent
settings. If changing the model family or preprocessing, call `configure_callbacks(config)`
from `smd.configs.distillation` after those changes, then apply any
callback overrides. This selects the fixed preview inputs and cache signature for the new model.
Resolved run settings are still saved as `config.yaml` for logging and reproducibility.

### Launch training

For Qwen-Image on two GPUs:

```bash
export SMD_OUTPUT_ROOT=/path/to/training_runs

torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/qwen_image.py \
  - dataloader_train.dataset_path=/path/to/prompts.parquet \
  dataloader_train.batch_size=1 log_config.name=qwen-image-my-data \
  trainer.max_iter=10000 trainer.save_ckpt_iter=500 trainer.resume=false
```

Choose a new run name for each experiment. Runs are saved under
`$SMD_OUTPUT_ROOT/<project>/<group>/<name>/`; the supplied configs use project
`simplified_distill` and group `reference`. Each run contains its resolved
`config.yaml`, `checkpoints/`, logs and `visualizations/`.

The main settings to adjust are:

| Setting | Meaning |
| --- | --- |
| `dataloader_train.batch_size` | Batch size per data-parallel rank |
| `trainer.grad_accum_rounds` | Gradient accumulation rounds per optimizer update |
| `trainer.max_iter` | Training iteration limit |
| `trainer.save_ckpt_iter` | Checkpoint interval |
| `model.net_optimizer.lr` | Student learning rate |
| `model.pivot_optimizer.lr` | Pivot learning rate |

Without context parallelism, the effective batch size is `batch_size × GPU count × grad_accum_rounds`. H3 splits each sample across context-parallel ranks; use its [training settings](#h3-training) when configuring GPU count. Mesh previews and TRELLIS training require the [3D environment](../../smd-setup/references/environment.md#3d-environment).

### Wan with cached video data

Our ordinary Wan1.3B and Wan14B runs use the RCM 248k dataset: 248,221 cached
records in 970 tar shards. Each record supplies video latents, text embeddings
and a prompt. The [data guide](data.md#cached-video-webdataset)
describes the file layout and tensor shapes. The full dataset is stored outside
this repository.

In `smd/configs/experiments/wan13.py`, replace the `config.dataloader_train` assignment inside `create_config()` with:

```python
from smd.utils import LazyCall as L

config.dataloader_train = L("smd.datasets.video_latent_loader.VideoLatentLoader")(
    datatags=["WDS:/path/to/rcm248k/video_shards"],
    batch_size=2,
    num_workers=1,
    sampler_start_idx=None,
    sampler_base_offset=0,
    shard_count_file=None,
    worker_timeout_s=300,
)
```

If available, set `shard_count_file` to a JSON mapping of tar filenames to record
counts to accelerate resume. For shards on shared storage, `cache_dir` can point
to a local scratch directory to stage the current and next shard.

Then launch the matching Python entry point:

```bash
export SMD_OUTPUT_ROOT=/path/to/training_runs
torchrun --standalone --nproc_per_node=8 train.py \
  --config smd/configs/experiments/wan13.py \
  - log_config.name=wan13-rcm248k trainer.grad_accum_rounds=1 \
  trainer.max_iter=10000 trainer.save_ckpt_iter=500 trainer.resume=false
```

Use `wan14.py` for the larger model; it inherits the Wan1.3B data loader. The small video
Parquet remains useful for trying prompt-only training, but it does not contain
the RCM video latents or cached text embeddings. Causal Wan has a separate
training configuration; this recipe describes the ordinary Wan models.

### H3 training

Use the [core SMD environment](../../smd-setup/references/environment.md#core-environment). The H3 runtime is included in `smd/networks/MiniMaxH3`. Check its imports and training configuration before loading weights:

```bash
python tools/download_resources.py --model h3_t2va_4step h3_ref2va_4step
python -m environments.checks.check_h3_environment
python train.py --config smd/configs/experiments/h3_t2va_4step.py --print-config
```

The current H3 loaders and network require `dataloader_train.batch_size=1`
per rank for each forward pass. The joint packed text/audio/video implementation
does not yet support multiple samples in one forward pass. This applies to the
H3 integration; other experiments use their own batch settings. Increase the
effective training batch with independent context-parallel groups and
`trainer.grad_accum_rounds`, as described below.

Each experiment uses one cached condition per context-parallel group, FSDP2 with CPU offload,
student/Pivot LoRA, and a frozen reverse-tied teacher. T2VA uses context-parallel
degree 2; Ref2VA uses degree 4 and should be trained on eight 80 GB GPUs.
The teacher follows `model.net` automatically. A different context-parallel
degree must be set on **both** `model.net.context_parallel_size` and
`dataloader_train.context_parallel_size`; the launch process count must be
divisible by it. Configure `trainer.grad_accum_rounds` directly. Distinct
conditions per update are `(world_size / context_parallel_size) * grad_accum_rounds`.

To train the four-step T2VA student on your cached conditions:

```bash
export SMD_OUTPUT_ROOT=/path/to/h3_training_runs
torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/h3_t2va_4step.py \
  - dataloader_train.dataset_path=/path/to/t2va_cache \
  log_config.name=h3-t2va trainer.grad_accum_rounds=2 \
  trainer.save_ckpt_iter=500 trainer.resume=false
```

Resume with the same output root and run identity, `trainer.resume=true`, and
a larger `trainer.max_iter`. Keep data, parallel degree and accumulation count
unchanged. For eight-step sampling, use the corresponding `h3_t2va_8step` experiment.

For Ref2VA, launch eight processes and point the dataset path at its `release/`
directory. Both the model and loader default to CP=4:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config smd/configs/experiments/h3_ref2va_4step.py \
  - dataloader_train.dataset_path=/path/to/ref2va/release \
  log_config.name=h3-ref2va trainer.grad_accum_rounds=2 \
  trainer.save_ckpt_iter=500 trainer.resume=false
```

Use `h3_ref2va_8step.py` for the eight-step student. Two-GPU Ref2VA inference is
supported separately; it does not establish that full-size training fits on two GPUs.

The base revision is pinned in the network configuration. To reuse an existing
local base, set `model.net.transformer_model_id=/path/to/base_root`; that root
must contain the chosen `transformer/` or `transformer_ref/` subdirectory.
Keep bases in the local/shared model cache, outside the release directory.

Checkpoints contain student LoRA, training Pivot parameters, optimizer,
scheduler and consumed-data progress. Resume starts a fresh trajectory cycle
at that saved data position. H3 exports contain the student LoRA and inference metadata. Keep the upstream base in the model cache. H3 uses clean time `t=1-sigma` and independent video/audio shifts
12/3. Preserve both shifts and the final-step scale when sampling the student.

### Resume

Keep the same output root, project, group and run name. Set `trainer.resume=true`; increase `max_iter` if the previous run reached its limit:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/qwen_image.py \
  - dataloader_train.dataset_path=/path/to/prompts.parquet \
  dataloader_train.batch_size=1 log_config.name=qwen-image-my-data \
  trainer.max_iter=20000 trainer.save_ckpt_iter=500 trainer.resume=true
```

The checkpointer restores model, optimizer, scheduler and data progress. Keep the dataset, seed, batch size, worker count, parallelism and accumulation settings unchanged to continue the same data stream. Ordinary training starts a fresh trajectory cycle at the restored data position.

Keep the complete checkpoint directory for resume: the `.pth` file and the
model/optimizer shard directories with the same numeric prefix belong together.
The `.bundle/` directory contains inference metadata and, for full-parameter
training, student weights. An exported inference package does not contain the
optimizer or Pivot state needed to resume training. See [checkpoint files](inference.md#checkpoint-files)
for the export inputs.

### W&B and sampling

W&B logging defaults to offline mode. To log online, run `wandb login` and add `log_config.wandb_mode=online` to the command.

Both sampling callbacks run after the first update and then every 500 updates. `samples` shows the current training batch; `prompt_samples` uses fixed cached inputs for comparison over time. To change both intervals to 100, add:

```text
trainer.callbacks.samples.every_n=100
trainer.callbacks.prompt_samples.every_n=100
```

Previews are saved under the run's `visualizations/` directory, including in offline mode. See [sampling callbacks](#sampling-callbacks) to change fixed inputs, sample counts or decoding settings.

Once you have a checkpoint, follow [weight export](inference.md#weight-export) to use the student for inference.

## Sampling callbacks

Training includes two callbacks for inspecting the student:

| Callback | Input | W&B key prefix |
| --- | --- | --- |
| `samples` | Current training batch | `samples/` |
| `prompt_samples` | Fixed, precomputed conditions | `prompt_samples/` |

Both use the live student and its few-step sampler. They run after the first optimizer update, including after resume, and then every 500 updates by default.

### Frequency and sample count

Override either callback in the training command:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/wan13.py - \
  trainer.callbacks.samples.every_n=100 \
  trainer.callbacks.prompt_samples.every_n=100 \
  trainer.callbacks.prompt_samples.grid_size=4
```

Set `every_n=0` to disable a callback, or `run_on_start=false` to use only the periodic schedule. `grid_size` limits the number of outputs collected across data-parallel ranks; a small batch may provide fewer. All ranks still participate in sampling under FSDP.

H3 defaults to two samples. Other families use four, except the paired image-edit and 3D fixed cases, which produce one result per input image. Set `seed` to control the sampling noise.

### Fixed inputs

`prompt_samples` reads encoded conditions from `assets/prompt_samples/cache/`. It does not consume training data or run a text/image encoder. Changing your training dataset therefore leaves the fixed previews unchanged.

Source files under `assets/prompt_samples/` are organized as follows:

| Source | Contents |
| --- | --- |
| `text_to_image.json` | 12 prompts, encoded separately for SD3.5, FLUX.1, FLUX.2, Lens and Qwen-Image |
| `text_to_video.json` | Eight Wan prompts and their negative conditions |
| `qwen_image_edit/` | Eight source images, edit instructions and target images |
| `hunyuan3d_shape/` | Eight images, DINOv2 features, reference latents and renders |
| `trellis2/` | Eight cases with DINOv3 features, voxel coordinates, shape/texture latents and renders |
| `h3_t2va.json`, `h3_ref2va.json` | Two cases each, using the H3 example condition caches and reference audio/video latents where applicable |

SD3.5 teacher variants share a student cache. H3 four- and eight-step variants share a cache for each conditioning mode. Causal Wan has its own cache because it uses a different negative prompt.

Each cache manifest records payload hashes and a signature covering the inputs, student model, preprocessing and precision. Startup rejects missing, corrupt or incompatible caches before loading model weights.

These fixed cases are for inspecting training, not a held-out benchmark.

First [download the selected experiment's runtime resources](#training-resources). To edit paired input images or features, download `--scope rebuild` before making local changes. After changing fixed prompts or source images, rebuild the affected cache before training:

```bash
python tools/cache_prompt_samples.py \
  --experiments qwen_image --device cuda:0 --force
```

Use `--experiments all` to prepare every cache. Preparation loads frozen encoders and reuses existing H3/TRELLIS features; base weights are separate downloads. If paired source files change, update their source-manifest hashes before rebuilding. Changing teachers, training data or sampling NFE does not require re-encoding.

For a custom cache, set `trainer.callbacks.prompt_samples.cache` to its manifest and `cache_signature` to its matching preprocessing signature. Startup validates the configured cache, including custom paths. Cached tensors stay on CPU between previews.

### Outputs

Images and videos are saved under:

```text
<run>/visualizations/
├── samples/media/iter_<iteration>/
├── prompt_samples/media/iter_<iteration>/
└── events/
```

The files are saved in offline W&B mode too. W&B keys stay the same across iterations, so each input keeps its own history. Image-edit previews include the source, student output and target when available. 3D previews show the input image and mesh views. H3 videos include audio; Ref2VA also includes the reference audio-video clip.

The `events/` directory contains decode manifests and failure reports. A decoder error is logged and training continues; errors during distributed sampling propagate to the training process. Wan, H3 and 3D callbacks allow up to 2,000 seconds for decoding; other families allow up to 540 seconds. Mesh previews require the [3D environment](../../smd-setup/references/environment.md#3d-environment).

Callbacks preserve training RNG state and module modes, use their own sampling noise, and leave cached trajectory endpoints unchanged. Fixed-noise comparisons assume the same parallel topology.
