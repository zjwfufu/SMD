# Export and inference

Use the export sections for checkpoints from this training repository, and the inference section for sampling the resulting student. Run commands from the repository root. For released Hub models, follow their standalone instructions.

## Weight export

SMD supports complete student denoisers and original student LoRA adapters. Text encoders, image encoders and VAEs come from the configured base model. Ordinary inference uses the student without the training Pivot.

### Checkpoint files

Checkpoints are saved in the run's `checkpoints/` directory. For a saved prefix
such as `0000500`, select the files according to the training format:

| Training format | Student weights | Sampling metadata |
| --- | --- | --- |
| LoRA, including H3 | `0000500.lora_net_model/` | `0000500.bundle/bundle.json` |
| Full-parameter, including the supplied SD3.5 experiments | `0000500.bundle/weights.safetensors` | `0000500.bundle/bundle.json` |

Use weights and metadata from the same saved prefix. The `.lora_net_model`
input is a distributed-checkpoint directory; keep all its shard files together.
The `.pth` file and optimizer/Pivot shards are needed for training resume, not
student export. Write exported packages outside the source checkout.

### Complete weights

Full-finetuning bundles can be passed directly to inference. For example, to
sample a saved SD3.5 self-distillation checkpoint:

```bash
python tools/download_resources.py --model sd35_self
python infer.py sd35_self --weights /path/to/checkpoints/0000500.bundle \
  --output /path/to/samples --prompt "A red ceramic mug on a wooden table."
```

To create a sharded package, supply the matching base transformer's architecture
`config.json` along with the saved student weights and sampling metadata:

```bash
python tools/weights.py --input /path/to/checkpoints/0000500.bundle/weights.safetensors \
  --config /path/to/base/transformer/config.json \
  --sampling /path/to/checkpoints/0000500.bundle/bundle.json \
  --output /path/to/exported_student
```

The input can also be a directory containing a native safetensors shard index. Exports include native architecture configuration, inference-only sampling metadata, a shard index and SHA-256 checksums. A package may contain the denoiser at its root or under `transformer/`; inference supports both layouts.

```bash
python infer.py sd35_self --weights /path/to/exported_student \
  --output /path/to/samples --prompt "A red ceramic mug on a wooden table."
```

Choose the identifier from [the model catalog](../../../../smd/configs/models.json). Keep the base components accessible through the configured repositories and cache. Inspect the decoded media and `inference.json` after generation.

### LoRA checkpoints

Export the original student adapter from a LoRA training checkpoint, using its actual rank, alpha and base identity:

```bash
python tools/export_lora.py --checkpoint /path/to/0000500.lora_net_model \
  --output /path/to/adapter --rank 256 --alpha 256 \
  --base-model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

This produces `lora.safetensors` and `adapter_config.json`. The `smd_lora` format contains LoRA matrices and any saved trainable replacement tensors. H3 loads its adapter directly; see [H3 audio-video](#h3-audio-video). For other families, use SMD's fusion tool to apply both to the exact training base:

```bash
python tools/weights.py --input /path/to/base/transformer \
  --adapter /path/to/adapter/lora.safetensors \
  --adapter-config /path/to/adapter/adapter_config.json \
  --config /path/to/base/transformer/config.json \
  --sampling /path/to/0000500.bundle/bundle.json \
  --output /path/to/merged_student
```

Then pass `/path/to/merged_student` to `infer.py`. These adapters use SMD's tensor names and may include replacement tensors, so merge them with `tools/weights.py`. They cannot be loaded directly with a generic PEFT or Diffusers LoRA loader.

Fusion computes LoRA products in float32, honors declared merge and delta precision, checks shapes and rejects unconsumed adapter tensors. Use the same base revision and precision as training. Rounding can make the result differ from a separately saved merged model.

When exporting, `--merge-precision` records the base/output dtype used for fusion;
`--delta-precision` records how the scaled LoRA product is rounded before addition.
Without these options the base dtype is preserved and the product stays float32
until the final sum is cast. Match the original export's arithmetic when
reconstructing an existing complete model. In particular:

| Adapter manifest fields | Fusion arithmetic |
|---|---|
| `merge_precision=bfloat16`, `delta_precision=float32` | Add base and LoRA product in FP32, then cast once to BF16 |
| `merge_precision=bfloat16`, `delta_precision=bfloat16` | Round the LoRA product to BF16 before adding it to the BF16 base |

The protocols can produce different weights. Use the precision settings in the adapter manifest when reconstructing a published model.

Full-finetuned students can be exported as complete weights without an adapter. For a Hunyuan PyTorch base, supply `--input /path/to/model.fp16.ckpt --state-key model`.

The six supplied SD3.5-M student experiments use full-parameter training. Their
reward LoRAs belong to the teachers; they are not student adapters to merge into
the distilled output.

### Causal video

Causal student adapters merge onto the causal initialization checkpoint `causal_ode.pt`. Their configured training teacher uses `ar_diffusion.pt`; the correction teacher is needed during training. Student inference uses the exported causal denoiser and its sampling metadata.

```bash
python tools/weights.py --input /path/to/causal_ode.pt --causal-base \
  --adapter /path/to/adapter/lora.safetensors \
  --adapter-config /path/to/adapter/adapter_config.json \
  --config /path/to/base_transformer_config.json \
  --sampling /path/to/checkpoint.bundle/bundle.json \
  --output /path/to/causal_student
```

Preserve the causal chunk size, schedule, transition and re-noise settings in the metadata when generating video.

### H3 audio-video

Export the student LoRA with the `bundle.json` saved alongside the same checkpoint. The supplied H3 configurations use rank 128 and alpha 128:

```bash
python tools/export_lora.py --checkpoint /path/to/0000500.lora_net_model \
  --sampling /path/to/0000500.bundle/bundle.json \
  --output /path/to/h3_student --rank 128 --alpha 128 \
  --base-model MiniMaxAI/MiniMax-H3

python tools/download_resources.py --model h3_t2va_4step
python infer.py h3_t2va_4step --n-gpus 1 \
  --weights /path/to/h3_student --dataset-case-index 0 \
  --output /path/to/h3_samples
```

Use `h3_t2va_8step`, `h3_ref2va_4step` or `h3_ref2va_8step` in both the download and inference commands for the other configurations. The identifier must match the mode and step count in the exported bundle. Keep all three export files together: `lora.safetensors`, `adapter_config.json` and `sampling.json`.

H3 inference uses the native runtime installed with SMD. Choose the number of visible GPUs with `--n-gpus N` (also accepted as `--n_gpus N`). A direct `python` launch defaults to one GPU. For `N > 1`, the entry point starts `torchrun` and uses FSDP and context parallelism across those GPUs; an existing `torchrun --nproc-per-node=N` launch is also supported. An explicit GPU count must match an existing launch. Use `CUDA_VISIBLE_DEVICES` to select devices.

Full-model T2VA 4-step inference has been validated on one 80 GB A100 with a rank-128 student LoRA, a 1344×768 input, 124 frames at 24 fps, FF chunks of 1024 and Q/K norm chunks of 2048. The transformer uses its native BF16 blocks and FP32 components, with all weights on that one GPU. Peak PyTorch tensor allocation was 68.2 GiB (68.6 GiB reserved); sampled whole-device use peaked at 69.1 GiB. Four student evaluations took about 96 seconds and produced a verified 5.17-second audio/video file. Earlier full-model inference was also validated on two 80 GB A100s. These are validation setups, not a two-GPU requirement or a guarantee that every mode, duration and geometry fits on one 80 GB GPU. Memory use depends on the cached case dimensions, frame count, and FF/QK chunk sizes. Single-GPU inference streams pretrained shards directly onto the selected GPU to avoid retaining a complete transformer in CPU memory; distributed inference loads on CPU before FSDP sharding. Decoders also load directly onto their selected device after the transformer is released. It loads the pinned base and applies the LoRA in memory, without writing a fused model. It needs the matching `transformer` (T2VA) or `transformer_ref` (Ref2VA), `vae` and `audio_vae`. The base downloads to the usual model cache; local snapshots can be supplied with `--base` and, if the transformer is stored separately, `--transformer-base`.

Inputs come from the same cached-condition formats as training. `--dataset-case-index` selects an unshuffled record from the bundled examples; `--condition-cache /path/to/dataset` selects your prepared T2VA dataset or Ref2VA release directory. New prompts and references need upstream encoding into the format described in the [H3 data guide](data.md#h3-cached-conditions); this repository's entry point reads those caches and does not encode raw prompts or reference videos. Frame count and dimensions follow the selected record; the exported student supplies the audio/video shifts and number of sampling steps.

The output includes `sample.mp4` with audio, `input_case.json` and `inference.json`. Ref2VA also decodes `reference.mp4` for checking the selected reference. Use a new output directory for each run.

### Share a model

Include the denoiser configuration, sampling metadata, base model identity and weight license. Adapters also need their manifest and exact base revision or checksum. Add a runnable example and its expected output so users can check their installation.

## Student inference

For models on Hugging Face, follow the standalone instructions in the [model card](../../../../MODEL_CARD.md). This guide covers students exported from your own SMD training runs.

Run these commands from the repository root with the model's environment activated. `infer.py` loads a complete student denoiser and its sampling metadata, or an [H3 adapter](#h3-audio-video). Export a training checkpoint or merge an adapter using the [weight tools](#weight-export) first.

Prepare the selected experiment with `python tools/download_resources.py --model EXPERIMENT` before loading its configuration; see [training resources](training.md#training-resources).

### Images and video

Use the experiment identifier from [the model catalog](../../../../smd/configs/models.json):

```bash
python infer.py wan13 --weights /path/to/student \
  --prompt "A kite flying over a meadow." --seed 42 \
  --output /path/to/samples
```

The output directory contains decoded media and `inference.json`, which records the sampling settings. Base encoders and decoders must be available through the configured model repositories or cache. The training Pivot is not needed for inference.

### Image editing and 3D

Select a record from the experiment's dataset:

```bash
python infer.py qwen_image_edit --weights /path/to/edit_student \
  --dataset-case-index 0 --output /path/to/edit_samples
```

The index selects an unshuffled dataset record. Without this option, image-edit and 3D experiments use record 0. For your own inputs, change the experiment's dataset path using the [documented data format](data.md).

Qwen-Image-Edit uses the source image and edit instruction. The paired target is included for comparison. Output dimensions follow the source aspect ratio at approximately one megapixel.

For 3D, use `hunyuan3d_shape`, `trellis2_shape` or `trellis2_texture` with the same `--dataset-case-index` option. TRELLIS Texture needs matching shape latents and sparse coordinates from the selected record. Mesh decoding and rendering require the [3D environment](../../smd-setup/references/environment.md#3d-environment).

### Sampling settings

Keep `sampling.json` with the weights. It carries the student's schedule and final-step settings; causal video also needs its chunk, transition and re-noise settings. Changing these can change the result even with the same weights and seed.

## Standalone Hub packages

The [model card](../../../../MODEL_CARD.md) is the model and example catalog. Each Hub package provides its own loading instructions, dependencies, sampling settings and example inputs; it does not require this training checkout. Keep the package sampler and matching base revision together. Use its provided loader so saved replacement tensors are applied along with LoRA weights.

The Lens package includes its original base, encoder, VAE and tokenizer. Its standalone `infer.py` accepts `--variant base`, `--variant teacher-lens` or `--variant teacher-flux2-9b`. H3 uses a separate pinned Diffusers runtime and a configurable GPU count; 3D packages require additional CUDA extensions. Follow the selected package's instructions for these dependencies.
