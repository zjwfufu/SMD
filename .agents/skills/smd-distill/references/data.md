# Data formats

The bundled examples let you try each data loader before preparing a larger dataset. To train on your own data, use the same file format and update `config.dataloader_train` inside the experiment's `create_config()` function. The loader assignments below use this Python configuration format. Commands for starting and resuming training are in the [training guide](training.md#training-and-resume).

Start with the example closest to your task:

| Example | What it shows |
| --- | --- |
| [Image prompts](../../../../assets/examples/text_to_image) | A Parquet table with one text prompt per row |
| [Video prompt example](../../../../assets/examples/text_to_video) | A minimal prompt-only input, separate from the cached dataset used for ordinary Wan training |
| [Cached video layout](#cached-video-webdataset) | The WebDataset format used for RCM 248k; documented below, with no tar sample bundled |
| [Image editing](#image-editing) | Image pairs and edit instructions packed into JSONL records inside tar shards |
| [3D objects](#image-conditioned-3d) | Condition images and sparse shape/texture arrays grouped by object key inside tar shards |
| [H3 T2VA and Ref2VA](#h3-cached-conditions) | Cached text/multimodal conditions, manifests and reference audio/video latents |

These are actual loader inputs. The sections below describe their layouts, required fields and how to prepare a larger dataset. The [cached video format](#cached-video-webdataset) is documented separately below; no video-latent tar sample is bundled here.

| Task | Loader | Data path field |
|---|---|---|
| Text-conditioned images or videos | `smd.datasets.prompt_only.PromptOnlyLoader` | `dataset_path` |
| Cached video latents and text embeddings | `smd.datasets.video_latent_loader.VideoLatentLoader` | `datatags` |
| Routed SD3.5 prompts | `smd.datasets.routed_prompt.RoutedPromptLoader` | `route_paths` |
| H3 T2VA cached conditions | `smd.datasets.minimax_h3_prompt_cache.MiniMaxH3PromptCacheLoader` | `dataset_path` |
| H3 Ref2VA cached conditions | `smd.datasets.minimax_h3_ref2va_cache.MiniMaxH3Ref2VACacheLoader` | `dataset_path` |
| Image editing | `smd.datasets.edit_image_loader.EditImageTarLoader` | `datatags` |
| Hunyuan3D image conditioning | `smd.datasets.hunyuan3d_shape_loader.Hunyuan3DShapeImageLoader` | `shard_dir` |
| TRELLIS shape | `smd.datasets.trellis_shape_loader.TrellisShapeWDSLoader` | `shard_dir` |
| TRELLIS texture | `smd.datasets.trellis_texture_loader.TrellisTextureWDSLoader` | `shard_dir` |

## Text prompts

`prompts.parquet` contains one row per sample, with a required `prompt` column
of Arrow type `string`. Each example file has four non-empty prompts. FLUX,
Qwen-Image and Lens share the image example. The default Wan configs point to
the small video prompt example so the prompt-only path can be tried without the
full cached dataset. Our ordinary Wan training runs instead use RCM 248k through
`VideoLatentLoader`, as described below.
The model encodes each prompt and starts an on-policy trajectory from noise.

Create a full dataset with the same schema:

```python
import pyarrow as pa
import pyarrow.parquet as pq

prompts = ["A red ceramic mug on a wooden table.", "A sailboat on a lake."]
table = pa.table({"prompt": pa.array(prompts, type=pa.string())})
pq.write_table(table, "prompts.parquet", compression="snappy")
```

```python
from smd.utils import LazyCall as L

config.dataloader_train = L("smd.datasets.prompt_only.PromptOnlyLoader")(
    dataset_path="/path/to/my_dataset/prompts.parquet",
    prompt_key="prompt",
    batch_size=2,
    num_workers=0,
    seed=0,
    shuffle=True,
)
```

For multiple Parquet shards, set `dataset_path` to their directory and
`config.dataloader_train.file_glob = "*.parquet"`. Additional columns are allowed; this loader reads only
`prompt_key`. If your prompt column has another name, set that key explicitly.

## Cached video WebDataset

Our ordinary Wan1.3B and Wan14B training runs use the RCM 248k cache:
248,221 records across 970 uncompressed tar shards. This dataset contains
precomputed video latents and text embeddings, as well as the original prompts.
It is separate from `examples/text_to_video/prompts.parquet` and is not bundled
in this repository. The same loader accepts other datasets with the following
layout:

```text
video_shards/
└── shard-00000.tar
    ├── KEY.latent.pt
    ├── KEY.embed.pt
    └── KEY.prompt.txt
```

Uncompressed tar shards can supply precomputed video latents and text conditions.
Each record contains consecutive `<key>.latent.pt`, `<key>.embed.pt` and
`<key>.prompt.txt` members. Tensors are stored without a batch dimension:
`latent.pt` is `[C,T,H,W]` and `embed.pt` is `[tokens,features]`. Wan 480p uses
`[16,21,60,104]` and `[512,4096]`. All tensors must be finite floating tensors;
records within a batch must have matching shapes.

Replace the experiment's complete `dataloader_train` mapping with the following.
Changing only `dataset_path` does not switch the default prompt-only loader:


```python
from smd.utils import LazyCall as L

config.dataloader_train = L("smd.datasets.video_latent_loader.VideoLatentLoader")(
    datatags=["WDS:/path/to/video_shards"],
    batch_size=2,
    num_workers=1,
    sampler_start_idx=None,
    sampler_base_offset=0,
    shard_count_file=None,
    worker_timeout_s=300,
)
```

The optional count file maps tar filenames to positive record counts, for example
`{"shard-00000.tar": 256, "shard-00001.tar": 128}`. It allows resume to skip
complete shards without reading their contents. Records follow sorted shard
filenames and member order, partitioned across ranks. One worker per rank keeps
this ordering stable; zero workers is available for local inspection. Corrupt or
incomplete records fail explicitly. Keep the dataset, count file, batch size and
world size unchanged when resuming from the saved global consumed-sample count.

Training reuses cached positive text embeddings and latents. Negative conditions
follow the model configuration. The student trajectory still begins from noise;
cached video latents provide geometry and the reference in batch previews.
Fixed prompt previews retain their separate prompts and condition cache.

## SD3.5 routed prompts

The four route files contain 30,000 prompts each; `combined_all.parquet` contains
the combined 120,000 prompts. A specialist can consume all four prompt routes while using one fixed
teacher adapter; prompt routing and teacher selection are separate settings.

## Image editing

`EditImageTarLoader` reads paired images and edit instructions from tar shards. The examples in `assets/examples/image_edit/` contain two shards with one record each.

```text
shards/
├── shard-000000.tar
│   └── records.jsonl
└── shard-000001.tar
    └── records.jsonl
```

Each JSONL line is an independent paired record:

```json
{"id":"sample_0001","source_img":"<base64 image bytes>","edited_img":"<base64 image bytes>","short_ins":"Replace the red mug with a blue mug."}
```

`source_img` and `edited_img` contain the original encoded image bytes as Base64,
without a data-URL prefix. Configure `caption_keys` to choose instruction fields;
these examples use `short_ins`. `id` is optional but recommended for tracing a
sample. The reference edited image is paired with the source; it is not a source
condition for the text encoder.

For example, create one shard from image pairs:

```python
import base64
import io
import json
from pathlib import Path
import tarfile

pairs = [("source.png", "edited.png", "Replace the red mug with a blue mug.")]
records = []
for index, (source, target, instruction) in enumerate(pairs):
    records.append({
        "id": f"sample_{index:06d}",
        "source_img": base64.b64encode(Path(source).read_bytes()).decode(),
        "edited_img": base64.b64encode(Path(target).read_bytes()).decode(),
        "short_ins": instruction,
    })
payload = "".join(json.dumps(record) + "\n" for record in records).encode()
with tarfile.open("shard-000000.tar", "w") as archive:
    member = tarfile.TarInfo("records.jsonl")
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))
```

Point the loader at the directory or explicit tar files:

```python
from smd.utils import LazyCall as L

config.dataloader_train = L("smd.datasets.edit_image_loader.EditImageTarLoader")(
    datatags=["WDS:/path/to/my_dataset/shards"],
    caption_keys=["short_ins"],
    batch_size=1,
    num_workers=0,
)
```

By default the VAE input preserves source aspect ratio, rounded to the model
grid at approximately one megapixel. The source PIL image remains available to
the model's vision-language preprocessor, which applies its own size policy.
The loader groups matching image sizes into batches. For larger datasets, add records to each shard and use more shards and workers.

## Image-conditioned 3D

The examples in `assets/examples/objects3d/` contain two objects in separate tar shards. Within a shard, files for one object share a key:

```text
shard-000000.tar
├── object_0001.cond_000.png
├── object_0001.shape.npz
├── object_0001.pbr.npz
└── object_0001.json
```

The files have these roles:

| File | Contents | Used by |
|---|---|---|
| `KEY.cond_000.png` | RGB conditioning image | All three loaders |
| `KEY.shape.npz` | Raw `coords: [N,3]`, `feats: [N,32]` | TRELLIS Shape and Texture |
| `KEY.pbr.npz` | Raw `coords: [N,3]`, `feats: [N,32]` | TRELLIS Texture |
| `KEY.json` | Optional object metadata | Inspection |

Coordinates are integer voxel positions; features are finite floating-point
arrays. Shape and PBR coordinates must match exactly, including row order.
Store the raw latent features: the loaders apply the corresponding per-channel
normalization once. The sample key must not contain a period.

Hunyuan3D reads only the condition image. TRELLIS Shape uses the sparse support;
TRELLIS Texture additionally receives the normalized shape features on that
same support. Each sample can have a different voxel count.

The examples use RGB condition images. `cond_000.png` contains the selected view; `original_condition_view` in the metadata records its source view number.

Pack your own precomputed samples, for example:

```python
from pathlib import Path
import webdataset as wds

with wds.ShardWriter("shard-%06d.tar", maxcount=1000) as writer:
    writer.write({
        "__key__": "object_0001",
        "cond_000.png": Path("condition.png").read_bytes(),
        "shape.npz": Path("shape.npz").read_bytes(),
        "pbr.npz": Path("pbr.npz").read_bytes(),
    })
```

For texture training:

```python
from smd.utils import LazyCall as L

config.dataloader_train = L("smd.datasets.trellis_texture_loader.TrellisTextureWDSLoader")(
    shard_dir="/path/to/my_dataset/shards",
    batch_size=1,
    num_workers=0,
    shuffle=True,
)
```

Use `TrellisShapeWDSLoader` for shape training and
`Hunyuan3DShapeImageLoader` for Hunyuan image conditioning, with the corresponding
module paths listed in the loader table above.

Workers and ranks partition the image-edit and 3D shards. Use enough shards for
parallel data loading; when there are fewer shards than ranks, the loader repeats
shards. Full datasets use more records per shard and more shards.

## H3 cached conditions

`assets/examples/h3/` contains four T2VA instructions and two Ref2VA instructions with precomputed conditions, about 91 MiB in total. Restore their binary payloads before loading a configuration:

```bash
python tools/download_resources.py --model h3_t2va_4step h3_ref2va_4step
```

See [training resources](training.md#training-resources) for authentication and the pinned revisions. You can inspect them on CPU without downloading the encoders or base model. Larger datasets use the same format.

The sections below describe the cache layout, fields and inspection workflow.

### T2VA

```text
t2va/
├── manifest.json
├── records.jsonl
└── conditions-000000000.safetensors
```

The schema-v1 manifest declares `storage: ragged_bfloat16`, `hidden_dim: 5120`
and fixed output geometry: 1344×768, 24 fps, 124 frames. Each JSONL record has
`shard`, `token_start`, `token_end`, `global_index`, `prompt` and
`prompt_sha256`. Token ranges are half-open offsets into the named shard.
The shard contains concatenated Qwen3-VL layer-50 `prompt_embeds` of shape
`[total_tokens, 5120]` in BF16 and `text_token_tags` of shape `[total_tokens]`
in int64. T2VA tags are all 1. The manifest records the encoder and tokenization settings used to create the cache.

The loader returns a `MiniMaxH3Condition`, video latent shape `[24,37,48,84]`,
audio latent shape `[2,32,207]`, fps, frame count and per-record metadata.
Shapes gain a leading batch dimension in a batch. There are no target latents:
the on-policy trajectory starts from noise.

### Ref2VA

```text
ref2va/
├── release/
│   ├── manifest.json
│   └── records.jsonl
├── qwen_conditions/
│   ├── manifest.json
│   └── parts/00000/ref2va_video_audio/<instruction_id>.{json,safetensors}
└── reference_av/
    ├── manifest.json
    └── records/<reference_id>.{json,safetensors}
```

The release manifest uses `minimax_h3.ref2va-onpolicy-set/v1`, `complete: true`
and `target_free: true`. It includes the exact SHA-256 of `records.jsonl` and
the condition manifest. Source roots are relative to `release/`, so the entire
dataset can move without rewriting paths.

Each instruction links a prompt, `instruction_id`, `clip_id`, `task_type`,
`output_geometry`, `reference_geometry`, a `condition` pointer and one
`references` entry. A condition pointer contains `condition_file`,
`metadata_file` and `condition_sha256`, relative to `qwen_conditions/`.
Each reference carries `reference_id`, its own geometry and a `compact_av`
pointer with the payload SHA-256. Files are named by those identifiers.

Condition tensors are BF16 `prompt_embeds: [tokens,5120]` and int64
`text_token_tags: [tokens]`. Tags 0 and 1 retain the encoded visual and textual
rows. Condition metadata uses `minimax_h3.multimodal-condition/v1` and binds
the instruction (`pair_id`), reference clip, prompt hash, encoder revision,
layer and payload hash. Editing a prompt requires recomputing its condition.

Reference payloads contain FP16 `ref2va_video: [24,Tv,H/16,W/16]` and
`ref2va_audio: [2,32,Ta]`, with exact shapes in `reference_geometry`.
Their metadata uses `minimax_h3.ref2va-compact-av/v1`; the containing manifest
uses `minimax_h3.ref2va-compact-av-set/v1`. These are reference conditions,
not training targets or raw MP4 files.

The two instructions demonstrate camera-shot transfer and composition
transfer. Both have 124 output frames at 24 fps; their output video shapes
are `[24,37,64,48]` and `[24,37,32,72]` respectively. Reference and output
geometry are separate and need not match.

### Read a batch

Run from the repository root in the `smd` environment:

```python
from smd.datasets.minimax_h3_prompt_cache import MiniMaxH3PromptCacheLoader
from smd.datasets.minimax_h3_ref2va_cache import MiniMaxH3Ref2VACacheLoader

loaders = [
    MiniMaxH3PromptCacheLoader(
        "assets/examples/h3/t2va", num_workers=0, pin_memory=False),
    MiniMaxH3Ref2VACacheLoader(
        "assets/examples/h3/ref2va/release", num_workers=0,
        pin_memory=False, verify_payload_hashes=True),
]
for loader in loaders:
    batch = next(iter(loader))
    batch["condition"].validate()
    print(batch["condition"].mode, batch["condition"].prompt_embeds.shape,
          batch["video_latent_shape"], batch["audio_latent_shape"])
```

The loaders are infinite; use a bounded number of batches for inspection.

To use your own data, produce cached conditions with the same encoder settings
and tensor contracts, add records/payloads in this structure, recompute counts
and hashes, and change `dataset_path`. For Ref2VA, write the condition manifest
first, then put its hash and the records-file hash into the release manifest.
Keep all three Ref2VA directories together or configure `condition_root` and
`compact_av_root` explicitly. Raw-media encoding is an upstream preparation
step and is not performed by either cache-only loader.

For training configurations, batch limits, parallelism and resume settings, see
[H3 training](training.md#h3-training).

## Resuming a stream

The supplied tar-loader configurations set `sampler_start_idx: null` so the
trainer resumes from its saved global consumed-sample count. The loader rebuilds
its seeded stream and skips previously consumed batches. This re-reads earlier
records at startup; it does not save all samples or prefetched batches.

Keep the shards, seed, world size, batch size and worker count unchanged when
resuming. Use `bucket_batch_timeout_s: 0` for deterministic image-edit batching.
Changing the dataset or topology should start a new data stream with an explicit
`sampler_start_idx: 0`. Training resumes with a fresh segment cycle.
