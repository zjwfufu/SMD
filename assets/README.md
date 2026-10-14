# Assets

[中文文档](README.zh-CN.md)

This directory contains preview inputs, small data examples, evaluation prompts and presentation resources used by SMD.

| Path | Contents |
| --- | --- |
| [prompt_samples/](prompt_samples/) | Fixed preview prompts, condition images and encoded condition caches |
| [examples/](examples/) | Small input examples for image/video prompts, image editing, 3D objects and H3 cached conditions |
| [sd35/](sd35/) | SD3.5 prompt tables for ClipScore, GenEval, OCR and preference routes, plus their combined table |
| [evaluation/](evaluation/) | SD3.5 evaluation prompts and GenEval metadata |
| [hdri/](hdri/) | Procedural environment lighting for 3D previews and its generator |
| [badges/](badges/) | SVG badges used in the project documentation |
| [smd-teaser.png](smd-teaser.png) | Overview image used in the main README |
| [resource_manifest.json](resource_manifest.json) | Downloadable resource file lists, sizes and checksums |
| [resource_lock.json](resource_lock.json) | Pinned repositories and revisions for resources and teacher adapters |

Some binary payloads are restored with `python tools/download_resources.py --model <model>` from the repository root; see [resource downloads](../.agents/skills/smd-distill/references/training.md#training-resources).

The examples illustrate input formats. Full training datasets such as RCM 248k are stored separately. For layouts and data preparation, see the [data guide](../.agents/skills/smd-distill/references/data.md); for training settings, see the [training guide](../.agents/skills/smd-distill/references/training.md#training-and-resume).

The procedural `hdri/forest.exr` is available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Its use and regeneration are described in [preview lighting](../.agents/skills/smd-setup/references/environment.md#decoders-and-lighting).
