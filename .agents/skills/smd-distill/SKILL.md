---
name: smd-distill
description: Train, resume and sample an existing SMD diffusion student, including pinned resource preparation, fixed previews, checkpoint export and LoRA fusion. Use for supported-model distillation and inference workflows.
---

# Distill and use an SMD student

Run commands from the repository root. Select the experiment from [smd/configs/models.json](../../../smd/configs/models.json) and its Python configuration under `smd/configs/experiments/`. Preserve the user's model, data and training choices.

## Prepare and train

Read [training and previews](references/training.md) for the operation being performed:

- [Resource preparation](references/training.md#training-resources) must happen before `create_config()` or `--print-config`. The downloader selects pinned preview assets and specialist teachers by experiment. Preserve SHA-256 and cache signature checks, H3's complete fixture trees and the shared HDRI; never place download metadata in signature-covered data directories.
- [Configuration and launch](references/training.md#training-and-resume) covers Python config composition, CLI overrides after a standalone `-`, data paths and parallelism. Use the [production data formats](references/data.md); ordinary Wan training uses cached video WebDataset shards, while the bundled video Parquet is a prompt-only example. Keep paired conditions and latents aligned.
- [Resume](references/training.md#resume) retains run identity and data-consumption settings. Check base-model caches and extra weights using the [setup reference](../smd-setup/references/environment.md#model-caches-and-extra-weights).
- [Sampling callbacks](references/training.md#sampling-callbacks) covers fixed-cache rebuilding, output keys and decode failures. Inspect both batch and fixed previews; a queued sample is not a published preview.

When interpreting losses or changing trajectory handling, read [method semantics](../smd-onboard-model/references/integration.md#method-and-code-structure). Ordinary training retains detached endpoints and conditions per accumulation slot; causal chunk/cache/correction semantics are separate.

## Export and generate

Read [export and inference](references/inference.md) for the actual checkpoint format, fusion and decoded sampling. Preserve base identity, replacement tensors, precision and sampling metadata. Teacher reward adapters are not student adapters; rounding can make full and fused weights differ.

For released models, use the [Hub package instructions](references/inference.md#standalone-hub-packages) and [model catalog](../../../MODEL_CARD.md). When preparing a model card, use the [template](assets/model_card_template.md), fill claims from evidence and publish only within the user's authorization. Public packages retain loading and inference metadata; keep training iterations, basis choices, experiment records and internal paths in private local evidence.

Write large exports outside the checkout. Honor user-run GPU workflows and run only affected or missing checks. Distinguish config parsing, optimizer updates, state restoration, decoded previews and visual review. Keep evidence in ignored `LOCAL_DEV/`; retain original weights when cleaning owned temporary exports.
