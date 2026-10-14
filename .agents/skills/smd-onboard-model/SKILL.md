---
name: smd-onboard-model
description: Add or validate a diffusion model in SMD, including its wrapper, condition and latent contracts, Pivot backend, training configuration, export and inference. Use when onboarding a model or diagnosing an integration mismatch.
---

# Onboard an SMD model

Read [model integration](references/integration.md) and use the closest existing network as the interface reference. Read only the files relevant to the current boundary; a local change does not require a full model matrix.

1. Establish an upstream inference reference with concrete conditions, noise, schedule and component revisions. Record latent packing/scaling, prediction sign, timestep conversion and guidance before implementing the wrapper.
2. Implement the wrapper and register its I/O contract. Keep variable-length conditions, masks and auxiliary inputs aligned. Reuse existing preprocessing, LoRA and FSDP support where the semantics match.
3. Add or reuse the Pivot backend. Preserve live student-feature gradients, local capture state and hook cleanup; the shared method owns segment detach boundaries and losses.
4. Add the Python configuration, catalog entry and examples in a production data format. Connect preprocessing and inference dispatch explicitly. Prepare fixed preview caches through the [training reference](../smd-distill/references/training.md#fixed-inputs); a catalog entry alone is insufficient.
5. Compare upstream and SMD predictions to find the first divergence, then validate affected training, gradients, teacher freezing, resume, export and decoded outputs. For ordinary training, check per-slot condition/endpoint reuse across a trajectory cycle; for causal training, check prefix-cache isolation.

Use [method ownership](references/integration.md#method-and-code-structure) and [export contracts](../smd-distill/references/inference.md#weight-export) as implementation constraints. Keep new integrations within the existing training methods and public runtime dependencies.

Honor the user's execution arrangement. Keep validation provenance intact while jobs run, and distinguish numerical checks from visual review. Store detailed commands and results in ignored `LOCAL_DEV/`; clean only owned temporary artifacts. Public instructions use portable paths and omit internal storage locations or development status counters.
