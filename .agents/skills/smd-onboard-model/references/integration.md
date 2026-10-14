# Model integration

Start with the closest network under `smd/networks/` and its experiment config. A model integration has three main pieces: a network wrapper, a condition/latent I/O contract and a Pivot backend. The segment loop and losses are shared.

## Network and I/O

Before implementing the wrapper, run the upstream model on a small input and record its revision, conditions, initial noise, schedule and decoded result. These details are particularly easy to get wrong:

| Component | Check |
| --- | --- |
| Latent | Shape, patch packing, normalization and decoder scaling |
| Condition | Masks, variable lengths and auxiliary inputs |
| Time | Sigma range, timestep scaling and schedule shifts |
| Prediction | Velocity/flow/epsilon convention and sign |
| Guidance | CFG, embedded guidance, rescaling and negative conditions |

Add `smd/networks/<ModelName>/` implementing `SMDNetwork`. The wrapper handles loading, forward passes, preprocessing, decoding and sampling. Expose the denoiser through `get_inference_module()` so inference and export use the same tensor names. Reuse the existing LoRA and FSDP hooks where possible.

Register the family in `smd/networks/io_contract.py` and define its network's `LazyCall` in the Python experiment configuration. Use `per_sample_list` for variable-length lists whose entries represent separate samples. Keep auxiliary data, such as source latents or sparse shape conditions, aligned when slicing a batch.

Use installed dependencies or vendored source with its original notices and licenses. Model code should not depend on paths to another checkout.

## Pivot backend

Implement `PivotBackend` from `smd/methods/simplified_distill/pivot/common.py` and register it in `pivot/factory.py`. The backend identifies the denoiser blocks, copies the Pivot tail, captures split-block features and decodes Pivot coefficients. It also handles model-specific masks, positional embeddings and output unpacking.

Keep student feature gradients live. The shared method owns the detach boundaries between segments. Forward hooks must be cleaned up after exceptions, and the backend must not retain capture state between trajectories.

See [method and code structure](#method-and-code-structure) for the shared implementation.

## Configuration and inputs

Add a Python module defining `create_config()` under `smd/configs/experiments/`, then add the experiment to [smd/configs/models.json](../../../../smd/configs/models.json). Include a small dataset using one of the [documented formats](../../smd-distill/references/data.md).

```bash
python train.py --config smd/configs/experiments/MODEL.py --print-config
```

Check the preprocessing dispatch in `smd/trainer.py` and sampling dispatch in `infer.py`. A catalog entry alone does not connect a new family to those paths. Add fixed preview inputs and build their cache as described in [sampling callbacks](../../smd-distill/references/training.md#sampling-callbacks).

## Check the integration

Compare upstream and SMD sampling with the same condition tensors, initial noise and schedule. Compare per-step predictions to locate the first divergence; identical random seeds alone may not reproduce the same inputs across different implementations.

Then run the [training and resume guide](../../smd-distill/references/training.md#training-and-resume) with the new config. Check that the student and Pivot receive gradients, teachers stay frozen, parameters update and optimizer/scheduler state restores. For ordinary training, cover a full trajectory cycle with gradient accumulation: each slot must retain its own condition and detached endpoint until the next cycle. Causal models also need checks that cached prefixes cannot attend to future chunks.

Finally, [export the student](../../smd-distill/references/inference.md#weight-export), generate samples through `infer.py`, and inspect the decoded output. Check both sampling callbacks as well. Keep commands and results in ignored `LOCAL_DEV/` so another developer can reproduce failures without adding machine-specific records to the release.

## Method and code structure

SMD trains a few-step student with a learned Pivot that refines the trajectory within each sampling segment. The student makes one prediction at a segment's input state and time. The policy constructs the segment trajectory and its velocity from that prediction and the Pivot output. Query times are sampled inside the segment; frozen teachers evaluate the corresponding trajectory states.

The ordinary objective combines two independent terms. Imitation minimizes the distance to the guided teacher velocity. Teacher repulsion maximizes the distance to a reference velocity produced with the configured guidance or skipped layers. Each distance averages squared error across features before applying `velocity_distance_beta / 2`; `beta=2` gives MSE. The weighted segment contribution is:

```text
0.5 * w_rescale / student_sample_steps
    * (imitation_weight * imitation - teacher_repulsion_weight * teacher_repulsion)
```

During an ordinary trajectory cycle, each gradient accumulation slot keeps a condition batch and a detached state. All slots train the same segment. After the optimizer update, each slot saves its detached endpoint and advances to the next segment. New conditions and initial noise are acquired after the last segment. Checkpoint resume restores consumed-condition progress and starts a fresh trajectory cycle.

Causal training rolls out temporal chunks, builds the teacher cache from completed prefixes and replays selected chunks with gradients. Its hybrid correction uses frozen conditional and skipped-layer negative teacher velocities to construct a detached pseudo-target for the live trajectory state. This term sends gradients through the student state; it does not train either teacher. Sampling uses the exported chunk size, transition, fresh/fixed noise and cache parameters.

Most of the implementation lives in [`smd/methods/simplified_distill/`](../../../../smd/methods/simplified_distill).

| Module | Responsibility |
|---|---|
| `distillation.py` | One training segment and shared training interface |
| `model.py` | Student, teacher and Pivot ownership; initialization and optimizers |
| `policy.py`, `basis.py`, `pivot/` | Segment states, velocities and Pivot construction |
| `teacher.py` | Frozen teacher velocity queries |
| `objectives.py`, `segment.py` | Independent losses and their segment aggregation |
| `execution.py` | Detached trajectory cycles and condition progress |
| `cross.py`, `multi_teacher.py`, `sparse.py` | Conditioning, routed adapters and sparse tensor adaptations |
| `causal.py`, `correction.py` | Chunk rollout/replay, caches and paired-teacher correction |
| `metrics.py` | Shared metric names and reductions |

W&B logging is implemented in [`smd/callbacks/simplified_wandb.py`](../../../../smd/callbacks/simplified_wandb.py).

The inference sampler reads its protocol from the weight package. It uses the complete student denoiser and the base condition encoders and decoder; Pivot modules are training components.
