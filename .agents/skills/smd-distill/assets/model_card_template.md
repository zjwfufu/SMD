---
model_name: "FILL: selected student name"
base_model: "FILL: student base repository"
license: "FILL: applicable weight license"
---

# FILL: selected student name

This package contains a few-step distilled student denoiser for SMD.

## Package identity

| Field | Verified value |
|---|---|
| Experiment identifier | FILL: identifier in smd/configs/models.json |
| Network family | FILL: bundle network_family |
| Full student artifact | FILL: file/index and checksum manifest |
| Base model and revision/checksum | FILL: exact student base |
| Adapter availability | FILL: original adapter / full fine-tuning only |
| Adapter rank/alpha and replacements | FILL: adapter_config.json or not applicable |
| Adapter fusion precision | FILL: merge_precision / delta_precision, or not applicable |
| Teacher identities | FILL: teacher bases and routed adapters used in training |
| Code version | FILL: release commit |

## Sampling

Describe the verified number of student calls, schedule, resolution and guidance.
For causal video include chunk size, frame count, transition, re-noise and cache
settings. Link the sampling metadata and identify the base encoders and decoder.

```bash
python infer.py EXPERIMENT --weights /path/to/package \
  --output /path/to/samples
```

## Training and data

Link the training configuration and bundled dataset description. Distinguish
the model’s data from the small examples supplied to exercise the
training pipeline. Identify the sources and applicable terms of routed prompt
tables and other included material.

## Validation

Record actual reports for native training, optimizer restoration, real-condition
training, selected-weight decoding and media inspection. State whether adapter
fusion was checked against the complete artifact and report its measured errors.
List pending checks explicitly. Add benchmark claims only with their own evidence.

## Requirements and limitations

List verified dependencies, device capacity and supported inputs. Describe
observed numerical differences and output limitations. Reference the applicable
base model, adapter and source terms without treating the code license as the
weight license.
