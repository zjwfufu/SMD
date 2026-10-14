# SMD model card

[中文文档](MODEL_CARD.zh-CN.md)

The SMD models are collected on [Hugging Face](https://huggingface.co/collections/zjwfufu/smd). Each repository contains weights, an inference script or ComfyUI workflow, and a worked example with its inputs and sampling settings. Follow the instructions in the model repository to run inference without installing this training repository.

## Images and image editing

All image students use four sampling steps. Repositories with multiple variants include examples for each variant.

| Model and weights | Variants | Format | Example |
| --- | --- | --- | --- |
| [FLUX.1-dev](https://huggingface.co/zjwfufu/SMD-FLUX.1-dev) | One student | LoRA | [Canvas tote bag](https://huggingface.co/zjwfufu/SMD-FLUX.1-dev#example) |
| [Qwen-Image](https://huggingface.co/zjwfufu/SMD-Qwen-Image) | One student | LoRA | [Afternoon tea](https://huggingface.co/zjwfufu/SMD-Qwen-Image#example) |
| [Qwen-Image-Edit-2511](https://huggingface.co/zjwfufu/SMD-Qwen-Image-Edit-2511) | One student | LoRA | [Source and edited image](https://huggingface.co/zjwfufu/SMD-Qwen-Image-Edit-2511#example) |
| [SD3.5-Medium](https://huggingface.co/zjwfufu/SMD-SD3.5-Medium) | `self`, `clipscore`, `geneval`, `ocr`, `pickscore`, `multi` | Full transformer | [Six-variant comparison](https://huggingface.co/zjwfufu/SMD-SD3.5-Medium#example) |
| [FLUX.2 Klein 4B](https://huggingface.co/zjwfufu/SMD-FLUX.2-Klein-4B) | `teacher-4b`, `teacher-9b` | LoRA | [Two-teacher comparison](https://huggingface.co/zjwfufu/SMD-FLUX.2-Klein-4B#example) |
| [Lens](https://huggingface.co/zjwfufu/SMD-Lens) | `teacher-lens`, `teacher-flux2-9b` | LoRA | [Original Lens and both students](https://huggingface.co/zjwfufu/SMD-Lens#reproducible-example) |

The Lens repository also includes the original Lens model under `base/`, with its own inference example.

## Video and audio-video

| Model and weights | Steps | Format | Example |
| --- | --- | --- | --- |
| [Wan2.1 1.3B](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B) | 4 | LoRA | [Delivery robot](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B#example) |
| [Wan2.1 14B](https://huggingface.co/zjwfufu/SMD-Wan2.1-14B) | 4 | LoRA | [Ferry on a lake](https://huggingface.co/zjwfufu/SMD-Wan2.1-14B#example) |
| [Causal Wan2.1 1.3B](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B-Causal) | 4 | LoRA | [Tabletop fountain](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B-Causal#example) |
| [MiniMax H3 T2VA · 4-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-4step) | 4 | LoRA | [Generated audio-video](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-4step#example) |
| [MiniMax H3 T2VA · 8-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-8step) | 8 | LoRA | [Generated audio-video](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-8step#example) |
| [MiniMax H3 Ref2VA · 4-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step) | 4 | LoRA | [Reference inputs and generated audio-video](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step#example) |
| [MiniMax H3 Ref2VA · 8-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step) | 8 | LoRA | [Reference inputs and generated audio-video](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step#example) |

H3 also has native ComfyUI workflows for [T2VA 4-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-4step-ComfyUI), [T2VA 8-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-8step-ComfyUI), [Ref2VA 4-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step-ComfyUI) and [Ref2VA 8-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI). Each package includes setup and model download instructions; no custom nodes are required.

## 3D

| Model and weights | Steps | Format | Example |
| --- | --- | --- | --- |
| [Hunyuan3D-2.1 Shape](https://huggingface.co/zjwfufu/SMD-Hunyuan3D-2.1-Shape) | 4 | LoRA | [Input image, mesh and turntable](https://huggingface.co/zjwfufu/SMD-Hunyuan3D-2.1-Shape#example) |
| [TRELLIS.2 Shape and Texture](https://huggingface.co/zjwfufu/SMD-TRELLIS.2) | 4 per distilled stage | LoRA | [Input, geometry and textured turntable](https://huggingface.co/zjwfufu/SMD-TRELLIS.2#example) |

TRELLIS.2 keeps the shape and texture adapters in `shape/` and `texture/`. Its example runs both stages to produce a textured asset.

## Using the weights

Download the model repository and follow its setup and inference commands. The examples link to case files containing the prompt or input references, seed and other generation settings. For repositories with several variants, select the variant shown in its model card.

> [!NOTE]
> LoRA packages require the matching base model. A full transformer contains the student denoiser; encoders and decoders still come from the base model. Use the base revision and sampling settings specified by the package. H3 student weights are released as LoRA adapters.

For training, see the [README](README.md#training); export recipes live in the [distillation skill](.agents/skills/smd-distill/SKILL.md#export-and-generate). Weight licenses and attribution are listed in each Hugging Face repository.
