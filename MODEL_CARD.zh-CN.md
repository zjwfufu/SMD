# SMD 模型说明

[English](MODEL_CARD.md)

所有 SMD 模型都可以在 [Hugging Face 合集](https://huggingface.co/collections/zjwfufu/smd) 中找到。每个模型仓库都附有推理脚本或 ComfyUI 工作流，以及示例输入和采样设置，下载后按说明运行即可，不需要安装这个训练仓库。

## 图像生成与编辑

图像模型均为四步采样。同一仓库有多个版本时，每个版本都附有示例。

| 模型与权重 | 版本 | 格式 | 示例 |
| --- | --- | --- | --- |
| [FLUX.1-dev](https://huggingface.co/zjwfufu/SMD-FLUX.1-dev) | 单一版本 | LoRA | [帆布手提袋](https://huggingface.co/zjwfufu/SMD-FLUX.1-dev#example) |
| [Qwen-Image](https://huggingface.co/zjwfufu/SMD-Qwen-Image) | 单一版本 | LoRA | [下午茶](https://huggingface.co/zjwfufu/SMD-Qwen-Image#example) |
| [Qwen-Image-Edit-2511](https://huggingface.co/zjwfufu/SMD-Qwen-Image-Edit-2511) | 单一版本 | LoRA | [源图与编辑结果](https://huggingface.co/zjwfufu/SMD-Qwen-Image-Edit-2511#example) |
| [SD3.5-Medium](https://huggingface.co/zjwfufu/SMD-SD3.5-Medium) | `self`、`clipscore`、`geneval`、`ocr`、`pickscore`、`multi` | 完整 Transformer | [六个版本对比](https://huggingface.co/zjwfufu/SMD-SD3.5-Medium#example) |
| [FLUX.2 Klein 4B](https://huggingface.co/zjwfufu/SMD-FLUX.2-Klein-4B) | `teacher-4b`、`teacher-9b` | LoRA | [两种教师蒸馏结果对比](https://huggingface.co/zjwfufu/SMD-FLUX.2-Klein-4B#example) |
| [Lens](https://huggingface.co/zjwfufu/SMD-Lens) | `teacher-lens`、`teacher-flux2-9b` | LoRA | [原始 Lens 与两个学生模型](https://huggingface.co/zjwfufu/SMD-Lens#reproducible-example) |

Lens 仓库的 `base/` 目录还包含原始 Lens 模型，也可以按其中的示例单独运行。

## 视频与音视频

| 模型与权重 | 采样步数 | 格式 | 示例 |
| --- | --- | --- | --- |
| [Wan2.1 1.3B](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B) | 4 | LoRA | [配送机器人](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B#example) |
| [Wan2.1 14B](https://huggingface.co/zjwfufu/SMD-Wan2.1-14B) | 4 | LoRA | [湖面渡轮](https://huggingface.co/zjwfufu/SMD-Wan2.1-14B#example) |
| [因果 Wan2.1 1.3B](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B-Causal) | 4 | LoRA | [桌面喷泉](https://huggingface.co/zjwfufu/SMD-Wan2.1-1.3B-Causal#example) |
| [MiniMax H3 T2VA · 4 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-4step) | 4 | LoRA | [生成的音视频](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-4step#example) |
| [MiniMax H3 T2VA · 8 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-8step) | 8 | LoRA | [生成的音视频](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-8step#example) |
| [MiniMax H3 Ref2VA · 4 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step) | 4 | LoRA | [参考输入与生成的音视频](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step#example) |
| [MiniMax H3 Ref2VA · 8 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step) | 8 | LoRA | [参考输入与生成的音视频](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step#example) |

H3 也提供使用原生节点的 ComfyUI 工作流：[T2VA 4 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-4step-ComfyUI)、[T2VA 8 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-T2VA-8step-ComfyUI)、[Ref2VA 4 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step-ComfyUI)和 [Ref2VA 8 步](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI)。各仓库都附有安装和模型下载说明，无需安装自定义节点。

## 3D

| 模型与权重 | 采样步数 | 格式 | 示例 |
| --- | --- | --- | --- |
| [Hunyuan3D-2.1 Shape](https://huggingface.co/zjwfufu/SMD-Hunyuan3D-2.1-Shape) | 4 | LoRA | [输入图像、网格与多角度展示](https://huggingface.co/zjwfufu/SMD-Hunyuan3D-2.1-Shape#example) |
| [TRELLIS.2 Shape 和 Texture](https://huggingface.co/zjwfufu/SMD-TRELLIS.2) | 每个蒸馏阶段 4 步 | LoRA | [输入、几何与带纹理的多角度展示](https://huggingface.co/zjwfufu/SMD-TRELLIS.2#example) |

TRELLIS.2 的形状和纹理 LoRA 分别放在 `shape/` 和 `texture/` 中。示例会依次运行两个阶段，生成带纹理的 3D 模型。

## 使用权重

进入对应模型仓库，按 README 安装依赖并运行示例。示例的 case 文件记录了提示词或参考输入、随机种子和采样参数，方便复现结果。同一仓库有多个版本时，按模型说明选择所需版本。

> [!NOTE]
> LoRA 必须搭配对应的基座模型使用。完整 Transformer 权重则已经包含学生去噪网络，但仍需从基座加载编码器和解码器。两种格式都应使用模型仓库指定的基座 revision 和采样参数。H3 的学生权重只提供 LoRA。

自己训练模型可参考 [README 的训练部分](README.zh-CN.md#训练)，权重导出见[蒸馏 skill](.agents/skills/smd-distill/SKILL.md#export-and-generate)。各模型的权重许可和版权声明以对应 Hugging Face 仓库为准。
