<h1 align="center">
  Simplifying and Improving On-Policy Distillation<br>
  for Few-Step Diffusion
</h1>

<p align="center">
  <a href="https://zjwfufu.github.io/SMD-Page/"><img src="assets/badges/project.zh-CN.svg" alt="项目主页" height="20"></a>
  <a href="https://arxiv.org/abs/2610.18472"><img src="assets/badges/arxiv.svg" alt="arXiv:2610.18472" height="20"></a>
  <a href="https://huggingface.co/zjwfufu/SMD"><img src="assets/badges/huggingface.svg" alt="Hugging Face" height="20"></a>
  <a href="https://huggingface.co/collections/zjwfufu/smd"><img src="assets/badges/models.zh-CN.svg" alt="模型合集" height="20"></a>
  <a href="https://huggingface.co/collections/zjwfufu/smd-data"><img src="assets/badges/datasets.zh-CN.svg" alt="数据集" height="20"></a>
  <a href="README.md"><img src="assets/badges/language.zh-CN.svg" alt="English" height="20"></a>
</p>

<p align="center">
  <a href="https://zjwfufu.github.io/SMD-Page/">
    <img src="assets/smd-teaser.png" alt="SMD 生成示例：左侧为图像，中间为 3D 资产，右侧为视频序列。" width="100%">
  </a>
</p>

*SMD 是一种简洁、易于扩展的少步扩散蒸馏方法，适用于图像、视频、音视频和 3D 生成。*

## 模型

模型权重、推理方法和生成示例见[模型说明](MODEL_CARD.zh-CN.md)。

下表中的模型名称链接到训练配置。除特别注明外，均为四步采样。

| 任务 | 模型 |
| --- | --- |
| 文生图 | [FLUX.1-dev](smd/configs/experiments/flux1_dev.py)、[Qwen-Image](smd/configs/experiments/qwen_image.py)、[SD3.5-Medium](smd/configs/experiments/sd35_self.py)、[FLUX.2 Klein 4B](smd/configs/experiments/flux2_4b.py)、[Lens](smd/configs/experiments/lens.py) |
| 图像编辑 | [Qwen-Image-Edit-2511](smd/configs/experiments/qwen_image_edit.py) |
| 文生视频 | Wan2.1 [1.3B](smd/configs/experiments/wan13.py) 和 [14B](smd/configs/experiments/wan14.py)、[因果 Wan2.1 1.3B](smd/configs/experiments/wan13_causal.py) |
| 音视频 | MiniMax H3 T2VA（[4 步](smd/configs/experiments/h3_t2va_4step.py)、[8 步](smd/configs/experiments/h3_t2va_8step.py)）和 Ref2VA（[4 步](smd/configs/experiments/h3_ref2va_4step.py)、[8 步](smd/configs/experiments/h3_ref2va_8step.py)） |
| 图像生成 3D | [Hunyuan3D-2.1 Shape](smd/configs/experiments/hunyuan3d_shape.py)、TRELLIS.2 [Shape](smd/configs/experiments/trellis2_shape.py) 和 [Texture](smd/configs/experiments/trellis2_texture.py) |

仓库还包含 SD3.5 的单个 specialist 教师和[多教师](smd/configs/experiments/sd35_multi.py)实验，以及用 FLUX.2 9B 蒸馏 [Klein 4B](smd/configs/experiments/flux2_9b_to_4b.py) 和 [Lens](smd/configs/experiments/flux2_9b_to_lens.py) 的配置。

## 安装

环境要求：

- Linux x86_64 和 Conda。
- NVIDIA GPU，驱动需兼容 CUDA 12.6。
- 从源码编译 FlashAttention 所需的 CUDA toolkit 和 C++ 编译器。

在仓库根目录运行以下命令，创建 `smd` 环境。

脚本会安装 Python 3.12、PyTorch 2.7.1、CUDA 12.6 对应的 wheel 包和 FlashAttention：

```bash
bash environments/setup_conda.sh
conda activate smd
```

Hunyuan3D 网格预览和 TRELLIS.2 需要额外的 3D 环境：

```bash
bash environments/setup_3d_environment.sh
conda activate smd_3d_extend
```

## 推理

使用已发布的权重时，进入 [Hugging Face 模型合集](https://huggingface.co/collections/zjwfufu/smd)，按对应模型仓库的示例运行。

使用自己训练的 checkpoint 时，先[导出学生权重](.agents/skills/smd-distill/references/inference.md#weight-export)，再运行[推理脚本](infer.py)。

以下以 Qwen-Image 为例：

```bash
python tools/download_resources.py --model qwen_image
python infer.py qwen_image --weights /path/to/exported_student \
  --output /path/to/samples --prompt "A red ceramic mug on a wooden table."
```

通过 `--weights` 指定导出的模型目录，通过 `--output` 指定生成结果的保存目录。模型目录需同时包含学生权重和采样设置。

其他模型和选项见[推理指南](.agents/skills/smd-distill/references/inference.md#student-inference)。

> [!NOTE]
> H3 使用学生 LoRA 和缓存条件，不需要保存完整的融合权重。导出时带上同一 checkpoint 的采样设置。
>
> 直接用 `python` 启动时默认为单卡，通过 `--n-gpus` 指定 GPU 数量。具体步骤见 [H3 导出与推理](.agents/skills/smd-distill/references/inference.md#h3-audio-video)。

## 训练

从[训练配置目录](smd/configs/experiments)选择配置，按[数据指南](.agents/skills/smd-distill/references/data.md)准备数据。

以下以双卡训练 Qwen-Image 为例，输入是含有 `prompt` 列的 Parquet 文件：

```bash
export SMD_OUTPUT_ROOT=/path/to/training_runs
torchrun --standalone --nproc_per_node=2 train.py \
  --config smd/configs/experiments/qwen_image.py --download-resources \
  - dataloader_train.dataset_path=/path/to/prompts.parquet \
  log_config.name=qwen-image-my-data
```

将示例中的数据文件和输出目录替换为自己的路径。

`--download-resources` 选项会自动下载所需资源。单独的 `-` 后面的参数用于覆盖所选配置。

环境变量 `SMD_OUTPUT_ROOT` 指定 checkpoint 和预览的保存根目录。W&B 默认离线记录。

输出目录、恢复训练、日志和预览设置见[训练指南](.agents/skills/smd-distill/references/training.md#training-and-resume)。

## Agent skills

为了方便使用和后续开发，我们在 [skills 目录](.agents/skills)中整理了[环境安装](.agents/skills/smd-setup/SKILL.md)、[训练与推理](.agents/skills/smd-distill/SKILL.md)和[模型接入](.agents/skills/smd-onboard-model/SKILL.md)的操作说明与参考资料。

你或编程助手都可以查阅和使用。这些 skills 目前为英文。

## 致谢与许可

SMD 的训练代码建立在 [NVIDIA FastGen](https://github.com/NVlabs/FastGen) 的基础上，模型实现大量借助了 [Hugging Face Diffusers](https://github.com/huggingface/diffusers)。

感谢这些项目的作者和维护者，让我们有了可以继续研究和开发的基础。

感谢 [FLUX.1](https://github.com/black-forest-labs/flux)、[FLUX.2](https://github.com/black-forest-labs/flux2)、[Qwen-Image 和 Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image)、[Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)、[Wan2.1](https://github.com/Wan-Video/Wan2.1)、[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)、[Lens](https://github.com/dxqb/Lens)、[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) 和 [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 的作者分享模型与代码。
这些开放的模型和代码，让我们得以在图像、视频、音视频和 3D 生成任务上探索 SMD。

也感谢 [Self Forcing](https://github.com/guandeh17/Self-Forcing) 和 [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) 的作者分享代码和 checkpoint。
我们的因果 Wan 实现基于 Self Forcing，并使用了 Causal Forcing 的 chunkwise 初始化和教师 checkpoint。

代码许可和第三方版权声明见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[第三方许可](licenses)。

模型权重、教师适配器和数据集沿用各自的上游许可。

## 引用

```bibtex
@article{smd,
  title   = {Simplifying and Improving On-Policy Distillation for Few-Step Diffusion},
  author  = {Jiawei Zhang and Ziyu Wan and Nanye Ma and Long Zhao and Dong Chen and Hongyu Liu and Qifeng Chen and Dongdong Chen},
  year    = {2026},
  journal = {arXiv preprint arXiv:2610.18472},
}
```
