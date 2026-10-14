# 资源清单

[English](README.md)

这个目录存放 SMD 使用的预览输入、少量数据示例、评测提示词和展示素材。

| 路径 | 内容 |
| --- | --- |
| [prompt_samples/](prompt_samples/) | 固定预览使用的提示词、条件图像和编码后的条件缓存 |
| [examples/](examples/) | 图像／视频提示词、图像编辑、3D 对象和 H3 缓存条件的小样本 |
| [sd35/](sd35/) | SD3.5 的 ClipScore、GenEval、OCR、偏好提示词表及合并表 |
| [evaluation/](evaluation/) | SD3.5 评测提示词和 GenEval 元数据 |
| [hdri/](hdri/) | 3D 预览使用的程序化环境灯光及生成脚本 |
| [badges/](badges/) | 项目文档使用的 SVG 徽章 |
| [smd-teaser.png](smd-teaser.png) | 主 README 使用的项目总览图 |
| [resource_manifest.json](resource_manifest.json) | 可下载资源的文件列表、大小和校验值 |
| [resource_lock.json](resource_lock.json) | 资源和教师适配器所用仓库及固定版本 |

部分二进制文件需在仓库根目录运行 `python tools/download_resources.py --model <model>` 恢复，详见[资源下载](../.agents/skills/smd-distill/references/training.md#training-resources)。

示例用于展示输入格式，RCM 248k 等完整训练数据集单独存放。数据布局和准备方法见[数据指南](../.agents/skills/smd-distill/references/data.md)，训练设置见[训练指南](../.agents/skills/smd-distill/references/training.md#training-and-resume)。

程序生成的 `hdri/forest.exr` 以 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 开放使用，使用和重新生成方法见[预览灯光](../.agents/skills/smd-setup/references/environment.md#decoders-and-lighting)。
