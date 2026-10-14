# SPDX-License-Identifier: Apache-2.0
"""On-policy distillation settings."""

import copy
import attrs
from typing import Any
from omegaconf import DictConfig
from smd.configs.config import BaseConfig, BaseModelConfig
from smd.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from smd.utils import LazyCall as L


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    pivot_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    pivot_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))
    student_init_checkpoint: str = ""
    pivot_hook_layer: int = -4
    pivot_num_blocks: int | None = None
    pivot_block_init: str = "pretrained"
    pivot_block_init_seed: int = 0
    no_pivot: bool = False
    final_step_size_scale: float = 0.5
    pivot_num_query_points: int = 4
    query_alpha_mode: str = "random"
    query_point_sampling: str = "raw_t"
    teacher_window_alpha: float = 0.0
    per_segment_backward: bool = True
    w_rescale: float = 30.0
    imitation_weight: float = 1.0
    teacher_repulsion_weight: float = 0.0
    pivot_parameterization: str = "residual"
    pivot_residual_scale: float = 1.0
    residual_basis: str = "bubble_path"
    basis_family: str = "taylor_path"
    num_basis_terms: int = 4
    residual_scale_mode: str = "sqrt_n"
    collocation_scheme: str = "gauss_legendre"
    velocity_distance_beta: float = 1.0
    velocity_distance_eps: float = 1e-06
    teacher_cfg_scale: float = 1.0
    teacher_repulsion_cfg_scale: float = 0.0
    teacher_repulsion_adaptive: bool = False
    log_velocity_diagnostics: bool = True
    embedded_guidance_scale: float | None = None
    cfg_norm_rescale: bool = False
    dynamic_shift: bool = False
    dynamic_shift_base_shift: float = 0.5
    dynamic_shift_max_shift: float = 1.15
    dynamic_shift_base_seq_len: int = 256
    dynamic_shift_max_seq_len: int = 4096
    dynamic_shift_empirical_steps: int = 0
    time_shift_type: str = "linear"
    freeze_backbone_for_lora: bool = True
    backbone_unfreeze_allowlist: list[str] = attrs.field(factory=lambda: ["norm_out"])
    lora_init_seed: int | None = 0
    pivot_train_mode: str = "full"
    pivot_lora_rank: int | None = None
    pivot_lora_alpha: int | None = None
    pivot_lora_dropout: float | None = None
    pivot_lora_init_seed: int | None = 1
    tie_backbone: bool = False


@attrs.define(slots=False)
class H3ModelConfig(ModelConfig):
    h3_condition_mode: str = "t2va"
    h3_video_loss_weight: float = 1.0
    h3_audio_loss_weight: float = 1.0
    h3_repulsion_velocity_mode: str = "segment_start"
    h3_fps: float = 24.0
    h3_num_frames: int = 124
    h3_height: int = 768
    h3_width: int = 1344


@attrs.define(slots=False)
class MultiTeacherModelConfig(ModelConfig):
    teacher_adapters: list[dict[str, Any]] = attrs.field(factory=list)
    route_to_lora: dict[str, list[str]] = attrs.field(factory=dict)
    base_teacher_ratio: float = 0.5


@attrs.define(slots=False)
class SparseModelConfig(ModelConfig):
    guidance_interval: list[float] | None = None
    guidance_rescale: float = 0.0


@attrs.define(slots=False)
class CausalModelConfig(ModelConfig):
    causal_correction_strong_teacher: dict | None = None
    causal_correction_strong_skip_layers: list[int] | None = None
    causal_correction_reference_mode: str = "strong_cfg_residual"
    causal_asymmetric_correction_scale: float = 0.2
    causal_asymmetric_correction_normalize: bool = False
    causal_asymmetric_correction_eps: float = 1e-6
    causal_teacher_type: str = "bidirectional"
    causal_teacher_checkpoint: str = ""
    causal_objective_mode: str = "velocity"
    causal_replay_chunks_per_step: int = 1
    causal_transition_type: str = "stochastic"
    causal_renoise_mode: str = "fresh"
    causal_cache_noise: float = 0.0
    causal_direction_only_loss: bool = False
    causal_fixed_trajectory_seed: int | None = None


def _model_class():
    from smd.methods.simplified_distill import SimplifiedDistillation

    return L(SimplifiedDistillation)(config=None)


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=_model_class)



def configure_callbacks(config):
    from smd.callbacks.simplified_wandb import SimplifiedWandbCallback
    from smd.callbacks.export_inference_bundle import ExportInferenceBundle
    from smd.callbacks.grad_clip import GradClipCallback
    from smd.callbacks.grad_norm import GradNormCallback
    from smd.methods.simplified_distill.execution import TrainingDataProgress
    from smd.configs.callbacks import preview_callbacks

    config.trainer.callbacks = {
        "wandb": L(SimplifiedWandbCallback)(sample_logging_iter=None),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net", metric_prefix="optimizer"),
        "pivot_grad_norm": L(GradNormCallback)(model_key="pivot", metric_prefix="optimizer"),
        "training_data_progress": L(TrainingDataProgress)(),
        "export_bundle": L(ExportInferenceBundle)(),
        **preview_callbacks(config),
    }
    return config


def create_config():
    return configure_callbacks(Config())
