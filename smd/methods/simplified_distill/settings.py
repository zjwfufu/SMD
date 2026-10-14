# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the supported execution contract before loading large networks."""

import math


def validate_config(cfg):
    if cfg.use_adaptive_loss:
        raise ValueError("SimplifiedDistillation does not yet support adaptive/logvar weighting.")
    if cfg.tie_backbone and (not cfg.net.use_lora or cfg.student_init_checkpoint):
        raise ValueError("Tying requires a pretrained LoRA student without student_init_checkpoint.")
    if cfg.pivot_train_mode not in {"full", "lora"}:
        raise ValueError("pivot_train_mode must be full or lora.")
    if cfg.pivot_block_init not in {"pretrained", "random"} or cfg.pivot_block_init_seed < 0:
        raise ValueError("Invalid Pivot block initialization.")
    if cfg.pivot_num_blocks is not None and cfg.pivot_num_blocks < 0:
        raise ValueError("pivot_num_blocks must be nonnegative.")
    if cfg.student_sample_steps < 1 or cfg.pivot_num_query_points < 1:
        raise ValueError("At least one student step and query are required.")
    if cfg.final_step_size_scale <= 0 or not math.isfinite(cfg.final_step_size_scale):
        raise ValueError("final_step_size_scale must be finite and positive.")
    if cfg.query_point_sampling not in {"raw_t", "sigma"}:
        raise ValueError("query_point_sampling must be raw_t or sigma.")
    if cfg.query_alpha_mode not in {"random", "random_endpoint"}:
        raise ValueError("Unknown query_alpha_mode.")
    if cfg.query_alpha_mode == "random_endpoint" and cfg.pivot_num_query_points != 1:
        raise ValueError("random_endpoint requires one query.")
    if not 0 < cfg.velocity_distance_beta <= 2:
        raise ValueError("velocity_distance_beta must be in (0, 2].")
    for name in [
        "velocity_distance_eps",
        "imitation_weight",
        "teacher_repulsion_weight",
        "teacher_window_alpha",
    ]:
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name in ["teacher_cfg_scale", "teacher_repulsion_cfg_scale", "w_rescale"]:
        if not math.isfinite(float(getattr(cfg, name))):
            raise ValueError(f"{name} must be finite.")
    if cfg.imitation_weight == cfg.teacher_repulsion_weight == 0:
        raise ValueError("At least one loss weight must be positive.")
