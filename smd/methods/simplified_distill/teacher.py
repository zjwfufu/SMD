# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Teacher velocity queries. No trajectory cache, losses or optimizer state."""

from __future__ import annotations
from typing import Any, Optional
import torch


class TeacherQuery:
    def __init__(self, teacher, config, device, precision):
        self.teacher, self.config = teacher, config
        self.device, self.precision = device, precision
        self.target_scale = config.teacher_cfg_scale
        self.reference_scale = config.teacher_repulsion_cfg_scale
        self.need_reference = config.teacher_repulsion_weight > 0
        self.need_target = config.imitation_weight > 0 or (self.need_reference and config.teacher_repulsion_adaptive)

    def __call__(self, x, sigma, condition, neg_condition):
        values = self._teacher_velocities(x.detach(), sigma, condition, neg_condition)
        if any(value is not None and not torch.isfinite(value).all() for value in values[:2]):
            raise RuntimeError("Non-finite teacher velocity.")
        return values

    def _make_guidance_kwargs(self, batch_size: int, scale: Optional[float | torch.Tensor]) -> dict[str, Any]:
        """Build the embedded-guidance tensor for student-like networks."""
        if scale is None:
            return {}
        if torch.is_tensor(scale):
            if tuple(scale.shape) != (batch_size,):
                raise ValueError(f"Expected guidance shape {(batch_size,)}, got {tuple(scale.shape)}.")
            return {"guidance": scale.to(device=self.device, dtype=self.precision)}
        return {"guidance": torch.full((batch_size,), float(scale), device=self.device, dtype=self.precision)}

    def _cfg_combine(self, out_cond: torch.Tensor, out_uncond: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        """Combine conditional / unconditional teacher velocities under CFG."""
        if cfg_scale == 0.0:
            return out_uncond
        if cfg_scale == 1.0:
            return out_cond
        comb = out_uncond + float(cfg_scale) * (out_cond - out_uncond)
        if not self.config.cfg_norm_rescale:
            return comb
        rescale = getattr(self.teacher, "rescale_cfg_velocity", None)
        if not callable(rescale):
            raise RuntimeError(f"cfg_norm_rescale=True requires {type(self.teacher).__name__}.rescale_cfg_velocity().")
        return rescale(out_cond, comb)

    @torch.no_grad()
    def _teacher_velocities(
        self, x: torch.Tensor, sigma: torch.Tensor, condition: Any, neg_condition: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        scales = []
        if self.need_target:
            scales.append(self.target_scale)
        if self.need_reference:
            scales.append(self.reference_scale)
        if not scales:
            return None, None, None, None
        x = x.to(dtype=self.precision)

        def forward(cond, **kwargs):
            value = self.teacher(x, sigma, condition=cond, fwd_pred_type="flow", **kwargs)
            return value[0] if isinstance(value, list) else value

        if self.config.embedded_guidance_scale is not None:
            outputs = {}
            for scale in scales:
                if scale not in outputs:
                    outputs[scale] = forward(condition, **self._make_guidance_kwargs(sigma.shape[0], scale))
            target = outputs[self.target_scale] if self.need_target else None
            reference = outputs[self.reference_scale] if self.need_reference else None
            return target, reference, target, reference

        need_conditional = any(scale != 0.0 for scale in scales)
        need_unconditional = any(scale != 1.0 for scale in scales)
        if need_unconditional and neg_condition is None:
            raise RuntimeError("Teacher CFG requires neg_condition for the requested target/reference scales.")
        conditional = forward(condition) if need_conditional else None
        unconditional = forward(neg_condition, **self._teacher_unconditional_kwargs()) if need_unconditional else None
        target = self._cfg_combine(conditional, unconditional, self.target_scale) if self.need_target else None
        reference = self._cfg_combine(conditional, unconditional, self.reference_scale) if self.need_reference else None
        return target, reference, conditional, unconditional

    def _teacher_unconditional_kwargs(self) -> dict[str, Any]:
        """Optional teacher kwargs applied only to the explicit-CFG negative branch."""
        skip_layers = self.config.skip_layers
        return {"skip_layers": skip_layers} if skip_layers is not None else {}
