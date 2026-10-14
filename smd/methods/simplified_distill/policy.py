# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existing Pivot segment operations, isolated from objectives and training state.

The numerical operations are ported unchanged; basis and capture/decode remain
owned by the existing Pivot components. No new curve parameterization is added.
"""

from __future__ import annotations
from typing import Any, Optional
import torch
from smd.networks.noise_schedule import BaseNoiseSchedule, SegmentSchedule
from .data import SegmentContext, SegmentState
from .objectives import velocity_distance


class SegmentPolicy:
    @staticmethod
    def loss_velocities(student, target, reference):
        return student, target, reference

    @staticmethod
    def distance_for(batch):
        return velocity_distance

    def __init__(self, net, pivot, backend, basis, config, device, precision):
        self.net, self.pivot, self.backend, self.basis = net, pivot, backend, basis
        self.config, self.device, self.precision = config, device, precision
        self.input_shape = list(config.input_shape)
        self._no_pivot = config.no_pivot
        self._hook_layer = config.pivot_hook_layer
        self._n_grid = basis.n_grid
        self._student_guidance_scale = config.embedded_guidance_scale
        self._num_query_points = config.pivot_num_query_points
        self._query_point_sampling = config.query_point_sampling
        self._teacher_window_alpha = config.teacher_window_alpha

    def prepare(self, batch, schedule, state, query_alphas):
        # Other accumulation slots (or validation) may have selected another
        # resolution's shift since this batch's schedule was cached.
        self._ensure_schedule_shift().resolve_time_shift(batch.input_shape)
        start = schedule.sigma_start.expand(batch.batch_size)
        end = schedule.sigma_end.expand(batch.batch_size)
        alphas = query_alphas.sort(dim=-1).values
        delta, queries = self._query_schedule(state, start, end, schedule.raw_start, schedule.raw_end, alphas)
        context = self._forward_segment_context(
            net=self.net,
            pivot=self.pivot,
            x=state,
            sigma_start=start,
            segment_size=schedule.size,
            condition=batch.condition,
            input_shape=batch.input_shape,
            return_logvar=False,
        )
        return SegmentState(
            schedule, state, start, end, delta, queries, context, None, query_alpha=alphas.float().mean().detach()
        )

    def endpoint(self, segment):
        return self._euler_update(
            segment.x, segment.guided_context.base_velocity, segment.sigma_end - segment.sigma_start
        ).detach()

    def _euler_update(self, state: torch.Tensor, velocity: torch.Tensor, delta_sigma: torch.Tensor) -> torch.Tensor:
        dt = delta_sigma.reshape(state.shape[0], *[1] * (state.ndim - 1)).to(dtype=state.dtype)
        return state + velocity * dt

    def _student_flow_velocity(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: Any,
        guidance_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """One student flow-velocity forward — the shared student-sampling primitive."""
        out = net(x.to(dtype=self.precision), sigma, condition=condition, fwd_pred_type="flow", **guidance_kwargs)
        return out[0] if isinstance(out, list) else out

    def _ensure_schedule_shift(self) -> "BaseNoiseSchedule":
        """Install the noise-scheduler time-shift policy from config (once)."""
        ns = self.net.noise_scheduler
        if not getattr(ns, "_time_shift_configured", False):
            cfg = self.config
            if cfg.dynamic_shift:
                ns.configure_time_shift(
                    dynamic=True,
                    base_shift=float(cfg.dynamic_shift_base_shift),
                    max_shift=float(cfg.dynamic_shift_max_shift),
                    base_seq_len=int(cfg.dynamic_shift_base_seq_len),
                    max_seq_len=int(cfg.dynamic_shift_max_seq_len),
                    time_shift_type=str(cfg.time_shift_type),
                    empirical_steps=int(cfg.dynamic_shift_empirical_steps),
                )
            else:
                ns.configure_time_shift(dynamic=False, fixed_shift=float(cfg.sample_t_cfg.shift))
        return ns

    def _segment_schedule(self, input_shape: list[int]) -> list[SegmentSchedule]:
        """Build the flow-matching Euler segment schedule for ``input_shape``."""
        ns = self._ensure_schedule_shift()
        configured_t_list = self.config.sample_t_cfg.t_list
        if configured_t_list is not None:
            ns.resolve_time_shift(input_shape)
            sigma_grid = torch.as_tensor(configured_t_list, device=self.device, dtype=ns.t_precision)
            expected_length = int(self.config.student_sample_steps) + 1
            if sigma_grid.ndim != 1 or sigma_grid.numel() != expected_length:
                raise ValueError(
                    f"sample_t_cfg.t_list must contain {expected_length} values for student_sample_steps={self.config.student_sample_steps}, got {tuple(sigma_grid.shape)}"
                )
            if not torch.isfinite(sigma_grid).all():
                raise ValueError("sample_t_cfg.t_list must contain only finite values.")
            if not torch.all(sigma_grid[:-1] > sigma_grid[1:]):
                raise ValueError("sample_t_cfg.t_list must be strictly decreasing.")
            if sigma_grid[-1].item() != 0.0:
                raise ValueError("sample_t_cfg.t_list must end at 0.0.")
            if sigma_grid[0].item() > ns.max_t or sigma_grid.min().item() < ns.min_t:
                raise ValueError(
                    f"sample_t_cfg.t_list must stay within [{ns.min_t}, {ns.max_t}], got [{sigma_grid.min().item()}, {sigma_grid.max().item()}]."
                )
            if not hasattr(ns, "unwarp_t"):
                raise TypeError(
                    f"{type(ns).__name__} cannot use sample_t_cfg.t_list in SMD because it has no unwarp_t()."
                )
            raw_grid = ns.unwarp_t(sigma_grid)
            segments = [
                SegmentSchedule(
                    index=index,
                    raw_start=float(raw_grid[index]),
                    raw_end=float(raw_grid[index + 1]),
                    sigma_start=sigma_grid[index],
                    sigma_end=sigma_grid[index + 1],
                    size=float(raw_grid[index] - raw_grid[index + 1]),
                )
                for index in range(expected_length - 1)
            ]
            return segments
        segments = ns.build_segment_schedule(
            input_shape=list(input_shape),
            num_steps=int(self.config.student_sample_steps),
            final_step_size_scale=float(self.config.final_step_size_scale),
            device=self.device,
        )
        return segments

    def _make_guidance_kwargs(self, batch_size: int, scale: Optional[float | torch.Tensor]) -> dict[str, Any]:
        """Build the embedded-guidance tensor for student-like networks."""
        if scale is None:
            return {}
        if torch.is_tensor(scale):
            if tuple(scale.shape) != (batch_size,):
                raise ValueError(f"Expected guidance shape {(batch_size,)}, got {tuple(scale.shape)}.")
            return {"guidance": scale.to(device=self.device, dtype=self.precision)}
        return {"guidance": torch.full((batch_size,), float(scale), device=self.device, dtype=self.precision)}

    def _student_forward_with_net(
        self,
        net: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        *,
        return_logvar: bool,
        guidance_scale: float | torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Student-like backbone → flow velocity. Triggers any hook registered on ``net``."""
        x_t_in = x_t.to(dtype=self.precision)
        if self.config.embedded_guidance_scale is not None:
            scale = self._student_guidance_scale if guidance_scale is None else guidance_scale
            kwargs = self._make_guidance_kwargs(t.shape[0], scale)
        else:
            kwargs = {}
        out = net(x_t_in, t, condition=condition, fwd_pred_type="flow", return_logvar=return_logvar, **kwargs)
        if return_logvar:
            v, logvar = out if isinstance(out, tuple) else (out, None)
            if logvar is None:
                raise RuntimeError(
                    "use_adaptive_loss=True but network did not return logvar. Ensure enable_logvar_linear=True on the backbone."
                )
            return (v, logvar)
        return out[0] if isinstance(out, list) else out

    def _build_coeffs_from_pivot(
        self,
        net: torch.nn.Module,
        pivot: torch.nn.Module,
        backend,
        hook_data: dict[str, torch.Tensor],
        x_t,
        sigma_t: torch.Tensor,
        seg_size: torch.Tensor,
        input_shape: list[int],
    ) -> dict[str, Any]:
        """Build basis coefficients from an arbitrary student-like net + pivot."""
        ns = net.noise_scheduler
        raw_tokens = backend.decode(pivot, hook_data)
        spatial = backend.unpack_to_spatial(net, raw_tokens, tuple(input_shape))
        flow = ns.convert_model_output(x_t, spatial, sigma_t, src_pred_type=net.net_pred_type, target_pred_type="flow")
        B = x_t.shape[0]
        flow_reshaped = flow.reshape(B, self._n_grid, *x_t.shape[1:])
        return self.basis.build_coeffs(flow_reshaped, x_t, sigma_t, seg_size)

    def _query_schedule(
        self,
        x: torch.Tensor,
        sigma_i: torch.Tensor,
        sigma_j: torch.Tensor,
        raw_t_i: float,
        raw_t_j: float,
        query_alphas: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Return broadcast segment delta and matched ``(sigma, sigma_alpha)`` query points."""
        ns = self.net.noise_scheduler
        batch_size = x.shape[0]
        expected_shape = (batch_size, self._num_query_points)
        if tuple(query_alphas.shape) != expected_shape:
            raise ValueError(f"Expected query_alphas shape {expected_shape}, got {tuple(query_alphas.shape)}")
        dtype = query_alphas.dtype
        alphas = query_alphas.to(device=self.device, dtype=dtype)
        spatial = [1] * (x.ndim - 1)
        dt = (sigma_j - sigma_i).reshape(batch_size, *spatial).to(dtype=dtype)
        dt_safe = dt.clone()
        dt_safe[dt_safe.abs() < 1e-06] = -1e-06
        queries: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k in range(self._num_query_points):
            alpha_k_per_sample = alphas[:, k]
            if self._query_point_sampling == "sigma":
                sigma_k = sigma_i + alpha_k_per_sample * (sigma_j - sigma_i)
                sigma_k = ns.safe_clamp(sigma_k.to(dtype=ns.t_precision), min=ns.min_t, max=ns.max_t)
                sigma_alpha_k = alpha_k_per_sample.reshape(batch_size, *spatial).to(dtype=dtype)
            else:
                raw_t_k = raw_t_i + alpha_k_per_sample * (raw_t_j - raw_t_i)
                sigma_k = ns.safe_clamp(ns.warp_t(raw_t_k).to(dtype=ns.t_precision), min=ns.min_t, max=ns.max_t)
                sigma_k_nd = sigma_k.reshape(batch_size, *spatial).to(dtype=dtype)
                sigma_i_nd = sigma_i.reshape(batch_size, *spatial).to(dtype=dtype)
                sigma_alpha_k = (sigma_k_nd - sigma_i_nd) / dt_safe
            queries.append((sigma_k, sigma_alpha_k))
        return (dt, queries)

    def _forward_segment_context(
        self,
        *,
        net: torch.nn.Module,
        pivot: torch.nn.Module,
        x,
        sigma_start: torch.Tensor,
        segment_size: float,
        condition: Any,
        input_shape: list[int],
        return_logvar: bool,
        guidance_scale: float | torch.Tensor | None = None,
    ) -> SegmentContext:
        if getattr(self, "_no_pivot", False):
            forward_output = self._student_forward_with_net(
                net, x, sigma_start, condition, return_logvar=return_logvar, guidance_scale=guidance_scale
            )
            if return_logvar:
                base_velocity, logvar = forward_output
            else:
                base_velocity = forward_output
                logvar = None
            return SegmentContext(base_velocity=base_velocity, pivot_coeffs={}, logvar=logvar)
        with self.backend.capture_pivot_features(net, self._hook_layer) as capture:
            forward_output = self._student_forward_with_net(
                net, x, sigma_start, condition, return_logvar=return_logvar, guidance_scale=guidance_scale
            )
            if return_logvar:
                base_velocity, logvar = forward_output
            else:
                base_velocity = forward_output
                logvar = None
            pivot_coeffs = self._build_coeffs_from_pivot(
                net,
                pivot,
                self.backend,
                capture.result,
                x,
                sigma_start,
                torch.full_like(sigma_start, segment_size),
                input_shape,
            )
        return SegmentContext(base_velocity=base_velocity, pivot_coeffs=pivot_coeffs, logvar=logvar)

    def _context_trajectory_state(
        self, segment: SegmentState, context: SegmentContext, sigma_alpha: torch.Tensor
    ) -> torch.Tensor:
        if getattr(self, "_no_pivot", False):
            return segment.x + sigma_alpha * segment.delta * context.base_velocity.detach()
        return self.basis.trajectory_state(
            segment.x, segment.delta, context.base_velocity.detach(), context.pivot_coeffs, sigma_alpha
        )

    def _context_proxy_state_for_loss(
        self, segment: SegmentState, context: SegmentContext, sigma_alpha: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate a proxy trajectory state with gradients to student and Pivot."""
        if getattr(self, "_no_pivot", False):
            return segment.x + sigma_alpha * segment.delta * context.base_velocity
        return self.basis.trajectory_state(
            segment.x, segment.delta, context.base_velocity, context.pivot_coeffs, sigma_alpha
        )

    def _context_velocity(self, context: SegmentContext, sigma_alpha: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_no_pivot", False):
            return context.base_velocity
        if self._teacher_window_alpha > 1e-06:
            sigma_alpha_end = (sigma_alpha + self._teacher_window_alpha).clamp(max=1.0)
            return self.basis.windowed_velocity(
                context.base_velocity, context.pivot_coeffs, sigma_alpha, sigma_alpha_end
            )
        return self.basis.evaluate_velocity(context.base_velocity, context.pivot_coeffs, sigma_alpha)

    def _student_guidance_kwargs(self, batch_size: int) -> dict[str, Any]:
        """Embedded-guidance kwargs for a student forward, or ``{}`` when disabled."""
        if self.config.embedded_guidance_scale is not None:
            return self._make_guidance_kwargs(batch_size, self._student_guidance_scale)
        return {}

    @torch.no_grad()
    def sample_student(self, noise: torch.Tensor, condition: Any, net) -> torch.Tensor:
        """Inference: standard student Euler sampling on the SegmentSchedule grid."""
        segments = self._segment_schedule(list(noise.shape[1:]))
        batch_size = noise.shape[0]
        guidance_kwargs = self._student_guidance_kwargs(batch_size)
        x_t = self._initial_state_from_noise(net, noise, segments[0].sigma_start)
        for seg in segments:
            sigma_i = seg.sigma_start.expand(batch_size)
            sigma_j = seg.sigma_end.expand(batch_size)
            v = self._student_flow_velocity(net, x_t, sigma_i, condition, guidance_kwargs)
            x_t = self._euler_update(x_t, v, sigma_j - sigma_i)
        return x_t.to(dtype=noise.dtype)

    def _initial_state_from_noise(self, net, noise, sigma_start):
        return net.noise_scheduler.latents(noise=noise, t_init=sigma_start).to(dtype=self.precision)
