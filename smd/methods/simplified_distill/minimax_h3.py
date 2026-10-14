# SPDX-License-Identifier: Apache-2.0
"""H3 paired clean-time dynamics on the shared SMD segment/training protocol.

The cycle stores a flat batch tensor so its detach/checkpoint/diagnostic contract
is unchanged. Geometry lives in each prepared batch/context, never global state.
Only network calls and basis evaluation unpack the video/audio pair. Losses
reduce each modality separately before weighting; video size cannot drown audio.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from smd.networks.MiniMaxH3.condition import MiniMaxH3Condition
from smd.networks.MiniMaxH3.network import MiniMaxH3Latents, MiniMaxH3Timesteps
from smd.networks.MiniMaxH3.geometry import audio_latent_num_frames, video_latent_num_frames
from smd.networks.noise_schedule import BaseNoiseSchedule
from .data import TrainingBatch, SegmentContext, SegmentState
from .distillation import SimplifiedDistillation
from .policy import SegmentPolicy
from .minimax_h3_lifecycle import H3LifecycleMixin
from .segment import SegmentObjective
from .teacher import TeacherQuery
from .objectives import velocity_distance


@dataclass(frozen=True)
class AVGeometry:
    video: tuple[int, ...]
    audio: tuple[int, ...]

    @property
    def video_size(self):
        return math.prod(self.video)

    def unpack(self, value):
        if value.ndim != 2 or value.shape[1] != self.video_size + math.prod(self.audio):
            raise ValueError("H3 flat state does not match its paired geometry.")
        return MiniMaxH3Latents(
            value[:, :self.video_size].reshape(value.shape[0], *self.video),
            value[:, self.video_size:].reshape(value.shape[0], *self.audio),
        )

    @staticmethod
    def pack(pair):
        return torch.cat((pair.video.flatten(1), pair.audio.flatten(1)), dim=1)


@dataclass(kw_only=True)
class H3TrainingBatch(TrainingBatch):
    geometry: AVGeometry


@dataclass(frozen=True)
class H3SegmentSchedule:
    index: int
    raw_start: torch.Tensor
    raw_end: torch.Tensor
    start: MiniMaxH3Timesteps
    end: MiniMaxH3Timesteps


def validate_h3_config(cfg):
    from .settings import validate_config

    validate_config(cfg)
    if cfg.h3_condition_mode not in {"t2va", "ref2va"}:
        raise ValueError("H3 training supports t2va or ref2va cached conditions.")
    if cfg.embedded_guidance_scale is not None or cfg.teacher_cfg_scale != 1 or cfg.cfg_norm_rescale:
        raise ValueError("H3 is guidance-distilled; use teacher_cfg_scale=1 and no external CFG.")
    if cfg.teacher_repulsion_cfg_scale != 0 or cfg.teacher_repulsion_adaptive:
        raise ValueError("H3 teacher repulsion uses native teacher velocities, without CFG or adaptive gating.")
    if cfg.query_point_sampling != "raw_t" or cfg.dynamic_shift or cfg.sample_t_cfg.t_list is not None:
        raise ValueError("H3 uses shared raw query time and its native paired shifted schedules.")
    if cfg.enable_preprocessors:
        raise ValueError("Cached H3 training must set enable_preprocessors=false.")
    if (not cfg.net.use_lora or not cfg.tie_backbone or not cfg.freeze_backbone_for_lora
            or cfg.backbone_unfreeze_allowlist):
        raise ValueError("H3 requires a LoRA-only student, frozen base and reverse-tied teacher.")
    if cfg.use_ema:
        raise ValueError("The H3 training lifecycle currently requires use_ema=false.")
    if cfg.teacher is not None:
        raise ValueError("H3 reverse-tied teacher is derived from model.net; set model.teacher=null.")
    if cfg.pivot_train_mode == "lora" and cfg.pivot_hook_layer == -1:
        raise ValueError("Pivot LoRA requires at least one copied block.")
    weights = (cfg.h3_video_loss_weight, cfg.h3_audio_loss_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("H3 modality weights must be finite, nonnegative and not both zero.")
    modes = {"query", "segment_start", "segment_endpoint_average"}
    if cfg.h3_repulsion_velocity_mode not in modes:
        raise ValueError(f"H3 teacher repulsion mode must be one of {sorted(modes)}.")
    if cfg.teacher_repulsion_weight and cfg.h3_repulsion_velocity_mode == "query" and not cfg.skip_layers:
        raise ValueError("H3 query teacher repulsion requires skip_layers to define a different teacher.")
    if cfg.h3_fps != 24:
        raise ValueError("The native H3 packed-layout contract requires 24 fps.")
    if cfg.net.transformer_subfolder != ("transformer" if cfg.h3_condition_mode == "t2va" else "transformer_ref"):
        raise ValueError("H3 condition mode and transformer subfolder disagree.")


class H3SegmentPolicy(SegmentPolicy):
    def schedules(self):
        raw = torch.tensor(BaseNoiseSchedule.compute_raw_grid_knots(
            self.config.student_sample_steps, self.config.final_step_size_scale), device=self.device, dtype=torch.float32)
        video, audio = self.net.build_paired_schedule(
            self.config.student_sample_steps, self.config.final_step_size_scale, device=self.device)
        return [H3SegmentSchedule(i, raw[i], raw[i+1],
                MiniMaxH3Timesteps((1-video[i]).reshape(1), (1-audio[i]).reshape(1)),
                MiniMaxH3Timesteps((1-video[i+1]).reshape(1), (1-audio[i+1]).reshape(1)))
                for i in range(len(raw)-1)]

    def distance_for(self, batch):
        n = batch.geometry.video_size

        def distance(left, right, beta=1.0, eps=1e-6):
            return (self.config.h3_video_loss_weight * velocity_distance(left[:, :n], right[:, :n], beta, eps)
                    + self.config.h3_audio_loss_weight * velocity_distance(left[:, n:], right[:, n:], beta, eps))
        return distance

    @staticmethod
    def broadcast_time(value, like):
        return value.reshape(like.shape[0], *([1] * (like.ndim-1))).to(like.dtype)

    def prepare(self, batch, schedule, state, query_alphas):
        geometry = batch.geometry
        pair = geometry.unpack(state)
        coeffs = {"geometry": geometry}
        if self._no_pivot:
            base = self.net(pair, schedule.start, batch.condition, fwd_pred_type="flow")
        else:
            with self.backend.capture_pivot_features(self.net, self._hook_layer) as capture:
                base = self.net(pair, schedule.start, batch.condition, fwd_pred_type="flow")
                grids = self.backend.unpack_pair_grid(self.net, self.backend.decode(self.pivot, capture.result), pair)
            for name in ("video", "audio"):
                start, end = getattr(schedule.start, name), getattr(schedule.end, name)
                coeffs[name] = self.basis.build_coeffs(
                    getattr(grids, name), getattr(pair, name), start, end-start)
        context = SegmentContext(geometry.pack(base), coeffs)
        queries = []
        for fraction in query_alphas.float().sort(dim=-1).values.unbind(1):
            raw = schedule.raw_start + fraction * (schedule.raw_end-schedule.raw_start)
            times = MiniMaxH3Timesteps(1-self.net._warp_raw_schedule(raw, self.net.video_shift),
                                      1-self.net._warp_raw_schedule(raw, self.net.audio_shift))
            alphas = tuple((getattr(times, name)-getattr(schedule.start, name)) /
                           (getattr(schedule.end, name)-getattr(schedule.start, name)) for name in ("video", "audio"))
            queries.append((times, alphas))
        return SegmentState(schedule, state, schedule.start, schedule.end, None, queries, context, None,
                            query_alpha=query_alphas.float().mean().detach())

    def endpoint(self, segment):
        geometry = segment.guided_context.pivot_coeffs["geometry"]
        state = geometry.unpack(segment.x)
        base = geometry.unpack(segment.guided_context.base_velocity)
        values = []
        for name in ("video", "audio"):
            value = getattr(state, name)
            dt = getattr(segment.schedule.end, name)-getattr(segment.schedule.start, name)
            values.append(value + self.broadcast_time(dt, value) * getattr(base, name).float())
        return geometry.pack(MiniMaxH3Latents(*values)).detach()

    def _context_trajectory_state(self, segment, context, alphas):
        geometry = context.pivot_coeffs["geometry"]
        state, base = geometry.unpack(segment.x), geometry.unpack(context.base_velocity)
        values = []
        for name, alpha in zip(("video", "audio"), alphas):
            value, velocity = getattr(state, name), getattr(base, name).detach()
            delta = self.broadcast_time(getattr(segment.schedule.end, name)-getattr(segment.schedule.start, name), value)
            alpha = self.broadcast_time(alpha, value)
            values.append(value + alpha * delta * velocity if self._no_pivot else
                          self.basis.trajectory_state(value, delta, velocity, context.pivot_coeffs[name], alpha))
        return geometry.pack(MiniMaxH3Latents(*values))

    def _context_velocity(self, context, alphas):
        if self._no_pivot:
            return context.base_velocity
        geometry = context.pivot_coeffs["geometry"]
        base = geometry.unpack(context.base_velocity)
        values = []
        for name, alpha in zip(("video", "audio"), alphas):
            velocity = getattr(base, name)
            alpha = self.broadcast_time(alpha, velocity)
            coeffs = context.pivot_coeffs[name]
            if self._teacher_window_alpha > 1e-6:
                value = self.basis.windowed_velocity(velocity, coeffs, alpha,
                                                    (alpha+self._teacher_window_alpha).clamp(max=1))
            else:
                value = self.basis.evaluate_velocity(velocity, coeffs, alpha)
            values.append(value)
        return geometry.pack(MiniMaxH3Latents(*values))

    @torch.no_grad()
    def sample_student(self, noise, condition, net):
        return net.sample(noise, condition, num_steps=self.config.student_sample_steps,
                          final_step_size_scale=self.config.final_step_size_scale)


class H3TeacherQuery(TeacherQuery):
    def for_batch(self, batch):
        bound = type(self)(self.teacher, self.config, self.device, self.precision)
        bound.batch = batch
        bound.reference = None
        return bound

    @torch.no_grad()
    def forward(self, state, times, condition, skip_layers=None):
        geometry = self.batch.geometry
        value = self.teacher(geometry.unpack(state.detach()), times, condition, fwd_pred_type="flow",
                             skip_layers=skip_layers)
        return geometry.pack(value)

    @torch.no_grad()
    def prepare(self, policy, segment):
        if not self.need_reference or self.config.h3_repulsion_velocity_mode == "query":
            return
        condition = self.batch.teacher_condition
        self.reference = self.forward(segment.x, segment.schedule.start, condition, self.config.skip_layers)
        if self.config.h3_repulsion_velocity_mode == "segment_endpoint_average":
            end = self.forward(policy.endpoint(segment), segment.schedule.end, condition, self.config.skip_layers)
            self.reference = (self.reference+end)*0.5

    @torch.no_grad()
    def _teacher_velocities(self, x, sigma, condition, neg_condition):
        if neg_condition is not None:
            raise ValueError("H3 has no negative-condition CFG branch.")
        target = self.forward(x, sigma, condition) if self.need_target else None
        reference = self.reference
        if self.need_reference and self.config.h3_repulsion_velocity_mode == "query":
            reference = self.forward(x, sigma, condition, self.config.skip_layers)
        return target, reference, target, reference


class H3SegmentObjective(SegmentObjective):
    def velocity_terms(self, policy, teacher, batch, segment):
        teacher.prepare(policy, segment)
        return super().velocity_terms(policy, teacher, batch, segment)


class MiniMaxH3Distillation(H3LifecycleMixin, SimplifiedDistillation):
    policy_class = H3SegmentPolicy
    teacher_query_class = H3TeacherQuery
    objective_class = H3SegmentObjective

    @property
    def teacher_config(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create(OmegaConf.to_container(self.config.net, resolve=True))
        config.use_lora = False
        return config

    def build_model(self):
        validate_h3_config(self.config)
        super().build_model()
        blocks = len(self.net.transformer.transformer_blocks)
        indices = list(self.config.skip_layers or [])
        if len(indices) != len(set(indices)) or any(i < 0 or i >= blocks for i in indices):
            raise ValueError("H3 skip_layers must be distinct valid zero-based block indices.")

    def prepare_batch(self, data, iteration):
        if data.get("neg_condition") is not None:
            raise ValueError("H3 does not accept negative conditions.")
        condition = data["condition"]
        if not isinstance(condition, MiniMaxH3Condition):
            raise TypeError("H3 requires cached MiniMaxH3Condition batches.")
        condition = condition.to(device=self.device)
        condition.validate()
        if condition.batch_size() != 1 or condition.mode != self.config.h3_condition_mode:
            raise ValueError("H3 requires one condition per rank matching h3_condition_mode.")
        if not torch.isfinite(condition.prompt_embeds).all():
            raise ValueError("H3 condition embeddings must be finite.")
        shapes = []
        for key, width in (("video_latent_shape",4),("audio_latent_shape",3)):
            value = data[key]
            if not torch.is_tensor(value) or value.shape != (1,width) or value.dtype != torch.int64:
                raise ValueError(f"H3 {key} must be an int64 [1,{width}] tensor.")
            shapes.append(tuple(int(v) for v in value[0].tolist()))
        geometry = AVGeometry(*shapes)
        frames = int(data["num_frames"].item())
        fps = float(data["fps"].item())
        if fps != 24 or min(*geometry.video,*geometry.audio) <= 0:
            raise ValueError("H3 requires positive latent dimensions and 24 fps.")
        if geometry.video[:2] != (24,video_latent_num_frames(frames)) or geometry.audio != (2,32,audio_latent_num_frames(frames,fps)):
            raise ValueError("H3 paired latent lengths disagree with output frame count.")
        patch = self.net.transformer.config.patch_size
        if any(n % p for n,p in zip(geometry.video[1:],patch)):
            raise ValueError("H3 video shape must be divisible by transformer patch size.")
        # This is called only when acquiring a new trajectory, once per accumulation slot.
        if condition.video_latents:
            t = self.net.condition_video_timestep
            videos = []
            for latent in condition.video_latents:
                noise = torch.randn_like(latent, dtype=torch.float32)
                self.net.context_parallel_state.broadcast(noise)
                videos.append((t*latent.float()+(1-t)*noise).to(latent.dtype))
            condition = replace(condition, video_latents=tuple(videos))
        return H3TrainingBatch(x0=None, condition=condition, neg_condition=None, batch_size=1,
                               input_shape=list(geometry.video), segments=self.policy.schedules(), iteration=iteration,
                               teacher_condition=condition, teacher_neg_condition=None, geometry=geometry)

    def initial_state(self, batch):
        values = []
        for shape in (batch.geometry.video,batch.geometry.audio):
            noise = torch.randn((batch.batch_size,*shape),device=self.device,dtype=torch.float32)
            self.net.context_parallel_state.broadcast(noise)
            values.append(noise)
        return batch.geometry.pack(MiniMaxH3Latents(*values))

    def query_alphas(self, batch, schedule, state):
        values = super().query_alphas(batch,schedule,state)
        self.net.context_parallel_state.broadcast(values)
        return values

    def teacher_for_batch(self, batch):
        return self.query_teacher.for_batch(batch)

    def get_sampler_snapshot(self):
        return {**super().get_sampler_snapshot(), "type":"minimax_h3_joint_av_euler_student",
                "model_id":self.net.model_id, "model_revision":self.net.model_revision,
                "condition_mode":self.config.h3_condition_mode,
                "transformer_subfolder":self.net.transformer_subfolder,
                "video_shift":self.net.video_shift, "audio_shift":self.net.audio_shift,
                "condition_video_timestep":self.net.condition_video_timestep,
                "fps":self.config.h3_fps, "num_frames":self.config.h3_num_frames,
                "height":self.config.h3_height, "width":self.config.h3_width,
                "guidance_distilled":True}
