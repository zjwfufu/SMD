# SPDX-License-Identifier: Apache-2.0
"""Causal chunk rollout/replay with the shared independent objectives and logging."""

from dataclasses import dataclass
from typing import Any
import torch

from smd.networks.network import CausalSMDNetwork
from .cross import CrossSimplifiedDistillation
from .data import SegmentContext, SegmentState
from .policy import SegmentPolicy
from .teacher import TeacherQuery
from .segment import SegmentObjective
from .execution import TrajectoryBatch, backward_substep
from .metrics import reduce_metrics


@dataclass
class CausalRollout:
    final_chunks: tuple[torch.Tensor, ...]
    cache_chunks: tuple[torch.Tensor, ...]
    segment_states: tuple[torch.Tensor, ...]
    segment_end_states: tuple[torch.Tensor, ...]
    query_states: tuple[tuple[torch.Tensor, ...], ...]
    query_sigmas: tuple[torch.Tensor, ...]
    query_alphas: torch.Tensor
    segment_index: int
    initial_noise: torch.Tensor
    transition_noises: tuple[tuple[torch.Tensor, ...], ...]


def causal_chunk_ranges(num_frames: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    """Match Self-Forcing's front-loaded remainder chunking."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    num_chunks, remainder = divmod(num_frames, chunk_size)
    if num_chunks == 0:
        return ((0, num_frames),)

    ranges = [(0, chunk_size + remainder)]
    start = chunk_size + remainder
    while start < num_frames:
        ranges.append((start, min(start + chunk_size, num_frames)))
        start += chunk_size
    return tuple(ranges)


def selected_segment(iteration, count):
    return (iteration * 2654435761 + 1013904223) % count


def causal_forward(net, x, sigma, condition, *, precision, start, store_kv, cache_tag="pos", skip_layers=None):
    kwargs = {"skip_layers": skip_layers} if skip_layers is not None else {}
    output = net(
        x.to(dtype=precision),
        sigma,
        condition=condition,
        fwd_pred_type="flow",
        cache_tag=cache_tag,
        cur_start_frame=start,
        store_kv=store_kv,
        is_ar=True,
        **kwargs,
    )
    return output[0] if isinstance(output, (list, tuple)) else output


class CausalSegmentPolicy(SegmentPolicy):
    def _causal_forward(self, net, x, sigma, condition, **kwargs):
        return causal_forward(net, x, sigma, condition, precision=self.precision, **kwargs)

    def loss_velocities(self, student, target, reference):
        if not self.config.causal_direction_only_loss:
            return student, target, reference

        def normalize(value):
            if value is None:
                return None
            scale = (
                value.float().reshape(value.shape[0], -1).square().mean(1).add(self.config.velocity_distance_eps).sqrt()
            )
            return value.float() / scale.reshape(value.shape[0], *([1] * (value.ndim - 1)))

        return tuple(normalize(value) for value in (student, target, reference))

    def live_segment(self, batch, rollout, chunk_index, ranges):
        self._rebuild_prefix_cache(batch, rollout, chunk_index, ranges)
        state = rollout.segment_states[chunk_index]
        schedule = batch.segments[rollout.segment_index]
        start, end = schedule.sigma_start.expand(batch.batch_size), schedule.sigma_end.expand(batch.batch_size)
        context = self._causal_segment_context(
            x=state,
            sigma_start=start,
            segment_size=schedule.size,
            condition=batch.condition,
            start=ranges[chunk_index][0],
        )
        _, queries = self._query_schedule(state, start, end, schedule.raw_start, schedule.raw_end, rollout.query_alphas)
        # Causal rollout/replay has historically kept the segment delta in float64.
        delta = (end - start).reshape(batch.batch_size, 1, 1, 1, 1)
        return SegmentState(
            schedule,
            state,
            start,
            end,
            delta,
            queries,
            context,
            None,
            query_alpha=rollout.query_alphas.float().mean().detach(),
        )

    @torch.no_grad()
    def sample_student(self, noise, condition, net, generator=None):
        schedules = self._segment_schedule(list(noise.shape[1:]))
        ranges = causal_chunk_ranges(noise.shape[2], net.chunk_size)
        chunks = []
        net.clear_caches()
        try:
            for start, end in ranges:
                initial = noise[:, :, start:end]
                state = net.noise_scheduler.latents(noise=initial, t_init=schedules[0].sigma_start).to(noise.dtype)
                fixed = initial if self.config.causal_renoise_mode == "fixed" else None
                for schedule in schedules:
                    sigma_start, sigma_end = (
                        schedule.sigma_start.expand(noise.shape[0]),
                        schedule.sigma_end.expand(noise.shape[0]),
                    )
                    velocity = self._causal_forward(net, state, sigma_start, condition, start=start, store_kv=False)
                    state = self._causal_transition(
                        net, state, velocity, sigma_start, sigma_end, generator=generator, renoise_noise=fixed
                    )
                chunks.append(state)
                cached, sigma = self._cache_input(
                    net, state, sigma_dtype=schedules[0].sigma_start.dtype, generator=generator
                )
                self._causal_forward(net, cached, sigma, condition, start=start, store_kv=True)
        finally:
            net.clear_caches()
        return torch.cat(chunks, dim=2)

    def _causal_segment_context(
        self,
        *,
        x: torch.Tensor,
        sigma_start: torch.Tensor,
        segment_size: float,
        condition: Any,
        start: int,
    ) -> SegmentContext:
        with self.backend.capture_pivot_features(self.net, self._hook_layer) as capture:
            base_velocity = self._causal_forward(
                self.net,
                x,
                sigma_start,
                condition,
                start=start,
                store_kv=False,
            )
            pivot_coeffs = self._build_coeffs_from_pivot(
                self.net,
                self.pivot,
                self.backend,
                capture.result,
                x,
                sigma_start,
                torch.full_like(sigma_start, segment_size),
                list(x.shape[1:]),
            )
        return SegmentContext(base_velocity=base_velocity, pivot_coeffs=pivot_coeffs)

    def _causal_transition(
        self,
        net: CausalSMDNetwork,
        state: torch.Tensor,
        velocity: torch.Tensor,
        sigma_start: torch.Tensor,
        sigma_end: torch.Tensor,
        generator: torch.Generator | None = None,
        renoise_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if str(self.config.causal_transition_type) == "deterministic":
            spatial = [1] * (state.ndim - 1)
            delta = (sigma_end - sigma_start).reshape(state.shape[0], *spatial)
            return (state.double() + delta.double() * velocity.double()).to(state.dtype)

        x0 = net.noise_scheduler.convert_model_output(
            state,
            velocity,
            sigma_start,
            src_pred_type="flow",
            target_pred_type="x0",
        )
        if renoise_noise is None:
            renoise_noise = torch.randn(
                x0.shape,
                device=x0.device,
                dtype=x0.dtype,
                generator=generator,
            )
        elif renoise_noise.shape != x0.shape:
            raise ValueError(
                f"renoise_noise shape {tuple(renoise_noise.shape)} does not match state shape {tuple(x0.shape)}."
            )
        return net.noise_scheduler.forward_process(
            x0,
            renoise_noise.to(device=x0.device, dtype=x0.dtype),
            sigma_end,
        ).to(dtype=state.dtype)

    def _cache_input(
        self,
        net: CausalSMDNetwork,
        clean_chunk: torch.Tensor,
        *,
        sigma_dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_noise = float(self.config.causal_cache_noise)
        cache_sigma = torch.full(
            (clean_chunk.shape[0],),
            cache_noise,
            device=clean_chunk.device,
            dtype=sigma_dtype,
        )
        if cache_noise == 0.0:
            return clean_chunk, cache_sigma
        noise = torch.randn(
            clean_chunk.shape,
            device=clean_chunk.device,
            dtype=clean_chunk.dtype,
            generator=generator,
        )
        cache_chunk = net.noise_scheduler.forward_process(clean_chunk, noise, cache_sigma)
        return cache_chunk.to(dtype=clean_chunk.dtype), cache_sigma

    @torch.no_grad()
    def _rollout_causal_student(
        self,
        batch: Any,
        iteration: int,
        *,
        forced_segment_index: int | None = None,
        forced_query_alphas: torch.Tensor | None = None,
        forced_noise: torch.Tensor | None = None,
        forced_transition_noises: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        generator: torch.Generator | None = None,
    ) -> CausalRollout:
        segments = batch.segments
        segment_index = (
            int(forced_segment_index)
            if forced_segment_index is not None
            else selected_segment(iteration, len(segments))
        )
        if not 0 <= segment_index < len(segments):
            raise ValueError(f"segment_index must be in [0, {len(segments)}), got {segment_index}.")
        if forced_query_alphas is None:
            query_batch_size = batch.batch_size
            sampled_query_alphas = (
                torch.rand(
                    query_batch_size,
                    self._num_query_points,
                    device=self.device,
                    dtype=self.precision,
                )
                .sort(dim=-1)
                .values
            )
            query_alphas = (
                sampled_query_alphas.expand(batch.batch_size, -1) if query_batch_size == 1 else sampled_query_alphas
            )
        else:
            expected_shape = (batch.batch_size, self._num_query_points)
            if tuple(forced_query_alphas.shape) != expected_shape:
                raise ValueError(
                    f"forced_query_alphas must have shape {expected_shape}, got {tuple(forced_query_alphas.shape)}."
                )
            query_alphas = forced_query_alphas.to(device=self.device, dtype=self.precision)
        if forced_noise is None:
            trajectory_generator = generator
            if self.config.causal_fixed_trajectory_seed is not None and generator is None:
                trajectory_generator = torch.Generator(device=self.device).manual_seed(
                    int(self.config.causal_fixed_trajectory_seed)
                )
            noise = torch.randn(
                (batch.batch_size, *batch.input_shape),
                device=self.device,
                dtype=self.precision,
                generator=trajectory_generator,
            )
        else:
            trajectory_generator = generator
            expected_shape = (batch.batch_size, *batch.input_shape)
            if tuple(forced_noise.shape) != expected_shape:
                raise ValueError(f"forced_noise must have shape {expected_shape}, got {tuple(forced_noise.shape)}.")
            noise = forced_noise.to(device=self.device, dtype=self.precision)
        chunk_ranges = causal_chunk_ranges(batch.input_shape[1], int(self.net.chunk_size))
        final_chunks: list[torch.Tensor] = []
        cache_chunks: list[torch.Tensor] = []
        transition_noises: list[tuple[torch.Tensor, ...]] = []
        segment_states: list[torch.Tensor] = []
        segment_end_states: list[torch.Tensor] = []
        query_states: list[list[torch.Tensor]] = [[] for _ in range(self._num_query_points)]
        query_sigmas: tuple[torch.Tensor, ...] | None = None

        self.net.clear_caches()
        try:
            for chunk_index, (start, end) in enumerate(chunk_ranges):
                initial_chunk_noise = noise[:, :, start:end]
                state = self.net.noise_scheduler.latents(
                    noise=initial_chunk_noise,
                    t_init=segments[0].sigma_start,
                ).to(dtype=self.precision)
                fixed_renoise_noise = initial_chunk_noise if str(self.config.causal_renoise_mode) == "fixed" else None
                chunk_transition_noises: list[torch.Tensor] = []

                for current_index, schedule in enumerate(segments):
                    is_selected_segment = current_index == segment_index
                    sigma_start = schedule.sigma_start.expand(batch.batch_size)
                    sigma_end = schedule.sigma_end.expand(batch.batch_size)
                    if is_selected_segment:
                        segment_states.append(state.detach().clone())
                        context = self._causal_segment_context(
                            x=state,
                            sigma_start=sigma_start,
                            segment_size=schedule.size,
                            condition=batch.condition,
                            start=start,
                        )
                        _, queries = self._query_schedule(
                            state,
                            sigma_start,
                            sigma_end,
                            schedule.raw_start,
                            schedule.raw_end,
                            query_alphas,
                        )
                        if query_sigmas is None:
                            query_sigmas = tuple(sigma for sigma, _ in queries)
                        for query_index, (_sigma, sigma_alpha) in enumerate(queries):
                            query_states[query_index].append(
                                self.basis.trajectory_state(
                                    state,
                                    (sigma_end - sigma_start).reshape(batch.batch_size, 1, 1, 1, 1),
                                    context.base_velocity,
                                    context.pivot_coeffs,
                                    sigma_alpha,
                                ).detach()
                            )
                        velocity = context.base_velocity
                    else:
                        velocity = self._causal_forward(
                            self.net,
                            state,
                            sigma_start,
                            batch.condition,
                            start=start,
                            store_kv=False,
                        )
                    if forced_transition_noises is not None:
                        transition_noise = forced_transition_noises[chunk_index][current_index].to(
                            device=state.device,
                            dtype=state.dtype,
                        )
                        if transition_noise.shape != state.shape:
                            raise ValueError(
                                "forced transition noise shape mismatch at "
                                f"chunk={chunk_index}, segment={current_index}: "
                                f"expected {tuple(state.shape)}, got {tuple(transition_noise.shape)}."
                            )
                    else:
                        transition_noise = (
                            fixed_renoise_noise
                            if fixed_renoise_noise is not None
                            else torch.randn(
                                state.shape,
                                device=state.device,
                                dtype=state.dtype,
                                generator=trajectory_generator,
                            )
                        )
                    chunk_transition_noises.append(transition_noise.detach().clone())
                    state = self._causal_transition(
                        self.net,
                        state,
                        velocity,
                        sigma_start,
                        sigma_end,
                        renoise_noise=transition_noise,
                    ).detach()
                    if is_selected_segment:
                        segment_end_states.append(state.detach().clone())

                final_chunk = state.detach()
                final_chunks.append(final_chunk)
                transition_noises.append(tuple(chunk_transition_noises))
                cache_chunk, cache_sigma = self._cache_input(
                    self.net,
                    final_chunk,
                    sigma_dtype=segments[0].sigma_start.dtype,
                    generator=trajectory_generator,
                )
                cache_chunks.append(cache_chunk.detach())
                self._causal_forward(
                    self.net,
                    cache_chunk,
                    cache_sigma,
                    batch.condition,
                    start=start,
                    store_kv=True,
                )
        finally:
            self.net.clear_caches()

        if query_sigmas is None:
            raise RuntimeError("Causal Wan rollout did not visit the selected diffusion segment.")
        return CausalRollout(
            final_chunks=tuple(final_chunks),
            cache_chunks=tuple(cache_chunks),
            segment_states=tuple(segment_states),
            segment_end_states=tuple(segment_end_states),
            query_states=tuple(tuple(states) for states in query_states),
            query_sigmas=query_sigmas,
            query_alphas=query_alphas,
            segment_index=segment_index,
            initial_noise=noise.detach(),
            transition_noises=tuple(transition_noises),
        )

    @torch.no_grad()
    def _rebuild_prefix_cache(
        self,
        batch: Any,
        rollout: CausalRollout,
        chunk_index: int,
        chunk_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        self.net.clear_caches()
        for prefix_index in range(chunk_index):
            start, _end = chunk_ranges[prefix_index]
            self._causal_forward(
                self.net,
                rollout.cache_chunks[prefix_index],
                torch.full(
                    (batch.batch_size,),
                    float(self.config.causal_cache_noise),
                    device=self.device,
                    dtype=batch.segments[0].sigma_start.dtype,
                ),
                batch.condition,
                start=start,
                store_kv=True,
            )


class CausalTeacherQuery(TeacherQuery):
    def targets(self, batch, rollout, selected, ranges):
        if not self.need_target and not self.need_reference:
            return {index: [(None, None, None, None)] * len(rollout.query_sigmas) for index in selected}
        if self.config.causal_teacher_type == "causal":
            return self.causal_targets(batch, rollout, selected, ranges)
        result = {index: [] for index in selected}
        for q, sigma in enumerate(rollout.query_sigmas):
            state = torch.cat(rollout.query_states[q], dim=2)
            values = self(state, sigma, batch.teacher_condition, batch.teacher_neg_condition)
            for index in selected:
                start, end = ranges[index]
                result[index].append(tuple(v[:, :, start:end].detach() if v is not None else None for v in values))
        return result

    @torch.no_grad()
    def causal_targets(self, batch, rollout, selected, ranges):
        result = {index: [] for index in selected}
        condition, negative = batch.teacher_condition, batch.teacher_neg_condition
        need_negative = (self.need_target and self.target_scale != 1) or (
            self.need_reference and self.reference_scale != 1
        )
        if need_negative and negative is None:
            raise ValueError("Causal teacher CFG requires a negative condition.")
        self.teacher.clear_caches()
        try:
            for index, (start, _) in enumerate(ranges):
                if index in result:
                    for q, sigma in enumerate(rollout.query_sigmas):
                        x = rollout.query_states[q][index]
                        positive = causal_forward(
                            self.teacher, x, sigma, condition, precision=self.precision, start=start, store_kv=False
                        )
                        neg = (
                            causal_forward(
                                self.teacher,
                                x,
                                sigma,
                                negative,
                                precision=self.precision,
                                start=start,
                                store_kv=False,
                                cache_tag="neg",
                                skip_layers=self.config.skip_layers,
                            )
                            if need_negative
                            else positive
                        )
                        target = self._cfg_combine(positive, neg, self.target_scale) if self.need_target else None
                        reference = (
                            self._cfg_combine(positive, neg, self.reference_scale) if self.need_reference else None
                        )
                        result[index].append((target, reference, positive, neg))
                zero = torch.zeros(batch.batch_size, device=self.device, dtype=batch.segments[0].sigma_start.dtype)
                for tag, cond, skip in [("pos", condition, None), ("neg", negative, self.config.skip_layers)]:
                    if cond is not None:
                        causal_forward(
                            self.teacher,
                            rollout.final_chunks[index],
                            zero,
                            cond,
                            precision=self.precision,
                            start=start,
                            store_kv=True,
                            cache_tag=tag,
                            skip_layers=skip,
                        )
        finally:
            self.teacher.clear_caches()
        return result


class ReplayTeacher:
    def __init__(self, targets):
        self.targets = iter(targets)

    def __call__(self, state, sigma, condition, negative):
        return next(self.targets)


class CausalSimplifiedDistillation(CrossSimplifiedDistillation):
    policy_class = CausalSegmentPolicy
    teacher_query_class = CausalTeacherQuery

    def __init__(self, config):
        super().__init__(config)
        # Temporal chunk replay retains its own rollout protocol, with fresh conditions per update.
        self.training_batch_reuse.length = 1

    def build_model(self):
        if self.config.causal_objective_mode not in {"velocity", "hybrid_causal_correction"}:
            raise ValueError("Unknown causal objective mode.")
        if self.config.causal_fixed_trajectory_seed is not None and self.config.causal_fixed_trajectory_seed < 0:
            raise ValueError("causal_fixed_trajectory_seed must be nonnegative.")
        if self.config.no_pivot or not self.config.net.use_lora:
            raise ValueError("Causal distillation requires a LoRA student and Pivot.")
        if self.config.causal_teacher_type not in {"causal", "bidirectional"}:
            raise ValueError("Unknown causal teacher type.")
        if self.config.causal_transition_type not in {
            "stochastic",
            "deterministic",
        } or self.config.causal_renoise_mode not in {"fresh", "fixed"}:
            raise ValueError("Invalid causal transition/re-noise configuration.")
        if not 0 <= self.config.causal_cache_noise <= 1 or self.config.causal_replay_chunks_per_step < 1:
            raise ValueError("Invalid causal cache noise or chunk replay count.")
        super().build_model()
        if not isinstance(self.net, CausalSMDNetwork):
            raise TypeError("Causal distillation requires a causal student network.")
        if self.net.total_num_frames != self.input_shape[1]:
            raise ValueError("Causal network frame count and training shape disagree.")
        if (self.config.causal_teacher_type == "causal") != isinstance(self.teacher, CausalSMDNetwork):
            raise TypeError("Teacher network and causal_teacher_type disagree.")

    def build_teacher_for_distill(self):
        super().build_teacher_for_distill()
        if self.config.causal_teacher_type == "causal":
            if not self.config.causal_teacher_checkpoint:
                raise ValueError("A causal teacher checkpoint is required.")
            self._load_pretrained_model(self.teacher, self.config.causal_teacher_checkpoint, fsdp_meta_init=False)
            self.teacher.eval().requires_grad_(False)
        if self.config.causal_objective_mode == "hybrid_causal_correction":
            if self.config.causal_correction_reference_mode != "strong_cfg_residual":
                raise ValueError("Hybrid correction requires strong_cfg_residual.")
            self.correction_strong_teacher = self._coordinated_instantiate(
                self.config.causal_correction_strong_teacher, all_rank_read=True
            )
            self.correction_strong_teacher.eval().requires_grad_(False)
            self.correction_strong_teacher.to(dtype=self.precision_fsdp)
            for name in ("text_encoder", "vae"):
                if hasattr(self.net, name):
                    setattr(self.correction_strong_teacher, name, getattr(self.net, name))

    @property
    def fsdp_dict(self):
        modules = super().fsdp_dict
        if hasattr(self, "correction_strong_teacher"):
            modules["correction_strong_teacher"] = self.correction_strong_teacher
        return modules

    @property
    def fsdp_no_sync_keys(self):
        keys = set(super().fsdp_no_sync_keys or ())
        if hasattr(self, "correction_strong_teacher"):
            keys.add("correction_strong_teacher")
        return keys or None

    @torch.no_grad()
    def correction_targets(self, batch, rollout, selected, ranges):
        if self.config.causal_objective_mode == "velocity":
            return None
        from .correction import strong_cfg_residual_targets
        result = {index: [] for index in selected}
        for q, sigma in enumerate(rollout.query_sigmas):
            state = torch.cat(rollout.query_states[q], dim=2)
            strong, reference = strong_cfg_residual_targets(
                self.correction_strong_teacher, state, sigma,
                batch.teacher_condition, batch.teacher_neg_condition,
                precision=self.precision, skip_layers=self.config.causal_correction_strong_skip_layers,
            )
            for index in selected:
                start, end = ranges[index]
                result[index].append((strong[:, :, start:end], reference[:, :, start:end]))
        return result

    def correction_objective(self, batch, segment, targets):
        from .correction import paired_teacher_state_correction_loss
        terms, gaps, directions = [], [], []
        for (_, alpha), (strong, reference) in zip(segment.queries, targets, strict=True):
            proxy = self.policy._context_proxy_state_for_loss(segment, segment.guided_context, alpha)
            loss, gap, direction = paired_teacher_state_correction_loss(
                proxy, reference, strong,
                normalize=self.config.causal_asymmetric_correction_normalize,
                eps=self.config.causal_asymmetric_correction_eps,
            )
            terms.append(loss); gaps.append(gap); directions.append(direction)
        return torch.stack(terms).mean(), {
            "diagnostic/correction/velocity_gap_rms": torch.stack(gaps).mean(),
            "diagnostic/correction/direction_rms": torch.stack(directions).mean(),
        }

    def single_train_step(
        self,
        data,
        iteration,
        *,
        per_segment_backward=False,
        grad_accum_rounds=1,
        sync_grads=True,
        is_fsdp=False,
        on_backward_begin=None,
        accumulation_index=0,
    ):
        training = torch.is_grad_enabled()
        if training and not per_segment_backward:
            raise ValueError("Causal replay requires per-chunk backward.")
        if training:
            data = self.training_batch_reuse.acquire(iteration, accumulation_index, lambda: data)
            entry = self.training_batch_reuse.batches[accumulation_index]
        else:
            entry = TrajectoryBatch(data)
        batch = self.prepare_batch(entry.data, iteration)
        ranges = causal_chunk_ranges(batch.input_shape[1], self.net.chunk_size)
        count = self.config.causal_replay_chunks_per_step
        if count > len(ranges):
            raise ValueError("Requested more replay chunks than the video contains.")
        selected = (
            tuple(range(len(ranges)))
            if count == len(ranges)
            else tuple((iteration + i) % len(ranges) for i in range(count))
        )
        rollout = self.policy._rollout_causal_student(batch, iteration)
        targets = self.query_teacher.targets(batch, rollout, selected, ranges)
        corrections = self.correction_targets(batch, rollout, selected, ranges)
        losses, statistics = [], []
        try:
            for position, chunk in enumerate(selected):
                segment = self.policy.live_segment(batch, rollout, chunk, ranges)
                loss, metrics = self.objective(self.policy, ReplayTeacher(targets[chunk]), batch, segment)
                if corrections is not None:
                    correction, diagnostics = self.correction_objective(batch, segment, corrections[chunk])
                    weighted = self.objective.weights.normalization * self.config.imitation_weight * self.config.causal_asymmetric_correction_scale * correction
                    loss = loss + weighted
                    metrics.update(diagnostics)
                    metrics["loss_raw/correction"] = correction.detach()
                    metrics["loss/correction"] = weighted.detach()
                    metrics["weights/correction"] = weighted.new_tensor(self.config.causal_asymmetric_correction_scale)
                    metrics["enabled/correction"] = weighted.new_tensor(float(self.config.causal_asymmetric_correction_scale > 0))
                scaled = loss / len(selected)
                if training:
                    backward_substep(
                        self,
                        scaled,
                        position,
                        len(selected),
                        grad_accum_rounds=grad_accum_rounds,
                        sync_grads=sync_grads,
                        is_fsdp=is_fsdp,
                        on_backward_begin=on_backward_begin,
                    )
                losses.append(scaled.detach())
                metrics.update(
                    {
                        f"chunks/{chunk}/{key}": value
                        for key, value in list(metrics.items())
                        if key.startswith("loss_raw/")
                    }
                )
                statistics.append(metrics)
                del segment, loss, scaled
        finally:
            self.net.clear_caches()
        if training:
            self.training_batch_reuse.stage(accumulation_index, rollout.final_chunks[-1])
        result = reduce_metrics(statistics)
        result["total_loss"] = torch.stack(losses).sum()
        result["progress/segment_index"] = result["total_loss"].new_tensor(rollout.segment_index)
        result["enabled/pivot"] = result["total_loss"].new_tensor(float(not self.config.no_pivot))
        result["diagnostic/causal/replayed_chunks"] = result["total_loss"].new_tensor(len(selected))
        return result, {"selected_chunk_indices": selected, "is_unconditional_step": False}

    def sample_student_with_generator(self, noise, condition, net, generator):
        return self.policy.sample_student(noise, condition, net, generator=generator)

    def next_segment_index(self, iteration):
        return selected_segment(iteration + 1, self.config.student_sample_steps)

    def get_sampler_snapshot(self):
        snapshot = super().get_sampler_snapshot()
        snapshot.update(
            type="causal_wan_student",
            chunk_size=self.net.chunk_size,
            transition=self.config.causal_transition_type,
            renoise=self.config.causal_renoise_mode,
            cache_noise=self.config.causal_cache_noise,
            execution="causal_chunk_replay",
        )
        return snapshot
