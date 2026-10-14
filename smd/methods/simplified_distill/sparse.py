# SPDX-License-Identifier: Apache-2.0
"""Sparse state boundaries and reductions; the segment objectives remain shared."""

from contextlib import nullcontext
from dataclasses import dataclass, fields
from functools import partial
import torch

from .data import TrainingBatch, SegmentState
from .distillation import SimplifiedDistillation
from .policy import SegmentPolicy
from .teacher import TeacherQuery
from .pivot.backends.sparse_utils import expand_to_voxels, is_sparse, register_sparse_pytree, sparse_feats_or_tensor


def reduce_sparse_pivot_coeffs(flow_feats: torch.Tensor, n_grid: int, channels: int) -> torch.Tensor:
    if flow_feats.ndim != 2:
        raise ValueError(
            f"Sparse pivot output must be a 2-D per-voxel tensor [N_total, C*n_grid], got {tuple(flow_feats.shape)}."
        )
    n_total, feat = flow_feats.shape
    expected = n_grid * channels
    if feat != expected:
        raise ValueError(
            f"Sparse pivot output has {feat} feature channels, expected "
            f"n_grid * C = {n_grid} * {channels} = {expected}."
        )
    return flow_feats.reshape(n_total, n_grid, channels)


@dataclass
class SparseTrainingBatch(TrainingBatch):
    structure_ref: object = None


@dataclass(frozen=True)
class SparseDistance:
    batch_map: torch.Tensor
    batch_size: int

    def __call__(self, left, right, beta=1.0, eps=1e-6):
        if left.shape != right.shape or left.shape[0] != self.batch_map.numel():
            raise ValueError("Sparse feature rows must match the shared coordinate layout.")
        per_voxel = (left.float() - right.float()).reshape(left.shape[0], -1).square().mean(1)
        sums = per_voxel.new_zeros(self.batch_size)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, self.batch_map, per_voxel)
        counts.scatter_add_(0, self.batch_map, torch.ones_like(per_voxel))
        mse = sums / counts.clamp_min(1.0)
        return mse if abs(beta - 2.0) < 1e-12 else (mse + eps).pow(beta / 2.0)


class SparseSegmentPolicy(SegmentPolicy):
    @staticmethod
    def distance_for(batch):
        return SparseDistance(batch.structure_ref.batch_boardcast_map, batch.batch_size)

    def prepare(self, batch, schedule, state, query_alphas):
        self._ensure_schedule_shift().resolve_time_shift(batch.input_shape)
        sparse = batch.structure_ref.replace(state)
        start, end = schedule.sigma_start.expand(batch.batch_size), schedule.sigma_end.expand(batch.batch_size)
        alphas = query_alphas.sort(dim=-1).values
        delta, queries = self._query_schedule(sparse, start, end, schedule.raw_start, schedule.raw_end, alphas)
        context = self._forward_segment_context(
            net=self.net,
            pivot=self.pivot,
            x=sparse,
            sigma_start=start,
            segment_size=schedule.size,
            condition=batch.condition,
            input_shape=batch.input_shape,
            return_logvar=False,
        )
        # All basis, loss and cached-trajectory state is ordinary feature tensors.
        return SegmentState(
            schedule, state, start, end, delta, queries, context, None, query_alpha=alphas.float().mean().detach()
        )

    def endpoint(self, segment):
        return (segment.x + segment.guided_context.base_velocity * segment.delta).detach()

    def _student_forward_with_net(self, *args, return_logvar, **kwargs):
        value = super()._student_forward_with_net(*args, return_logvar=return_logvar, **kwargs)
        if return_logvar:
            return sparse_feats_or_tensor(value[0]), value[1]
        return sparse_feats_or_tensor(value)

    def _broadcast_batch_scalar(self, scalar, state, batch_size):
        return expand_to_voxels(scalar, state)

    def _euler_update(self, state, velocity, delta_sigma):
        delta = expand_to_voxels(delta_sigma, state).to(state.feats.dtype)
        return state.replace(state.feats + velocity * delta)

    def _student_flow_velocity(self, *args, **kwargs):
        return sparse_feats_or_tensor(super()._student_flow_velocity(*args, **kwargs))

    @torch.no_grad()
    def sample_student(self, noise, condition, net):
        segments = self._segment_schedule(list(self.config.input_shape))
        batch_size = noise.shape[0]
        sigma_min = getattr(net.noise_scheduler, "_sigma_min", 0.0)
        t = expand_to_voxels(segments[0].sigma_start.expand(batch_size), noise)
        state = noise.replace(((sigma_min + (1.0 - sigma_min) * t) * noise.feats).to(self.precision))
        for segment in segments:
            start, end = segment.sigma_start.expand(batch_size), segment.sigma_end.expand(batch_size)
            velocity = self._student_flow_velocity(net, state, start, condition, {})
            state = self._euler_update(state, velocity, end - start)
        return state

    def _query_schedule(self, x, sigma_i, sigma_j, raw_t_i, raw_t_j, query_alphas):
        ns = self.net.noise_scheduler
        batch_size = x.shape[0]
        expected_shape = (batch_size, self._num_query_points)
        if tuple(query_alphas.shape) != expected_shape:
            raise ValueError(f"Expected query_alphas shape {expected_shape}, got {tuple(query_alphas.shape)}")

        dtype = query_alphas.dtype
        alphas = query_alphas.to(device=self.device, dtype=dtype)
        dt = self._broadcast_batch_scalar(sigma_j - sigma_i, x, batch_size).to(dtype=dtype)
        dt_safe = dt.clone()
        dt_safe[dt_safe.abs() < 1e-6] = -1e-6

        queries: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k in range(self._num_query_points):
            alpha_k_per_sample = alphas[:, k]
            if self._query_point_sampling == "sigma":
                sigma_k = sigma_i + alpha_k_per_sample * (sigma_j - sigma_i)
                sigma_k = ns.safe_clamp(sigma_k.to(dtype=ns.t_precision), min=ns.min_t, max=ns.max_t)
                sigma_alpha_k = self._broadcast_batch_scalar(alpha_k_per_sample, x, batch_size).to(dtype=dtype)
            else:
                raw_t_k = raw_t_i + alpha_k_per_sample * (raw_t_j - raw_t_i)
                sigma_k = ns.safe_clamp(
                    ns.warp_t(raw_t_k).to(dtype=ns.t_precision),
                    min=ns.min_t,
                    max=ns.max_t,
                )
                sigma_k_nd = self._broadcast_batch_scalar(sigma_k, x, batch_size).to(dtype=dtype)
                sigma_i_nd = self._broadcast_batch_scalar(sigma_i, x, batch_size).to(dtype=dtype)
                sigma_alpha_k = (sigma_k_nd - sigma_i_nd) / dt_safe
            queries.append((sigma_k, sigma_alpha_k))
        return dt, queries

    def _build_coeffs_from_pivot(
        self,
        net,
        pivot,
        backend,
        hook_data,
        x_t,
        sigma_t,
        seg_size,
        input_shape,
    ) -> dict:
        if not is_sparse(x_t):
            raise TypeError("Sparse pivot coefficient construction requires a SparseTensor state.")
        del input_shape

        ns = net.noise_scheduler
        raw_tokens = backend.decode(pivot, hook_data)
        flow_feats = sparse_feats_or_tensor(raw_tokens)
        if net.net_pred_type != "flow":
            flow_feats = ns.convert_model_output(
                x_t.feats,
                flow_feats,
                sigma_t,
                src_pred_type=net.net_pred_type,
                target_pred_type="flow",
            )
        channels = x_t.feats.shape[-1]
        flow_reshaped = reduce_sparse_pivot_coeffs(flow_feats, self._n_grid, channels)
        return self.basis.build_coeffs(flow_reshaped, x_t.feats, sigma_t, seg_size)


class SparseTeacherQuery(TeacherQuery):
    @torch.no_grad()
    def query_features(self, structure, features, sigma, condition, negative):
        values = self._teacher_velocities(structure.replace(features.detach()), sigma, condition, negative)
        result = tuple(sparse_feats_or_tensor(value) if value is not None else None for value in values)
        if any(value is not None and not torch.isfinite(value).all() for value in result[:2]):
            raise RuntimeError("Non-finite sparse teacher velocity.")
        return result

    def _broadcast_batch_scalar(self, scalar, state, batch_size):
        return expand_to_voxels(scalar, state)

    @torch.no_grad()
    def _teacher_velocities(self, x, sigma, condition, negative):
        x = x.to(dtype=self.precision)
        conditional = self.teacher(x, sigma, condition=condition, fwd_pred_type="flow")
        conditional = conditional[0] if isinstance(conditional, list) else conditional
        target_scale, reference_scale = self.target_scale, self.reference_scale
        if self.config.guidance_interval is not None:
            low, high = self.config.guidance_interval
            inside = (sigma >= low) & (sigma <= high)
            target_scale = torch.where(inside, torch.full_like(sigma, target_scale), torch.ones_like(sigma))
            reference_scale = torch.where(inside, torch.full_like(sigma, reference_scale), torch.ones_like(sigma))
        need_negative = (self.need_target and self.target_scale != 1) or (
            self.need_reference and self.reference_scale != 1
        )
        if need_negative and negative is None:
            raise ValueError("Sparse teacher CFG requires a negative image condition.")
        if negative is None:
            unconditional = conditional
        else:
            unconditional = self.teacher(x, sigma, condition=negative, fwd_pred_type="flow")
            unconditional = unconditional[0] if isinstance(unconditional, list) else unconditional
        target = self._cfg_combine(conditional, unconditional, target_scale) if self.need_target else None
        reference = self._cfg_combine(conditional, unconditional, reference_scale) if self.need_reference else None
        if self.config.guidance_rescale > 0 and need_negative:
            if target is not None:
                target = self._apply_guidance_rescale(x, conditional, target, sigma, self.config.guidance_rescale)
            if reference is not None:
                reference = self._apply_guidance_rescale(x, conditional, reference, sigma, self.config.guidance_rescale)
        return target, reference, conditional, unconditional

    def _cfg_combine(self, out_cond, out_uncond, cfg_scale):
        if torch.is_tensor(cfg_scale):
            scale = self._broadcast_batch_scalar(cfg_scale.to(self.device), out_cond, cfg_scale.shape[0])
            comb = out_uncond + scale * (out_cond - out_uncond)
        else:
            comb = out_uncond + float(cfg_scale) * (out_cond - out_uncond)
        if not self.config.cfg_norm_rescale:
            return comb
        rescale = getattr(self.teacher, "rescale_cfg_velocity", None)
        if not callable(rescale):
            raise RuntimeError(f"cfg_norm_rescale=True requires {type(self.teacher).__name__}.rescale_cfg_velocity().")
        return rescale(out_cond, comb)

    def _apply_guidance_rescale(self, x_t, out_cond, result, t, weight):
        if not is_sparse(x_t) or not is_sparse(out_cond) or not is_sparse(result):
            raise TypeError("Sparse guidance rescale requires SparseTensor inputs.")
        batch_size = x_t.shape[0]
        batch_map = x_t.batch_boardcast_map
        t_vox = expand_to_voxels(t, x_t).to(torch.float64)
        sigma_min = getattr(self.teacher.noise_scheduler, "_sigma_min", 0.0)

        def _flow_to_x0(velocity):
            x_t_f64 = x_t.feats.to(torch.float64)
            velocity_f64 = velocity.feats.to(torch.float64)
            numerator = (1.0 - sigma_min) * x_t_f64 - (sigma_min + (1.0 - sigma_min) * t_vox) * velocity_f64
            denominator = (1.0 - sigma_min * t_vox).clamp_min(1e-8)
            return (numerator / denominator).to(x_t.feats.dtype)

        def _per_sample_std(features):
            channels = features.shape[-1]
            flat = features.to(torch.float64).reshape(-1)
            expanded_batch_map = batch_map.unsqueeze(-1).expand(-1, channels).reshape(-1)
            sample_sum = torch.zeros(batch_size, device=features.device, dtype=torch.float64)
            sample_sum_sq = torch.zeros_like(sample_sum)
            sample_sum.scatter_add_(0, expanded_batch_map, flat)
            sample_sum_sq.scatter_add_(0, expanded_batch_map, flat.square())
            counts = torch.bincount(batch_map.to(torch.long), minlength=batch_size).to(torch.float64) * channels
            mean = sample_sum / counts.clamp_min(1.0)
            variance = (sample_sum_sq / counts.clamp_min(1.0) - mean.square()).clamp_min(0.0)
            return variance.sqrt().clamp_min(1e-8).to(features.dtype)[batch_map].unsqueeze(-1)

        x0_cond = _flow_to_x0(out_cond)
        x0_cfg = _flow_to_x0(result)
        x0_blend = (
            weight * x0_cfg.float() * (_per_sample_std(x0_cond) / _per_sample_std(x0_cfg))
            + (1.0 - weight) * x0_cfg.float()
        )

        x0_f64 = x0_blend.to(torch.float64)
        x_t_f64 = x_t.feats.to(torch.float64)
        epsilon_denominator = (sigma_min + (1.0 - sigma_min) * t_vox).clamp_min(1e-8)
        epsilon = (x_t_f64 - (1.0 - t_vox) * x0_f64) / epsilon_denominator
        features = ((1.0 - sigma_min) * epsilon - x0_f64).to(x_t.feats.dtype)
        return result.replace(features)


class SparseSimplifiedDistillation(SimplifiedDistillation):
    policy_class = SparseSegmentPolicy
    teacher_query_class = SparseTeacherQuery

    def __init__(self, config):
        register_sparse_pytree()
        self._elastic_controller = None
        super().__init__(config)

    def build_model(self):
        if self.config.no_pivot or self.config.embedded_guidance_scale is not None:
            raise ValueError("Sparse distillation currently requires Pivot and explicit CFG.")
        if self.config.query_alpha_mode != "random":
            raise ValueError("Sparse distillation currently supports random query alphas.")
        super().build_model()
        if hasattr(self.net.model, "register_memory_controller"):
            from smd.utils.elastic_memory import LinearMemoryController

            self._elastic_controller = LinearMemoryController(target_ratio=0.8, buffer_size=1000, update_every=100)
            self.net.model.register_memory_controller(self._elastic_controller)

    def prepare_batch(self, data, iteration):
        batch = super().prepare_batch(data, iteration)
        structure = data.get("real")
        if not is_sparse(structure) or structure.shape[0] != batch.batch_size:
            raise ValueError("Sparse distillation requires a real SparseTensor coordinate template for each condition.")
        return SparseTrainingBatch(
            **{f.name: getattr(batch, f.name) for f in fields(TrainingBatch)}, structure_ref=structure
        )

    def initial_state(self, batch):
        template = batch.structure_ref
        sigma_min = getattr(self.net.noise_scheduler, "_sigma_min", 0.0)
        t = expand_to_voxels(batch.segments[0].sigma_start.expand(batch.batch_size), template)
        return ((sigma_min + (1.0 - sigma_min) * t) * torch.randn_like(template.feats)).to(self.precision)

    def teacher_for_batch(self, batch):
        return partial(self.query_teacher.query_features, batch.structure_ref)

    def single_train_step(self, *args, **kwargs):
        context = self._elastic_controller.record() if self._elastic_controller is not None else nullcontext()
        with context:
            return super().single_train_step(*args, **kwargs)
