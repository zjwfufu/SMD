# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Few-step on-policy distillation with independent losses and explicit trajectory cycles."""

from dataclasses import replace
import torch
from smd.networks.io_contract import NetworkIOAdapter
from .data import TrainingBatch
from .sampling import sample_query_alphas
from .model import StudentPivotModel
from .policy import SegmentPolicy
from .teacher import TeacherQuery
from .segment import SegmentObjective
from .execution import TrajectoryCycle, TrajectoryBatch, backward_substep
from .metrics import reduce_metrics


class SimplifiedDistillation(StudentPivotModel):
    method_name = "simplified_distill"
    policy_class = SegmentPolicy
    teacher_query_class = TeacherQuery
    objective_class = SegmentObjective

    def __init__(self, config):
        super().__init__(config)
        self.io_adapter = NetworkIOAdapter.from_model(self.net)
        self.policy = self.policy_class(
            self.net, self.pivot, self.backend, self.basis, config, self.device, self.precision
        )
        self.query_teacher = self.teacher_query_class(self.teacher, config, self.device, self.precision)
        self.objective = self.objective_class(config)
        self.training_batch_reuse = TrajectoryCycle(
            config.student_sample_steps,
            self.io_adapter.batch_size,
        )

    def requires_internal_backward(self, iteration):
        return bool(self.config.per_segment_backward)

    def prepare_batch(self, data, iteration):
        condition = self.io_adapter.pack_condition(data["condition"], data)
        negative = data.get("neg_condition")
        negative = self.io_adapter.pack_condition(negative, data) if negative is not None else None
        shape = self.io_adapter.input_shape_from_data(data, default_input_shape=self.input_shape)
        return TrainingBatch(
            x0=None,
            condition=condition,
            neg_condition=negative,
            batch_size=self.io_adapter.batch_size(condition),
            input_shape=shape,
            segments=self.policy._segment_schedule(shape),
            iteration=iteration,
            teacher_condition=condition,
            teacher_neg_condition=negative,
            route=data.get("route"),
        )

    def initial_state(self, batch):
        noise = torch.randn((batch.batch_size, *batch.input_shape), device=self.device, dtype=self.precision)
        return self.net.noise_scheduler.latents(noise=noise, t_init=batch.segments[0].sigma_start).to(self.precision)

    def train_segment(self, entry, iteration, segment_index, *, backward=None):
        batch = replace(entry.prepared, iteration=iteration)
        teacher = self.teacher_for_batch(batch)
        state = entry.state if entry.state is not None else self.initial_state(batch)
        schedule = batch.segments[segment_index]
        alphas = self.query_alphas(batch, schedule, state)
        segment = self.policy.prepare(batch, schedule, state, alphas)
        loss, metrics = self.objective(self.policy, teacher, batch, segment)
        metrics.update(getattr(teacher, "diagnostics", {}))
        endpoint = self.policy.endpoint(segment)
        if backward is not None:
            backward(loss, 0, 1)
            loss = loss.detach()
        return loss, metrics, endpoint

    def query_alphas(self, batch, schedule, state):
        return sample_query_alphas(
            self.config.query_alpha_mode,
            iteration=batch.iteration,
            segment_index=schedule.index,
            batch_size=batch.batch_size,
            num_query_points=self.config.pivot_num_query_points,
            device=self.device,
            dtype=state.dtype,
        )

    def teacher_for_batch(self, batch):
        return self.query_teacher

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
        cycle = self.training_batch_reuse
        if training:
            data = cycle.acquire(iteration, accumulation_index, lambda: data)
            entry = cycle.batches[accumulation_index]
            index = cycle.segment_index
        else:
            entry, index = TrajectoryBatch(data), 0
        if entry.prepared is None:
            entry.prepared = self.prepare_batch(data, iteration)

        def backward(loss, position, count):
            backward_substep(
                self,
                loss,
                position,
                count,
                grad_accum_rounds=grad_accum_rounds,
                sync_grads=sync_grads,
                is_fsdp=is_fsdp,
                on_backward_begin=on_backward_begin,
            )

        total, metrics, endpoint = self.train_segment(
            entry, iteration, index, backward=backward if training and per_segment_backward else None
        )
        if training:
            cycle.stage(accumulation_index, endpoint)
        loss_map = metrics
        loss_map["total_loss"] = total
        loss_map["enabled/pivot"] = total.new_tensor(float(not self.config.no_pivot))
        loss_map["progress/segment_index"] = total.new_tensor(index)
        return loss_map, {}

    @staticmethod
    def reduce_training_metrics(records):
        return reduce_metrics(records)

    def next_segment_index(self, iteration):
        return self.training_batch_reuse.segment_index

    def sample_student(self, *args, **kwargs):
        return self.policy.sample_student(*args, **kwargs)

    def get_sampler_snapshot(self):
        return {
            "type": "flow_match_euler_student",
            "student_sample_steps": self.config.student_sample_steps,
            "final_step_size_scale": self.config.final_step_size_scale,
            "embedded_guidance_scale": self.config.embedded_guidance_scale,
            "input_shape": list(self.input_shape),
        }
