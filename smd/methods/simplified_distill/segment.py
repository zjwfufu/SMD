# SPDX-License-Identifier: Apache-2.0
"""Evaluate velocity imitation and teacher repulsion on one segment."""

import torch
from .objectives import ImitationLoss, TeacherRepulsionLoss, LossWeights
from .metrics import objective_metrics
from .diagnostics import VelocityDiagnostics


def average(values):
    return values[0] if len(values) == 1 else torch.stack(values).mean()


class SegmentObjective:
    def __init__(self, config):
        self.log_velocity_diagnostics = config.log_velocity_diagnostics
        self.imitation = ImitationLoss(config.velocity_distance_beta, config.velocity_distance_eps)
        self.teacher_repulsion = TeacherRepulsionLoss(
            config.velocity_distance_beta, config.velocity_distance_eps, config.teacher_repulsion_adaptive
        )
        self.weights = LossWeights(
            config.imitation_weight,
            config.teacher_repulsion_weight,
            0.5 * config.w_rescale / config.student_sample_steps,
        )
    def velocity_terms(self, policy, teacher, batch, segment):
        distance = policy.distance_for(batch)
        zero = segment.guided_context.base_velocity.float().new_zeros(())
        imitation, repulsion = zero, zero
        observations = VelocityDiagnostics(
            distance,
            segment.guided_context.base_velocity,
            enabled=self.log_velocity_diagnostics,
            eps=self.imitation.eps,
        )
        if self.weights.imitation == self.weights.teacher_repulsion == 0:
            return imitation, repulsion, observations.result()
        query_observations = {}
        for sigma, alpha in segment.queries:
            state = policy._context_trajectory_state(segment, segment.guided_context, alpha)
            if not torch.isfinite(state).all():
                raise RuntimeError("Non-finite teacher query state.")
            target, reference, _, _ = teacher(state, sigma, batch.teacher_condition, batch.teacher_neg_condition)
            velocity = policy._context_velocity(segment.guided_context, alpha)
            observations.observe(velocity, target, reference)
            for name, value in getattr(teacher, "query_diagnostics", {}).items():
                query_observations.setdefault(name, []).append(value)
            velocity, target, reference = policy.loss_velocities(velocity, target, reference)
            if self.weights.imitation > 0:
                imitation = imitation + self.imitation(velocity, target, distance=distance).mean()
            if self.weights.teacher_repulsion > 0:
                value, gate = self.teacher_repulsion(velocity, reference, anchor=target, distance=distance)
                repulsion = repulsion + value.mean()
                observations.observe_repulsion(
                    velocity, reference, gate, beta=self.teacher_repulsion.beta, eps=self.teacher_repulsion.eps
                )
        count = len(segment.queries)
        diagnostics = observations.result()
        diagnostics.update({name: average(values) for name, values in query_observations.items()})
        return imitation / count, repulsion / count, diagnostics

    def __call__(self, policy, teacher, batch, segment):
        imitation, repulsion, diagnostics = self.velocity_terms(policy, teacher, batch, segment)
        loss = self.weights(imitation, repulsion)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite segment objective.")
        raw = {"imitation": imitation, "teacher_repulsion": repulsion}
        metrics = objective_metrics(raw, self.weights, segment_index=segment.schedule.index)
        metrics.update({f"diagnostic/{name}": value.detach() for name, value in diagnostics.items()})
        metrics["diagnostic/query/alpha"] = segment.query_alpha
        return loss, metrics
