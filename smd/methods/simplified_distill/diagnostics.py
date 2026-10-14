# SPDX-License-Identifier: Apache-2.0
"""Optional observations of existing velocity tensors, without teacher calls or autograd."""

import torch


class VelocityDiagnostics:
    def __init__(self, distance, base, *, enabled, eps):
        self.distance, self.enabled, self.eps = distance, enabled, eps
        self.base = base.detach() if enabled else None
        self.previous = None
        self.values = {}
        if enabled:
            self.add("student/base_velocity_rms", self.rms(self.base))

    def rms(self, value):
        # Mean of per-condition RMS, including ragged sparse layouts.
        return self.distance(value.detach(), torch.zeros_like(value), 1.0, 0.0).mean()

    def add(self, name, value):
        self.values.setdefault(name, []).append(value.detach())

    @torch.no_grad()
    def observe(self, velocity, target, reference):
        if not self.enabled:
            return
        self.add("student/query_velocity_rms", self.rms(velocity))
        self.add("pivot/residual_velocity_rms", self.rms(velocity - self.base))
        if self.previous is not None:
            self.add("pivot/adjacent_velocity_delta_rms", self.rms(velocity - self.previous))
        self.previous = velocity.detach()
        if target is not None:
            self.add("teacher/target_velocity_rms", self.rms(target))
        if reference is not None:
            self.add("teacher/reference_velocity_rms", self.rms(reference))
        if target is not None and reference is not None:
            self.add("teacher/target_reference_gap_rms", self.rms(target - reference))

    @torch.no_grad()
    def observe_repulsion(self, velocity, reference, gate, *, beta, eps):
        if self.enabled:
            # Unlike loss_raw/teacher_repulsion, this distance excludes the adaptive gate.
            self.add("teacher_repulsion/distance_before_gate", self.distance(velocity, reference, beta, eps).mean())
            self.add("teacher_repulsion/gate", gate.mean())

    def result(self):
        result = {name: torch.stack(values).mean() for name, values in self.values.items()}
        if "pivot/adjacent_velocity_delta_rms" in result:
            result["pivot/smoothness_relative"] = result["pivot/adjacent_velocity_delta_rms"] / (
                result["student/query_velocity_rms"] + self.eps
            )
        return result
