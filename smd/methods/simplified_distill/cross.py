# SPDX-License-Identifier: Apache-2.0
"""Independent teacher construction and dual conditioning; shared segment/loss execution."""

from dataclasses import replace
import torch

from smd.networks.io_contract import NetworkIOAdapter
from .distillation import SimplifiedDistillation


class CrossSimplifiedDistillation(SimplifiedDistillation):
    def build_model(self):
        if self.config.tie_backbone:
            raise ValueError("Cross distillation requires independent teacher parameters (tie_backbone=False).")
        super().build_model()

    def build_student(self):
        super().build_student()
        if self.config.enable_preprocessors:
            for name in ("text_encoder", "vae"):
                module = getattr(self.net, name, None)
                if module is not None:
                    module.to(device=self.device)

    def build_teacher_for_distill(self):
        self.build_teacher(force_all_rank_load=True)
        self.teacher.eval().requires_grad_(False)
        self.teacher_io_adapter = NetworkIOAdapter.from_model(self.teacher)
        if not self.config.enable_preprocessors:
            return
        # Wan2.1 sizes use identical UMT5/VAE; Flux2 sizes and Lens have distinct encoders.
        shared_wan = all(getattr(net, "network_family", None) == "wan" for net in (self.net, self.teacher))
        if shared_wan:
            self.teacher.text_encoder = self.net.text_encoder
            self.teacher.vae = self.net.vae
        else:
            self._build_preprocessors(self.teacher)
            if getattr(self.teacher, "text_encoder", None) is not None:
                self.teacher.text_encoder.to(device=self.device)
            if getattr(self.net, "vae", None) is not None and getattr(self.teacher, "vae", None) is not None:
                self.teacher.vae = self.net.vae
                torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_teacher_conditions(self, data):
        if "teacher_condition" in data:
            return data["teacher_condition"], data.get("teacher_neg_condition")
        encoder = getattr(self.teacher, "text_encoder", None)
        prompts, negative = data.get("condition_raw"), data.get("neg_condition_raw")
        if encoder is not None and isinstance(prompts, (list, tuple)) and all(isinstance(p, str) for p in prompts):
            condition = encoder.encode(prompts, precision=self.precision)
            uncond = encoder.encode(negative, precision=self.precision) if negative is not None else None
            return condition, uncond
        shared_encoder = encoder is not None and encoder is getattr(self.net, "text_encoder", None)
        if shared_encoder:
            return data["condition"], data.get("neg_condition")
        raise ValueError("Cross distillation requires raw prompts or explicit teacher_condition/teacher_neg_condition.")

    def prepare_batch(self, data, iteration):
        batch = super().prepare_batch(data, iteration)
        if self.config.imitation_weight == self.config.teacher_repulsion_weight == 0:
            return batch
        condition, negative = self.encode_teacher_conditions(data)
        condition = self.teacher_io_adapter.pack_condition(condition, data)
        negative = self.teacher_io_adapter.pack_condition(negative, data) if negative is not None else None
        if self.teacher_io_adapter.batch_size(condition) != batch.batch_size:
            raise ValueError("Student and teacher conditions have different batch sizes.")
        return replace(batch, teacher_condition=condition, teacher_neg_condition=negative)
