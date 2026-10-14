# SPDX-License-Identifier: Apache-2.0
"""One independent teacher backbone with routed reward LoRAs and a shared student/Pivot."""

from contextlib import contextmanager
import numpy as np
import torch

from .distillation import SimplifiedDistillation


class LoRATeacherPool:
    def __init__(self, transformer, adapters):
        from peft import PeftModel

        first, *others = adapters
        self.model = PeftModel.from_pretrained(transformer, first["path"], adapter_name=first["name"])
        for adapter in others:
            self.model.load_adapter(adapter["path"], adapter_name=adapter["name"])
        self.names = tuple(adapter["name"] for adapter in adapters)
        self.model.eval().requires_grad_(False)

    @contextmanager
    def use(self, name):
        previous = self.model.active_adapter
        try:
            if name == "base":
                self.model.disable_adapter_layers()
            else:
                self.model.set_adapter(name)
            # PEFT set_adapter also toggles trainability; these adapters are teachers.
            self.model.requires_grad_(False)
            yield
        finally:
            self.model.enable_adapter_layers()
            self.model.set_adapter(previous)
            self.model.requires_grad_(False)


class RoutedTeacherQuery:
    def __init__(self, query, pool, name):
        self.query, self.pool, self.name = query, pool, name
        self.diagnostics = {
            f"diagnostic/teacher/selected/{candidate}": torch.tensor(float(candidate == name), device=query.device)
            for candidate in ("base", *pool.names)
        }

    def __call__(self, *args, **kwargs):
        with self.pool.use(self.name):
            return self.query(*args, **kwargs)


class MultiTeacherSimplifiedDistillation(SimplifiedDistillation):
    def build_teacher_for_distill(self):
        # Reward adapters need a real independent backbone, including when student/EMA tie.
        self.build_teacher(force_all_rank_load=True)
        self.teacher.eval().requires_grad_(False)

    def build_model(self):
        if not 0 <= self.config.base_teacher_ratio <= 1:
            raise ValueError("base_teacher_ratio must be in [0, 1].")
        names = [adapter["name"] for adapter in self.config.teacher_adapters]
        if not names or len(set(names)) != len(names) or "base" in names:
            raise ValueError("Teacher adapters require unique names, with 'base' reserved for the frozen base teacher.")
        super().build_model()
        self.teacher_pool = LoRATeacherPool(self.teacher.transformer, self.config.teacher_adapters)
        self.teacher.transformer = self.teacher_pool.model
        self.teacher.eval().requires_grad_(False)

    def select_teacher(self, route, iteration):
        # Preserve the old independent selection RNG and its draw order.
        rng = np.random.default_rng([42, int(iteration)])
        if rng.random() < self.config.base_teacher_ratio:
            return "base"
        available = [name for name in self.config.route_to_lora.get(route, []) if name in self.teacher_pool.names]
        return available[int(rng.integers(len(available)))] if available else "base"

    def teacher_for_batch(self, batch):
        if self.config.imitation_weight == self.config.teacher_repulsion_weight == 0:
            return self.query_teacher
        return RoutedTeacherQuery(
            self.query_teacher, self.teacher_pool, self.select_teacher(batch.route, batch.iteration)
        )
