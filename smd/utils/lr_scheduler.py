# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Borrowed from: https://huggingface.co/spaces/multimodalart/latentdiffusion/blob/main/latent-diffusion/ldm/lr_scheduler.py
"""

import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    """
    Implements a schedule with linear warm-up and decay in each cycle.

    Let n_local be the step within the current cycle.
    1. Warm-up (n_local < warm_up_steps):
       f = f_start + (f_max - f_start) * (n_local / warm_up_steps)

    2. Decay (n_local >= warm_up_steps):
       decay_steps = cycle_length - warm_up_steps
       t = (n_local - warm_up_steps) / decay_steps
       f = f_max - (f_max - f_min) * t

    This formulation is continuous at n_local == warm_up_steps and reaches f_min
    at the end of the cycle.
    """

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            decay_steps = self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            if decay_steps <= 0:
                f = self.f_max[cycle]
            else:
                t = (n - self.lr_warm_up_steps[cycle]) / decay_steps
                t = min(max(t, 0.0), 1.0)
                f = self.f_max[cycle] - (self.f_max[cycle] - self.f_min[cycle]) * t
            self.last_f = f
            return f


class LambdaInverseSquareRootScheduler:
    """
    Learning rate decay schedule used in the paper "Analyzing and Improving
    the Training Dynamics of Diffusion Models".
    """

    def __init__(self, warm_up_steps, decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.verbosity_interval = verbosity_interval
        self.last_f = 0.0

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}")

        f = 1.0
        if n > self.decay_steps and self.decay_steps > 0:
            f /= np.sqrt(n / self.decay_steps)

        if n < self.lr_warm_up_steps:
            f *= n / self.lr_warm_up_steps

        self.last_f = f
        return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
