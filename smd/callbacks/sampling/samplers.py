# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch and offline fixed-input previews sharing one sampler and renderer."""

from dataclasses import dataclass, field

from smd.callbacks.sampling.base import SampleCallbackBase, preview_mode, resolve_net
from smd.callbacks.sampling.engine import copy_tree, decoder_spec, render_items, sample_records, topology
from smd.utils import logging_utils as logger


@dataclass
class _SampleCallback(SampleCallbackBase):
    include_real: bool = True

    def decoder_spec(self, net):
        if not self._decoder_probed:
            self._decoder = decoder_spec(net)
            self._decoder_probed = True
            if self._decoder is None:
                raise ValueError(f"No preview decoder for {net.network_family}.")
        return self._decoder

    def _sample_items(self, model, net, data, *, index, prefix, omit_single_index=False):
        seed = self.seed + index * 1000
        records = sample_records(model, net, data, grid_size=self.grid_size, seed=seed, include_real=self.include_real)
        return render_items(
            records,
            family=net.network_family,
            prefix=prefix,
            fps=self.fps,
            omit_single_index=omit_single_index,
        )


@dataclass
class BatchSampleCallback(_SampleCallback):
    """Current conditions; fixed noise streams make changes easier to compare."""

    key_prefix: str = "samples"

    def on_training_step_end(self, model, data_batch, output_batch, loss_dict, iteration=0):
        if not self.should_sample(iteration):
            return
        net = resolve_net(model)
        with preview_mode(model, net, self.seed):
            items = self._sample_items(model, net, data_batch, index=0, prefix=self.key_prefix)
        self.publish(items, iteration=iteration, net=net)
        logger.info(f"[{self.key_prefix}] iter={iteration}: queued batch preview")


@dataclass
class PromptSampleCallback(_SampleCallback):
    """Read complete offline conditions; never load inputs or run encoders.

    Missing, stale or corrupted caches fail before training. Rebuild them with
    ``python tools/cache_prompt_samples.py`` before launching the job.
    """

    key_prefix: str = "prompt_samples"
    cache: str | None = None
    cache_signature: str | None = None
    _encoded_cases: list = field(default_factory=list, init=False, repr=False)

    def on_app_begin(self):
        super().on_app_begin()
        if not self.cache:
            raise ValueError("Fixed previews require a precomputed cache manifest.")

    def prepare(self, model):
        import math
        from smd.callbacks.sampling.inputs.cache import read_cache
        from smd.utils.condition_utils import cond_repeat

        net = resolve_net(model)
        manifest, cases = read_cache(self.cache, expected_signature=self.cache_signature)
        if manifest["repeat_text"]:
            repeats = max(1, math.ceil(self.grid_size / topology(net)[1]))
            for case in cases:
                for key in ("condition", "neg_condition"):
                    if key in case:
                        case[key] = cond_repeat(case[key], repeats)
                for key in ("condition_raw", "neg_condition_raw", "fname"):
                    if key in case:
                        case[key] = case[key] * repeats
        self._encoded_cases = cases

    def on_training_step_end(self, model, data_batch, output_batch, loss_dict, iteration=0):
        if not self.should_sample(iteration):
            return
        net = resolve_net(model)
        items = []
        for index, cached in enumerate(self._encoded_cases):
            with preview_mode(model, net, self.seed + index * 1000):
                data = copy_tree(cached, model.device)
                items.extend(
                    self._sample_items(
                        model, net, data, index=index, prefix=f"{self.key_prefix}/{index:02d}",
                        omit_single_index=self.grid_size == 1,
                    )
                )
        self.publish(items, iteration=iteration, net=net)
        logger.info(f"[{self.key_prefix}] iter={iteration}: queued {len(self._encoded_cases)} cached cases")
