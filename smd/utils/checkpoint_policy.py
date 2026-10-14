# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit checkpoint policy for LoRA-only save/load.

The FSDP checkpointer used to *infer* how to save/load each module in
``model_dict`` from string patterns (``key.startswith("ema")`` and
``key.endswith("_pivot")``). That coupled the checkpointer to a naming
convention and silently mis-handled any module whose name did not match.

Instead the **model / EMA mixin declares an explicit policy**: a mapping from
``model_dict`` key to a :class:`ModuleCheckpointPolicy`. The checkpointer only
consumes this policy — it never guesses from key strings.

This module deliberately has no heavy imports so both
``smd.methods.model`` (producer) and ``smd.utils.checkpointer``
(consumer) can depend on it without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class CheckpointSaveMode(str, Enum):
    """How a single ``model_dict`` module is saved/loaded in LoRA-only mode."""

    #: Fully-trainable module (e.g. the live pivot or an EMA pivot). Every
    #: parameter is written and read back.
    FULL = "full"

    #: Module that owns its own ``requires_grad`` flags with a frozen base
    #: (e.g. the LoRA student ``net``). Only the trainable (non-frozen) params
    #: are saved / loaded via ``ignore_frozen_params=True``.
    TRAINABLE_SUBSET = "trainable_subset"

    #: A frozen EMA student whose *saved subset* mirrors another module's
    #: trainable parameter names. The EMA has ``requires_grad=False`` on every
    #: parameter, so the checkpointer temporarily marks the mirror source's
    #: trainable names as trainable, saves that subset, then restores the flags.
    EMA_TRAINABLE_MIRROR = "ema_trainable_mirror"


@dataclass(frozen=True)
class ModuleCheckpointPolicy:
    """Per-module checkpoint policy.

    Attributes:
        mode: the :class:`CheckpointSaveMode` for this module.
        mirror_source: for :attr:`CheckpointSaveMode.EMA_TRAINABLE_MIRROR`, the
            ``model_dict`` key whose trainable parameter names define the subset
            to save/load. Required (and only meaningful) for that mode.
    """

    mode: CheckpointSaveMode
    mirror_source: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode == CheckpointSaveMode.EMA_TRAINABLE_MIRROR and not self.mirror_source:
            raise ValueError(
                "ModuleCheckpointPolicy(EMA_TRAINABLE_MIRROR) requires a mirror_source "
                "module key whose trainable names define the saved subset."
            )
        if self.mode != CheckpointSaveMode.EMA_TRAINABLE_MIRROR and self.mirror_source is not None:
            raise ValueError(
                f"mirror_source is only valid for EMA_TRAINABLE_MIRROR, got mode={self.mode} "
                f"with mirror_source={self.mirror_source!r}."
            )


#: Full checkpoint policy: ``model_dict`` key -> per-module policy.
CheckpointPolicy = Dict[str, ModuleCheckpointPolicy]


def structural_default_policy(module) -> ModuleCheckpointPolicy:
    """Derive a policy from a module's ``requires_grad`` structure (no key strings).

    Used only as a fallback when a module has no explicit policy entry (e.g. the
    single-key ``model_dict`` built by ``Trainer.load_pretrained_ckpt``). A module
    with any frozen parameter is treated as :attr:`TRAINABLE_SUBSET`; a fully
    trainable module is :attr:`FULL`. This never inspects the module's *name*.
    """
    total = 0
    trainable = 0
    for p in module.parameters():
        total += 1
        if p.requires_grad:
            trainable += 1
    has_frozen = trainable < total
    mode = CheckpointSaveMode.TRAINABLE_SUBSET if has_frozen else CheckpointSaveMode.FULL
    return ModuleCheckpointPolicy(mode=mode)
