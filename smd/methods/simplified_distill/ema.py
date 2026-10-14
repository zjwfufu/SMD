# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-lifecycle Pivot EMA mixin.

:class:`PivotEMAMixin` owns the *complete* Pivot EMA lifecycle so that the
orchestration class (:class:`~smd.methods.simplified_distill.distillation.SimplifiedDistillation`)
only calls a few clear lifecycle hooks and never touches EMA internals.

The mixin owns, per EMA name in ``self.use_ema``:

* an **EMA student** built from the *final* student structure (after LoRA +
  freeze), reverse-tying the immutable base to the student-owned base in the
  ``tie_backbone`` path, or an independent all-rank copy otherwise;
* an **EMA pivot** (``{name}_pivot``) deep-copied from the live pivot;
* independent EMA LoRA / norm / allowlist storage initialized from the live
  student (never shared — only the *immutable* frozen base is shared);
* preprocessor sharing (text_encoder / vae / ...);
* device / dtype placement;
* post-FSDP reverse unshard/reshard hooks for reverse-tied EMAs;
* the unified per-step update of both the EMA student and the EMA pivot; and
* explicit checkpoint-policy metadata + model/checkpoint module exposure.

Ownership invariant:
*EMA always derives from the student.* Only immutable frozen base weights may
be tied/shared with the student; EMA LoRA and EMA allowlist/norm parameters
always have independent storage.

The host class must provide: ``net``, ``pivot``, ``backend`` (a pivot backend
with ``get_transformer``), ``config``, ``device``, ``precision``,
``precision_fsdp``, ``use_ema`` (from :class:`SMDModel`), ``ema_dict``
(from :class:`SMDModel`), ``_coordinated_instantiate`` (from
:class:`SMDModel`), and ``_reverse_tied`` (set by the student-first build
lifecycle).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch

import smd.utils.logging_utils as logger
from smd.utils import instantiate
from smd.utils.checkpoint_policy import CheckpointSaveMode, ModuleCheckpointPolicy
from smd.utils.distributed import synchronize
from smd.utils.distributed.tie_weights import (
    materialize_independent_params,
    register_reverse_unshard_hooks,
    reverse_tie_frozen_base,
)
from smd.utils.lora_utils import freeze_for_lora

log = logger

_PREPROCESSOR_ATTRS = ("text_encoder", "vae", "image_encoder", "image_cond_model", "image_processor")


class PivotEMAMixin:
    """Owns the complete Pivot EMA lifecycle (student EMA + pivot EMA)."""

    # ------------------------------------------------------------------
    # Construction-time state
    # ------------------------------------------------------------------

    def _init_pivot_ema_state(self) -> None:
        """Initialize EMA bookkeeping. Call BEFORE ``SMDModel.__init__``.

        ``SMDModel.__init__`` runs ``build_model()`` (which calls
        :meth:`build_ema`), so this must set up the EMA registries first.
        """
        self._ema_pivot_names: List[str] = []
        self._ema_reverse_results: Dict[str, Any] = {}
        self._ema_reverse_hook_handles: List = []

    # ------------------------------------------------------------------
    # Build lifecycle
    # ------------------------------------------------------------------

    def build_ema(self) -> None:
        """Build the complete Pivot EMA structure for every name in ``use_ema``.

        Invoked from ``build_model()`` AFTER the student's LoRA/freeze and the
        teacher are configured, so each EMA mirrors the *final* student
        structure. No-op when EMA is disabled.
        """
        if not self.use_ema:
            return
        use_lora = bool(getattr(self.config.net, "use_lora", False))
        reverse = bool(getattr(self, "_reverse_tied", False))
        for name in self.use_ema:
            self._build_ema_student(name, use_lora=use_lora, reverse=reverse)
            self._build_ema_pivot(name)
        synchronize()

    def _build_ema_student(self, name: str, *, use_lora: bool, reverse: bool) -> None:
        """Build (or refresh) one EMA student net mirroring the final student."""
        if not hasattr(self, name):
            log.info(f"SMD EMA: building EMA student {name!r} (reverse={reverse}, use_lora={use_lora})")
            if reverse:
                # Frozen meta skeleton: base is reverse-tied to the student-owned
                # base; only independent LoRA/norm params are materialized.
                with torch.device("meta"):
                    ema = instantiate(self.config.net)
                ema.to(dtype=self.precision_fsdp)
            else:
                ema = self._coordinated_instantiate(self.config.net, all_rank_read=True)
            if use_lora:
                if not hasattr(ema, "attach_lora"):
                    raise RuntimeError("SMD EMA requires attach_lora() when config.net.use_lora=True.")
                ema.attach_lora()
                if self.config.freeze_backbone_for_lora:
                    freeze_for_lora(ema, allowlist=list(self.config.backbone_unfreeze_allowlist))
                if reverse:
                    ema_result = reverse_tie_frozen_base(
                        self.backend.get_transformer(ema), self.backend.get_transformer(self.net)
                    )
                    materialize_independent_params(ema, self.net, requires_grad=False, device=self.device)
                    self._ema_reverse_results[name] = ema_result
            setattr(self, name, ema)
        else:
            log.warning(f"SMD EMA student {name!r} already exists; refreshing from the live student net.")

        ema = getattr(self, name)
        if not reverse:
            # Independent EMA: copy the full (base + trainable) state from the student.
            load_info = ema.load_state_dict(self.net.state_dict(), strict=False)
            log.info(f"Loaded SMD EMA student {name!r}: {load_info}")
        self._share_preprocessors_with_ema(ema)
        ema.eval().requires_grad_(False)

    def _build_ema_pivot(self, name: str) -> None:
        """Build (or refresh) the EMA pivot ``{name}_pivot`` from the live pivot."""
        ema_pivot_name = f"{name}_pivot"
        if not hasattr(self, ema_pivot_name):
            log.info(f"SMD EMA: building EMA pivot {ema_pivot_name!r}")
            ema_pivot = copy.deepcopy(self.pivot)
            setattr(self, ema_pivot_name, ema_pivot)
        ema_pivot = getattr(self, ema_pivot_name)
        ema_pivot.load_state_dict(self.pivot.state_dict(), strict=True)
        ema_pivot.eval().requires_grad_(False)
        if ema_pivot_name not in self._ema_pivot_names:
            self._ema_pivot_names.append(ema_pivot_name)

    def _share_preprocessors_with_ema(self, ema: torch.nn.Module) -> None:
        """Share auxiliary preprocessors with the EMA instead of loading duplicates."""
        if not self.config.enable_preprocessors:
            return
        for attr_name in _PREPROCESSOR_ATTRS:
            if hasattr(self.net, attr_name):
                setattr(ema, attr_name, getattr(self.net, attr_name))

    # ------------------------------------------------------------------
    # Device / dtype placement (called from on_train_begin)
    # ------------------------------------------------------------------

    def place_ema_modules(self, is_fsdp: bool) -> None:
        """Move every EMA student + EMA pivot to the training device/dtype.

        EMA modules are excluded from FSDP, so their placement is handled here
        rather than by the FSDP wrapper.
        """
        if not self.use_ema:
            return
        target_dtype = self.precision_fsdp if is_fsdp else self.precision
        for name, ema in self.ema_dict.items():
            log.info(f"Placing SMD EMA student {name!r} on device={self.device}, dtype={target_dtype}.")
            ema.to(device=self.device, dtype=target_dtype)
            self._share_preprocessors_with_ema(ema)
            ema.eval().requires_grad_(False)
        for ema_pivot_name, ema_pivot in self.ema_pivot_dict.items():
            ema_pivot.to(device=self.device, dtype=target_dtype)
            ema_pivot.eval().requires_grad_(False)
        synchronize()

    # ------------------------------------------------------------------
    # Post-FSDP reverse unshard/reshard hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _recompute_shared_param_ids(consumer: torch.nn.Module, owner: torch.nn.Module) -> set:
        """Recompute owner-param ids the consumer references, AFTER FSDP wrapping.

        Reverse-tie binds by *module identity*, so FSDP's ``to_empty`` +
        ``set_model_state_dict`` swap the owner's params for fresh DTensors and
        the consumer transparently sees the new objects. Build-time
        ``shared_param_ids`` are therefore stale; the live shared set is the
        intersection of the consumer's and owner's current params.
        """
        owner_ids = {id(p) for p in owner.parameters()}
        return {id(p) for p in consumer.parameters() if id(p) in owner_ids}

    def register_ema_reverse_hooks(self, student_tf: torch.nn.Module) -> List:
        """Register reverse unshard/reshard hooks for each reverse-tied EMA.

        Returns the list of registered hook handles (also tracked on the mixin
        via ``self._ema_reverse_hook_handles``). Only relevant when the EMA is a
        reverse-tied consumer of the FSDP-sharded student; independent EMAs need
        no hooks. Raises if a reverse-tied EMA shares no params with the student
        (reverse-tie lost across FSDP wrapping).
        """
        self._ema_reverse_hook_handles = []
        for ema_name, ema in self.ema_dict.items():
            if not hasattr(ema, "transformer"):
                continue
            if ema_name not in self._ema_reverse_results:
                # Independent (non-reverse-tied) EMA — nothing to unshard.
                continue
            ema_tf = self.backend.get_transformer(ema)
            ema_ids = self._recompute_shared_param_ids(ema_tf, student_tf)
            if not ema_ids:
                raise RuntimeError(
                    f"register_ema_reverse_hooks: reverse-tied EMA {ema_name!r} shares no params with "
                    f"the FSDP student. Reverse-tie was lost across FSDP wrapping."
                )
            handles = register_reverse_unshard_hooks(ema_tf, student_tf, ema_ids)
            self._ema_reverse_hook_handles.extend(handles)
            log.info(f"Registered {len(handles)} reverse unshard hook pairs for EMA {ema_name!r}")
        return self._ema_reverse_hook_handles

    # ------------------------------------------------------------------
    # Per-step update (single unified interface for EMACallback)
    # ------------------------------------------------------------------

    def update_ema(self, ema_name: str, beta: float, initialize: bool, is_fsdp: bool) -> None:
        """Update both the EMA student and the matching EMA pivot in one call.

        This is the single interface :class:`EMACallback` uses: the callback
        computes ``beta`` and calls here; it never touches EMA internals. Missing
        target keys or structural mismatches raise (strict — no silent skips).
        """
        ema_student = getattr(self, ema_name, None)
        if ema_student is None:
            raise RuntimeError(
                f"update_ema: EMA student {ema_name!r} not found (use_ema={self.use_ema}). "
                f"EMA must be built before it can be updated."
            )
        self._ema_update_module(self.net, ema_student, beta=beta, initialize=initialize, is_fsdp=is_fsdp)

        ema_pivot_name = f"{ema_name}_pivot"
        ema_pivot = getattr(self, ema_pivot_name, None)
        if ema_pivot is None:
            raise RuntimeError(
                f"update_ema: EMA pivot {ema_pivot_name!r} not found for EMA {ema_name!r}. "
                f"Every EMA student must have a matching EMA pivot."
            )
        self._ema_update_module(self.pivot, ema_pivot, beta=beta, initialize=initialize, is_fsdp=is_fsdp)

    @staticmethod
    def _ema_update_module(
        source: torch.nn.Module,
        target: torch.nn.Module,
        *,
        beta: float,
        initialize: bool,
        is_fsdp: bool,
    ) -> None:
        """EMA-update ``target`` from ``source`` (trainable params + buffers).

        Frozen source params are skipped: they are copied once at initialization
        and never change, so re-copying would all-gather the frozen backbone
        every step (fatal for LoRA/FSDP throughput). Every *trainable* source
        param and every source buffer MUST map to a target key — a missing key
        raises rather than silently skipping.
        """
        target_state = target.state_dict()
        for name, p_src in source.named_parameters():
            if not p_src.requires_grad:
                continue
            if is_fsdp and hasattr(p_src, "full_tensor"):
                # Gather the full tensor from all ranks (FSDP DTensor). full_tensor()
                # all_gathers, which needs a CUDA backend, so move CPU shards first.
                src_tensor = p_src.to("cuda").full_tensor() if p_src.device.type == "cpu" else p_src.full_tensor()
            else:
                src_tensor = p_src
            key = name.replace("_checkpoint_wrapped_module.", "")
            if key not in target_state:
                raise KeyError(
                    f"EMA update: trainable source param {name!r} (key {key!r}) has no matching target key "
                    f"in {type(target).__name__}. The EMA must mirror the student's trainable structure exactly."
                )
            dst = target_state[key]
            src = src_tensor.to(device=dst.device, dtype=dst.dtype)
            if initialize:
                dst.copy_(src)
            else:
                dst.lerp_(src, 1.0 - beta)

        # state_dict omits non-persistent buffers (e.g. Wan's RoPE tables),
        # although named_buffers includes them on both sides.
        target_buffers = dict(target.named_buffers())
        for name, b_src in source.named_buffers():
            key = name.replace("_checkpoint_wrapped_module.", "")
            if key not in target_buffers:
                raise KeyError(
                    f"EMA update: source buffer {name!r} (key {key!r}) has no matching target key in "
                    f"{type(target).__name__}. The EMA must mirror the buffer structure exactly."
                )
            dst = target_buffers[key]
            dst.copy_(b_src.to(device=dst.device, dtype=dst.dtype))

    # ------------------------------------------------------------------
    # Model / checkpoint exposure + explicit checkpoint policy
    # ------------------------------------------------------------------

    @property
    def net_ema(self) -> torch.nn.Module | None:
        """The first EMA student network, or ``None`` when EMA is disabled."""
        if self.use_ema:
            return getattr(self, self.use_ema[0], None)
        return None

    @property
    def ema_pivot_dict(self) -> Dict[str, torch.nn.Module]:
        """Mapping ``{name}_pivot -> EMA pivot module`` for every EMA pivot."""
        return {name: getattr(self, name) for name in self._ema_pivot_names}

    @property
    def ema_module_dict(self) -> Dict[str, torch.nn.Module]:
        """All EMA modules (students + pivots) for model/checkpoint exposure."""
        modules = dict(self.ema_dict)
        modules.update(self.ema_pivot_dict)
        return modules

    def ema_checkpoint_policy(self, mirror_source: str = "net") -> Dict[str, ModuleCheckpointPolicy]:
        """Explicit checkpoint policy for every EMA module.

        * Each EMA student is a frozen mirror of ``mirror_source``'s trainable
          params -> :attr:`CheckpointSaveMode.EMA_TRAINABLE_MIRROR`.
        * Each EMA pivot is full in full mode, or mirrors the live Pivot trainable
          subset when Pivot LoRA is enabled.
        """
        policy: Dict[str, ModuleCheckpointPolicy] = {}
        for name in self.use_ema:
            policy[name] = ModuleCheckpointPolicy(CheckpointSaveMode.EMA_TRAINABLE_MIRROR, mirror_source=mirror_source)
        for ema_pivot_name in self._ema_pivot_names:
            if getattr(self, "_pivot_train_mode", "full") == "lora":
                policy[ema_pivot_name] = ModuleCheckpointPolicy(
                    CheckpointSaveMode.EMA_TRAINABLE_MIRROR,
                    mirror_source="pivot",
                )
            else:
                policy[ema_pivot_name] = ModuleCheckpointPolicy(CheckpointSaveMode.FULL)
        return policy
