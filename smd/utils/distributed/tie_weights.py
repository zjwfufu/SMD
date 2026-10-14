# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LakonLab-style structural weight tying for FSDP2.

``tie_untrained_submodules`` (before FSDP wrap): Python identity tie of frozen sub-modules.
After both teacher and student are FSDP-wrapped, ``tie_fsdp2_modules`` registers forward
hooks on student blocks to unshard/reshard the corresponding teacher blocks — replicating
the LakonLab FSDP1 ``tie_fsdp_modules`` pattern for FSDP2.

Memory model:
  - Teacher is wrapped first → shared params become sharded DTensors
  - Student is wrapped with ``ignored_params`` for the shared DTensors
  - Student forward hooks explicitly unshard teacher blocks as needed
  - Result: only 1 copy of backbone, sharded across ranks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

import smd.utils.logging_utils as logger


def _has_trainable_params(module: nn.Module) -> bool:
    """Return True if the module (non-recursively) has any trainable parameters."""
    return any(p.requires_grad for p in module.parameters(recurse=False))


def _is_lora_layer(module: nn.Module) -> bool:
    """Return True if the module is a peft LoRA layer (has a ``base_layer`` attribute)."""
    return hasattr(module, "base_layer") and hasattr(module, "lora_A")


def tie_untrained_submodules(
    tgt: nn.Module,
    src: nn.Module,
    tie_lora_base_layer: bool = True,
    _path: str = "",
) -> Set[str]:
    """Recursively tie frozen sub-modules of *tgt* to *src* (Python identity).

    Rules:
    - If ``tie_lora_base_layer`` and *tgt* child is a LoRA layer:
      tie just the ``base_layer`` (frozen) to src's corresponding module.
    - If *tgt* child has any trainable parameters (e.g. LoRA adapters, ``norm_out``):
      recurse into children instead of tying the whole module.
    - If *tgt* child is fully frozen: replace ``tgt._modules[key]`` with
      ``src._modules[key]`` (Python identity, shared GPU memory).

    This logic is architecture-agnostic — no need to annotate specific block classes.

    Note: this function does NOT mark ``_no_shard``. After teacher FSDP wraps
    (which shards the shared params into DTensors), use ``collect_shared_params``
    on the student to detect them via ``isinstance(param, DTensor)``.

    Returns:
        Set of dotted paths that were tied.
    """
    tied: Set[str] = set()

    for key in list(tgt._modules.keys()):
        tgt_child = tgt._modules[key]
        src_child = src._modules.get(key)

        if tgt_child is None or src_child is None:
            continue

        full_path = f"{_path}.{key}" if _path else key

        # LoRA layer: tie the frozen base_layer only
        if tie_lora_base_layer and _is_lora_layer(tgt_child):
            # src_child is the original (non-LoRA) linear — tie it as base_layer
            tgt_child.base_layer = src_child
            # Also share the weight/bias references for backward compat
            if hasattr(src_child, "weight"):
                tgt_child.base_layer.weight = src_child.weight
            tied.add(full_path + ".base_layer")
            logger.debug(f"[tie] LoRA base_layer tied: {full_path}")
            continue

        # If the tgt child has any trainable params anywhere, recurse instead of tying
        has_trainable = any(p.requires_grad for p in tgt_child.parameters())
        if has_trainable:
            child_tied = tie_untrained_submodules(
                tgt_child, src_child, tie_lora_base_layer=tie_lora_base_layer, _path=full_path
            )
            tied.update(child_tied)
            continue

        # Fully frozen module — tie it (Python identity)
        tgt._modules[key] = src_child
        tied.add(full_path)
        logger.debug(f"[tie] Module tied (identity): {full_path}")

    return tied


def collect_no_shard_params(module: nn.Module) -> set:
    """Return the set of parameters that should be ignored by FSDP.

    Detects two kinds of shared params:
    1. Legacy ``_no_shard=True`` marker (for backward compat).
    2. DTensor params — already sharded by another module's FSDP (e.g. teacher).
       These must be ``ignored_params`` so student's FSDP doesn't try to shard them again.
    """
    from torch.distributed.tensor import DTensor

    ignored = set()
    for p in module.parameters():
        if getattr(p, "_no_shard", False):
            ignored.add(p)
        elif isinstance(p, DTensor):
            ignored.add(p)
    return ignored


# ---------------------------------------------------------------------------
# FSDP2 forward-hook tying (replaces LakonLab's tie_fsdp_modules for FSDP2)
# ---------------------------------------------------------------------------


def tie_fsdp2_modules(
    student_modules: List[nn.Module],
    teacher_modules: List[nn.Module],
) -> List[Tuple[torch.utils.hooks.RemovableHook, torch.utils.hooks.RemovableHook]]:
    """Register pre/post forward hooks on student modules to unshard/reshard teacher modules.

    After FSDP2 wraps both teacher and student, teacher modules have sharded
    DTensor params. Student modules share (via tying) the same param objects.
    During student forward, teacher's FSDP hooks don't fire (different Python
    objects). We explicitly call ``teacher_module.unshard()`` /
    ``teacher_module.reshard()`` to make the shared params available.

    This should be called with ALL FSDP units that have shared params, including
    both per-block modules AND the root transformer module (which owns ``img_in``,
    ``txt_in``, ``time_text_embed``, etc.).

    Args:
        student_modules: List of student modules (blocks + root transformer).
        teacher_modules: List of corresponding teacher modules (FSDP-wrapped, same length).

    Returns:
        List of (pre_hook_handle, post_hook_handle) tuples for cleanup.
    """
    from torch.distributed._composable.fsdp import FSDPModule

    if len(student_modules) != len(teacher_modules):
        raise ValueError(f"Module count mismatch: student={len(student_modules)}, teacher={len(teacher_modules)}")

    handles = []
    n_registered = 0

    for stu_mod, tea_mod in zip(student_modules, teacher_modules):
        # Only register if teacher module is an FSDP module
        if not isinstance(tea_mod, FSDPModule):
            logger.debug(f"[tie_fsdp2] Skipping non-FSDP teacher module: {type(tea_mod).__name__}")
            continue

        def _make_hooks(tb):
            """Capture teacher module in closure."""

            def pre_hook(module, args):
                tb.unshard()

            def post_hook(module, args, output):
                tb.reshard()

            return pre_hook, post_hook

        pre_hook, post_hook = _make_hooks(tea_mod)
        h_pre = stu_mod.register_forward_pre_hook(pre_hook)
        h_post = stu_mod.register_forward_hook(post_hook)
        handles.append((h_pre, h_post))
        n_registered += 1

        # Also hook the inner module if wrapped by CheckpointWrapper.
        # During activation checkpointing recompute, checkpoint() calls
        # _checkpoint_wrapped_module(*args) directly, bypassing the
        # outer wrapper's hooks.  We need unshard/reshard there too.
        inner = getattr(stu_mod, "_checkpoint_wrapped_module", None)
        if inner is not None and inner is not stu_mod:
            h_pre_inner = inner.register_forward_pre_hook(pre_hook)
            h_post_inner = inner.register_forward_hook(post_hook)
            handles.append((h_pre_inner, h_post_inner))

    logger.info(f"[tie_fsdp2] Registered unshard/reshard hooks on {n_registered} student modules")
    return handles


# ---------------------------------------------------------------------------
# Student-owned reverse tying
# ---------------------------------------------------------------------------
#
# The forward-tying utilities above make the *student* borrow a *teacher*-owned
# FSDP backbone. Reverse tying flips ownership: the **student** is the canonical
# parameter owner and is FSDP-sharded first; frozen consumers (teacher, EMA)
# are meta skeletons that bind their immutable base sub-modules/params back to
# the student, materialize their own independent parameters, and unshard the
# student-owned FSDP units on demand during their own forward.
#
# Design rules (all strict — structural mismatches raise, never fall back):
#   * A frozen sub-tree of the owner is shared into the consumer by Python
#     identity (``consumer._modules[k] = owner._modules[k]``).
#   * A LoRA-wrapped owner layer exposes an immutable ``base_layer``:
#       - a *plain* consumer layer (teacher) is REPLACED by that ``base_layer``;
#       - a *LoRA* consumer layer (EMA) keeps its own ``lora_A``/``lora_B`` and
#         only rebinds ``base_layer`` to the owner's immutable base.
#   * Any owner sub-tree that is trainable but contains no LoRA layer (e.g.
#     ``norm_out``/allowlist params) stays independent: the consumer keeps its
#     own storage, materialized from the owner's values.


def _module_has_trainable(module: nn.Module) -> bool:
    """Return True if *module* (recursively) owns any trainable parameter."""
    return any(p.requires_grad for p in module.parameters())


def _subtree_contains_lora(module: nn.Module) -> bool:
    """Return True if *module* or any descendant is a LoRA layer."""
    return any(_is_lora_layer(m) for m in module.modules())


def _param_shapes(module: nn.Module) -> Dict[str, Tuple[int, ...]]:
    return {n: tuple(p.shape) for n, p in module.named_parameters()}


def _assert_structural_match(consumer_child: nn.Module, owner_source: nn.Module, path: str) -> None:
    """Raise if the consumer module we are about to replace/bind is structurally
    incompatible with the owner module/param it will share."""
    c_shapes = _param_shapes(consumer_child)
    o_shapes = _param_shapes(owner_source)
    if c_shapes != o_shapes:
        raise RuntimeError(
            f"reverse_tie structural mismatch at {path!r}: consumer param shapes {c_shapes} "
            f"!= owner param shapes {o_shapes}. Reverse tying refuses to bind mismatched modules."
        )


@dataclass
class ReverseTieResult:
    """Bookkeeping for a reverse-tie pass.

    Attributes:
        identity_modules: dotted paths shared into the consumer as whole frozen modules.
        base_layer_binds: dotted paths where the consumer bound to an owner ``base_layer``.
        independent_params: dotted parameter names the consumer owns independently
            (still on meta after tying — to be materialized from the owner).
        shared_param_ids: ``id()`` of every owner parameter now referenced by the consumer.
    """

    identity_modules: List[str] = field(default_factory=list)
    base_layer_binds: List[str] = field(default_factory=list)
    independent_params: List[str] = field(default_factory=list)
    shared_param_ids: Set[int] = field(default_factory=set)


def reverse_tie_frozen_base(
    consumer: nn.Module,
    owner: nn.Module,
    *,
    _path: str = "",
    _result: ReverseTieResult | None = None,
) -> ReverseTieResult:
    """Bind *consumer*'s frozen base to *owner* (the canonical parameter owner).

    *consumer* is typically a meta-device skeleton (teacher or EMA). *owner* is a
    real, possibly FSDP-sharded, student. After this call the consumer references
    the owner's immutable base modules/params by Python identity, while any
    independent consumer parameters (LoRA adapters on an EMA, ``norm_out``
    snapshots on a teacher) are left on meta for :func:`materialize_independent_params`.

    Raises:
        RuntimeError: if the consumer is missing a module the owner has, or if a
            module the consumer would bind is structurally incompatible.
    """
    if _result is None:
        _result = ReverseTieResult()

    for key in list(owner._modules.keys()):
        owner_child = owner._modules[key]
        consumer_child = consumer._modules.get(key)
        if owner_child is None:
            continue
        full = f"{_path}.{key}" if _path else key

        if consumer_child is None:
            raise RuntimeError(
                f"reverse_tie: consumer is missing sub-module {full!r} present on the owner. "
                f"Consumer skeleton must mirror the owner's module structure."
            )

        # 1. Owner LoRA layer — the immutable ``base_layer`` is the shared base.
        if _is_lora_layer(owner_child):
            base = owner_child.base_layer
            if _is_lora_layer(consumer_child):
                # EMA: keep independent adapters, rebind the immutable base.
                _assert_structural_match(consumer_child.base_layer, base, full + ".base_layer")
                consumer_child.base_layer = base
                _result.base_layer_binds.append(full + ".base_layer")
                for pn, _ in consumer_child.named_parameters():
                    if "base_layer" not in pn:
                        _result.independent_params.append(f"{full}.{pn}")
            else:
                # Teacher: replace the plain consumer layer with the owner base.
                _assert_structural_match(consumer_child, base, full)
                consumer._modules[key] = base
                _result.base_layer_binds.append(full)
            for p in base.parameters():
                _result.shared_param_ids.add(id(p))
            continue

        # 2. Owner sub-tree containing LoRA — recurse (e.g. a block / ModuleList).
        if _subtree_contains_lora(owner_child):
            reverse_tie_frozen_base(consumer_child, owner_child, _path=full, _result=_result)
            continue

        # 3. Fully frozen owner sub-tree — identity-share the whole module.
        if not _module_has_trainable(owner_child):
            _assert_structural_match(consumer_child, owner_child, full)
            consumer._modules[key] = owner_child
            _result.identity_modules.append(full)
            for p in owner_child.parameters():
                _result.shared_param_ids.add(id(p))
            continue

        # 4. Trainable owner sub-tree with no LoRA — independent consumer storage.
        for pn, _ in consumer_child.named_parameters():
            _result.independent_params.append(f"{full}.{pn}")

    return _result


def _find_owner_param(name: str, owner_params: Dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Map a consumer parameter name to the owner's value (handles LoRA naming)."""
    if name in owner_params:
        return owner_params[name]
    stripped = name.replace(".base_layer.", ".")
    if stripped in owner_params:
        return owner_params[stripped]
    return None


def gather_full_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a full, unsharded, replicated parameter dict — correct under FSDP2 + CPU offload.

    Reading a CPU-offloaded ``DTensor`` via ``full_tensor()`` dispatches an
    all-gather on CPU tensors, which NCCL cannot service. FSDP's
    ``get_model_state_dict`` gathers the full (padding-corrected) tensors through
    the proper collective path, so it is the safe way to read owner values for
    materialization / EMA updates when the owner is FSDP-sharded.

    For a non-FSDP module this is just ``named_parameters`` (no collective).
    """
    from torch.distributed.fsdp import FSDPModule

    if not any(isinstance(m, FSDPModule) for m in module.modules()):
        return {n: p.detach() for n, p in module.named_parameters()}

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=False))


def materialize_independent_params(
    consumer: nn.Module,
    owner: nn.Module,
    *,
    requires_grad: bool,
    device: torch.device | str | None = None,
) -> int:
    """Materialize the consumer's remaining meta params from the owner's values.

    After :func:`reverse_tie_frozen_base`, only the consumer's *independent*
    parameters remain on meta (shared base params are real owner tensors). Each
    is replaced by a full, replicated tensor copied from the matching owner
    parameter (gathered via :func:`gather_full_state_dict` so it is correct even
    when the owner is FSDP-sharded with CPU offload). The consumer's independent
    copy is a plain replicated tensor usable in its unsharded forward.

    Args:
        requires_grad: ``requires_grad`` for the materialized params. Consumers
            (teacher / EMA) are frozen, so this is normally ``False``.

    Raises:
        RuntimeError: if a meta consumer param has no matching owner param, or if
            any parameter is still on meta afterwards. No silent zero-init.
    """
    owner_values = gather_full_state_dict(owner)
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    materialized = 0
    for mod_name, module in consumer.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if not param.is_meta:
                continue
            full = f"{mod_name}.{param_name}" if mod_name else param_name
            src = _find_owner_param(full, owner_values)
            if src is None:
                raise RuntimeError(
                    f"materialize_independent_params: consumer meta param {full!r} has no matching "
                    f"owner parameter. Independent params must map to a real owner value — no zero-init fallback."
                )
            new_data = src.detach().to(device=device, dtype=param.dtype).clone()
            setattr(module, param_name, nn.Parameter(new_data, requires_grad=requires_grad))
            materialized += 1

    remaining = [n for n, p in consumer.named_parameters() if p.is_meta]
    if remaining:
        raise RuntimeError(
            f"materialize_independent_params: {len(remaining)} params still on meta after materialization. "
            f"First 5: {remaining[:5]}"
        )
    return materialized


def _fsdp_unit_owned_param_ids(owner: nn.Module) -> List[Tuple[str, nn.Module, Set[int]]]:
    """Return (path, unit, owned_param_ids) for each FSDP unit in *owner*.

    ``owned_param_ids`` excludes params managed by nested FSDP units so each
    parameter is attributed to exactly one (innermost) unit.
    """
    from torch.distributed.fsdp import FSDPModule

    units = [(name, m) for name, m in owner.named_modules() if isinstance(m, FSDPModule)]
    out: List[Tuple[str, nn.Module, Set[int]]] = []
    for name, unit in units:
        owned = {id(p) for p in unit.parameters()}
        for other_name, other in units:
            if other is unit:
                continue
            nested = other_name.startswith(f"{name}.") if name else True
            if nested:
                for p in other.parameters():
                    owned.discard(id(p))
        out.append((name, unit, owned))
    return out


def register_reverse_unshard_hooks(
    consumer: nn.Module,
    owner: nn.Module,
    shared_param_ids: Set[int],
) -> List[Tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]]:
    """Register unshard/reshard hooks so *consumer*'s forward can use owner FSDP params.

    For every owner FSDP unit whose *own* parameters are shared into the consumer:

    * If the consumer references that unit as a whole module by identity
      (a fully frozen block), no hook is registered — the unit's own FSDP
      forward hooks already fire (the "direct shared-module" path).
    * Otherwise (the consumer only shares an inner ``base_layer`` — e.g. a teacher
      plain ``Linear`` or an EMA LoRA layer), an explicit forward pre/post hook is
      registered on the consumer's corresponding module to ``unshard()`` the owner
      unit before its forward and ``reshard()`` it afterwards.

    Args:
        shared_param_ids: ``id()`` set from :attr:`ReverseTieResult.shared_param_ids`.

    Raises:
        RuntimeError: if a needed owner FSDP unit has no matching consumer module.
    """
    consumer_mods = dict(consumer.named_modules())
    handles: List[Tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]] = []
    direct: List[str] = []
    explicit: List[str] = []

    for name, unit, owned in _fsdp_unit_owned_param_ids(owner):
        if owned.isdisjoint(shared_param_ids):
            continue  # consumer does not reference this unit's own params
        cons_mod = consumer if name == "" else consumer_mods.get(name)
        if cons_mod is None:
            raise RuntimeError(
                f"register_reverse_unshard_hooks: owner FSDP unit {name!r} owns shared params but "
                f"the consumer has no module at that path. Reverse-tie structure is inconsistent."
            )
        if cons_mod is unit:
            direct.append(name or "<root>")
            continue

        def _make_hooks(fsdp_unit):
            def pre_hook(module, args):
                fsdp_unit.unshard()

            def post_hook(module, args, output):
                fsdp_unit.reshard()

            return pre_hook, post_hook

        pre_hook, post_hook = _make_hooks(unit)
        h_pre = cons_mod.register_forward_pre_hook(pre_hook)
        h_post = cons_mod.register_forward_hook(post_hook)
        handles.append((h_pre, h_post))
        explicit.append(name or "<root>")

    logger.info(
        f"[reverse_tie] unshard hooks — explicit={explicit} direct(identity)={direct} "
        f"(consumer={type(consumer).__name__})"
    )
    return handles
