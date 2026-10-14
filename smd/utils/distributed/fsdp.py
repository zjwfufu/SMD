# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from functools import partial
from contextlib import contextmanager
import time
from typing import TYPE_CHECKING, Optional, Callable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
)

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


from smd.networks.network import SMDNetwork
from smd.utils.distributed import world_size, synchronize, is_rank0
from smd.utils.distributed.tie_weights import collect_no_shard_params
import smd.utils.logging_utils as logger

if TYPE_CHECKING:
    from smd.methods import SMDModel


def apply_fsdp_checkpointing(module: torch.nn.Module, check_fn: Optional[Callable[[torch.nn.Module], bool]] = None):
    """
    Apply FSDP checkpointing to a module.

    Follows overall approach outlined in https://pytorch.org/blog/maximizing-training/, without adaptive selection.

    Args:
        module: The module to wrap with activation checkpointing.
        check_fn: A function (Module -> bool) that takes a module and returns True if it should be wrapped.
            If None, wraps transformer blocks (Block in class name) only.
    """
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    if check_fn is None:
        # Default: only checkpoint block-level modules  (e.g. WanTransformerBlock in Wan network)
        # This is the correct granularity for FSDP + checkpointing
        def check_fn(submodule):
            # Check for common transformer block class names
            class_name = submodule.__class__.__name__
            return "Block" in class_name

        logger.info("Using default check_fn: checkpointing modules with 'Block' in class name")

    apply_activation_checkpointing(module, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


def model_to_fsdp(
    model: SMDModel,
    min_num_params: int = 10_000_000,
    apply_cpu_offload: bool = False,
    sync_module_states: bool = False,
    sharding_group_size: Optional[int] = None,
    no_sync_keys: set | None = None,
):
    """Convert model to FSDP.

    Args:
        model: The SMD model to wrap with FSDP.
        min_num_params: Minimum number of parameters for a module to be wrapped as a separate FSDP unit.
            Default is 10M, which wraps large models quite finely.
            For a 14B model, each transformer block has ~300-400M params, so a larger value may
            be preferable (e.g., 100M-500M).
        apply_cpu_offload: Whether to offload parameters to CPU.
        sync_module_states: If True, broadcast module states from rank 0 to all other ranks.
            Enable this when using memory-efficient loading where only rank 0 loads weights.
        no_sync_keys: Set of fsdp_dict keys that should skip broadcast even when
            sync_module_states=True. These modules are assumed to have identical weights
            on all ranks (e.g., teacher loaded from pretrained by every rank).

    Memory-Efficient Loading with Meta Device:
        To use memory-efficient loading for large models (14B+):
        1. Set `fsdp_meta_init=True` in trainer config
        2. Network classes should check `is_rank0()` and:
           - Rank 0: load full weights
           - Other ranks: use `with torch.device("meta"): load_model()` for ZERO memory allocation
        4. FSDP will materialize meta tensors and broadcast weights from rank 0

        This reduces initialization from N*model_size RAM to 1*model_size RAM,
        and avoids N parallel disk reads (major I/O contention).
    """
    fsdp_dict = model.fsdp_dict

    total_world_size = world_size()
    if sharding_group_size is None:
        # Full sharding over the world
        logger.info(f"Fully sharding model with {total_world_size} ranks...")
        device_mesh = init_device_mesh("cuda", (total_world_size,))
    else:
        if total_world_size % sharding_group_size != 0:
            raise ValueError(
                f"World size {total_world_size} must be divisible by shard group size {sharding_group_size}"
            )
        replica_group_size = total_world_size // sharding_group_size
        logger.info(f"Sharding model with {replica_group_size} sharding groups of size {sharding_group_size}...")
        device_mesh = init_device_mesh(
            "cuda", (replica_group_size, sharding_group_size), mesh_dim_names=("replicate", "shard")
        )

    # Mixed precision policy for FSDP2
    mp_policy = MixedPrecisionPolicy(
        param_dtype=model.precision,
        reduce_dtype=model.precision_fsdp,
        # We avoid casting all inputs so we can control t precision etc.
        output_dtype=None,
        cast_forward_inputs=False,
    )

    offload_policy = CPUOffloadPolicy() if apply_cpu_offload else None

    def _restore_unmanaged_submodules_to_gpu(module: nn.Module) -> None:
        """After CPU-offload init, move non-FSDP submodules back to GPU.

        CPU offload uses ``to_empty(device="cpu")`` which moves the ENTIRE
        network to CPU, then ``set_model_state_dict`` restores all weights —
        but with ``cpu_offload=True`` it leaves everything on CPU.  FSDP-managed
        parameters (DTensors) are automatically prefetched to GPU during forward;
        however, auxiliary frozen submodules (e.g. text_encoder, vae,
        image_cond_model) are plain Tensors and will stay on CPU forever.

        This function identifies submodules whose parameters are ALL plain
        Tensors (not DTensor) and moves them back to CUDA, where they belong.
        It also restores plain CPU buffers because FSDP only manages
        parameters: e.g. SD3.5's PatchEmbed has a DTensor convolution
        parameter but a plain positional-embedding buffer that would otherwise
        stay on CPU.  It also handles non-nn.Module preprocessor objects (e.g.
        QwenImageTextEncoder) that have a ``.to()`` method but aren't
        registered as nn.Module children.
        """
        cuda_device = torch.cuda.current_device()
        restored = []
        for name, child in module.named_children():
            if getattr(child, "_smd_keep_cpu_after_fsdp_init", False):
                continue
            child_params = list(child.parameters())
            if not child_params:
                continue
            # If ANY parameter is a DTensor, FSDP manages this subtree — skip it
            if any(isinstance(p.data, DTensor) for p in child_params):
                continue
            # All plain Tensors, stuck on CPU → move to GPU
            if all(p.device.type == "cpu" for p in child_params):
                child.to(device=cuda_device)
                n_params = sum(p.numel() for p in child_params)
                restored.append(f"{name}({n_params / 1e6:.0f}M)")

        # Also restore non-nn.Module preprocessors (e.g. QwenImageTextEncoder,
        # QwenImageImageEncoder) that have a .to() method but aren't registered
        # as nn.Module children — named_children() won't find them.
        for attr_name in ("text_encoder", "vae", "image_encoder"):
            attr = getattr(module, attr_name, None)
            if attr is None or isinstance(attr, nn.Module):
                continue  # nn.Module ones already handled above
            if getattr(attr, "_smd_keep_cpu_after_fsdp_init", False):
                continue
            if hasattr(attr, "to"):
                attr.to(device=cuda_device)
                restored.append(f"{attr_name}(non-Module)")

        if restored:
            logger.info(f"  Restored unmanaged submodules to GPU: {', '.join(restored)}")

        cpu_only_modules = {
            id(descendant)
            for child in module.modules()
            if getattr(child, "_smd_keep_cpu_after_fsdp_init", False)
            for descendant in child.modules()
        }
        restored_buffers = []
        for module_name, child in module.named_modules():
            if id(child) in cpu_only_modules:
                continue
            for buffer_name, buffer in child._buffers.items():
                if buffer is None or isinstance(buffer, DTensor) or buffer.device.type != "cpu":
                    continue
                child._buffers[buffer_name] = buffer.to(device=cuda_device)
                numel = buffer.numel()
                full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                restored_buffers.append(f"{full_name}({numel / 1e6:.0f}M)")

        if restored_buffers:
            logger.info(f"  Restored CPU buffers to GPU: {', '.join(restored_buffers)}")

    @contextmanager
    def _detach_frozen_preprocessors(module: nn.Module):
        detached: list[tuple[str, nn.Module]] = []
        for attr_name in ("text_encoder", "vae", "image_encoder"):
            child = getattr(module, attr_name, None)
            if isinstance(child, nn.Module) and not any(parameter.requires_grad for parameter in child.parameters()):
                detached.append((attr_name, child))
                setattr(module, attr_name, None)
        try:
            yield
        finally:
            for attr_name, child in detached:
                setattr(module, attr_name, child)

    for k, v in fsdp_dict.items():
        if k.startswith("ema"):
            logger.warning("EMA network stored in fsdp_dict will be skipped during FSDP2 wrap.")
            continue

        # Collect shared/tied params that should be excluded from this FSDP wrap.
        # Two cases: (1) legacy _no_shard marker, (2) DTensors already sharded
        # by another module's FSDP (e.g. teacher wrapped before student).
        ignored = collect_no_shard_params(v)
        if ignored:
            logger.info(f"[FSDP] '{k}': {len(ignored)} ignored_params (shared/tied)")
            for p in ignored:
                # Pre-move plain tensors to CUDA; skip DTensors (FSDP manages them)
                if not isinstance(p, DTensor) and not p.is_cuda:
                    p.data = p.data.to(torch.cuda.current_device())

        num_params = sum(p.numel() for p in v.parameters()) / 1e9
        logger.info(f"Starting FSDP2 wrap for '{k}' ({num_params:.2f}B params)...")
        t0 = time.time()

        # Step 1: Extract full state dict from rank 0 BEFORE sharding
        # Rank 0 has real weights, other ranks have meta tensors
        # Per-key sync: skip broadcast for keys in no_sync_keys (all ranks have identical weights)
        key_needs_sync = sync_module_states and (no_sync_keys is None or k not in no_sync_keys)
        if key_needs_sync:
            if is_rank0():
                # Rank 0 has the real weights - extract them
                # TODO: This can be slow. Instead, we should shard before loading pre-trained weights.
                state_dict = copy.deepcopy(v.state_dict())
                logger.info(f"  [Rank 0] Extracted state dict with {len(state_dict)} tensors")
            else:
                state_dict = None
        else:
            state_dict = None

        # Step 2: Apply fully_shard to create DTensor structure
        if isinstance(v, SMDNetwork) or getattr(v, "_smd_custom_fully_shard", False):
            # We use the network's custom fully_shard method
            v.fully_shard(
                mesh=device_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                ignored_params=ignored if ignored else None,
            )
        else:
            # Apply activation checkpointing before sharding (order matters for FSDP2).
            # Default check_fn wraps modules with "Block" in class name.
            apply_fsdp_checkpointing(v)

            # Shard each direct child, skipping containers (ModuleList, Sequential, etc.)
            # that don't implement forward. For containers, shard their children instead.
            def _shard_children(module: torch.nn.Module, _ignored=ignored) -> None:
                for child in module.children():
                    if isinstance(child, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                        _shard_children(child)  # recurse into container
                    elif sum(p.numel() for p in child.parameters()) >= min_num_params:
                        fully_shard(
                            child,
                            mesh=device_mesh,
                            mp_policy=mp_policy,
                            offload_policy=offload_policy,
                            reshard_after_forward=True,
                            **({"ignored_params": _ignored} if _ignored else {}),
                        )

            _shard_children(v)

            fully_shard(
                v,
                mesh=device_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=True,
                **({"ignored_params": ignored} if ignored else {}),
            )
        logger.info("Completed sharding")

        # Step 3: Move parameters to correct device and optionally broadcast state dict
        # With CPU offloading, use CPU as target device so FSDP can manage GPU placement
        # Without CPU offloading, use CUDA directly
        target_device = "cpu" if apply_cpu_offload else torch.cuda.current_device()

        if key_needs_sync:
            logger.info("Syncing module states from rank 0 to all ranks")
            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                cpu_offload=apply_cpu_offload,
            )

            logger.debug(f"Moving all ranks to target device: {target_device}")
            v.to_empty(device=target_device)
            if hasattr(v, "reset_parameters"):
                # We need this to reinitialize non-persistent buffers like RoPE freqs_cos/freqs_sin
                # These aren't stored in the state dict, so we need to reinitialize them after to_empty()
                v.reset_parameters()
            else:
                logger.warning(
                    f"Network {v.__class__.__name__} does not implement the reset_parameters method. "
                    "This may cause unexpected behavior with FSDP2, like non-persistent buffers not "
                    "being initialized correctly."
                )
            synchronize()
            logger.debug("Moved all ranks to target devices")

            logger.debug("Broadcasting the state dict to all ranks")

            set_model_state_dict(v, model_state_dict=state_dict, options=options)
            torch.cuda.empty_cache()
            if apply_cpu_offload:
                _restore_unmanaged_submodules_to_gpu(v)
        else:
            if sync_module_states and no_sync_keys and k in no_sync_keys:
                logger.info(f"Skipping broadcast for '{k}' (no_sync_keys): all ranks loaded independently")
            if apply_cpu_offload:
                # CPU offload for no_sync modules: we must go through
                # set_model_state_dict (same as the sync path) so that FSDP2
                # internal bookkeeping knows params live on CPU and will
                # migrate them to GPU during forward.  Simply moving the
                # underlying _local_tensor to CPU is NOT enough — FSDP2 won't
                # know to bring them back.
                #
                # The difference from the sync path: we use each rank's own
                # state dict (no broadcast_from_rank0) since all ranks loaded
                # weights independently.
                logger.info(f"Applying CPU offload to no_sync module '{k}' via set_model_state_dict")
                with _detach_frozen_preprocessors(v):
                    local_state_dict = {n: p.detach().clone() for n, p in v.state_dict().items()}
                    options = StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=False,
                        cpu_offload=True,
                    )
                    v.to_empty(device="cpu")
                    if hasattr(v, "reset_parameters"):
                        v.reset_parameters()
                    synchronize()
                    set_model_state_dict(v, model_state_dict=local_state_dict, options=options)
                    del local_state_dict
                torch.cuda.empty_cache()
                _restore_unmanaged_submodules_to_gpu(v)
                logger.info(f"CPU offload applied to '{k}'")
            else:
                v.to(device=target_device)
        synchronize()
        logger.info(f"FSDP2 wrapped {k} in {time.time() - t0:.1f}s")

    # Post-FSDP hook: allow model to perform additional tying (e.g., LoRA structural tying)
    if hasattr(model, "post_fsdp_wrap"):
        model.post_fsdp_wrap()

    return model


def _get_submodules_to_shard(module: torch.nn.Module, min_num_params: int) -> list[torch.nn.Module]:
    """Get list of submodules that should be sharded based on parameter count."""
    modules_to_shard = []

    def _count_params(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters(recurse=False))

    def _recurse(m: torch.nn.Module) -> None:
        for child in m.children():
            _recurse(child)
        own_params = _count_params(m)
        if own_params >= min_num_params:
            modules_to_shard.append(m)

    _recurse(module)
    return modules_to_shard


@contextmanager
def fsdp_sync_grad(model: SMDModel, enabled: bool):
    """
    Context manager to enable/disable gradient synchronization for FSDP2 modules.

    This mirrors DDP's no_sync behavior for gradient accumulation: for non-last
    microbatches, set_requires_gradient_sync(False) to skip communication.
    """
    fsdp_modules = [
        m
        for m in model.fsdp_dict.values()
        if hasattr(m, "set_requires_gradient_sync") and any(p.requires_grad for p in m.parameters())
    ]
    if fsdp_modules:
        for module in fsdp_modules:
            module.set_requires_gradient_sync(enabled, recurse=True)
            if hasattr(module, "set_is_last_backward"):
                module.set_is_last_backward(enabled)
    try:
        yield
    finally:
        if fsdp_modules and not enabled:
            for module in fsdp_modules:
                module.set_requires_gradient_sync(True, recurse=True)
                if hasattr(module, "set_is_last_backward"):
                    module.set_is_last_backward(True)


# ---------------------------------------------------------------------------
# FSDP2 unshard context for gradient checkpointing compatibility
# ---------------------------------------------------------------------------


@contextmanager
def fsdp_unshard_for_checkpoint(module: nn.Module):
    """Temporarily unshard all FSDP units and disable their hooks for
    ``torch.utils.checkpoint`` compatibility.

    ``torch.utils.checkpoint`` recomputes the forward during backward.  FSDP2's
    pre-forward hooks expect a clean forward-pass state machine, causing
    DTensor/Tensor mixing errors during recompute.  This context manager:

    1. Calls ``unshard()`` on every FSDP unit → params become plain Tensors.
    2. Clears pre/post forward hooks → FSDP won't re-shard during recompute.
    3. Sets ``reshard_after_backward = False`` → backward won't re-shard either.

    **IMPORTANT**: ``backward()`` MUST be called INSIDE this context (before
    ``__exit__`` restores hooks and reshards). Any active integration must expose
    that choice explicitly at the Trainer/model call boundary; do not encode
    backward control flow in the loss map.

    After exit, hooks are restored and all units are resharded back to normal.

    Usage::

        with fsdp_unshard_for_checkpoint(model.net):
            # forward with torch.utils.checkpoint
            x = checkpoint(step_fn, x, ...)
            loss = compute_loss(x)
            # backward MUST happen here
            loss.backward()
        # FSDP state is restored; subsequent steps work normally.
    """
    from torch.distributed._composable.fsdp import FSDPModule

    fsdp_modules = [mod for mod in module.modules() if isinstance(mod, FSDPModule)]

    # 1. Unshard all FSDP units (params become plain Parameter, not DTensor)
    for mod in fsdp_modules:
        mod.unshard()

    # 2. Save and clear forward hooks (prevent FSDP from re-sharding during recompute)
    saved_hooks = {}
    for i, mod in enumerate(fsdp_modules):
        state = mod._get_fsdp_state()
        saved_hooks[i] = {
            "pre": dict(mod._forward_pre_hooks),
            "post": dict(mod._forward_hooks),
            "reshard_bwd": state._fsdp_param_group.reshard_after_backward if state._fsdp_param_group else True,
        }
        mod._forward_pre_hooks.clear()
        mod._forward_hooks.clear()
        if state._fsdp_param_group:
            state._fsdp_param_group.reshard_after_backward = False

    try:
        yield
    finally:
        # 3. Restore hooks and reshard (after backward is done)
        for i, mod in enumerate(fsdp_modules):
            state = mod._get_fsdp_state()
            mod._forward_pre_hooks.update(saved_hooks[i]["pre"])
            mod._forward_hooks.update(saved_hooks[i]["post"])
            if state._fsdp_param_group:
                state._fsdp_param_group.reshard_after_backward = saved_hooks[i]["reshard_bwd"]
        for mod in fsdp_modules:
            mod.reshard()
