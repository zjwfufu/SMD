# SPDX-License-Identifier: Apache-2.0
"""Numerical reductions and model ownership invariants."""

import torch
import torch.nn as nn


def per_sample_velocity_distance(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    batch_size: int,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Return Moment-style ``MSE(lhs, rhs) ** (beta / 2)`` per sample.

    Squared error is averaged over all non-batch dimensions before applying
    ``beta / 2``. ``beta=2`` returns exact MSE without epsilon smoothing.
    """
    if not 0.0 < beta <= 2.0:
        raise ValueError(f"Expected beta in (0, 2], got {beta}")
    if lhs.shape != rhs.shape:
        raise ValueError(f"Velocity shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    if lhs.shape[0] != batch_size:
        raise ValueError(f"Expected batch-first velocity with B={batch_size}, got {tuple(lhs.shape)}")
    mse = (lhs.float() - rhs.float()).reshape(batch_size, -1).square().mean(dim=1)
    if abs(beta - 2.0) < 1e-12:
        return mse
    return (mse + eps).pow(beta / 2.0)


def assert_lora_invariants(
    *,
    net: nn.Module,
    pivot: nn.Module,
    teacher: nn.Module | None,
    use_lora: bool,
    freeze_backbone_for_lora: bool,
    backbone_unfreeze_allowlist: list[str],
    pivot_train_mode: str,
    pivot_lora_backbone: nn.Module,
) -> None:
    """Verify teacher/pivot/student LoRA configuration is correct.

    Raises:
        RuntimeError: if any invariant is violated.
    """
    if teacher is not None:
        for n, p in teacher.named_parameters():
            if "lora_" in n:
                raise RuntimeError(f"teacher contains LoRA param: {n}")
            if p.requires_grad:
                raise RuntimeError(f"teacher param is trainable: {n}")

    pivot_block_ids = {id(parameter) for parameter in pivot_lora_backbone.parameters()}
    if pivot_train_mode == "full":
        for name, parameter in pivot.named_parameters():
            if "lora_" in name:
                raise RuntimeError(f"full Pivot contains LoRA param: {name}")
            if not parameter.requires_grad:
                raise RuntimeError(f"full Pivot param is frozen: {name}")
    elif pivot_train_mode == "lora":
        lora_count = 0
        for name, parameter in pivot.named_parameters():
            is_block = id(parameter) in pivot_block_ids
            is_lora = "lora_" in name
            if is_block:
                if is_lora:
                    lora_count += 1
                    if not parameter.requires_grad:
                        raise RuntimeError(f"Pivot LoRA param is frozen: {name}")
                elif parameter.requires_grad:
                    raise RuntimeError(f"Pivot block base param is unexpectedly trainable: {name}")
            elif not parameter.requires_grad:
                raise RuntimeError(f"Pivot non-block output/norm/modulation param is frozen: {name}")
        if lora_count == 0:
            raise RuntimeError("Pivot LoRA mode produced no trainable adapter parameters.")
    else:
        raise ValueError(f"Unknown pivot_train_mode={pivot_train_mode!r}")

    if not use_lora or not freeze_backbone_for_lora:
        return

    allow = list(backbone_unfreeze_allowlist)
    bad_trainable: list[str] = []
    bad_frozen: list[str] = []
    for n, p in net.named_parameters():
        is_lora = "lora_" in n
        is_allow = any(key in n for key in allow)
        if p.requires_grad and not (is_lora or is_allow):
            bad_trainable.append(n)
        if (is_lora or is_allow) and not p.requires_grad:
            bad_frozen.append(n)
    if bad_trainable:
        raise RuntimeError(
            f"net contains {len(bad_trainable)} unexpectedly trainable params, first 5: {bad_trainable[:5]}"
        )
    if bad_frozen:
        raise RuntimeError(f"net contains {len(bad_frozen)} unexpectedly frozen params, first 5: {bad_frozen[:5]}")


def validate_optimizer_matches_trainable(
    optimizers: list[torch.optim.Optimizer],
    modules: list[nn.Module],
) -> None:
    """Verify the union of optimizer param sets equals the trainable param set.

    Raises:
        RuntimeError: if the optimizer param set and the ``requires_grad`` set
            across *modules* differ (freeze ordering or optimizer construction
            is wrong).
    """
    opt_ids: set[int] = set()
    for opt in optimizers:
        for g in opt.param_groups:
            for p in g["params"]:
                opt_ids.add(id(p))
    trainable_ids: set[int] = set()
    for mod in modules:
        for p in mod.parameters():
            if p.requires_grad:
                trainable_ids.add(id(p))
    missing = trainable_ids - opt_ids
    extra = opt_ids - trainable_ids
    if missing or extra:
        raise RuntimeError(
            f"optimizer param set != trainable param set "
            f"(missing_from_opt={len(missing)}, extra_in_opt={len(extra)}); "
            f"freeze ordering or optimizer construction is wrong"
        )
