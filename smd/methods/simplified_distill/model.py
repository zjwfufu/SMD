# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Student/Pivot model ownership, independent of objectives and trajectory execution.

"""

from __future__ import annotations
from contextlib import contextmanager
import torch
from smd.methods.model import SMDModel
from .ema import PivotEMAMixin
from .pivot import create_pivot_backend
from .basis import create_basis
from .utils import assert_lora_invariants, validate_optimizer_matches_trainable
from smd.configs.opt import get_scheduler
from smd.utils import instantiate
from smd.utils.distributed import synchronize, world_size
from smd.utils.distributed.tie_weights import (
    materialize_independent_params,
    register_reverse_unshard_hooks,
    reverse_tie_frozen_base,
)
from smd.utils.checkpoint_policy import CheckpointPolicy, CheckpointSaveMode, ModuleCheckpointPolicy
from smd.utils.lora_utils import freeze_for_lora
import smd.utils.logging_utils as log
from .settings import validate_config


@contextmanager
def _optional_init_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


class StudentPivotModel(PivotEMAMixin, SMDModel):
    """Own network modules and lifecycle; no training step or loss computation."""

    def __init__(self, config):
        self._init_pivot_ema_state()
        self._reverse_tied: bool = False
        self._teacher_reverse_result = None
        self._reverse_hook_handles: list = []
        super().__init__(config)
        self.config = config

    def build_model(self):
        validate_config(self.config)
        self.build_student()
        cfg = self.config
        self.basis = create_basis(cfg)
        self._n_grid = self.basis.n_grid
        self._proj_out_init_mode = "zero"
        self._proj_out_init_plan = None
        self._hook_layer = int(cfg.pivot_hook_layer)
        self._pivot_num_blocks = cfg.pivot_num_blocks
        self._pivot_block_init = cfg.pivot_block_init
        self._pivot_block_init_seed = cfg.pivot_block_init_seed
        self._no_pivot = cfg.no_pivot
        self._pivot_train_mode = cfg.pivot_train_mode
        self.build_pivot()
        self._configure_pivot_training()
        self._configure_student_lora()
        self.build_teacher_for_distill()
        self.build_ema()
        assert_lora_invariants(
            net=self.net,
            pivot=self.pivot,
            teacher=self.teacher,
            use_lora=bool(getattr(self.net, "use_lora", False)),
            freeze_backbone_for_lora=cfg.freeze_backbone_for_lora,
            backbone_unfreeze_allowlist=list(cfg.backbone_unfreeze_allowlist),
            pivot_train_mode=self._pivot_train_mode,
            pivot_lora_backbone=self.backend.get_lora_backbone(self.pivot),
        )
        self._log_trainable_breakdown()

    @property
    def is_lora_mode(self) -> bool:
        return bool(getattr(self.net, "use_lora", False)) or getattr(self, "_pivot_train_mode", "full") == "lora"

    def build_student(self):
        """Instantiate + initialize the real student on all ranks (no rank0 broadcast)."""
        log.info("SimplifiedDistill: instantiating student on all ranks (force_all_rank_load)")
        self.net = self._coordinated_instantiate(self.config.net, all_rank_read=True)
        self.net.train().requires_grad_(True)
        self._build_preprocessors(self.net)
        self.initialize_student()

    def initialize_student(self):
        """Apply the Pivot-specific ``student_init_checkpoint`` (all ranks, no broadcast)."""
        ckpt = self.config.student_init_checkpoint
        if not ckpt:
            return
        log.info(f"SimplifiedDistill: loading student_init_checkpoint from {ckpt}")
        SMDModel._load_pretrained_model(self.net, ckpt, fsdp_meta_init=False)
        synchronize()

    def build_pivot(self):
        """Build the deep pivot decoder from the initialized, pre-LoRA student."""
        self.backend = create_pivot_backend(self.net)
        self.pivot = self.backend.build_pivot(
            self.net,
            self._hook_layer,
            self._n_grid,
            proj_out_init_mode=self._proj_out_init_mode,
            proj_out_init_plan=self._proj_out_init_plan,
        )
        self.backend.configure_pivot_blocks(
            self.pivot,
            self.net,
            self._hook_layer,
            self._pivot_num_blocks,
            self._pivot_block_init,
            self._pivot_block_init_seed,
        )
        self.pivot.requires_grad_(True)
        self._assert_pivot_from_student()
        log.info(
            f"Basis={type(self.basis).__name__}, proj_out_init_mode={self._proj_out_init_mode}, no_pivot={self._no_pivot}"
        )

    def _assert_pivot_from_student(self):
        """Strict proof that the pivot's backend + block shapes originate from the student."""
        student_backend = create_pivot_backend(self.net)
        if type(self.backend) is not type(student_backend):
            raise RuntimeError(
                f"pivot backend {type(self.backend).__name__} != student backend {type(student_backend).__name__}: pivot origin is not the student."
            )
        if self.backend.get_transformer(self.net) is not student_backend.get_transformer(self.net):
            raise RuntimeError("pivot backend transformer is not the student's transformer.")
        student_blocks = self.backend.enumerate_blocks(self.net)
        n = len(student_blocks)
        pos = self._hook_layer if self._hook_layer >= 0 else n + self._hook_layer
        if not 0 <= pos < n:
            raise IndexError(f"pivot_hook_layer={self._hook_layer} out of range for {n} student blocks")
        tail = student_blocks[pos + 1 :]
        pivot_blocks = list(getattr(self.pivot, "blocks", []))
        expected_count = len(tail) if self._pivot_num_blocks is None else self._pivot_num_blocks
        if len(pivot_blocks) != expected_count:
            raise RuntimeError(
                f"pivot has {len(pivot_blocks)} decoder blocks but pivot_num_blocks resolves to {expected_count} (hook_layer={self._hook_layer}, init={self._pivot_block_init})."
            )
        if self._pivot_block_init == "pretrained":
            if expected_count != len(tail):
                raise RuntimeError(
                    f"pretrained Pivot blocks must exactly match the natural student tail: expected {len(tail)}, got {expected_count}."
                )
            reference_blocks = tail
        else:
            reference_blocks = [student_blocks[-1]] * expected_count
        for i, (pb, sb) in enumerate(zip(pivot_blocks, reference_blocks)):
            p_shapes = {name: tuple(p.shape) for name, p in pb.named_parameters()}
            s_shapes = {name: tuple(p.shape) for name, p in sb.named_parameters()}
            if p_shapes != s_shapes:
                raise RuntimeError(
                    f"pivot decoder block {i} parameter shapes differ from the student block architecture."
                )

    def _configure_student_lora(self):
        """Attach LoRA to the student and freeze the backbone (except the allowlist)."""
        if not bool(getattr(self.config.net, "use_lora", False)):
            return
        if self.config.lora_init_seed is None:
            log.warning(
                "lora_init_seed=None uses ambient model-construction RNG before Trainer seeding; trainer.seed alone will not reproduce the initial LoRA weights. Under no-sync FSDP, rank-local initializations are sharded into one patchwork global adapter."
            )
        with _optional_init_seed(self.config.lora_init_seed):
            self.net.attach_lora()
        if self.config.freeze_backbone_for_lora:
            freeze_for_lora(self.net, allowlist=list(self.config.backbone_unfreeze_allowlist))

    def _validate_lora_init_seeds_for_runtime(self, is_fsdp: bool) -> None:
        """Reject ambient LoRA initialization before multi-rank FSDP sharding."""
        if not is_fsdp or world_size() <= 1:
            return
        unseeded = []
        if bool(getattr(self.config.net, "use_lora", False)) and self.config.lora_init_seed is None:
            unseeded.append("lora_init_seed")
        if self._pivot_train_mode == "lora" and self.config.pivot_lora_init_seed is None:
            unseeded.append("pivot_lora_init_seed")
        if unseeded:
            fields = ", ".join(unseeded)
            raise ValueError(
                f"Multi-rank FSDP requires explicit construction-time LoRA seeds; got {fields}=None. Ambient rank-local initializations would be sharded into a patchwork global adapter. Set deterministic integer seeds before launching or override them when resuming an old checkpoint."
            )

    def _configure_pivot_training(self) -> None:
        """Keep the Pivot full-rank or train block adapters plus full output components."""
        if self._pivot_train_mode == "full":
            self.pivot.requires_grad_(True)
            return
        if self._pivot_train_mode != "lora":
            raise ValueError(f"Unknown pivot_train_mode={self._pivot_train_mode!r}")
        student_rank = getattr(self.config.net, "lora_rank", None)
        student_alpha = getattr(self.config.net, "lora_alpha", None)
        student_dropout = getattr(self.config.net, "lora_dropout", None)
        rank_value = self.config.pivot_lora_rank
        alpha_value = self.config.pivot_lora_alpha
        dropout_value = self.config.pivot_lora_dropout
        rank_source = student_rank if rank_value is None else rank_value
        if rank_source is None:
            raise ValueError("Pivot LoRA rank is unset and config.net exposes no lora_rank to inherit.")
        rank = int(rank_source)
        alpha_source = student_alpha if alpha_value is None else alpha_value
        alpha = rank if alpha_source is None else int(alpha_source)
        dropout_source = student_dropout if dropout_value is None else dropout_value
        dropout = 0.0 if dropout_source is None else float(dropout_source)
        if rank < 1:
            raise ValueError(f"pivot_lora_rank must resolve to >= 1, got {rank}")
        if alpha < 1:
            raise ValueError(f"pivot_lora_alpha must resolve to >= 1, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"pivot_lora_dropout must resolve to [0, 1), got {dropout}")
        backbone = self.backend.get_lora_backbone(self.pivot)
        exact_targets = self.net.resolve_lora_targets_for_module(backbone, recipe_prefix="blocks.")
        with _optional_init_seed(self.config.pivot_lora_init_seed):
            injected = self.net.attach_lora_to_module(
                backbone, rank=rank, alpha=alpha, dropout=dropout, target_modules=exact_targets
            )
        self.backend.set_lora_backbone(self.pivot, injected)
        backbone = self.backend.get_lora_backbone(self.pivot)
        pivot_dtype = next(self.pivot.parameters()).dtype
        backbone.to(dtype=pivot_dtype)
        freeze_for_lora(backbone, allowlist=[])
        backbone_param_ids = {id(value) for value in backbone.parameters()}
        for _name, parameter in self.pivot.named_parameters():
            if id(parameter) not in backbone_param_ids:
                parameter.requires_grad_(True)
        log.info(
            f"Pivot LoRA configured: rank={rank}, alpha={alpha}, dropout={dropout}, copied block bases frozen; non-block output components full-trainable"
        )

    def build_teacher_for_distill(self):
        """Build the same-architecture teacher."""
        if bool(self.config.tie_backbone):
            self._build_reverse_tied_teacher()
        else:
            self.build_teacher(force_all_rank_load=True)
            self.teacher.eval().requires_grad_(False)

    def _build_reverse_tied_teacher(self):
        """Build a frozen meta-skeleton teacher whose immutable base reverse-ties to the student."""
        log.info("tie_backbone: building meta-skeleton teacher (reverse-tie to student-owned base)")
        if self.precision != self.precision_fsdp:
            raise ValueError(
                f"tie_backbone reverse-tie requires precision == precision_fsdp (got precision={self.precision}, precision_fsdp={self.precision_fsdp}). The teacher shares the student's parameter storage, so mixed precision between the shared base and the FSDP shard is not supported."
            )
        with torch.device("meta"):
            self.teacher = instantiate(self.teacher_config)
        self.teacher.to(dtype=self.precision_fsdp)
        student_tf = self.backend.get_transformer(self.net)
        teacher_tf = self.backend.get_transformer(self.teacher)
        result = reverse_tie_frozen_base(teacher_tf, student_tf)
        materialize_independent_params(self.teacher, self.net, requires_grad=False, device=self.device)
        self.teacher.eval().requires_grad_(False)
        self._teacher_reverse_result = result
        self._reverse_tied = True
        log.info(
            f"tie_backbone reverse-tie: identity_modules={len(result.identity_modules)} base_layer_binds={len(result.base_layer_binds)} independent={len(result.independent_params)}"
        )

    def init_optimizers(self):
        super().init_optimizers()
        self.pivot_optimizer = instantiate(self.config.pivot_optimizer, model=self.pivot)
        self.pivot_lr_scheduler = get_scheduler(self.pivot_optimizer, self.config.pivot_scheduler)
        validate_optimizer_matches_trainable([self.net_optimizer, self.pivot_optimizer], [self.net, self.pivot])

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self.net_optimizer, self.pivot_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler._LRScheduler]:
        del iteration
        return [self.net_lr_scheduler, self.pivot_lr_scheduler]

    @property
    def model_dict(self):
        modules = {"net": self.net, "pivot": self.pivot, **self.ema_module_dict}
        return torch.nn.ModuleDict(modules)

    def checkpoint_policy(self) -> CheckpointPolicy:
        """Explicit LoRA-only checkpoint policy for the Pivot module set."""
        policy: CheckpointPolicy = {
            "net": ModuleCheckpointPolicy(CheckpointSaveMode.TRAINABLE_SUBSET),
            "pivot": ModuleCheckpointPolicy(
                CheckpointSaveMode.TRAINABLE_SUBSET if self._pivot_train_mode == "lora" else CheckpointSaveMode.FULL
            ),
        }
        policy.update(self.ema_checkpoint_policy(mirror_source="net"))
        return policy

    @property
    def net_inference(self):
        """Return live student for SimplifiedDistill inference; EMA is exposed as ``net_ema``."""
        return self.net

    @property
    def fsdp_dict(self):
        d = {"pivot": self.pivot}
        if getattr(self, "_reverse_tied", False):
            d["net"] = self.net
        else:
            d["net"] = self.net
            if getattr(self, "teacher", None) is not None and self.config.add_teacher_to_fsdp_dict:
                d["teacher"] = self.teacher
        return d

    @property
    def fsdp_no_sync_keys(self) -> set | None:
        keys = {"net"}
        if not getattr(self, "_reverse_tied", False):
            if getattr(self, "teacher", None) is not None and self.config.add_teacher_to_fsdp_dict:
                keys.add("teacher")
        return keys if keys else None

    def post_fsdp_wrap(self):
        """After the student is FSDP-wrapped, register reverse unshard/reshard hooks."""
        if not getattr(self, "_reverse_tied", False):
            return
        self._reverse_hook_handles = []
        student_tf = self.backend.get_transformer(self.net)
        teacher_tf = self.backend.get_transformer(self.teacher)
        shared_ids = self._recompute_shared_param_ids(teacher_tf, student_tf)
        if not shared_ids:
            raise RuntimeError(
                "post_fsdp_wrap: reverse-tied teacher shares no params with the FSDP student. Reverse-tie was lost across FSDP wrapping."
            )
        self._reverse_hook_handles.extend(register_reverse_unshard_hooks(teacher_tf, student_tf, shared_ids))
        self._reverse_hook_handles.extend(self.register_ema_reverse_hooks(student_tf))

    def on_train_begin(self, is_fsdp=False):
        """Move pivot to correct device; handle meta tensors for FSDP."""
        self._validate_lora_init_seeds_for_runtime(is_fsdp)
        target_dtype = self.precision_fsdp if is_fsdp else self.precision
        is_meta = any((p.is_meta for p in self.pivot.parameters()))
        if is_meta:
            self.pivot.to_empty(device=self.device)
            self.pivot.to(dtype=target_dtype)
        else:
            self.pivot.to(dtype=target_dtype, device=self.device)
        super().on_train_begin(is_fsdp=is_fsdp)
        self.place_ema_modules(is_fsdp)

    @property
    def optimizer_dict(self):
        return {"net": self.net_optimizer, "pivot": self.pivot_optimizer}

    @property
    def scheduler_dict(self):
        return {"net": self.net_lr_scheduler, "pivot": self.pivot_lr_scheduler}

    def _log_trainable_breakdown(self) -> None:
        """Log trainable vs total param count for each fsdp_dict submodule."""
        modules: list[tuple[str, torch.nn.Module]] = []
        for name in ("net", "pivot", "teacher"):
            mod = getattr(self, name, None)
            if isinstance(mod, torch.nn.Module) and mod not in [m for _, m in modules]:
                modules.append((name, mod))
        for name, mod in modules:
            n_trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in mod.parameters())
            pct = (100.0 * n_trainable / n_total) if n_total > 0 else 0.0
            log.info(
                f"[trainable_breakdown] {name}: trainable={n_trainable / 1e6:.2f}M / "
                f"total={n_total / 1e6:.2f}M ({pct:.2f}%)"
            )
