# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import io
import shutil
import torch.nn
import torch.distributed as dist
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Dict, Union, Any

from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

from smd.configs.config import BaseCheckpointerConfig
from smd.utils.checkpoint_policy import (
    CheckpointPolicy,
    CheckpointSaveMode,
    ModuleCheckpointPolicy,
    structural_default_policy,
)
from smd.utils.distributed.s3_filesystem import S3StorageWriter, S3StorageReader
from smd.utils.io_utils import s3_load, s3_save, latest_checkpoint
import smd.utils.logging_utils as logger
from smd.utils.distributed import synchronize, is_rank0
from smd.callbacks.callback import CallbackDict


def _normalize_param_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "")


def _match_optimizer_state_template(state_dict: dict, metadata) -> list[str]:
    """Retain only parameter states actually saved by the optimizer.

    Adam state is lazy: an unused trainable branch has no moments or step. The
    DCP template initializer creates state for every parameter, so prune those
    wholly absent states. Entries of a present state remain strict load keys.
    """
    keys = metadata.state_dict_metadata
    states = state_dict.get("state", {})
    names = set(states)
    for group in state_dict.get("param_groups", []):
        names.update(group["params"])
    absent = [name for name in names if not any(key.startswith(f"state.{name}.") for key in keys)]
    for name in absent:
        states.pop(name, None)
    return absent


def _restore_uninitialized_optimizer_states(state_dict: dict, absent: list[str]) -> None:
    # PyTorch's FQN-to-parameter mapper expects an entry for every trainable
    # parameter. An empty entry means Adam has never observed a gradient.
    for name in absent:
        state_dict["state"][name] = {}


def _drop_empty_optimizer_states(optimizer) -> None:
    for parameter in list(optimizer.state):
        if not optimizer.state[parameter]:
            del optimizer.state[parameter]


def _is_local_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _copy_local_dcp_staging(staging_dir: str, final_path: str, *, include_metadata: bool) -> None:
    for entry in os.scandir(staging_dir):
        if not entry.is_file():
            continue
        if entry.name == ".metadata" and not include_metadata:
            continue
        destination = os.path.join(final_path, entry.name)
        if os.path.exists(destination):
            raise RuntimeError(f"Duplicate staged DCP file while merging nodes: {destination}")
        shutil.copy2(entry.path, destination)


def _missing_dcp_files(checkpoint_path: str) -> list[str]:
    metadata = FileSystemReader(checkpoint_path).read_metadata()
    expected = {os.fspath(info.relative_path) for info in metadata.storage_data.values()}
    return sorted(
        relative_path for relative_path in expected if not os.path.isfile(os.path.join(checkpoint_path, relative_path))
    )


def _validate_dcp_files(checkpoint_path: str) -> None:
    missing = _missing_dcp_files(checkpoint_path)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise RuntimeError(f"Incomplete DCP checkpoint at {checkpoint_path}: missing {preview}{suffix}")


def _trainable_param_names(module: torch.nn.Module) -> set[str]:
    return {_normalize_param_name(name) for name, p in module.named_parameters() if p.requires_grad}


def _trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
    state = get_model_state_dict(module, options=options)
    trainable_names = _trainable_param_names(module)
    return {name: value for name, value in state.items() if _normalize_param_name(name) in trainable_names}


@contextmanager
def _detach_frozen_preprocessors(module: torch.nn.Module):
    detached: list[tuple[str, torch.nn.Module]] = []
    for name in ("text_encoder", "vae", "image_encoder"):
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.Module) and not any(parameter.requires_grad for parameter in child.parameters()):
            detached.append((name, child))
            setattr(module, name, None)
    try:
        yield
    finally:
        for name, child in detached:
            setattr(module, name, child)


def _temporarily_mark_trainable(module: torch.nn.Module, names: set[str]) -> list[tuple[torch.nn.Parameter, bool]]:
    changed: list[tuple[torch.nn.Parameter, bool]] = []
    for name, p in module.named_parameters():
        if _normalize_param_name(name) in names and not p.requires_grad:
            changed.append((p, p.requires_grad))
            p.requires_grad_(True)
    return changed


def _restore_requires_grad(changed: list[tuple[torch.nn.Parameter, bool]]) -> None:
    for p, requires_grad in changed:
        p.requires_grad_(requires_grad)


@contextmanager
def _bind_optimizer_to_current_parameters(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    param_names: list[list[str]],
):
    current = {_normalize_param_name(name): parameter for name, parameter in module.named_parameters()}
    original_groups = [list(group["params"]) for group in optimizer.param_groups]
    replacements: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
    for group, names, original_parameters in zip(optimizer.param_groups, param_names, original_groups, strict=True):
        try:
            current_parameters = [current[name] for name in names]
        except KeyError as error:
            raise RuntimeError(
                f"Checkpoint optimizer parameter {error.args[0]!r} is missing from the module."
            ) from error
        group["params"] = current_parameters
        replacements.update(zip(original_parameters, current_parameters, strict=True))

    original_state = optimizer.state
    rebound_state = defaultdict(dict)
    for parameter, state in original_state.items():
        rebound_state[replacements.get(parameter, parameter)] = state
    optimizer.state = rebound_state
    try:
        yield
    finally:
        for group, parameters in zip(optimizer.param_groups, original_groups, strict=True):
            group["params"] = parameters
        optimizer.state = original_state


def _resolve_policy(
    checkpoint_policy: Optional[CheckpointPolicy], key: str, module: torch.nn.Module
) -> ModuleCheckpointPolicy:
    """Return the explicit policy for ``key``, or a structural fallback.

    The checkpointer never infers behavior from key strings: an explicit policy
    entry wins, and any key without one (e.g. the single-key dict built by
    ``Trainer.load_pretrained_ckpt``) falls back to a structural policy derived
    purely from the module's ``requires_grad`` layout.
    """
    if checkpoint_policy is not None and key in checkpoint_policy:
        return checkpoint_policy[key]
    return structural_default_policy(module)


def _mirror_trainable_names(policy: ModuleCheckpointPolicy, model_dict: "torch.nn.ModuleDict") -> set[str]:
    """Trainable param names of an EMA policy's mirror source (or empty set)."""
    source = model_dict[policy.mirror_source] if policy.mirror_source in model_dict else None
    return _trainable_param_names(source) if source is not None else set()


def _group_key(name: str) -> str:
    """Bucket a parameter / buffer FQN into a coarse group for diagnostic logging.

    Groups are chosen to make LoRA / pivot / teacher mismatches obvious
    when loading checkpoints with strict=False.
    """
    if "lora_" in name or ".lora_A" in name or ".lora_B" in name:
        return "lora"
    if name.startswith("pivot") or ".pivot" in name:
        return "pivot"
    if name.startswith("teacher") or ".teacher" in name:
        return "teacher"
    # First two dotted segments form a stable bucket (e.g. transformer.blocks).
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "<root>")


def _log_load_mismatches(model_key: str, load_info) -> None:
    """Group-log the missing/unexpected keys from a ``load_state_dict`` result.

    Suspicious buckets (anything other than ``lora`` / ``pivot``) are
    escalated to ERROR so that silent backbone drift in strict=False loads is
    immediately visible. LoRA + pivot buckets are expected to be
    missing when loading a vanilla pretrained checkpoint into a LoRA-attached
    student, and are logged at INFO instead.
    """
    missing = list(getattr(load_info, "missing_keys", []) or [])
    unexpected = list(getattr(load_info, "unexpected_keys", []) or [])
    if not missing and not unexpected:
        logger.info(f"Model {model_key}: state_dict loaded cleanly (no missing/unexpected keys).")
        return

    def _bucket(keys):
        groups: Dict[str, int] = {}
        for k in keys:
            g = _group_key(k)
            groups[g] = groups.get(g, 0) + 1
        return groups

    miss_groups = _bucket(missing)
    unex_groups = _bucket(unexpected)
    benign = {"lora", "pivot"}

    if missing:
        logger.info(f"Model {model_key}: {len(missing)} missing keys by group: {miss_groups}")
        suspicious = {g: n for g, n in miss_groups.items() if g not in benign}
        if suspicious:
            logger.error(f"Model {model_key}: SUSPICIOUS missing keys (non-LoRA/non-pivot): {suspicious}")
            # Show up to 10 sample FQNs to aid debugging without flooding logs.
            samples = [k for k in missing if _group_key(k) not in benign][:10]
            logger.error(f"Model {model_key}: sample suspicious missing keys: {samples}")

    if unexpected:
        logger.info(f"Model {model_key}: {len(unexpected)} unexpected keys by group: {unex_groups}")
        suspicious = {g: n for g, n in unex_groups.items() if g not in benign}
        if suspicious:
            logger.error(f"Model {model_key}: SUSPICIOUS unexpected keys (non-LoRA/non-pivot): {suspicious}")
            samples = [k for k in unexpected if _group_key(k) not in benign][:10]
            logger.error(f"Model {model_key}: sample suspicious unexpected keys: {samples}")


class Checkpointer:
    """Class to save and load model checkpoints"""

    def __init__(self, config: BaseCheckpointerConfig):
        self.config = config

    def _save_checkpoint(self, save_dict: Dict[str, Any], path: str):
        assert path.endswith(".pth"), f"{path} does not end with .pth"
        if self.config.use_s3:
            assert path.startswith("s3://"), f"{path} does not start with s3:// when using s3 storage"
            logger.info(f"Saving the model to {path}")
            # convert the save_dict to bytes
            buffer = io.BytesIO()
            torch.save(save_dict, buffer)
            s3_save(path, buffer.getvalue())
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(save_dict, path)

    def _load_checkpoint(self, path: str, device: torch.device = "cpu") -> Dict[str, Any]:
        assert path.endswith(".pth"), f"{path} does not end with .pth"
        if self.config.use_s3:
            assert path.startswith("s3://"), f"{path} does not start with s3:// when using s3 storage"
            state = torch.load(s3_load(path), map_location=device)
        else:
            assert os.path.exists(path), f"{path} does not exist"
            state = torch.load(path, map_location=device, weights_only=False)
        return state

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """
        synchronize()

        if is_rank0():
            model_state = {k: v.state_dict() for k, v in model_dict.items()}
            optim_state = None if optimizer_dict is None else {k: v.state_dict() for k, v in optimizer_dict.items()}
            scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
            grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
            callbacks_state = None if callbacks is None else callbacks.state_dict()
            save_dict = {
                "model": model_state,
                "optimizer": optim_state,
                "scheduler": scheduler_state,
                "grad_scaler": grad_scaler_state,
                "callbacks": callbacks_state,
                "iteration": iteration,
            }

            if path is None:
                if self.config.use_s3:
                    path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}.pth")
                else:
                    path = os.path.join(self.config.save_dir, f"{iteration:07d}.pth")

            if not self.config.use_s3:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Saving the model to {path}")
            self._save_checkpoint(save_dict, path)

        logger.success(f"Model saved at iteration {iteration}")
        synchronize()
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: Optional[torch.device] = "cpu",
        checkpoint_policy: Optional[CheckpointPolicy] = None,
    ) -> int:
        """Load the model checkpoint

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): The path to the checkpoint file
            device (Optional[torch.device]): The device to load the model to. Defaults to "cpu".
            checkpoint_policy: Accepted for interface parity with FSDPCheckpointer; basic checkpoints are full-state.

        Returns:
            int: The iteration number

        """
        del checkpoint_policy
        # TODO: rank 0 load and broadcast to all other ranks
        if path is None:
            if self.config.use_s3:
                checkpoint_path = os.path.join(self.config.s3_container, self.config.save_dir)
            else:
                checkpoint_path = self.config.save_dir
            path = latest_checkpoint(checkpoint_path) + ".pth"
            if path == ".pth":
                # no checkpoint found, starting from iteration 0
                return 0
        if not self.config.use_s3 and not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        logger.info(f"Loading model from {path}")
        state = self._load_checkpoint(path, device=device)

        logger.info("Loading the model_dict...")
        for k, v in model_dict.items():
            if k in state["model"] and v is not None:
                # strict is False to allow evaluating external checkpoints without, e.g., logvar parameters
                model_load_info = v.load_state_dict(state["model"][k], strict=False)
                logger.info(f"Model {k}, loading info: {model_load_info}")
                _log_load_mismatches(k, model_load_info)
            else:
                logger.warning(f"Model {k} not found in checkpoint.")

        if optimizer_dict is not None:
            logger.info("Loading the optimizer_dict...")
            for k, v in optimizer_dict.items():
                if k in state["optimizer"] and v is not None:
                    v.load_state_dict(state["optimizer"][k])
                else:
                    logger.warning(f"Optimizer {k} not found in checkpoint.")

        if scheduler_dict is not None:
            logger.info("Loading the scheduler_dict...")
            for k, v in scheduler_dict.items():
                if k in state["scheduler"] and v is not None:
                    v.load_state_dict(state["scheduler"][k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            # Check if saved grad_scaler state is non-empty (disabled scalers save empty state)
            if state.get("grad_scaler") and len(state["grad_scaler"]) > 0:
                grad_scaler.load_state_dict(state["grad_scaler"])
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            if "callbacks" in state:
                callbacks.load_state_dict(state["callbacks"])
            else:
                logger.warning("Callbacks not found in checkpoint.")

        if "iteration" not in state:
            logger.warning("Iteration not found in checkpoint.")
            return 0
        return state["iteration"]


class ModelWrapper(Stateful):
    """
    Wrapper for model state dict handling

    Code taken from this tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.
    """

    def __init__(self, model: torch.nn.Module, options: StateDictOptions | None = None):
        self.model = model
        self.options = options

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, and sets the default state dict type to FSDP.SHARDED_STATE_DICT
        return get_model_state_dict(self.model, options=self.options)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_model_state_dict(self.model, model_state_dict=state_dict, options=self.options)


class OptimizerWrapper(Stateful):
    """
    Wrapper for optimizer state dict handling

    Code taken from this tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.
    """

    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, options: StateDictOptions | None = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.options = options

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, and sets the default state dict type to FSDP.SHARDED_STATE_DICT
        optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=self.options)
        return optimizer_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=state_dict, options=self.options)


class FSDPCheckpointer(Checkpointer):
    """Class to save and load model checkpoints"""

    # Optional staging; by default DCP writes directly to the configured checkpoint directory.
    _STAGING_ROOT = None

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.config.use_s3:
            return S3StorageWriter(
                credential_path=self.config.s3_credential,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.config.use_s3:
            return S3StorageReader(
                credential_path=self.config.s3_credential,
                path=checkpoint_path,
            )
        _validate_dcp_files(checkpoint_path)
        return FileSystemReader(checkpoint_path)

    def _staged_dcp_save(self, state_dict: Dict[str, Any], final_path: str) -> None:
        """Save tensor files directly, or use explicitly configured staging.

        Each writer thread packs its assigned tensors into a shard. Every
        rank contributes disjoint files; rank zero validates the complete
        checkpoint after writes finish.
        """
        if self.config.use_s3:
            storage_writer = self.get_storage_writer(checkpoint_path=final_path)
            dcp.save(state_dict, storage_writer=storage_writer)
            return

        if self._STAGING_ROOT is None:
            writer = FileSystemWriter(path=final_path, single_file_per_rank=True, thread_count=4)
            dcp.save(state_dict, storage_writer=writer)
            synchronize()
            if is_rank0():
                _validate_dcp_files(final_path)
            synchronize()
            return

        # Stage to local SSD
        stage_root = self._STAGING_ROOT or os.path.join(self.config.save_dir, ".staging")
        staging_dir = os.path.join(stage_root, os.path.basename(final_path))
        synchronize()
        if _is_local_rank0():
            shutil.rmtree(staging_dir, ignore_errors=True)
        synchronize()
        staging_writer = FileSystemWriter(path=staging_dir, single_file_per_rank=True, thread_count=4)
        dcp.save(state_dict, storage_writer=staging_writer)

        # All ranks finished node-local writes. Global rank 0 prepares the shared
        # destination, then one leader per node contributes its local shard set.
        synchronize()
        if is_rank0():
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.rmtree(final_path, ignore_errors=True)
            os.makedirs(final_path)
        synchronize()
        if _is_local_rank0():
            _copy_local_dcp_staging(staging_dir, final_path, include_metadata=is_rank0())
            logger.info(f"Node-local checkpoint shards copied: {staging_dir} → {final_path}")

        # Validate on global rank 0 and broadcast the result so every rank either
        # proceeds or raises before the scalar .pth completion marker is written.
        synchronize()
        validation_error: list[str | None] = [None]
        if is_rank0():
            try:
                _validate_dcp_files(final_path)
            except Exception as exc:
                validation_error[0] = str(exc)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(validation_error, src=0)
        if validation_error[0] is not None:
            raise RuntimeError(validation_error[0])

        synchronize()
        if _is_local_rank0():
            shutil.rmtree(staging_dir, ignore_errors=True)
        synchronize()

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """

        if path is None:
            if self.config.use_s3:
                path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}")
            else:
                path = os.path.join(self.config.save_dir, f"{iteration:07d}")
                if not os.path.exists(self.config.save_dir):
                    os.makedirs(self.config.save_dir, exist_ok=True)
        if path.endswith(".pth"):  # In the case of autoresume
            path = path[:-4]
        logger.info(f"Saving FSDP model to prefix {path}")

        synchronize()
        # fsdp should save on all ranks
        for k, v in model_dict.items():
            model_state_dict = ModelWrapper(model=v).state_dict()
            self._staged_dcp_save(model_state_dict, final_path=f"{path}.{k}_model")

        if optimizer_dict is not None:
            for k, v in optimizer_dict.items():
                optim_state_dict = OptimizerWrapper(model=model_dict[k], optimizer=v).state_dict()
                self._staged_dcp_save(optim_state_dict, final_path=f"{path}.{k}_optim")

        # other scalars only save on rank 0
        if is_rank0():
            scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
            grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
            callbacks_state = None if callbacks is None else callbacks.state_dict()
            save_dict = {
                "scheduler": scheduler_state,
                "grad_scaler": grad_scaler_state,
                "callbacks": callbacks_state,
                "iteration": iteration,
            }
            self._save_checkpoint(save_dict, f"{path}.pth")

        logger.success(f"Model saved at iteration {iteration}")
        synchronize()
        return path

    def save_lora_only(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
        checkpoint_policy: Optional[CheckpointPolicy] = None,
    ) -> str:
        """Save only LoRA adapter weights + trainable (non-base) modules, skipping frozen base model.

        This dramatically reduces checkpoint size for LoRA training (e.g. from 12GB to ~200MB-2GB).
        All ``dcp.save`` calls go through ``_staged_dcp_save`` which writes to local SSD first,
        then copies to the final (possibly blobfuse) path to avoid NCCL/gloo timeouts.

        Each module's save mode comes from the explicit ``checkpoint_policy`` (produced by
        ``model.checkpoint_policy()``), never from key-string inference. Keys without an
        explicit policy fall back to a structural (``requires_grad``-derived) default.

        Saved layout (same prefix convention as ``save``):
            {path}.lora_model/   — LoRA + trainable param state_dict via dcp.save
            {path}.lora_optim/   — optimizer state for trainable params via dcp.save
            {path}.pth           — scheduler, grad_scaler, callbacks, iteration (rank 0 only)
        """
        if path is None:
            if self.config.use_s3:
                path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}")
            else:
                path = os.path.join(self.config.save_dir, f"{iteration:07d}")
                if not os.path.exists(self.config.save_dir):
                    os.makedirs(self.config.save_dir, exist_ok=True)
        if path.endswith(".pth"):
            path = path[:-4]
        logger.info(f"Saving LoRA-only FSDP checkpoint to prefix {path}")

        synchronize()

        for k, v in model_dict.items():
            policy = _resolve_policy(checkpoint_policy, k, v)
            total_params = sum(1 for _ in v.parameters())
            total_numel = sum(p.numel() for p in v.parameters())

            # Resolve the save options + (for EMA mirror) temporarily mark the mirror
            # source's trainable names, all driven by the explicit policy.
            restore_requires_grad: list[tuple[torch.nn.Parameter, bool]] = []
            if policy.mode == CheckpointSaveMode.EMA_TRAINABLE_MIRROR:
                mirror_names = _mirror_trainable_names(policy, model_dict)
                restore_requires_grad = _temporarily_mark_trainable(v, mirror_names)
                save_options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
                label = f"EMA mirror of '{policy.mirror_source}' trainable subset"
            elif policy.mode == CheckpointSaveMode.TRAINABLE_SUBSET:
                save_options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
                label = "trainable-only"
            else:  # CheckpointSaveMode.FULL
                save_options = StateDictOptions(full_state_dict=False)
                label = "full"

            trainable_numel = sum(p.numel() for p in v.parameters() if p.requires_grad)
            if policy.mode in (CheckpointSaveMode.TRAINABLE_SUBSET, CheckpointSaveMode.EMA_TRAINABLE_MIRROR):
                with _detach_frozen_preprocessors(v):
                    save_state = _trainable_state_dict(v)
            else:
                save_state = get_model_state_dict(v, options=save_options)

            if save_state:
                self._staged_dcp_save(save_state, final_path=f"{path}.lora_{k}_model")
                saved_numel = sum(t.numel() for t in save_state.values())
                logger.info(
                    f"Saved '{k}' [FSDP, {label}]: {len(save_state)} tensors ({saved_numel / 1e6:.2f}M params) "
                    f"out of {total_params} tensors ({total_numel / 1e6:.2f}M params), "
                    f"trainable={trainable_numel / 1e6:.2f}M ({100.0 * trainable_numel / max(total_numel, 1):.2f}%)"
                )
            else:
                logger.warning(f"No parameters to save for model '{k}' (mode={policy.mode}), skipping")

            _restore_requires_grad(restore_requires_grad)

        if optimizer_dict is not None:
            for k, v in optimizer_dict.items():
                # Some FSDP lifecycles keep stable optimizer-owned DTensors while
                # checkpoint collection temporarily changes module parameter views.
                rebind_names = getattr(v, "_smd_checkpoint_rebind_param_names", None)
                if rebind_names is None:
                    rebind_names = getattr(v, "_ltx2_checkpoint_rebind_param_names", None)
                if rebind_names is None:
                    optim_state = OptimizerWrapper(model=model_dict[k], optimizer=v).state_dict()
                else:
                    with _bind_optimizer_to_current_parameters(model_dict[k], v, rebind_names):
                        optim_state = OptimizerWrapper(model=model_dict[k], optimizer=v).state_dict()
                self._staged_dcp_save(optim_state, final_path=f"{path}.lora_{k}_optim")

        # Scalars on rank 0 only
        if is_rank0():
            scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
            grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
            callbacks_state = None if callbacks is None else callbacks.state_dict()
            save_dict = {
                "scheduler": scheduler_state,
                "grad_scaler": grad_scaler_state,
                "callbacks": callbacks_state,
                "iteration": iteration,
                "lora_only": True,  # marker so load knows this is a LoRA-only checkpoint
            }
            self._save_checkpoint(save_dict, f"{path}.pth")

        logger.success(f"LoRA-only checkpoint saved at iteration {iteration}")
        synchronize()
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: Optional[torch.device] = "cpu",
        checkpoint_policy: Optional[CheckpointPolicy] = None,
    ) -> int:
        """Load the model checkpoint

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): The path to the checkpoint file
            device (Optional[torch.device]): The device to load the model to. Defaults to "cpu".

        Returns:
            int: The iteration number

        """
        if path is None:
            if self.config.use_s3:
                checkpoint_path = os.path.join(self.config.s3_container, self.config.save_dir)
            else:
                checkpoint_path = self.config.save_dir
            path = latest_checkpoint(checkpoint_path)
            if path == "":
                # no checkpoint found, starting from iteration 0
                return 0
        if path.endswith(".pth"):
            # switch to basic checkpointer
            logger.debug(f"Loading non-FSDP model from {path}")
            return super().load(
                model_dict,
                optimizer_dict=optimizer_dict,
                scheduler_dict=scheduler_dict,
                grad_scaler=grad_scaler,
                callbacks=callbacks,
                path=path,
                device=device,
            )
        if not self.config.use_s3 and not os.path.exists(f"{path}.pth"):
            raise FileNotFoundError(f"Checkpoint file not found at {path}.pth")
        logger.info(f"Loading FSDP model from prefix {path}")

        # Detect LoRA-only checkpoint by checking .pth metadata or file naming
        pth_state = self._load_checkpoint(f"{path}.pth", device="cpu")
        is_lora_only = pth_state.get("lora_only", False)
        key_prefix = "lora_" if is_lora_only else ""
        if is_lora_only:
            logger.info("Detected LoRA-only checkpoint — using 'lora_' key prefix for model/optim shards")

        for k, v in model_dict.items():
            model_shard_path = f"{path}.{key_prefix}{k}_model"
            logger.info(f"Loading the FSDP model dict for key {k} from {model_shard_path}...")
            if not os.path.exists(model_shard_path):
                if is_lora_only:
                    logger.warning(
                        f"LoRA shard not found for '{k}' at {model_shard_path}, skipping (base weights already loaded)"
                    )
                    continue
                raise AssertionError(f"Key {k} does not exist in FSDP model dict at {model_shard_path}")

            if is_lora_only:
                policy = _resolve_policy(checkpoint_policy, k, v)
                total_params = sum(1 for _ in v.parameters())

                if policy.mode == CheckpointSaveMode.FULL:
                    # Fully-trainable module (e.g. pivot / EMA pivot): load everything.
                    model_wrapper = ModelWrapper(model=v)
                    model_state_dict = model_wrapper.state_dict()
                    storage_reader = self.get_storage_reader(checkpoint_path=model_shard_path)
                    dcp.load(state_dict=model_state_dict, storage_reader=storage_reader)
                    model_wrapper.load_state_dict(model_state_dict)
                    logger.info(f"Loaded full state for '{k}' (all {total_params} params)")
                else:
                    # TRAINABLE_SUBSET (LoRA student) or EMA_TRAINABLE_MIRROR (EMA student):
                    # load only the trainable subset. For the EMA mirror, temporarily mark
                    # the mirror source's trainable names so its subset matches what was saved.
                    restore_requires_grad: list[tuple[torch.nn.Parameter, bool]] = []
                    if policy.mode == CheckpointSaveMode.EMA_TRAINABLE_MIRROR:
                        mirror_names = _mirror_trainable_names(policy, model_dict)
                        restore_requires_grad = _temporarily_mark_trainable(v, mirror_names)
                    trainable_params = sum(1 for p in v.parameters() if p.requires_grad)
                    with _detach_frozen_preprocessors(v):
                        model_state_dict = _trainable_state_dict(v)
                        storage_reader = self.get_storage_reader(checkpoint_path=model_shard_path)
                        dcp.load(state_dict=model_state_dict, storage_reader=storage_reader)
                        # strict=False so frozen params missing from the saved subset don't error.
                        load_options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False)
                        set_model_state_dict(v, model_state_dict=model_state_dict, options=load_options)
                    logger.info(
                        f"Loaded trainable-only state for '{k}' [{policy.mode.value}] "
                        f"({trainable_params}/{total_params} params)"
                    )
                    _restore_requires_grad(restore_requires_grad)
            else:
                model_wrapper = ModelWrapper(model=v)
                model_state_dict = model_wrapper.state_dict()
                storage_reader = self.get_storage_reader(checkpoint_path=model_shard_path)
                dcp.load(state_dict=model_state_dict, storage_reader=storage_reader)
                model_wrapper.load_state_dict(model_state_dict)

        if optimizer_dict is not None:
            for k, v in optimizer_dict.items():
                optim_shard_path = f"{path}.{key_prefix}{k}_optim"
                logger.info(f"Loading the FSDP optimizer dict for key {k} from {optim_shard_path}...")
                if not os.path.exists(optim_shard_path):
                    raise FileNotFoundError(f"Requested optimizer {k} is missing from checkpoint: {optim_shard_path}")

                # get_optimizer_state_dict initializes a fresh optimizer with
                # zero-like (including DTensor) gradients and lr=0. Pre-filling
                # grads here makes PyTorch skip that initialization: DCP then
                # loads an empty state template and silently loses all moments.

                if is_lora_only:
                    # Use ignore_frozen_params for modules with frozen base params
                    total_params = sum(1 for _ in model_dict[k].parameters())
                    trainable_params = sum(1 for p in model_dict[k].parameters() if p.requires_grad)
                    has_frozen = trainable_params < total_params
                    if has_frozen:
                        trainable_options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
                    else:
                        trainable_options = StateDictOptions(full_state_dict=False)
                    optim_state_dict = get_optimizer_state_dict(model_dict[k], v, options=trainable_options)
                    storage_reader = self.get_storage_reader(checkpoint_path=optim_shard_path)
                    absent_states = _match_optimizer_state_template(optim_state_dict, storage_reader.read_metadata())
                    try:
                        dcp.load(state_dict=optim_state_dict, storage_reader=storage_reader)
                        _restore_uninitialized_optimizer_states(optim_state_dict, absent_states)
                        set_optimizer_state_dict(model_dict[k], v, optim_state_dict, options=trainable_options)
                        _drop_empty_optimizer_states(v)
                        logger.success(f"Successfully loaded LoRA optimizer state for {k}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to restore optimizer {k} from {optim_shard_path}") from e
                else:
                    optim_wrapper = OptimizerWrapper(model=model_dict[k], optimizer=v)
                    optim_state_dict = optim_wrapper.state_dict()
                    storage_reader = self.get_storage_reader(checkpoint_path=optim_shard_path)
                    absent_states = _match_optimizer_state_template(optim_state_dict, storage_reader.read_metadata())

                    try:
                        dcp.load(
                            state_dict=optim_state_dict,
                            storage_reader=storage_reader,
                        )
                        _restore_uninitialized_optimizer_states(optim_state_dict, absent_states)
                        optim_wrapper.load_state_dict(optim_state_dict)
                        _drop_empty_optimizer_states(v)
                        logger.success(f"Successfully loaded optimizer state for {k}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to restore optimizer {k} from {optim_shard_path}") from e

        # Re-load .pth with target device (we loaded with cpu earlier for lora_only detection)
        state = pth_state if str(device) == "cpu" else self._load_checkpoint(f"{path}.pth", device=device)

        if scheduler_dict is not None:
            logger.info("Loading the scheduler_dict...")
            for k, v in scheduler_dict.items():
                if k in state["scheduler"]:
                    v.load_state_dict(state["scheduler"][k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            # Check if saved grad_scaler state is non-empty (disabled scalers save empty state)
            if state.get("grad_scaler") and len(state["grad_scaler"]) > 0:
                grad_scaler.load_state_dict(state["grad_scaler"])
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            if "callbacks" in state:
                callbacks.load_state_dict(state["callbacks"])
            else:
                logger.warning("Callbacks not found in checkpoint.")

        return state["iteration"]
