# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import socket
import subprocess
import argparse
from datetime import datetime, timezone

import torch

from smd.configs.config_utils import serialize_config
from smd.configs import config_utils
from smd.configs.config_validation import validate_callback_configs, validate_h3_launch_config
import smd.utils.logging_utils as logger
from smd.utils.basic_utils import get_batch_size_total
from smd.utils.distributed import ddp, launch_rank, launch_world_size, world_size
from smd.utils.io_utils import set_env_vars
from smd.configs.config import BaseConfig


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--config", default="configs.config", help="Path to the config file")
    parser.add_argument("--config_yaml", type=str, default=None, help="Optional YAML file to override config values.")
    parser.add_argument("--download-resources", action="store_true", help="Download pinned assets and teachers before loading the config.")
    parser.add_argument("--resource-model", help="Resource experiment name when using --download-resources with a custom config.")
    parser.add_argument("--log_level", default="INFO", help="Log level (e.g. DEBUG, INFO)")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="""Modify config options at the end of the command.
                                For Yacs configs, use space-separated "PATH.KEY VALUE" pairs.
                                For python-based LazyConfig, use "path.key=value".""",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help=(
            "Resolve the config (python + CLI overrides), print it as YAML to stdout, and exit. "
            "No DDP/FSDP init or torchrun needed. --download-resources may populate missing assets first."
        ),
    )
    args = parser.parse_args()
    return args


def set_cuda_backend(deterministic: bool = True, benchmark: bool = True, tf32_enabled: bool = True):
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = tf32_enabled
    logger.critical(f"cuDNN deterministic: {deterministic}, cuDNN benchmark: {benchmark}, enable TF32: {tf32_enabled}")


def _validate_global_batch_size(config: BaseConfig) -> int | None:
    value = config.trainer.batch_size_global
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "trainer.batch_size_global must be an integer or null, "
            f"got {value!r} ({type(value).__name__}). "
            "For no global-batch override, omit the CLI option or use trainer.batch_size_global=null; "
            "Hydra does not interpret Python-style None as null."
        )
    if value <= 0:
        raise ValueError(f"trainer.batch_size_global must be positive when set, got {value}.")
    return value


def _configure_global_batch_size(config: BaseConfig, *, launch_size: int | None = None) -> None:
    requested = _validate_global_batch_size(config)
    if requested is None:
        return

    effective_world_size = launch_world_size() if launch_size is None else launch_size
    if effective_world_size < 1:
        raise ValueError(f"launch world size must be positive, got {effective_world_size}.")
    batch_size = config.dataloader_train.batch_size * effective_world_size
    accum_rounds = max(requested // batch_size, 1)
    new_batch_size_global = accum_rounds * batch_size
    if new_batch_size_global != requested:
        logger.warning(
            f"[batch_size_global] Requested {requested} is not divisible by "
            f"per-step batch (dataloader_train.batch_size={config.dataloader_train.batch_size} * "
            f"world_size={effective_world_size} = {batch_size}). "
            f"Silently rounding DOWN to {new_batch_size_global} "
            f"(accum_rounds={accum_rounds}). "
            f"To suppress this warning, set batch_size_global to a multiple of {batch_size}."
        )

    if accum_rounds != config.trainer.grad_accum_rounds:
        logger.info(
            f"Changing gradient accumulation rounds from {config.trainer.grad_accum_rounds} "
            f"to {accum_rounds} to match requested global batch size."
        )
        config.trainer.grad_accum_rounds = accum_rounds


def setup(args: argparse.Namespace, evaluation: bool = False) -> BaseConfig:
    if hasattr(args, "log_level"):
        # set log level for logger (INFO, by default)
        logger.set_log_level(args.log_level)

    if getattr(args, "download_resources", False):
        from tools.download_resources import prepare_for_config

        result = prepare_for_config(args.config, model=getattr(args, "resource_model", None))
        logger.info(f"Training resources verified: {result['files']} files; installed {result['installed']}.")

    # Import the config only after optional resource preparation has completed.
    from smd.configs.preflight import import_training_config, validate_preview_resources, validate_training_weights

    config: BaseConfig = import_training_config(args.config, model=getattr(args, "resource_model", None))
    if hasattr(args, "config_yaml") and args.config_yaml:
        # Apply YAML overrides before CLI opts so CLI always has highest priority
        config = config_utils.override_config_with_yaml(config, args.config_yaml)
    if hasattr(args, "opts"):
        # Override the config with the command line arguments
        config = config_utils.override_config_with_opts(config, args.opts)

    _validate_global_batch_size(config)
    # Fail here rather than inside Trainer.__init__: a stale callback argument
    # is a config bug, and it must not survive --print-config / --dryrun.
    validate_callback_configs(config)
    validate_h3_launch_config(config)
    validate_training_weights(config, config_path=args.config, model=getattr(args, "resource_model", None))
    validate_preview_resources(config, config_path=args.config, model=getattr(args, "resource_model", None))

    # --print-config: after optional downloads, skip run outputs and DDP initialization.
    if getattr(args, "print_config", False):
        print(serialize_config(config, return_type="yaml"))
        sys.exit(0)

    _configure_global_batch_size(config)

    # Update checkpointer save_dir
    config.trainer.checkpointer.save_dir = f"{config.log_config.save_path}/{config.trainer.checkpointer.save_dir}"

    # save config
    config_save_path = config.log_config.save_path
    if evaluation:
        config_save_path = os.path.join(config_save_path, config.eval.samples_dir)
    if launch_rank() == 0:
        serialize_config(config, return_type="file", path=config_save_path, filename="config.yaml")
        _dump_run_info(config_save_path, args)

    # Check for dryrun
    if getattr(args, "dryrun", False):
        if launch_rank() == 0:
            logger.info("Dryrun")
            logger.info(serialize_config(config, return_type="yaml"))
            logger.info(f"config.yaml is saved at {config_save_path}")
        sys.exit(0)

    # distributed setup
    if config.trainer.ddp or config.trainer.fsdp:
        # check if ddp is available
        if not torch.distributed.is_available():
            raise RuntimeError("Distributed training is not available, please check your PyTorch installation.")
        # initialize DDP
        ddp.init()
        logger.info(f"Distributed training initialized, world size: {world_size()}")
    else:
        logger.info("No DDP or FSDP parallelism")

    # Check if we can use memory-efficient FSDP meta init
    if config.model.fsdp_meta_init:
        if not config.trainer.fsdp:
            logger.warning("Ignoring fsdp_meta_init since FSDP is disabled.")
            config.model.fsdp_meta_init = False
        elif evaluation:
            logger.warning("Ignoring fsdp_meta_init for evaluation/inference.")
            config.model.fsdp_meta_init = False
        elif config.trainer.checkpointer.pretrained_ckpt_path:
            # TODO: this functionality is not implemented yet
            logger.warning("Ignoring fsdp_meta_init for loading pretrained checkpoint.")
            config.model.fsdp_meta_init = False

    logger.critical(
        f"Global batch size: {get_batch_size_total(config)} (Batch size per GPU: {config.dataloader_train.batch_size}, Gradient accumulation rounds: {config.trainer.grad_accum_rounds}, World size: {world_size()})"
    )

    # Set up s3 environmental variables
    set_env_vars(config.trainer.checkpointer.s3_credential)

    # Set up CUDA backend
    set_cuda_backend(config.trainer.cudnn.deterministic, config.trainer.cudnn.benchmark, config.trainer.tf32_enabled)

    return config


def _git_capture(cmd: list[str], cwd: str) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _dump_run_info(save_path: str, args: argparse.Namespace) -> None:
    """Dump a run_info.json next to config.yaml capturing what is needed to reproduce a run.

    Includes: git SHA, git dirty status, list of modified/untracked files (truncated),
    the launch command, hostname, world size, timestamp, and a few key env vars.

    Best-effort: logs failures without blocking training startup.
    """
    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        # Walk up to find the actual repo root (where .git lives).
        for _ in range(6):
            if os.path.exists(os.path.join(repo_root, ".git")):
                break
            repo_root = os.path.dirname(repo_root)

        sha = _git_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
        branch = _git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        status_porcelain = _git_capture(["git", "status", "--porcelain"], cwd=repo_root)
        dirty_lines = [line for line in (status_porcelain or "").splitlines() if line.strip()]
        # Cap the dirty list to keep run_info.json small.
        dirty_truncated = dirty_lines[:50]
        dirty_overflow = max(0, len(dirty_lines) - len(dirty_truncated))

        info = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "world_size": launch_world_size(),
            "git": {
                "sha": sha,
                "branch": branch,
                "is_dirty": bool(dirty_lines) if status_porcelain is not None else None,
                "dirty_files": dirty_truncated,
                "dirty_files_overflow": dirty_overflow,
            },
            "launch": {
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "config_arg": getattr(args, "config", None),
                "config_yaml_arg": getattr(args, "config_yaml", None),
                "opts": getattr(args, "opts", None),
            },
            "env": {
                "SMD_OUTPUT_ROOT": os.environ.get("SMD_OUTPUT_ROOT"),
                "DATA_ROOT_DIR": os.environ.get("DATA_ROOT_DIR"),
                "CKPT_ROOT_DIR": os.environ.get("CKPT_ROOT_DIR"),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
                "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
                "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
                "WANDB_MODE": os.environ.get("WANDB_MODE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "TORCHELASTIC_RESTART_COUNT": os.environ.get("TORCHELASTIC_RESTART_COUNT"),
                "TORCHELASTIC_MAX_RESTARTS": os.environ.get("TORCHELASTIC_MAX_RESTARTS"),
                "WANDB_RUN_ID": os.environ.get("WANDB_RUN_ID"),
            },
        }
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, "run_info.json")
        with open(out_path, "w") as f:
            json.dump(info, f, indent=2, sort_keys=False)
        logger.info(f"Wrote run_info.json to {out_path}")
        if info["git"]["is_dirty"] is None:
            logger.warning(
                "[run_info] git status is unavailable; source cleanliness is unknown. "
                "Retain the immutable uploaded code asset identity for this run."
            )
        elif info["git"]["is_dirty"]:
            logger.warning(
                f"[run_info] git working tree is DIRTY ({len(dirty_lines)} entries). "
                f"This run may not be exactly reproducible from the commit SHA alone."
            )
    except Exception as e:
        logger.warning(f"_dump_run_info failed (non-fatal): {e}")
