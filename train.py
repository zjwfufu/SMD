# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The training entry script for the SMD project. Works for both DDP and FSDP training.
"""

import argparse
import gc
import traceback
import warnings

from smd.configs.config import BaseConfig
from smd.utils import instantiate
from smd.trainer import Trainer
import smd.utils.logging_utils as logger
from smd.utils.distributed import synchronize, clean_up
from smd.utils.scripts import parse_args, setup

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 2.6.


def main(config: BaseConfig):
    # initiate the model
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    synchronize()

    # initiate the trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(config)
    logger.success("Trainer initialized successfully")
    synchronize()

    # Start training
    trainer.run(model)
    synchronize()


def _finish_wandb_run(exit_code: int) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
    except Exception as e:
        logger.warning(f"Failed to finish wandb run during training cleanup: {type(e).__name__}: {e}")


def _cleanup_runtime(exit_code: int) -> None:
    try:
        clean_up()
    except Exception as e:
        logger.warning(f"Distributed cleanup failed after training attempt: {type(e).__name__}: {e}")

    _finish_wandb_run(exit_code=exit_code)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"CUDA cache cleanup failed after training attempt: {type(e).__name__}: {e}")


def _run_once(args: argparse.Namespace) -> None:
    config = setup(args)
    main(config)
    _cleanup_runtime(exit_code=0)
    logger.info("Training finished.")


def run(args: argparse.Namespace) -> None:
    try:
        _run_once(args)
    except Exception as e:
        logger.error(f"Training failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        # A failed rank must exit promptly so torchrun can stop its peers.
        # NCCL destruction/CUDA cleanup can block when another rank is already
        # inside a collective. Print on every rank before propagating the error.
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    args = parse_args(parser)
    run(args)
