# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, List, Optional, Callable
import io
import os
import threading
import time
import gc

import torch
from torch.utils.data import DataLoader

from smd.methods import SMDModel
from smd.utils.basic_utils import set_random_seed, set_tmp_random_seed
from smd.utils.checkpointer import Checkpointer, FSDPCheckpointer
import smd.utils.logging_utils as logger
from smd.configs.config import BaseConfig
from smd.callbacks.callback import CallbackDict
from smd.utils import instantiate, basic_utils
import smd.utils.distributed.ddp as ddp
import smd.utils.distributed.fsdp as fsdp
from smd.utils.distributed import synchronize, is_rank0, world_size
import torch.distributed as dist
from smd.utils import set_global_vars, set_temp_global_vars
from smd.utils import global_vars
from smd.utils.autoresume import AutoResumeInterface, create_auto_resume


def resolve_sampler_start_idx(configured_start_idx, base_offset: int, resumed_samples: int) -> int:
    if configured_start_idx is not None:
        return int(configured_start_idx)
    base_offset = int(base_offset)
    if base_offset < 0:
        raise ValueError(f"sampler_base_offset must be >= 0, got {base_offset}")
    return base_offset + int(resumed_samples)


def _decode_condition_image_bytes(image_bytes: bytes):
    """Decode condition image bytes while preserving alpha for image-conditioned models."""
    from PIL import Image as PILImage

    return PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")


class Trainer:
    def __init__(self, config: BaseConfig, auto_resume: Optional[AutoResumeInterface] = None):
        """
        Initialize the Trainer.

        Args:
            config (BaseConfig): SMD config
            auto_resume (Optional[AutoResumeInterface]): Custom auto-resume implementation.
                If None, defaults to NoOpAutoResume (auto-resume disabled).
                See smd.utils.autoresume for the interface and examples.
        """

        self.config = config
        set_global_vars(self.config.trainer.global_vars)

        # Initialize auto-resume (defaults to NoOpAutoResume if not provided)
        self.auto_resume = create_auto_resume(auto_resume)
        logger.info(f"Auto-resume: {type(self.auto_resume).__name__}")
        self._last_data_origin = "<none>"
        if float(self.config.trainer.data_wait_warn_s) < 0:
            raise ValueError(f"trainer.data_wait_warn_s must be >= 0, got {self.config.trainer.data_wait_warn_s}")

        # Set random seed
        set_random_seed(config.trainer.seed, by_rank=True)

        # Initialize the callback functions.
        logger.info("Initializing callbacks (including wandb)...")
        self.callbacks = CallbackDict(config=config, trainer=self)
        logger.success("Callbacks initialized successfully")

        # Synchronize after callback initialization to handle wandb timing differences
        synchronize()
        logger.info("Callback synchronization complete")

        # Initialize the checkpointer.
        logger.info("Initializing checkpointer...")
        if self.config.trainer.fsdp:
            self.checkpointer = FSDPCheckpointer(self.config.trainer.checkpointer)
        else:
            self.checkpointer = Checkpointer(self.config.trainer.checkpointer)
        logger.success("Checkpointer initialized successfully")

    def run(
        self,
        model: SMDModel,
    ) -> None:
        """
        Run the training loop

        Args:
            model (SMDModel): Distillation model.
        """
        logger.info("Starting training")

        iter_start = 0
        logger.info("Initializing callbacks and model ...")
        self.callbacks.on_model_init_start(model)
        if self.config.trainer.checkpointer.pretrained_ckpt_path:
            # This typically only affects the first job in the auto-resume chain
            self.load_pretrained_ckpt(model)

        if self.config.trainer.fsdp and (
            self.config.model.precision_amp is not None
            and self.config.model.precision_amp != self.config.model.precision
        ):
            logger.warning(
                f"Autocast to {self.config.model.precision_amp} is enabled and FSDP is enabled. "
                f"While this is possible, it is not recommended."
            )

        logger.info("Starting model.on_train_begin ...")
        synchronize()
        model.on_train_begin(is_fsdp=self.config.trainer.fsdp)
        synchronize()
        logger.info("model.on_train_begin completed")

        # wrap model into DDP or FSDP
        assert not (self.config.trainer.ddp and self.config.trainer.fsdp), (
            "Model cannot be wrapped into both DDP and FSDP"
        )
        if self.config.trainer.ddp:
            logger.info("Wrapping model into ddp ..")
            model_ddp = ddp.model_to_ddp(model)
            logger.info("DDP wrapping completed")
        elif self.config.trainer.fsdp:
            logger.info("Wrapping model into fsdp ..")
            model_ddp = fsdp.model_to_fsdp(
                model,
                min_num_params=self.config.trainer.fsdp_min_num_params,
                apply_cpu_offload=self.config.trainer.fsdp_cpu_offload,
                sync_module_states=self.config.model.fsdp_meta_init,
                sharding_group_size=self.config.trainer.fsdp_sharding_group_size,
                no_sync_keys=getattr(model, "fsdp_no_sync_keys", None),
            )
            logger.info("FSDP wrapping completed")
            if hasattr(model, "log_preprocessor_devices"):
                model.log_preprocessor_devices("after_fsdp_wrap")
        else:
            model_ddp = model
        self.callbacks.on_model_init_end(model_ddp)
        synchronize()

        self.callbacks.on_optimizer_init_start(model)
        model.init_optimizers()
        self.callbacks.on_optimizer_init_end(model)

        self.callbacks.on_load_checkpoint_start(model)

        # Check if we are resuming from an auto-resume checkpoint
        self.auto_resume.init()
        auto_resume_details = self.auto_resume.get_resume_details()
        logger.info(f"Auto-Resume Details: {auto_resume_details}")
        autoresume_ckpt = auto_resume_details["save_path"] if auto_resume_details else None

        if self.config.trainer.resume or autoresume_ckpt is not None:
            # Warn if pretrained_ckpt_path was just loaded
            # and we're about to silently overwrite those weights with on-disk auto-resume.
            if self.config.trainer.checkpointer.pretrained_ckpt_path:
                logger.warning(
                    f"[resume-over-pretrained] pretrained_ckpt_path="
                    f"{self.config.trainer.checkpointer.pretrained_ckpt_path!r} was loaded above, "
                    f"but trainer.resume=True (and/or auto-resume found a checkpoint at "
                    f"{autoresume_ckpt!r}). The on-disk checkpoint WILL OVERWRITE the pretrained "
                    f"weights. To avoid this, either clear save_dir, unset pretrained_ckpt_path "
                    f"after the first job, or set trainer.resume=False."
                )
            logger.info("Loading checkpoints for resuming ..")

            # load previous checkpoint
            iter_start = self.checkpointer.load(
                model.model_dict,
                optimizer_dict=model.optimizer_dict,
                scheduler_dict=model.scheduler_dict,
                grad_scaler=model.grad_scaler,
                callbacks=self.callbacks,
                path=autoresume_ckpt,
                device=model.device,
                checkpoint_policy=model.checkpoint_policy(),
            )
        self.callbacks.on_load_checkpoint_end(model, iteration=iter_start)

        # re-seed based on the current iteration for resuming
        set_random_seed(self.config.trainer.seed, iteration=iter_start, by_rank=True)

        # resume samplers and initiate the dataloaders
        self.callbacks.on_dataloader_init_start(model, iteration=iter_start)
        nimg = (
            iter_start * self.config.dataloader_train.batch_size * self.config.trainer.grad_accum_rounds * world_size()
        )
        nimg = getattr(self, "training_samples_consumed", nimg)
        for loader in ["dataloader_train", "dataloader_val"]:
            dataloader_config = getattr(self.config, loader, None)
            if getattr(dataloader_config, "sampler_start_idx", 0) is None:
                base_offset = int(getattr(dataloader_config, "sampler_base_offset", 0))
                start_idx = resolve_sampler_start_idx(None, base_offset, nimg)
                logger.info(
                    f"Setting sampler start index to {start_idx} images for {loader} "
                    f"(base_offset={base_offset}, resumed_samples={nimg})"
                )
                dataloader_config.sampler_start_idx = start_idx

        logger.info("Instantiating dataloader...")
        dataloader_train = instantiate(self.config.dataloader_train)
        dataloader_val = (
            instantiate(self.config.dataloader_val) if getattr(self.config, "dataloader_val", None) else None
        )
        augment_pipe = instantiate(self.config.trainer.augment_pipe)
        self.callbacks.on_dataloader_init_end(model, dataloader_train, dataloader_val, iteration=iter_start)

        self.callbacks.on_train_begin(model, iteration=iter_start)
        logger.info(f"iter_start: {iter_start}")

        if iter_start == 0 and dataloader_val is not None:
            # validation before first training step
            self.validate(model_ddp, model, dataloader_val, iteration=iter_start)

        dataloader_train_iter = iter(dataloader_train)
        for iter_cur in range(iter_start + 1, self.config.trainer.max_iter):
            self.callbacks.on_training_step_begin(model, iteration=iter_cur)
            # Sync route-homogeneous loaders to the authoritative global iteration so
            # all ranks pick the same route per step (no-op for ordinary loaders).
            if hasattr(dataloader_train, "set_step"):
                dataloader_train.set_step(iter_cur)
            reduce_metrics = getattr(model, "reduce_training_metrics", None)
            iteration_metrics = [] if callable(reduce_metrics) else None
            for grad_accum_iter in range(self.config.trainer.grad_accum_rounds):
                data = self._acquire_training_batch(
                    model,
                    dataloader_train_iter,
                    dataloader_train,
                    augment_pipe,
                    iteration=iter_cur,
                    grad_accum_iter=grad_accum_iter,
                )
                logger.debug(
                    f"iteration: {iter_cur} | grad_accum_iter: {grad_accum_iter} | data: {basic_utils.to_str(data)}"
                )

                # single training step
                self.callbacks.on_training_accum_step_begin(model, data, iteration=iter_cur, accum_iter=grad_accum_iter)
                loss_map, outputs = self.train_step(model_ddp, model, data, iter_cur, grad_accum_iter)
                if iteration_metrics is not None:
                    iteration_metrics.append(loss_map)

            if iteration_metrics is not None:
                loss_map = reduce_metrics(iteration_metrics)

            self.callbacks.on_training_step_end(
                model=model,
                data_batch=data,
                output_batch=outputs,
                loss_dict=loss_map,
                iteration=iter_cur,
            )

            # validation
            if iter_cur % self.config.trainer.validation_iter == 0 and dataloader_val is not None:
                self.validate(model_ddp, model, dataloader_val, iteration=iter_cur)

            # save checkpoint
            just_saved_checkpoint = False
            latest_checkpoint_path = None
            if iter_cur % self.config.trainer.save_ckpt_iter == 0:
                latest_checkpoint_path = self.save_checkpoint(model, iter_cur)
                just_saved_checkpoint = True

            if self.auto_resume_exit(
                model, iter_cur, skip_if_just_saved=just_saved_checkpoint, recent_checkpoint_path=latest_checkpoint_path
            ):
                # termination requested
                self.callbacks.on_train_end(model, iteration=iter_cur)
                self.callbacks.on_app_end(model, iteration=iter_cur)
                logger.info("Taking a 10 sec nap and exiting training.")
                time.sleep(10)
                return

        logger.info("Training complete.")
        # validation in the end
        if dataloader_val is not None:
            self.validate(model_ddp, model, dataloader_val, iteration=self.config.trainer.max_iter)
        if self.config.trainer.save_final_checkpoint:
            self.save_checkpoint(model, self.config.trainer.max_iter)
        else:
            logger.info("Final checkpoint disabled by trainer.save_final_checkpoint=False.")
        self.callbacks.on_train_end(model, iteration=self.config.trainer.max_iter)
        self.callbacks.on_app_end(model, iteration=self.config.trainer.max_iter)
        logger.info("Taking a 10 sec nap and exiting training.")
        time.sleep(10)

    def _acquire_training_batch(self, model, dataloader_iter, dataloader, augment_pipe, *, iteration, grad_accum_iter):
        """Ordinary methods load every step; an opt-in cycle may reuse prepared data."""

        def load_fresh():
            data = self._next_training_batch(
                dataloader_iter, dataloader, iteration=iteration, grad_accum_iter=grad_accum_iter
            )
            return self.preprocess_data(model, data, augment_pipe)

        reuse = getattr(model, "training_batch_reuse", None)
        if reuse is None:
            return load_fresh()
        return reuse.acquire(iteration, grad_accum_iter, load_fresh)

    @staticmethod
    def _data_loader_description(dataloader: Any) -> str:
        fields = [f"type={type(dataloader).__name__}"]
        for name in ("datatags", "dataset_path", "num_workers", "worker_timeout_s"):
            if hasattr(dataloader, name):
                fields.append(f"{name}={getattr(dataloader, name)!r}")
        return " ".join(fields)

    @staticmethod
    def _summarize_data_origin(data: Any) -> str:
        if not isinstance(data, dict):
            return f"type={type(data).__name__}"

        def first_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return "<empty>"
                return value.reshape(-1)[0].item()
            if isinstance(value, (list, tuple)):
                return first_value(value[0]) if value else "<empty>"
            return value

        parts = []
        for key in ("shard", "fname"):
            if key in data:
                parts.append(f"{key}={first_value(data[key])!r}")
        meta = data.get("meta")
        if isinstance(meta, dict):
            for key in ("source", "id"):
                if key in meta:
                    parts.append(f"meta.{key}={first_value(meta[key])!r}")
        return " ".join(parts) if parts else f"keys={sorted(data.keys())!r}"

    def _next_training_batch(
        self,
        dataloader_iter: Any,
        dataloader: Any,
        *,
        iteration: int,
        grad_accum_iter: int,
    ) -> Any:
        warn_after_s = float(self.config.trainer.data_wait_warn_s)
        if warn_after_s < 0:
            raise ValueError(f"trainer.data_wait_warn_s must be >= 0, got {warn_after_s}")

        rank = int(os.environ.get("RANK", "0"))
        started = time.monotonic()
        stall_reported = threading.Event()
        timer: threading.Timer | None = None

        if warn_after_s > 0:

            def report_stall() -> None:
                stall_reported.set()
                logger.warning(
                    "[data-wait] "
                    f"rank={rank} iteration={iteration} grad_accum={grad_accum_iter} "
                    f"elapsed={time.monotonic() - started:.1f}s "
                    f"last_batch=({self._last_data_origin}) "
                    f"loader=({self._data_loader_description(dataloader)})"
                )

            timer = threading.Timer(warn_after_s, report_stall)
            timer.daemon = True
            timer.start()

        try:
            data = next(dataloader_iter)
        except StopIteration:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - started
            message = (
                "[data-fetch-error] "
                f"rank={rank} iteration={iteration} grad_accum={grad_accum_iter} elapsed={elapsed:.1f}s "
                f"last_batch=({self._last_data_origin}) "
                f"loader=({self._data_loader_description(dataloader)}) error={exc!r}"
            )
            logger.error(message)
            raise RuntimeError(message) from exc
        finally:
            if timer is not None:
                timer.cancel()

        elapsed = time.monotonic() - started
        self._last_data_origin = self._summarize_data_origin(data)
        if stall_reported.is_set():
            logger.warning(
                "[data-wait-recovered] "
                f"rank={rank} iteration={iteration} grad_accum={grad_accum_iter} "
                f"elapsed={elapsed:.1f}s current_batch=({self._last_data_origin})"
            )
        return data

    def load_pretrained_ckpt(self, model: SMDModel, device: torch.device | str = "cpu"):
        """
        Load pretrained model weights from a checkpoint.
        """
        key_map = self.config.trainer.checkpointer.pretrained_ckpt_key_map
        # use FSDP checkpointer to load the pretrained checkpoint
        # (which falls back to the basic checkpointer if the checkpoint ends with .pth)
        _checkpointer = FSDPCheckpointer(self.config.trainer.checkpointer)
        resume_iter = None
        for k_model, k_ckpt in key_map.items():
            if hasattr(model, k_model):
                model_dict = torch.nn.ModuleDict({k_ckpt: getattr(model, k_model)})
                resume_iter = _checkpointer.load(
                    model_dict,
                    path=self.config.trainer.checkpointer.pretrained_ckpt_path,
                    device=device,
                )
                logger.info(
                    f"Loaded {k_model} model from {k_ckpt} in {self.config.trainer.checkpointer.pretrained_ckpt_path} "
                    f"at iteration {resume_iter}"
                )
            else:
                logger.warning(
                    f"Model does not have submodule {k_model}. Skipping loading {k_ckpt} from "
                    f"{self.config.trainer.checkpointer.pretrained_ckpt_path}."
                )
        if resume_iter is not None:
            logger.info(f"Setting resume_iter for model to {resume_iter}.")
            model.resume_iter = resume_iter

    def save_checkpoint(self, model: SMDModel, iteration: int, path: str | None = None) -> str:
        logger.info(f"Saving checkpoint iteration {iteration}")
        self.callbacks.on_save_checkpoint_start(model, iteration=iteration)
        # awaken the dataloader to avoid timeout

        # LoRA-only save: skip frozen base model weights for much faster I/O
        use_lora_save = (
            model.is_lora_mode
            and isinstance(self.checkpointer, FSDPCheckpointer)
            and hasattr(self.checkpointer, "save_lora_only")
        )

        if use_lora_save:
            logger.info("Using LoRA-only checkpoint save (skipping frozen base model weights)")
            path = self.checkpointer.save_lora_only(
                model.model_dict,
                optimizer_dict=model.optimizer_dict,
                scheduler_dict=model.scheduler_dict,
                grad_scaler=model.grad_scaler,
                callbacks=self.callbacks,
                path=path,
                iteration=iteration,
                checkpoint_policy=model.checkpoint_policy(),
            )
        else:
            path = self.checkpointer.save(
                model.model_dict,
                optimizer_dict=model.optimizer_dict,
                scheduler_dict=model.scheduler_dict,
                grad_scaler=model.grad_scaler,
                callbacks=self.callbacks,
                path=path,
                iteration=iteration,
            )
        self.callbacks.on_save_checkpoint_success(model, iteration=iteration, path=path)
        # Explicitly clear memory after checkpointing: we need this to
        # avoid OOM during wandb logging where the VAE is loaded and
        # used for decoding
        gc.collect()
        torch.cuda.empty_cache()
        self.callbacks.on_save_checkpoint_end(model, iteration=iteration)
        return path

    def train_step(
        self,
        model_ddp: SMDModel | torch.nn.parallel.DistributedDataParallel,
        model: SMDModel,
        data: Dict[str, Any],
        iteration: int,
        grad_accum_iter: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Single training step

        Args:
            model_ddp (SMDModel | torch.nn.parallel.DistributedDataParallel): Distillation model with ddp wraaper.
            model (SMDModel): Distillation model.
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration: Current training iteration
            grad_accum_iter (int): Gradient accumulation iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output
        """
        grad_accum_rounds = self.config.trainer.grad_accum_rounds
        sync_grads = grad_accum_iter == grad_accum_rounds - 1
        use_fsdp = self.config.trainer.fsdp
        reuse = getattr(model, "training_batch_reuse", None)
        step_kwargs = {} if reuse is None else {"accumulation_index": grad_accum_iter}

        if model.requires_internal_backward(iteration):
            if self.config.trainer.ddp:
                raise RuntimeError(
                    "SMD per-segment internal backward does not support DDP. Use FSDP or single GPU."
                )
            with model.autocast():
                loss_map, outputs = model.single_train_step(
                    data,
                    iteration,
                    per_segment_backward=True,
                    grad_accum_rounds=grad_accum_rounds,
                    sync_grads=sync_grads,
                    is_fsdp=use_fsdp,
                    **step_kwargs,
                    on_backward_begin=lambda: self.callbacks.on_backward_begin(
                        model,
                        data,
                        {},
                        {},
                        iteration=iteration,
                        accum_iter=grad_accum_iter,
                    ),
                )
        else:
            # Ordinary methods retain the original one-forward, one-backward path.
            if not use_fsdp:
                with ddp.ddp_sync_grad(model_ddp, sync_grads):
                    with model.autocast():
                        loss_map, outputs = model_ddp.single_train_step(data, iteration, **step_kwargs)
                    self.callbacks.on_backward_begin(
                        model,
                        data,
                        outputs,
                        loss_map,
                        iteration=iteration,
                        accum_iter=grad_accum_iter,
                    )
                    model.grad_scaler.scale(loss_map["total_loss"] / grad_accum_rounds).backward()
            else:
                with model.autocast():
                    loss_map, outputs = model_ddp.single_train_step(data, iteration, **step_kwargs)
                self.callbacks.on_backward_begin(
                    model,
                    data,
                    outputs,
                    loss_map,
                    iteration=iteration,
                    accum_iter=grad_accum_iter,
                )
                with fsdp.fsdp_sync_grad(model, sync_grads):
                    model.grad_scaler.scale(loss_map["total_loss"] / grad_accum_rounds).backward()

        if grad_accum_iter == grad_accum_rounds - 1:
            # optimizer step, scheduler step, and more
            if os.environ.get("PIVOT_DEBUG_SYNC") == "1" and torch.cuda.is_available():
                torch.cuda.synchronize()
                _m = torch.cuda.memory_allocated() / (1024**3)
                print(f"[rank {os.environ.get('RANK', 0)}] MEM before_opt_step: alloc={_m:.2f}GB", flush=True)
            self.callbacks.on_optimizer_step_begin(model=model, iteration=iteration)
            model.optimizers_schedulers_step(iteration)
            if os.environ.get("PIVOT_DEBUG_SYNC") == "1" and torch.cuda.is_available():
                torch.cuda.synchronize()
                _m = torch.cuda.memory_allocated() / (1024**3)
                print(f"[rank {os.environ.get('RANK', 0)}] MEM after_opt_step: alloc={_m:.2f}GB", flush=True)
            # Zero after step to free memory on active optimizers
            model.optimizers_zero_grad(iteration)
            if reuse is not None:
                reuse.finish_iteration(iteration, grad_accum_rounds)
            if os.environ.get("PIVOT_DEBUG_SYNC") == "1" and torch.cuda.is_available():
                torch.cuda.synchronize()
                _m = torch.cuda.memory_allocated() / (1024**3)
                print(f"[rank {os.environ.get('RANK', 0)}] MEM after_zero_grad: alloc={_m:.2f}GB", flush=True)

        # detach loss_map and outputs
        return basic_utils.detach(loss_map), basic_utils.detach(outputs)

    @torch.no_grad()
    def validate(
        self,
        model_ddp: SMDModel | torch.nn.parallel.DistributedDataParallel,
        model: SMDModel,
        dataloader_val: DataLoader,
        iteration: int = 0,
    ) -> None:
        for idx, val_vars in enumerate(self.config.trainer.global_vars_val):
            with (
                set_temp_global_vars(val_vars),
                set_tmp_random_seed(
                    self.config.trainer.val_seed,
                    by_rank=True,
                    devices=[model.device] if model.device.type == "cuda" else [],
                ),
            ):
                self.callbacks.on_validation_begin(model, iteration=iteration, idx=idx)
                logger.info(f"Validation iteration {iteration}")
                for step, data in enumerate(dataloader_val):
                    if getattr(global_vars, "MAX_VAL_STEPS", None) is not None and step >= getattr(
                        global_vars, "MAX_VAL_STEPS"
                    ):
                        break

                    self.callbacks.on_validation_step_begin(model, data, step=step, iteration=iteration, idx=idx)
                    data = self.preprocess_data(model, data)
                    logger.debug(f"iteration: {iteration} | validation step: {step} | data: {basic_utils.to_str(data)}")
                    with model.autocast():
                        loss_map, outputs = model.single_train_step(data, iteration)
                    self.callbacks.on_validation_step_end(
                        model, data, outputs, loss_map, step=step, iteration=iteration, idx=idx
                    )
                self.callbacks.on_validation_end(model, iteration=iteration, idx=idx)
                synchronize()

    @torch.no_grad()
    def preprocess_data(
        self, model: SMDModel, data: Dict[str, Any], augment_pipe: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Preprocess the data before passing to the model.
        Args:
            model: SMDModel
            data: Dict[str, Any]

        Returns:
            Dict[str, Any]: Preprocessed data

        """
        ctx = dict(device=model.device, dtype=model.precision)
        data = basic_utils.to(data, **ctx)
        if getattr(model.net, "network_family", None) == "minimax_h3":
            if augment_pipe is not None:
                raise ValueError("H3 cached conditions do not support image augmentation pipelines.")
            from smd.networks.MiniMaxH3.condition import MiniMaxH3Condition

            condition = data.get("condition")
            if not isinstance(condition, MiniMaxH3Condition):
                raise TypeError("H3 training requires the production cached-condition loader.")
            # Keep the cached reference precision; H3 owns its model-input casts.
            data["condition"] = condition.to(device=model.device)
            data["condition"].validate()
            return data
        if augment_pipe is not None:
            data = augment_pipe(data)

        # we do not use torch.inference_mode here since resulting inference tensors
        # give runtime errors with gradient computations/checkpointing
        with torch.autocast(
            device_type=model.device.type,
            dtype=model.precision_amp_enc or torch.bfloat16,
            enabled=model.precision_amp_enc is not None,
        ):
            # Data/noise
            raw = "{}_raw".format
            for k in ["real", "noise"]:
                if k in data and raw(k) not in data:
                    data[raw(k)] = data[k]
                    # Skip VAE encoding for SparseTensor data (e.g. TRELLIS.2)
                    if basic_utils._is_sparse_tensor(data[k]):
                        continue
                    # dataloader returns real of shape [B, C, T, H, W]
                    if hasattr(model.net, "vae") and data[k].shape[1] != model.input_shape[0]:
                        # Encode the data/noise to latent space
                        data[k] = model.net.vae.encode(data[k])

            # Fill neg_condition from model config if dataloader didn't provide it
            if "neg_condition" not in data or data["neg_condition"] is None:
                neg_prompt = getattr(model.config, "neg_prompt", None)
                if neg_prompt is not None:
                    batch_size = (
                        data["condition"].shape[0]
                        if isinstance(data["condition"], torch.Tensor)
                        else len(data["condition"])
                    )
                    data["neg_condition"] = [neg_prompt] * batch_size
                    if not getattr(self, "_neg_prompt_injected_logged", False):
                        self._neg_prompt_injected_logged = True
                        logger.info(
                            f"[Trainer] Injected neg_condition from config.model.neg_prompt "
                            f"({len(neg_prompt)} chars, batch_size={batch_size}): "
                            f"{repr(neg_prompt[:80])}{'...' if len(neg_prompt) > 80 else ''}"
                        )

            # Text conditions
            for k in ["condition", "neg_condition"]:
                if k in data and raw(k) not in data:
                    data[raw(k)] = data[k]
                    if hasattr(model.net, "text_encoder") and isinstance(data[k], List):
                        # Encode the prompt to embedding
                        if getattr(model.net, "network_family", None) == "sana_wm":
                            data[k] = model.net.text_encoder.encode(
                                data[k],
                                add_chi_prompt=k == "condition",
                            )
                        elif getattr(model.net, "network_family", None) == "ltx2":
                            # LTX owns the Gemma autocast and SDPA backend. Disable
                            # the surrounding generic encoder autocast so its
                            # numerical policy is identical in Trainer and direct
                            # inference/precache paths.
                            with torch.autocast(device_type=model.device.type, enabled=False):
                                data[k] = model.net.text_encoder.encode(data[k])
                        else:
                            data[k] = model.net.text_encoder.encode(data[k])
                        if not getattr(self, f"_{k}_encoded_logged", False):
                            setattr(self, f"_{k}_encoded_logged", True)
                            shape = data[k].shape if isinstance(data[k], torch.Tensor) else type(data[k]).__name__
                            logger.info(f"[Trainer] Encoded {k} → {shape}")

            if getattr(model.net, "network_family", None) == "ltx2" and torch.distributed.is_initialized():
                synchronize()

            # Image conditions: encode raw PNGs on-GPU via the network-specific image encoder.
            if "condition_images" in data and hasattr(model.net, "encode_image"):
                pil_images = [_decode_condition_image_bytes(b) for b in data["condition_images"]]
                cond_features = model.net.encode_image(pil_images)
                data["condition"] = cond_features
                data["neg_condition"] = [torch.zeros_like(c) for c in cond_features]
                data["condition_images_raw"] = data["condition_images"]
                del data["condition_images"]
                if not getattr(self, "_image_cond_encoded_logged", False):
                    self._image_cond_encoded_logged = True
                    logger.info(
                        f"[Trainer] Encoded {len(pil_images)} condition images (feature shape={cond_features[0].shape})"
                    )

            # Edit conditioning: source image through VLM text encoder + VAE
            if getattr(model.net, "is_edit", False) and "source" in data:
                # 1. VAE-encode source images to latent space
                source_raw = data.pop("source")
                if hasattr(model.net, "vae") and source_raw.shape[1] != model.input_shape[0]:
                    data["source_latents"] = model.net.vae.encode(source_raw, mode="argmax")
                else:
                    data["source_latents"] = source_raw

                # 2. Get source PIL images for VLM text encoder
                source_pil = data.pop("source_pil", None)

                # 3. Re-encode text conditions with source images through VLM
                for k in ["condition", "neg_condition"]:
                    raw_key = f"{k}_raw"
                    if raw_key in data and isinstance(data[raw_key], list):
                        data[k] = model.net.text_encoder.encode(data[raw_key], images=source_pil)
                        if not getattr(self, f"_{k}_edit_encoded_logged", False):
                            setattr(self, f"_{k}_edit_encoded_logged", True)
                            shape = data[k][0].shape if isinstance(data[k], tuple) else data[k].shape
                            logger.info(f"[Trainer] Edit-encoded {k} with source images → {shape}")

            # Context for i2v/vid2vid
            if "real_raw" in data:
                if getattr(model.net, "is_i2v", False):  # extra vid context for i2v
                    # compute input for I2V models
                    real_raw_first_frame = data["real_raw"][:, :, 0:1]
                    bsz, channels = real_raw_first_frame.shape[0:2]
                    num_frames, height, width = data["real_raw"].shape[2:]
                    first_frame_cond = real_raw_first_frame

                    # Wan 2.1 I2V model concatenates first_frame_cond with noisy latents and mask.
                    # Wan 2.2 5B model replaces the first noisy latent frame with the first clean latent frame.
                    if model.net.concat_mask:
                        padding_shape = (bsz, channels, num_frames - 1, height, width)
                        first_frame_cond = torch.cat(
                            [real_raw_first_frame, real_raw_first_frame.new_zeros(*padding_shape)], dim=2
                        )

                    if hasattr(model.net, "vae"):
                        # Official Wan I2V implementation uses the VAE encoder with "argmax" mode to avoid stochasticity
                        data["first_frame_cond"] = model.net.vae.encode(first_frame_cond, mode="argmax")
                    else:
                        data["first_frame_cond"] = first_frame_cond

                if hasattr(model.net, "image_encoder"):
                    # Encode the first video frame with CLIP
                    data["encoder_hidden_states_image"] = model.net.image_encoder.encode(data["real_raw"][:, :, 0])

                if getattr(model.net, "is_vid2vid", False):  # extra vid context for vid2vid
                    assert hasattr(model.net, "prepare_vid_conditioning"), (
                        "model.net must have prepare_vid_conditioning method"
                    )
                    if "depth_latent" in data:
                        data["vid_context"] = model.net.prepare_vid_conditioning(
                            data["real_raw"], condition_latents=data["depth_latent"]
                        )
                    else:
                        data["vid_context"] = model.net.prepare_vid_conditioning(data["real_raw"])

                # Cosmos video2world conditioning: use first a few frames as conditioning
                if getattr(model.net, "is_video2world", False):
                    num_cond_frames = getattr(model.net, "num_conditioning_frames", 1)
                    real_raw_first_frames = data["real_raw"][:, :, :num_cond_frames]
                    bsz, channels, _, height, width = data["real_raw"].shape

                    # Encode conditioning frames with VAE
                    if hasattr(model.net, "vae"):
                        data["conditioning_latents"] = model.net.vae.encode(real_raw_first_frames, mode="argmax")
                    else:
                        data["conditioning_latents"] = real_raw_first_frames

                    # Create condition mask: 1 for conditioning frames, 0 for generated frames
                    t_latent = data["real"].shape[2]
                    t_cond_latent = data["conditioning_latents"].shape[2]
                    condition_mask = torch.zeros(bsz, 1, t_latent, height // 8, width // 8, device=data["real"].device)
                    condition_mask[:, :, :t_cond_latent] = 1.0
                    data["condition_mask"] = condition_mask

        # Move encoded data to dtype and device
        data = basic_utils.to(data, **ctx)

        return data

    def auto_resume_exit(
        self, model: SMDModel, iteration: int, skip_if_just_saved: bool = False, recent_checkpoint_path: str = None
    ) -> bool:
        """
        Check if the training should be terminated and auto-resume should be triggered.

        Args:
            model (SMDModel): Distillation model.
            iteration (int): Current training iteration
            skip_if_just_saved (bool): Skip saving checkpoint if we just saved one
            recent_checkpoint_path (str): Path to the most recently saved checkpoint

        Returns:
            bool: True if the training should be terminated, False otherwise

        """
        # Check termination on rank 0 and broadcast to all ranks
        termination_requested = False

        # Ensure all ranks are ready before rank 0 checks termination
        synchronize()

        if is_rank0():
            termination_requested = self.auto_resume.termination_requested()

        # Broadcast the decision from rank 0 to all other ranks
        if world_size() > 1:
            termination_tensor = torch.tensor([1.0 if termination_requested else 0.0], device=model.device)
            dist.broadcast(termination_tensor, src=0)
            termination_requested = termination_tensor.item() > 0.5

        # Ensure all ranks have received the broadcast before proceeding
        synchronize()

        if not termination_requested:
            return False

        # Termination requested - save checkpoint and request resume
        ar_details = self.auto_resume.get_resume_details() or {}

        # Only save checkpoint if we haven't just saved one
        if not skip_if_just_saved:
            save_path = self.save_checkpoint(
                model, iteration, path=os.path.join(self.config.trainer.checkpointer.save_dir, "latest_ar.pth")
            )
            ar_details["save_path"] = save_path
        else:
            # Use the most recent checkpoint path
            logger.info("Skipping AutoResume checkpoint save as we just saved a regular checkpoint")
            if recent_checkpoint_path:
                save_path = recent_checkpoint_path
                logger.info(f"Using recently saved checkpoint: {save_path}")
            else:
                # Fallback: construct the path (this should rarely happen)
                logger.warning("No recent checkpoint path provided, constructing path")
                if isinstance(self.checkpointer, FSDPCheckpointer):
                    save_path = os.path.join(self.config.trainer.checkpointer.save_dir, f"{iteration:07d}")
                else:
                    save_path = os.path.join(self.config.trainer.checkpointer.save_dir, f"{iteration:07d}.pth")
            ar_details["save_path"] = save_path

        if is_rank0():
            self.auto_resume.request_resume(user_dict=ar_details)

        logger.info("Autoresume requested. Terminating training.")
        return True
