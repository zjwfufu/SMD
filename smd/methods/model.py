# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING, Callable, List
import contextlib
import os

import torch

from smd.configs.opt import get_scheduler
from smd.utils import instantiate
from smd.utils.checkpoint_policy import CheckpointPolicy, CheckpointSaveMode, ModuleCheckpointPolicy
from smd.utils.distributed import synchronize, world_size
from smd.utils.hf_utils import rank0_first_load
from smd.utils.io_utils import s3_load
import smd.utils.logging_utils as logger
import smd.utils.basic_utils as basic_utils
from smd.utils.distributed import is_rank0

if TYPE_CHECKING:
    from smd.configs.config import BaseModelConfig
    from smd.networks.network import SMDNetwork


class SMDModel(torch.nn.Module):
    def __init__(self, config: BaseModelConfig):
        """SMDModel class for implementing training interface for all smd networks.

        Args:
            config (BaseModelConfig): The configuration for the SMD model
        """
        super().__init__()
        self.config = config

        # device
        self.device = torch.device(config.device)
        if self.device.type not in ["cuda", "cpu"]:
            raise ValueError(f"SMDModel models only support cuda and cpu devices, got {self.device.type}")

        # precision and autocast
        self.set_precision(
            precision=self.config.precision,
            precision_amp=self.config.precision_amp,
            precision_amp_infer=self.config.precision_amp_infer,
            precision_amp_enc=self.config.precision_amp_enc,
            precision_fsdp=self.config.precision_fsdp,
        )

        # input shape
        self.input_shape = config.input_shape
        logger.info(f"Input shape is {self.input_shape}.")

        # define the name of the EMA networks to use
        use_ema = config.use_ema
        if isinstance(use_ema, bool):
            use_ema = ["ema"] if use_ema else []
        if not all(isinstance(name, str) and name.startswith("ema") for name in use_ema):
            raise ValueError(f"use_ema must be a bool or a list of strings starting with `ema`, got {use_ema}.")
        self.use_ema = use_ema

        # instantiate all necessary nets and submodules
        self.build_model()

    @property
    def is_lora_mode(self) -> bool:
        """Whether the model uses LoRA adapters. Override in subclasses that support LoRA."""
        return False

    def _setup_ema(self):
        """Initialize EMA networks. Only call during build_model(), before checkpoint loading."""
        for name in self.use_ema:
            if not hasattr(self, name):
                logger.info(f"Initializing EMA network {name}")
                ema = instantiate(self.config.net)
                ema.eval().requires_grad_(False)
                setattr(self, name, ema)
            else:
                logger.warning(
                    f"EMA network {name} already exists, skipping initialization. "
                    "This is expected if loading pretrained network weights"
                )
            ema = getattr(self, name)
            # Only rank 0 loads weights if using meta initialization (non-rank-0 has meta tensors in self.net)
            if (not self.config.fsdp_meta_init) or is_rank0():
                net_load_info = ema.load_state_dict(self.net.state_dict(), strict=False)
                logger.success(f"Loaded EMA network {name}. Loading info: {net_load_info}")
            # Broadcast EMA weights from rank 0 to all other ranks when using meta init
            if world_size() > 1 and self.config.fsdp_meta_init:
                ema.to(device=self.device)
                for param in ema.parameters():
                    torch.distributed.broadcast(param.data, src=0)
                for buffer in ema.buffers():
                    torch.distributed.broadcast(buffer.data, src=0)
        synchronize()

    def _get_meta_init_context(self, fsdp_meta_init: bool = None):
        """Get context manager for FSDP meta initialization.

        When fsdp_meta_init is enabled, non-rank-0 processes use meta device
        for memory-efficient loading. Rank 0 loads weights normally, then
        FSDP syncs weights to other ranks via sync_module_states.

        Args:
            fsdp_meta_init: Whether to use meta initialization. If None, uses
                self.config.fsdp_meta_init.
        """
        if fsdp_meta_init is None:
            fsdp_meta_init = self.config.fsdp_meta_init
        use_meta = fsdp_meta_init and not is_rank0()
        if use_meta:
            return torch.device("meta")
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Coordinated model loading (decoupled from FSDP / tie / sharding)
    #
    # These protected methods are the SINGLE place that couples "read weights
    # from disk" with "distributed coordination + fault tolerance". Every method's
    # build_model should route network instantiation / preprocessor loading
    # through them instead of hand-rolling the rank gating. They follow the
    # symmetry template of ``_load_pretrained_model`` (gate the disk read, but
    # have every rank hit exactly one barrier), so they are deadlock-safe.
    # ------------------------------------------------------------------

    def _coordinated_instantiate(self, config, *, all_rank_read: bool) -> torch.nn.Module:
        """Instantiate a network with coordinated, fault-tolerant weight reads.

        A network's ``__init__`` reads its weights from the HF cache (via
        ``safe_load_from_pretrained``, which already retries). This wrapper adds the
        *distributed coordination* around that read so 8 ranks don't stampede the
        same cache concurrently:

        * ``all_rank_read=True`` — every rank reads real weights from disk (e.g. the
          ``force_all_rank_load`` teacher, or any non-meta build). We serialize via
          :func:`rank0_first_load`: rank 0 constructs first (warming the cache),
          the others wait, then construct from the warm cache. Exactly ONE barrier
          per rank.

          **Requirement:** the network ``__init__`` must issue NO collective
          (barrier/broadcast/all-*). Otherwise rank 0 would block inside its
          ``__init__`` waiting for peers that are stuck at the outer barrier. All
          current networks satisfy this (their ``__init__`` only does
          ``from_pretrained``/``from_config``).

        * ``all_rank_read=False`` — meta-init path: only rank 0 reads; non-rank0
          allocate meta (zero-byte) tensors and receive real weights later via the
          FSDP broadcast (``model_to_fsdp`` / ``sync_module_states``). We still emit
          one unconditional barrier (matching ``_load_pretrained_model``'s trailing
          ``synchronize()``) so this branch is symmetric with the other and with
          callers that previously relied on ``build_teacher``'s trailing barrier.

        Both branches => exactly one barrier per rank => symmetric => deadlock-safe.

        Note: pass ``all_rank_read = not meta_init`` (NOT ``force_all_rank_load``).
        A network reads on all ranks whenever it is NOT meta-initialized, which
        includes ``fsdp_meta_init=False`` runs — using ``force_all_rank_load`` would
        miss those and reopen the concurrent-read race.
        """
        if all_rank_read:
            with rank0_first_load():
                net = instantiate(config)
        else:
            with self._get_meta_init_context():
                net = instantiate(config)
            synchronize()
        return net

    def _build_preprocessors(self, net: torch.nn.Module) -> None:
        """Load a network's preprocessors (text_encoder / vae) with rank0-first coordination.

        Preprocessors are ALWAYS read on every rank (there is no meta optimization
        for them — every rank needs the real text_encoder/vae to encode prompts /
        decode latents). That makes them the single most frequent concurrent-read
        race point across all methods. We serialize the whole ``init_preprocessors``
        block via :func:`rank0_first_load` (one barrier per rank), so rank 0 warms
        the cache and the others read from it.

        Coordination lives HERE (not inside the network's ``init_preprocessors``) so
        every network is treated uniformly and there is exactly one barrier layer.
        """
        if hasattr(net, "init_preprocessors") and self.config.enable_preprocessors:
            with rank0_first_load():
                net.init_preprocessors()

    def set_precision(
        self,
        precision: str = "float32",
        precision_amp: str | None = None,
        precision_amp_infer: str | None = None,
        precision_amp_enc: str | None = None,
        precision_fsdp: str | None = None,
    ):
        """Set the model/data precision and automatic mixed precision (AMP) precision for training and inference.

        All precision arguments are strings that are mapped to torch dtypes according to PRECISION_MAP:
            "float16" -> torch.float16
            "bfloat16" -> torch.bfloat16
            "float32" -> torch.float32
            "float64" -> torch.float64

        Note that the precision of the time steps is handled in the noise scheduler (defaulting to float64 for numerical stability).

        Args:
            precision: Precision for model/optimizer states and data. Recommended to be float32 if precision_amp is not None.
            precision_amp: Precision for AMP during training. If None or equal to precision, AMP is disabled during training.
            precision_amp_infer: Precision for AMP during inference. If None or equal to precision, AMP is disabled during inference.
            precision_amp_enc: Precision for AMP en-/decoder (e.g., for VAEs or text encoders).
                If None or equal to precision, AMP is disabled during en-/decoding.
            precision_fsdp: Precision for FSDP2 parameter storage and gradient reduction.
                If None, defaults to `precision`.
        """

        # precision for model/optimizer states and data
        self.precision = basic_utils.PRECISION_MAP[precision]

        # precision for FSDP2 parameter storage and gradient reduction (defaults to model precision)
        self.precision_fsdp = (
            basic_utils.PRECISION_MAP[precision_fsdp] if precision_fsdp is not None else self.precision
        )

        # precision for AMP training
        if precision_amp is None or precision_amp == precision:
            # AMP is disabled during training
            self.precision_amp = None
        else:
            self.precision_amp = basic_utils.PRECISION_MAP[precision_amp]
            if self.precision != torch.float32:
                logger.warning(
                    f"Autocast to {self.precision_amp} is enabled and model and data are cast to {self.precision}. "
                    f"It is recommended to set `config.model.precision` to `float32`."
                )

        # precision for AMP inference
        if precision_amp_infer is None or precision_amp_infer == precision:
            # AMP is disabled during inference
            self.precision_amp_infer = None
        else:
            self.precision_amp_infer = basic_utils.PRECISION_MAP[precision_amp_infer]

        # precision for AMP en-/decoder (e.g., for VAEs or text encoders)
        if precision_amp_enc is None or precision_amp_enc == precision:
            # AMP is disabled during en-/decoding
            self.precision_amp_enc = None
        else:
            self.precision_amp_enc = basic_utils.PRECISION_MAP[precision_amp_enc]

        logger.critical(
            f"Model and data precision: {self.precision}. AMP training precision: {self.precision_amp}. "
            f"AMP en-/decoder precision: {self.precision_amp_enc}. AMP inference precision: {self.precision_amp_infer}. "
            f"FSDP precision: {self.precision_fsdp}."
        )

    @property
    def teacher_config(self) -> dict:
        teacher_config = self.config.net
        if self.config.teacher is not None:
            logger.critical("Using teacher config (usually due to teacher architecture being different from student)")
            teacher_config = self.config.teacher
        return teacher_config

    def build_teacher(self, force_all_rank_load: bool = False):
        # instantiate the teacher
        logger.info("Instantiating the teacher")
        meta_init_teacher = self.config.add_teacher_to_fsdp_dict and self.config.fsdp_meta_init
        if force_all_rank_load:
            meta_init_teacher = False  # All ranks load from disk independently, no broadcast needed
        logger.info(
            f"build_teacher: add_teacher_to_fsdp_dict={self.config.add_teacher_to_fsdp_dict}, "
            f"fsdp_meta_init={self.config.fsdp_meta_init}, meta_init_teacher={meta_init_teacher}, "
            f"force_all_rank_load={force_all_rank_load}"
        )
        # Coordinated instantiate: when not meta-initialized, ALL ranks read the
        # teacher's weights from the HF cache, so serialize rank0-first to avoid the
        # concurrent-read shard race. The meta path keeps its single barrier inside
        # _coordinated_instantiate, so build_teacher still ends synchronized either
        # way (the old trailing bare synchronize() is therefore removed).
        self.teacher = self._coordinated_instantiate(self.teacher_config, all_rank_read=not meta_init_teacher)
        logger.info(
            f"Teacher guidance scale set to {self.config.guidance_scale} (skip-layer guidance: {self.config.skip_layers})"
        )

        # load pre-trained teacher model (external checkpoint via torch.load; this
        # has its own rank gating + trailing barrier in _load_pretrained_model).
        model_path = self.config.pretrained_model_path
        if model_path is not None and len(model_path) > 0:
            SMDModel._load_pretrained_model(self.teacher, model_path, fsdp_meta_init=meta_init_teacher)
        self.teacher.eval().requires_grad_(False)

    def load_student_weights_and_ema(self):
        # path to an external network ckpt different from teacher (e.g. pretrained kd, pretrained self-forcing, etc.)
        pretrained_student_net_path = self.config.pretrained_student_net_path
        has_student_path = pretrained_student_net_path is not None and len(pretrained_student_net_path) > 0
        # path to the pretrained teacher model ckpt
        pretrained_model_path = self.config.pretrained_model_path
        has_model_path = pretrained_model_path is not None and len(pretrained_model_path) > 0

        if self.config.load_student_weights:
            logger.info("Loading student weights")
            if has_student_path:
                SMDModel._load_pretrained_model(
                    self.net, pretrained_student_net_path, fsdp_meta_init=self.config.fsdp_meta_init
                )
            elif has_model_path:
                if getattr(self, "teacher", None) is not None:
                    logger.info("Loading student weights from teacher weights")
                    # initialize the consistency network with the teacher weights
                    # Only rank 0 loads weights if using meta initialization
                    if (not self.config.fsdp_meta_init) or is_rank0():
                        net_load_info = self.net.load_state_dict(self.teacher.state_dict(), strict=False)
                        logger.success(f"Net initializing info: {net_load_info}")
                else:
                    SMDModel._load_pretrained_model(
                        self.net, pretrained_model_path, fsdp_meta_init=self.config.fsdp_meta_init
                    )
            else:
                logger.warning(
                    "No student weights specified. This might be intended if the student initialization already "
                    "loads pretrained weights (e.g., from diffusers)."
                )

            if has_student_path or has_model_path:
                synchronize()

        elif has_student_path:
            logger.warning("Ignoring `pretrained_student_net_path` since `load_student_weights` is False.")
        elif has_model_path and getattr(self, "teacher", None) is None:
            logger.warning("Ignoring `pretrained_model_path` since `load_student_weights` is False.")

        # load EMA weights
        self._setup_ema()

    def build_model(self):
        # instantiate the generator network
        logger.info("Instantiating the generator network")
        with self._get_meta_init_context():
            self.net = instantiate(self.config.net)
        no_grad_params = [n for n, p in self.net.named_parameters() if not p.requires_grad]
        if any(no_grad_params):
            logger.warning(
                f"The `requires_grad` attribute of these parameters is `False` at initialization and will be set to `True`: {no_grad_params}"
            )
        self.net.train().requires_grad_(True)

        # initialize the preprocessors if they exist, only in the net model
        # this is useful for models that require specific preprocessing. e.g. SD model for image / text encoding
        # Routed through the coordinated loader so the all-rank text_encoder/vae
        # read is serialized rank0-first (the most frequent concurrent-read race).
        self._build_preprocessors(self.net)

    def on_train_begin(self, is_fsdp=False):
        self._is_fsdp = is_fsdp  # Store for later use (e.g., to skip EMA during inference)
        ctx = dict(dtype=self.precision, device=self.device)

        if is_fsdp:
            # Cast fsdp_dict modules to precision_fsdp before FSDP wrapping (when AMP is disabled).
            # This sets the parameter storage dtype that FSDP will preserve for shards and gradient reduction.
            for net_name, net in self.fsdp_dict.items():
                logger.debug(f"Casting {net_name} to dtype={self.precision_fsdp} (pre-FSDP).")
                net.to(dtype=self.precision_fsdp)
                synchronize()

            # Move EMA networks as they aren't handled by FSDP
            for net_name, net in self.ema_dict.items():
                logger.debug(f"Starting moving EMA {net_name} to device: {self.device}.")
                net.to(device=self.device)
                synchronize()
                logger.debug(f"Completed moving EMA {net_name} to device: {self.device}.")
        else:
            # If no FSDP, we need to manually handle casting and device management
            for net_name, net in self.fsdp_dict.items():
                logger.debug(f"Starting moving {net_name} to context: {ctx}.")
                net.to(**ctx)
                synchronize()
                logger.debug(f"Completed moving {net_name} to context: {ctx}.")

        # Handle teacher separately if it's not in the FSDP dict
        if getattr(self, "teacher", None) is not None:
            fsdp_dict_keys = list(self.fsdp_dict.keys())
            logger.info(
                f"Teacher check: add_teacher_to_fsdp_dict={self.config.add_teacher_to_fsdp_dict}, "
                f"fsdp_dict keys={fsdp_dict_keys}, teacher in fsdp_dict={'teacher' in fsdp_dict_keys}"
            )
            if "teacher" not in self.fsdp_dict:
                # No gradients for teacher, can put in lower precision
                logger.info(f"Started converting teacher to context: {ctx}.")
                self.teacher.to(**ctx)
                synchronize()

        # For networks that don't need gradients, we always manually handle casting and device management
        if hasattr(self.net, "init_preprocessors") and self.config.enable_preprocessors:
            logger.debug(f"Starting moving preprocessors to context: {ctx}.")
            if hasattr(self.net, "vae"):
                if getattr(self.net.vae, "_smd_keep_cpu_after_fsdp_init", False):
                    self.net.vae.to(device="cpu")
                else:
                    self.net.vae.to(**ctx)
                synchronize()
            if hasattr(self.net, "text_encoder"):
                self.net.text_encoder.to(**ctx)
                synchronize()
            if hasattr(self.net, "image_encoder"):
                self.net.image_encoder.to(**ctx)
                synchronize()
            logger.debug(f"Completed moving preprocessors to context: {ctx}.")
            self.log_preprocessor_devices("after_on_train_begin")

        synchronize()

    @staticmethod
    def _object_device_summary(obj: Any) -> str:
        """Best-effort device summary for nn.Modules and preprocessor wrappers."""
        if obj is None:
            return "missing"
        if torch.is_tensor(obj):
            return str(obj.device)
        if isinstance(obj, torch.nn.Module):
            tensors = list(obj.parameters(recurse=True)) + list(obj.buffers(recurse=True))
            if not tensors:
                return "no_tensors"
            devices = sorted({str(t.device) for t in tensors})
            return ",".join(devices[:4]) + ("..." if len(devices) > 4 else "")
        for inner_name in ("text_encoder", "vae", "image_encoder", "model"):
            inner = getattr(obj, inner_name, None)
            if inner is not None and inner is not obj:
                inner_summary = SMDModel._object_device_summary(inner)
                if inner_summary not in {"missing", "no_tensors"}:
                    return f"{inner_name}:{inner_summary}"
        return type(obj).__name__

    def log_preprocessor_devices(self, tag: str) -> None:
        """Log where auxiliary preprocessors live (text encoders, VAE, image encoders)."""
        entries: list[str] = []
        for owner_name in ("net", "teacher"):
            owner = getattr(self, owner_name, None)
            if owner is None:
                continue
            for attr_name in ("text_encoder", "vae", "image_encoder", "image_cond_model", "image_processor"):
                if hasattr(owner, attr_name):
                    entries.append(f"{owner_name}.{attr_name}={self._object_device_summary(getattr(owner, attr_name))}")
        if entries:
            logger.info(f"[preprocessor_devices:{tag}] " + "; ".join(entries))

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        gen_data = self.net(input_student, t_student, condition=condition, fwd_pred_type="x0")
        return gen_data

    @classmethod
    def _student_sample_loop(
        cls,
        net: SMDNetwork,
        x: torch.Tensor,
        t_list: torch.Tensor,
        condition: Any = None,
        student_sample_type: str = "sde",
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample loop for the student network.

        Args:
            net: The SMDNetwork network
            x: The latents to start from
            t_list: Timesteps to sample
            condition: Optional conditioning information
            student_sample_type: Type of student multistep sampling

        Returns:
            The sampled data
        """
        batch_size = x.shape[0]

        # Check if network has custom conditioning preservation hooks
        # This allows video I2V/v2w models to handle conditioning without
        # complicating this generic loop with model-specific logic
        has_preserve_hook = hasattr(net, "preserve_conditioning")

        x_pred = x
        for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
            # Forward pass to get x0 prediction
            t_batch = t_cur.expand(batch_size)
            x_pred = net(x, t_batch, condition=condition, fwd_pred_type="x0")

            # Allow network to preserve conditioning frames
            if has_preserve_hook:
                x_pred = net.preserve_conditioning(x_pred, condition)

            # One step reverse process
            if t_next > 0:
                t_next_batch = t_next.expand(batch_size)
                if student_sample_type == "sde":
                    eps_infer = torch.randn_like(x_pred)
                elif student_sample_type == "ode":
                    eps_infer = net.noise_scheduler.x0_to_eps(xt=x, x0=x_pred, t=t_batch)
                else:
                    raise NotImplementedError(
                        f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                    )
                x = net.noise_scheduler.forward_process(x_pred, eps_infer, t_next_batch)

                # Preserve conditioning frames after adding noise
                if has_preserve_hook:
                    x = net.preserve_conditioning(x, condition)

        return x_pred

    @classmethod
    def generator_fn(
        cls,
        net: SMDNetwork,
        noise: torch.Tensor,
        student_sample_steps: int = 1,
        t_list: Optional[List[float]] = None,
        data: torch.Tensor = None,
        precision_amp: Optional[torch.dtype] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Single-step or multistep generation with the distilled network.

        Args:
            net: The SMDNetwork network
            noise: Pure noise to start from (zero-mean, unit-variance Gaussian)
            student_sample_steps: Number of student diffusion steps
            t_list: Timesteps to sample (defaults to None: use noise_schedule.get_t_list() instead)
            data (torch.Tensor, optional): Additional data to add to initial latents.
                Useful for inpainting or other conditional tasks. Defaults to None.
            precision_amp (torch.dtype, optional): If not None, uses autocast with this dtype for inference.
            **kwargs: Additional keyword arguments passed to the network.

        Returns:
            Generated sample from the distilled single-step or multistep student.
        """
        with basic_utils.inference_mode(net, precision_amp=precision_amp, device_type=noise.device.type):
            # Default timestep schedule
            if t_list is None:
                t_list = net.noise_scheduler.get_t_list(sample_steps=student_sample_steps, device=noise.device)
            else:
                assert len(t_list) - 1 == student_sample_steps, (
                    f"t_list length (excluding zero) != student_sample_steps: {len(t_list) - 1} != {student_sample_steps}"
                )
                t_list = torch.tensor(t_list, device=noise.device, dtype=net.noise_scheduler.t_precision)
            assert t_list[-1].item() == 0, "t_list[-1] must be zero"

            # Initialize with noise scaling
            latents = net.noise_scheduler.latents(noise=noise, t_init=t_list[0])

            # Add optional data (e.g., for inpainting)
            if data is not None:
                latents = latents + data

            # Multistep sampling loop
            return cls._student_sample_loop(net, latents, t_list=t_list, **kwargs).to(dtype=noise.dtype)

    def sample(self, net: SMDNetwork, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hasattr(net, "sample")
        with basic_utils.inference_mode(net, precision_amp=self.precision_amp_infer):
            return net.sample(
                noise,
                guidance_scale=self.config.guidance_scale,
                **kwargs,
            ).to(dtype=noise.dtype)

    def sample_student(self, noise: torch.Tensor, condition: Any, net: Optional[SMDNetwork] = None) -> torch.Tensor:
        """Sample student latents from noise with model-default student sampling settings."""
        net = net or self.net_inference
        return self.generator_fn(
            net=net,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            student_sample_type=self.config.student_sample_type,
            t_list=getattr(self.config.sample_t_cfg, "t_list", None),
            precision_amp=self.precision_amp_infer,
        )

    def _prepare_training_data(self, data: Dict[str, Any]) -> tuple[torch.Tensor, Any, Any]:
        """Prepare training data and conditions from input data dict.

        Args:
            data: Data dict containing real data, conditions, etc.

        Returns:
            tuple of (real_data, condition, neg_condition)
        """
        real_data = data["real"]
        if getattr(self.net, "is_vid2vid", False):
            # handle vid2vid
            vid_context = data["vid_context"]  # this is processed in trainer.py
            condition = {
                "text_embeds": data["condition"],
                "vid_context": vid_context,
            }
            neg_condition = {
                "text_embeds": data["neg_condition"],
                "vid_context": vid_context,
            }
        elif getattr(self.net, "is_i2v", False):
            # handle i2v (WanI2V style)
            first_frame_cond = data["first_frame_cond"]  # this is processed in trainer.py
            condition = {
                "text_embeds": data["condition"],
                "first_frame_cond": first_frame_cond,
            }
            neg_condition = {
                "text_embeds": data["neg_condition"],
                "first_frame_cond": first_frame_cond,
            }
            if hasattr(self.net, "image_encoder"):
                condition["encoder_hidden_states_image"] = data["encoder_hidden_states_image"]
                neg_condition["encoder_hidden_states_image"] = data["encoder_hidden_states_image"]
        elif getattr(self.net, "is_video2world", False):
            # handle video2world (Cosmos style)
            conditioning_latents = data["conditioning_latents"]  # this is processed in trainer.py
            condition_mask = data["condition_mask"]  # this is processed in trainer.py
            condition = {
                "text_embeds": data["condition"],
                "conditioning_latents": conditioning_latents,
                "condition_mask": condition_mask,
            }
            neg_condition = {
                "text_embeds": data["neg_condition"],
                "conditioning_latents": conditioning_latents,
                "condition_mask": condition_mask,
            }
        else:
            # handle other cases
            condition = data["condition"]
            neg_condition = data["neg_condition"]

        return real_data, condition, neg_condition

    @abstractmethod
    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        """
        Get model outputs as a dictionary of tensors.
        """

    @abstractmethod
    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the SMD model.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output
        """

    def requires_internal_backward(self, iteration: int) -> bool:
        """Whether this iteration performs backward inside ``single_train_step``."""
        del iteration
        return False

    # ------------------------------------------------------------------
    # Bundle export contract
    # ------------------------------------------------------------------
    # Subclasses that use ExportInferenceBundle must define ``method_name``
    # (a class variable) and implement ``get_sampler_snapshot()``.

    method_name: str  # e.g. "simplified_distill" — no default, subclass must set

    def get_sampler_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict describing the sampler state.

        The dict MUST contain a ``"type"`` key so the inference-side
        dispatcher knows which sampler to reconstruct.

        The default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not implement "
            "get_sampler_snapshot(). Add it to support inference-bundle export."
        )

    def init_optimizers(self):
        """Initialize optimizers, lr_schedulers and grad_scalers"""
        # instantiate the optimizer for the generator network
        self.net_optimizer = instantiate(self.config.net_optimizer, model=self.net)
        # instantiate the lr scheduler for the generator network
        self.net_lr_scheduler = get_scheduler(self.net_optimizer, self.config.net_scheduler)
        # instantiate the gradient scaler (only fp16 needs grad_scaler)
        grad_scaler_required = self.precision == torch.float16 or self.precision_amp == torch.float16
        grad_scaler_enabled = self.config.grad_scaler_enabled and grad_scaler_required
        if grad_scaler_required:
            if grad_scaler_enabled:
                logger.info(
                    f"Grad scaler enabled with init scale {self.config.grad_scaler_init_scale} and growth interval {self.config.grad_scaler_growth_interval}."
                )
            else:
                logger.warning(
                    f"Grad scaler disabled but recommended when using float16 precision (precision={self.precision}, precision_amp={self.precision_amp})."
                )
        self.grad_scaler = torch.amp.GradScaler(
            init_scale=self.config.grad_scaler_init_scale,
            growth_interval=self.config.grad_scaler_growth_interval,
            enabled=grad_scaler_enabled,
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration
        """
        return [self.net_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration
        """
        return [self.net_lr_scheduler]

    def optimizers_zero_grad(self, iteration: int) -> None:
        """
        Zero the gradients of the optimizers based on the iteration
        """
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad(set_to_none=True)

    def should_use_grad_scaler(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Check if grad_scaler should be used for the given optimizer.

        GradScaler only works with float32 gradients. When model weights are in
        FP16/BF16 (e.g., with FSDP2 mixed precision), gradients are also in that
        dtype and grad_scaler cannot be used.

        Args:
            optimizer: The optimizer to check

        Returns:
            True if grad_scaler should be used, False otherwise
        """
        if not self.grad_scaler.is_enabled():
            return False

        # Check if any gradient is not float32
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None and param.grad.dtype != torch.float32:
                    return False
        return True

    def optimizers_schedulers_step(self, iteration: int) -> None:
        """
        Step the optimizer and scheduler step based on the iteration,
        and gradient scaler is also updated
        """
        for optimizer in self.get_optimizers(iteration):
            if self.should_use_grad_scaler(optimizer):
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                optimizer.step()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    @staticmethod
    def _load_pretrained_model(
        model: torch.nn.Module,
        pretrained_model_path: str,
        device: Optional[torch.device] = "cpu",
        fsdp_meta_init: bool = False,
    ) -> None:
        """
        Load the pre-trained model from the given path
        Args:
            model (torch.nn.Module): The model to load
            pretrained_model_path (str): The path to the pretrained model
            device (Optional[torch.device]): The device to load the model on
            fsdp_meta_init (bool): Whether to use meta initialization for FSDP
        """
        # Only rank-0 loads weights if using meta initialization
        if (not fsdp_meta_init) or is_rank0():
            logger.info(f"Loading the pretrained diffusion model from {pretrained_model_path}")
            if pretrained_model_path.startswith("s3://"):
                key = pretrained_model_path.split("/")[-1]
                local_path = os.path.join(
                    os.environ.get("SMD_OUTPUT_ROOT", "SMD_OUTPUT"), "model", key.split("/")[-1]
                )
                if os.path.exists(local_path):
                    logger.info(f"Model already exists at {local_path}, loading from local cache")
                    model_dict = torch.load(local_path, weights_only=True, map_location=device)
                else:
                    model_dict = torch.load(s3_load(pretrained_model_path), weights_only=True, map_location=device)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    torch.save(model_dict, local_path)
            else:
                assert os.path.isfile(pretrained_model_path), f"{pretrained_model_path} is not a valid file"
                model_dict = torch.load(pretrained_model_path, weights_only=True, map_location=device)

            for k, v in model_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    # since FSDP2 cannot handle 0-dim. tensors, we adapted all network definitions to use 1-dim.
                    # tensors with numel equal to 1
                    model_dict[k] = v.unsqueeze(0)
                    logger.debug(f"Changed {k} from 0-dim. tensor to 1-dim. tensor with numel equal to 1.")

            model_load_info = model.load_state_dict(model_dict, strict=False)
            torch.cuda.empty_cache()
            logger.success(f"Model loading info: {model_load_info}")
        synchronize()

    def autocast(self):
        """Return the autocast context manager for training"""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.precision_amp,
            enabled=self.precision_amp is not None,
        )

    @property
    def ema_dict(self):
        """Return dict containing all EMA networks"""
        return {name: getattr(self, name) for name in self.use_ema}

    def update_ema(self, ema_name: str, beta: float, initialize: bool, is_fsdp: bool) -> None:
        """Update a single EMA network from ``self.net`` (student EMA).

        This is the unified per-step EMA interface: :class:`EMACallback` computes
        ``beta`` and calls here, so the callback never re-implements the update.
        Methods that own auxiliary EMA modules (e.g. SMD's pivot EMA)
        override this to update those in the same call.

        Frozen student params are skipped — they are copied once at
        initialization and never change, so re-copying would all-gather the
        frozen backbone every step (fatal for LoRA/FSDP throughput).
        """
        ema = getattr(self, ema_name, None)
        if ema is None:
            raise RuntimeError(f"update_ema: EMA network {ema_name!r} not found (use_ema={self.use_ema}).")
        ema_state_dict = ema.state_dict()

        for name, p_net in self.net.named_parameters():
            if not p_net.requires_grad:
                continue
            if is_fsdp and hasattr(p_net, "full_tensor"):
                # Gather the full tensor from all ranks (FSDP DTensor). full_tensor()
                # all_gathers, which requires a CUDA backend, so move CPU shards first.
                full_tensor = p_net.to("cuda").full_tensor() if p_net.device.type == "cpu" else p_net.full_tensor()
            else:
                full_tensor = p_net
            key = name.replace("_checkpoint_wrapped_module.", "")
            if key in ema_state_dict:
                ema_param = ema_state_dict[key]
                if initialize:
                    ema_param.copy_(full_tensor.to(device=ema_param.device, dtype=ema_param.dtype))
                else:
                    ema_param.lerp_(full_tensor.to(device=ema_param.device, dtype=ema_param.dtype), 1.0 - beta)
            elif initialize:
                logger.warning(f"EMA parameter {key} not found in EMA state dict, skipping update.")

        # FSDP2 does not shard buffers, so a plain copy is correct.
        for name, b_net in self.net.named_buffers():
            key = name.replace("_checkpoint_wrapped_module.", "")
            if key in ema_state_dict:
                ema_param = ema_state_dict[key]
                ema_param.copy_(b_net.to(device=ema_param.device, dtype=ema_param.dtype))
            elif initialize:
                logger.warning(f"EMA buffer {key} not found in EMA state dict, skipping update.")

    def checkpoint_policy(self) -> CheckpointPolicy:
        """Explicit per-module policy for LoRA-only checkpoint save/load.

        The FSDP checkpointer consumes this instead of inferring behavior from
        module-key strings. The default covers the base ``model_dict``: the
        student ``net`` saves its trainable subset (LoRA), and each EMA student
        mirrors the student's trainable names. Subclasses with extra modules
        (e.g. SMD's pivot / EMA pivot) extend this.
        """
        policy: CheckpointPolicy = {"net": ModuleCheckpointPolicy(CheckpointSaveMode.TRAINABLE_SUBSET)}
        for name in self.use_ema:
            policy[name] = ModuleCheckpointPolicy(CheckpointSaveMode.EMA_TRAINABLE_MIRROR, mirror_source="net")
        return policy

    @property
    def net_inference(self):
        """Return the network to use for inference.

        Uses EMA network when available and not using FSDP.
        TODO: When FSDP is enabled, EMA networks are not wrapped and have dtype issues,
        so we fall back to the main network which is properly FSDP-wrapped.
        Standalone inference uses the selected exported weights through ``infer.py``.
        """
        use_ema_for_inference = self.use_ema and not getattr(self, "_is_fsdp", False)
        return getattr(self, self.use_ema[0]) if use_ema_for_inference else self.net

    @property
    def fsdp_dict(self):
        """Return dict containing all networks to be sharded.

        By default, this is the same as the model dict.
        If the model has a teacher and add_teacher_to_fsdp_dict is True, the teacher is added to the dict.
        """
        model_dict = self.model_dict
        if getattr(self, "teacher", None) is not None and self.config.add_teacher_to_fsdp_dict:
            model_dict["teacher"] = self.teacher
        return model_dict

    @property
    def fsdp_no_sync_keys(self) -> set | None:
        """Keys in fsdp_dict that should skip broadcast (all ranks load independently).

        Default: None (all keys are broadcast from rank 0).
        Subclasses override to skip broadcast for modules where all ranks
        already have identical weights (e.g., teacher loaded from pretrained).
        """
        return None

    @property
    def model_dict(self):
        """Return the model dict containing the student and EMA networks"""
        return torch.nn.ModuleDict({"net": self.net, **self.ema_dict})

    @property
    def optimizer_dict(self):
        """Return a dict containing all the optimizers"""
        return {
            "net": self.net_optimizer,
        }

    @property
    def scheduler_dict(self):
        """Return a dict containing all the lr schedulers"""
        return {
            "net": self.net_lr_scheduler,
        }
