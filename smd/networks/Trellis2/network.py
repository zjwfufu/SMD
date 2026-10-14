# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TrellisShape — SMDNetwork wrapper for TRELLIS.2 Shape DiT.

Wraps the TRELLIS.2 ``SLatFlowModel`` (1.3 B sparse-voxel DiT) so that it
conforms to the :class:`SMDNetwork` interface used by SMD.

Key design decisions:

* **SparseTensor in, SparseTensor out** — no padding to dense ``[B, N_max, C]``.
* Time rescaling follows the official TRELLIS.2 sampler: the noise scheduler
  applies the ``rescale_t`` Möbius warp to the Euler time grid, and
  ``forward()`` only converts the warped value to the backbone's ``×1000``
  timestep embedding range.
* CFG is applied per-sample over ``SparseTensor.feats``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from smd.networks.network import SMDNetwork
from smd.networks.io_contract import unpack_condition_container
from smd.utils.hf_utils import get_hf_search_dirs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SparseLoraLinear — LoRA for SparseLinear (peft-incompatible layers)
# ---------------------------------------------------------------------------


class SparseLoraLinear(nn.Module):
    """Drop-in replacement for SparseLinear with LoRA adapters.

    peft's ``LoraLinear`` cannot handle ``SparseTensor`` input because its
    forward path passes ``x`` directly to ``nn.functional.linear`` (which
    expects a plain Tensor).  This wrapper explicitly extracts ``.feats``,
    runs the LoRA branch on the plain Tensor, and wraps back.

    Parameters
    ----------
    base : SparseLinear
        The original SparseLinear layer (weight & bias are copied, then frozen).
    r : int
        LoRA rank.
    alpha : int
        LoRA alpha (scaling = alpha / r).
    """

    def __init__(self, base: nn.Module, r: int = 128, alpha: int = 128):
        super().__init__()
        in_features = base.in_features
        out_features = base.out_features

        # Freeze original weight/bias (shared with base, not duplicated)
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA low-rank adapters (trainable)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

        # Init: A ~ N(0, 1/sqrt(r)), B = 0 → initial LoRA output is zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """Forward pass for SparseTensor or VarLenTensor input."""
        from smd.networks.Trellis2.sparse import SparseTensor, VarLenTensor

        if isinstance(x, (VarLenTensor, SparseTensor)):
            feats = x.feats
            base_out = nn.functional.linear(feats, self.weight, self.bias)
            lora_out = self.lora_B(self.lora_A(feats)) * self.scaling
            return x.replace(base_out + lora_out)
        else:
            base_out = nn.functional.linear(x, self.weight, self.bias)
            lora_out = self.lora_B(self.lora_A(x)) * self.scaling
            return base_out + lora_out


def _replace_sparse_linear_with_lora(
    model: nn.Module,
    target_names: list[str],
    r: int = 128,
    alpha: int = 128,
) -> int:
    """Replace matching SparseLinear modules with SparseLoraLinear in-place.

    Parameters
    ----------
    model : nn.Module
        Root module to search.
    target_names : list[str]
        Substrings to match against fully-qualified module names.
    r, alpha : int
        LoRA rank and alpha.

    Returns
    -------
    int
        Number of modules replaced.
    """
    from smd.networks.Trellis2.sparse import SparseLinear

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, SparseLinear):
            continue
        if not any(t in name for t in target_names):
            continue
        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]
        # Replace with SparseLoraLinear
        lora_module = SparseLoraLinear(module, r=r, alpha=alpha)
        setattr(parent, attr_name, lora_module)
        replaced += 1

    return replaced


class Trellis2(SMDNetwork):
    """SMD wrapper around the TRELLIS.2 Shape DiT (``SLatFlowModel``).

    Parameters
    ----------
    model_config_path : str
        Path to the TRELLIS.2 JSON config (``*.json`` next to ``*.safetensors``).
    pretrained_path : str
        Path to the ``.safetensors`` weight file, **without** the extension.
        If ``model_config_path`` is given, ``pretrained_path`` defaults to
        the same stem (replace ``.json`` → ``.safetensors``).
    latent_channels : int
        Number of latent channels (32 for Shape).
    rescale_t : float
        Official sampler Möbius rescaling factor. Used as the noise-scheduler
        ``shift`` default when no explicit ``shift`` is provided.
    sigma_min : float
        ``σ_min`` used in the TRELLIS.2 flow-matching formulation.
    cond_channels : int
        Condition embedding dimension (1024 for DINOv3 features).
    net_pred_type : str
        Prediction type (``"flow"``).
    schedule_type : str
        Noise schedule type (``"trellis_flow"``).
    """

    network_family: str = "trellis2"

    # Default HF repo and checkpoint names
    MODEL_ID = "microsoft/TRELLIS.2-4B"
    SHAPE_CKPT = "slat_flow_img2shape_dit_1_3B_1024_bf16"
    TEXTURE_CKPT = "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        ckpt_name: str = SHAPE_CKPT,
        model_config_path: str = "",
        pretrained_path: str = "",
        latent_channels: int = 32,
        rescale_t: float = 3.0,
        sigma_min: float = 1e-5,
        cond_channels: int = 1024,
        net_pred_type: str = "flow",
        schedule_type: str = "trellis_flow",
        # LoRA configuration
        use_lora: bool = False,
        lora_rank: int = 128,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,
        **kwargs,
    ):
        # Pop noise-schedule kwargs before passing to super
        ns_kwargs = {
            "sigma_min": sigma_min,
            "shift": kwargs.pop("shift", rescale_t),
        }

        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            **ns_kwargs,
            **kwargs,
        )

        self.model_id = model_id
        self.ckpt_name = ckpt_name
        self.latent_channels = latent_channels
        self.rescale_t = float(rescale_t)
        self._sigma_min = float(sigma_min)
        self.cond_channels = cond_channels

        # LoRA configuration (adapter injection deferred to attach_lora())
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or []
        self._lora_attached = False

        # Resolve paths from model_id if not explicitly given
        if not model_config_path:
            model_config_path, pretrained_path = self._resolve_from_hf(model_id, ckpt_name)

        # Defer model loading when in meta-device context (FSDP)
        if self._is_in_meta_context():
            log.info("Trellis2: meta-device init — deferring model load")
            self.model = nn.Identity()  # placeholder
            return

        self.model = self._load_model(model_config_path, pretrained_path)

        # DINOv3 image condition encoder — loaded at init, stays on GPU.
        # Used by preprocess_data when dataloader provides raw images
        # instead of precomputed cond_dinov3.npy features.
        self._init_image_cond_model()

    @staticmethod
    def _resolve_from_hf(model_id: str, ckpt_name: str) -> tuple[str, str]:
        """Resolve config + weights paths from a HuggingFace repo ID.

        Searches ``$SMD_OUTPUT_ROOT/.cache/hub/``, ``$HF_HOME/hub/``,
        and ``~/.cache/huggingface/hub/`` for the downloaded snapshot.
        """
        # model_id "microsoft/TRELLIS.2-4B" → dir name "models--microsoft--TRELLIS.2-4B"
        dir_name = f"models--{model_id.replace('/', '--')}"

        # get_hf_search_dirs() already returns paths ending with /hub
        search_dirs = [os.path.join(d, dir_name) for d in get_hf_search_dirs()]

        for base in search_dirs:
            snapshots_dir = os.path.join(base, "snapshots")
            if not os.path.isdir(snapshots_dir):
                continue
            # Pick the first (usually only) snapshot
            snaps = sorted(os.listdir(snapshots_dir))
            if not snaps:
                continue
            ckpts_dir = os.path.join(snapshots_dir, snaps[0], "ckpts")
            config_path = os.path.join(ckpts_dir, f"{ckpt_name}.json")
            weights_path = os.path.join(ckpts_dir, f"{ckpt_name}.safetensors")
            if os.path.exists(config_path):
                log.info(f"Trellis2: resolved {model_id}/{ckpt_name} from {ckpts_dir}")
                return config_path, weights_path

        raise FileNotFoundError(
            f"Cannot find {model_id} checkpoint '{ckpt_name}' in any of: {search_dirs}. "
            f"Download with: huggingface-cli download {model_id}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(config_path: str, pretrained_path: str) -> nn.Module:
        """Load the TRELLIS.2 ``SLatFlowModel`` from config + safetensors."""

        from .model import SLatFlowModel

        if not config_path:
            raise ValueError("model_config_path is required for TrellisShape")

        with open(config_path) as f:
            cfg = json.load(f)

        model_args = cfg.get("args", cfg)
        model = SLatFlowModel(**model_args)

        # Load weights
        weights_path = pretrained_path
        if not weights_path:
            weights_path = config_path.replace(".json", "")

        safetensors_path = weights_path if weights_path.endswith(".safetensors") else weights_path + ".safetensors"
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
            load_info = model.load_state_dict(state_dict, strict=False)
            log.info(f"Trellis2 loaded weights from {safetensors_path}: {load_info}")
        else:
            log.warning(f"Trellis2: weight file not found at {safetensors_path}")

        return model

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def attach_lora(self) -> None:
        """Inject LoRA adapters into ``self.model`` (SLatFlowModel).

        Must be called exactly once, AFTER pivot construction (deepcopy)
        and BEFORE FSDP wrap. SparseLinear inherits nn.Linear, so peft's
        LoraConfig recognizes and wraps them automatically.
        """
        if not self.use_lora:
            return
        if self._lora_attached:
            raise RuntimeError("attach_lora() called twice on the same Trellis2 network")

        self.model = self.attach_lora_to_module(
            self.model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self._lora_attached = True

        if hasattr(self.model, "dtype"):
            self.model.to(self.model.dtype)
        n_sparse_lora = sum(1 for _, module in self.model.named_modules() if isinstance(module, SparseLoraLinear))
        log.info(
            f"Attached LoRA to Trellis2: r={self.lora_rank}, alpha={self.lora_alpha}, "
            f"peft_targets={self.effective_lora_target_modules}, sparse_lora_layers={n_sparse_lora}"
        )

    @property
    def effective_lora_target_modules(self) -> list[str]:
        return list(self.lora_target_modules or ["to_qkv", "to_q", "to_kv", "to_out"])

    def attach_lora_to_module(
        self,
        module: nn.Module,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: str | list[str] | None = None,
    ) -> nn.Module:
        """Inject TRELLIS sparse and dense LoRA adapters into ``module``."""
        if dropout != 0.0:
            raise ValueError("TRELLIS SparseTensor LoRA requires dropout=0.")

        from peft import LoraConfig, inject_adapter_in_model

        _replace_sparse_linear_with_lora(
            module,
            ["input_layer", "out_layer", "mlp.mlp.0", "mlp.mlp.2"],
            r=rank,
            alpha=alpha,
        )
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=self.effective_lora_target_modules if target_modules is None else target_modules,
            lora_dropout=0.0,  # SparseTensor incompatible with nn.Dropout
        )
        injected = inject_adapter_in_model(lora_cfg, module)
        if not any("lora_" in name for name, _ in injected.named_parameters()):
            raise RuntimeError("TRELLIS Pivot LoRA recipe matched no modules.")
        return injected

    # ------------------------------------------------------------------
    # Time rescaling
    # ------------------------------------------------------------------

    def _rescale_time(self, t: torch.Tensor) -> torch.Tensor:
        """Scale timestep to [0, 1000] for the backbone.

        The Möbius warp (official ``rescale_t=3.0``) is handled by the noise
        scheduler's ``warp_t`` (via ``shift=3.0``), so all ``t`` values
        arriving here are already warped. We only need ``×1000``.

        Previously this method applied the Möbius warp redundantly, causing
        a double-warp when shift was also set to 3.0, or incorrect Euler
        step sizes when shift was 1.0 (uniform steps instead of warped).
        """
        return t * 1000.0

    # ------------------------------------------------------------------
    # DINOv3 image conditioning
    # ------------------------------------------------------------------

    def _init_image_cond_model(self) -> None:
        """Load DINOv3 ViT-L feature extractor (303M params, ~0.6GB fp16)."""
        from .image_encoder import DinoV3FeatureExtractor

        self.image_cond_model = DinoV3FeatureExtractor("facebook/dinov3-vitl16-pretrain-lvd1689m", image_size=1024)
        self.image_cond_model.cuda()
        n_params = sum(p.numel() for p in self.image_cond_model.model.parameters())
        log.info(f"Trellis2: loaded DINOv3 image_cond_model ({n_params / 1e6:.0f}M params)")

    @torch.no_grad()
    def encode_image(self, images: list) -> list[torch.Tensor]:
        """Encode PIL images → list of DINOv3 feature tensors.

        Args:
            images: list of PIL.Image.Image

        Returns:
            list of Tensor [L, 1024], one per image.
        """
        features = self.image_cond_model(images)  # [B, L, 1024]
        return [features[i] for i in range(features.shape[0])]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t,
        t: torch.Tensor,
        condition=None,
        r=None,
        return_features_early: bool = False,
        feature_indices=None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ):
        """Run the TRELLIS.2 Shape DiT.

        Parameters
        ----------
        x_t : SparseTensor
            Noisy latent (``feats: [N_total, 32]``, ``coords: [N_total, 4]``).
        t : Tensor
            Warped timesteps ``[B]`` in ``[0, 1]`` space.
        condition : Tensor or list[Tensor]
            Condition features ``[B, L, 1024]`` (DINOv3 embeddings).

        Returns
        -------
        SparseTensor
            Model output in ``fwd_pred_type`` (default: ``"flow"``).
        """
        t_model = self._rescale_time(t)

        # Unpack an optional condition container: the DINOv3 features ride under
        # the primary key with ``concat_cond`` (texture path) alongside them.
        condition, _extra_inputs = unpack_condition_container(condition)
        if _extra_inputs:
            fwd_kwargs = {**_extra_inputs, **fwd_kwargs}

        # Elastic memory: dynamically adjust per-block checkpointing based on N_total
        controller = getattr(self.model, "_memory_controller", None)
        use_elastic = (
            controller is not None and torch.is_grad_enabled() and self.model.training and hasattr(x_t, "feats")
        )

        def _run_model(**kw):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**kw)

        model_kwargs = dict(x=x_t, t=t_model, cond=condition, **fwd_kwargs)

        if use_elastic:
            input_size = x_t.feats.shape[0]
            mem_ratio = controller.get_mem_ratio(input_size)
            with self.model.with_mem_ratio(mem_ratio) as exact_ratio:
                output = _run_model(**model_kwargs)
            controller.update_run_states(input_size, exact_ratio)
        else:
            output = _run_model(**model_kwargs)

        # pred type conversion if needed
        target = fwd_pred_type or self.net_pred_type
        if target != "flow":
            ns = self.noise_scheduler
            converted_feats = ns.convert_model_output(
                x_t.feats,
                output.feats,
                t,
                src_pred_type="flow",
                target_pred_type=target,
            )
            output = output.replace(converted_feats)

        return output

    # ------------------------------------------------------------------
    # Inference convenience
    # ------------------------------------------------------------------

    def get_inference_module(self) -> nn.Module:
        return self.model

    def fully_shard(self, **kwargs):
        """Fully shard the TRELLIS.2 SLatFlowModel for FSDP2.

        Requires ``register_sparse_pytree()`` to have been called beforehand
        so that FSDP2's ``tree_flatten(output)`` can extract ``.feats`` from
        SparseTensor outputs and register backward hooks on them.
        """
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

        # Apply FSDP-compatible activation checkpointing to each transformer block
        # BEFORE sharding (order matters for FSDP2).
        # Without this, all 28 blocks' activations (~0.12GB each for FFN) are kept
        # for backward, consuming ~33GB+ for a single 8K-voxel sample.
        block_cls = type(self.model.blocks[0])
        apply_activation_checkpointing(
            self.model,
            check_fn=lambda m: isinstance(m, block_cls),
        )
        log.info(
            f"Trellis2: applied activation checkpointing to {len(self.model.blocks)} blocks ({block_cls.__name__})"
        )

        for block in self.model.blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.model, **kwargs)
        log.info(f"Trellis2: FSDP2 sharded {len(self.model.blocks)} blocks + root")

    @property
    def bundle_skip_prefixes(self) -> tuple[str, ...]:
        return ()

    def get_noise_scheduler_snapshot(self) -> dict[str, Any]:
        return {
            "type": "trellis_flow",
            "sigma_min": self._sigma_min,
            "rescale_t": self.rescale_t,
            "shift": float(self.noise_scheduler.shift),
            "min_t": float(self.noise_scheduler.min_t),
            "max_t": float(self.noise_scheduler.max_t),
        }
