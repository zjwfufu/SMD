# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMD network wrapper for the vendored Hunyuan3D 2.1 Shape model.

This is the boundary between SMD and the vendored ``HunYuanDiTPlain``. The
time/sign convention adapter lives INSIDE ``dit.HunYuanDiTPlain.forward`` (see
the convention note in ``dit.py``), so this wrapper does **no** time reversal or
sign flip — it simply forwards SMD ODE-time ``t`` and the conditioning, and
(optionally) converts the returned ``flow`` to another prediction type.

Conditioning: Hunyuan uses frozen DINOv2-Large image features ``[B, 1370, 1024]``
fed to the DiT as ``contexts={"main": ...}``. The wrapper accepts ``condition``
as a bare tensor, a ``{"main": tensor}`` dict, or a length-B list of ``[1370,
1024]`` tensors (SMD's per-sample condition format), and normalizes to the
contexts dict.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import smd.utils.logging_utils as log
from smd.networks.network import SMDNetwork


class Hunyuan3DShape(SMDNetwork):
    """SMD wrapper around the Hunyuan3D 2.1 Shape DiT (``HunYuanDiTPlain``).

    Latent is a dense ``[B, 4096, 64]`` vec2set token tensor (NOT sparse). The
    noise schedule is standard rectified flow (``schedule_type="rf"``,
    ``shift=1.0``), matching Hunyuan's ``FlowMatchEulerDiscreteScheduler`` with
    no shift, mapped into SMD's convention by the DiT forward adapter.
    """

    network_family: str = "hunyuan3d_shape"

    MODEL_ID = "tencent/Hunyuan3D-2.1"
    DINO_ID = "facebook/dinov2-large"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        latent_tokens: int = 4096,
        latent_channels: int = 64,
        cond_channels: int = 1024,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        # LoRA configuration (adapter injection deferred to attach_lora())
        use_lora: bool = False,
        lora_rank: int = 128,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,
        # Whether to load the DINOv2 image encoder at init (for runtime cond encoding)
        load_image_encoder: bool = True,
        **kwargs,
    ):
        # Rectified-flow schedule kwargs (shift=1.0 => Hunyuan's no-shift schedule).
        ns_kwargs = {
            "shift": kwargs.pop("shift", 1.0),
            "min_t": kwargs.pop("min_t", 1e-4),
            "max_t": kwargs.pop("max_t", 0.999),
        }
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **ns_kwargs, **kwargs)

        self.model_id = model_id
        self.latent_tokens = latent_tokens
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels

        # LoRA configuration
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            # Attention (self-attn attn1 + cross-attn attn2, all 21 blocks)
            "to_q",
            "to_k",
            "to_v",
            "out_proj",
            # FFN — plain MLP blocks 0-14
            "fc1",
            "fc2",
            # FFN — MoE expert/shared_expert blocks 15-20 (8 experts + 1 shared per block)
            "net.0.proj",
            "net.2",
        ]
        self._lora_attached = False

        self._load_image_encoder = load_image_encoder
        self.image_cond_model: nn.Module | None = None

        # Defer model loading when in meta-device context (FSDP).
        if self._is_in_meta_context():
            log.info("Hunyuan3DShape: meta-device init — deferring model load")
            self.model = nn.Identity()  # placeholder
            return

        from .weight_loader import load_dit

        self.model = load_dit(model_id)
        if self._load_image_encoder:
            self._init_image_cond_model()

    # ------------------------------------------------------------------
    # Image conditioning (frozen DINOv2-Large)
    # ------------------------------------------------------------------
    def _init_image_cond_model(self) -> None:
        from .conditioner import SingleImageEncoder
        from .image_processor import ImageProcessorV2

        self.image_cond_model = SingleImageEncoder(version=self.DINO_ID, image_size=518, use_cls_token=True)
        self.image_processor = ImageProcessorV2(size=512, border_ratio=0.15)
        if torch.cuda.is_available():
            self.image_cond_model.cuda()
        n_params = sum(p.numel() for p in self.image_cond_model.parameters())
        log.info(f"Hunyuan3DShape: loaded DINOv2 image encoder ({n_params / 1e6:.0f}M params)")

    @torch.no_grad()
    def encode_image(self, images: list) -> list[torch.Tensor]:
        """Encode raw PIL images → list of DINOv2 feature tensors ``[1370, 1024]``.

        Each PIL image is preprocessed (recenter/pad/white-composite → [-1,1])
        then encoded by the frozen DINOv2-Large. Returns one tensor per image so
        the result slots straight into SMD's per-sample condition list.
        """
        assert self.image_cond_model is not None, "image encoder not loaded"
        device = next(self.image_cond_model.parameters()).device
        tensors = []
        for img in images:
            out = self.image_processor(img)  # {"image": [1,3,512,512] in [-1,1], "mask": ...}
            tensors.append(out["image"])
        batch = torch.cat(tensors, dim=0).to(device)
        feats = self.image_cond_model(batch)["main"]  # [B, 1370, 1024]
        return [feats[i] for i in range(feats.shape[0])]

    @torch.no_grad()
    def unconditional_embedding(self, batch_size: int) -> torch.Tensor:
        """Zeros conditioning ``[B, 1370, 1024]`` for classifier-free guidance."""
        assert self.image_cond_model is not None, "image encoder not loaded"
        return self.image_cond_model.unconditional_embedding(batch_size)["main"]

    # ------------------------------------------------------------------
    # Condition normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _to_contexts(condition: Any) -> dict:
        """Normalize ``condition`` to the Hunyuan ``{"main": [B, L, D]}`` dict.

        Accepts: a dict (already contexts), a bare ``[B, L, D]`` tensor, or a
        length-B list/tuple of ``[L, D]`` tensors (SMD per-sample format).
        """
        if isinstance(condition, dict):
            return condition
        if isinstance(condition, (list, tuple)):
            main = torch.stack(list(condition), dim=0)
        else:
            main = condition
        return {"main": main}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices=None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ):
        """Run the Hunyuan3D Shape DiT.

        x_t       : ``[B, 4096, 64]`` dense latent tokens.
        t         : ``[B]`` SMD ODE-time in ``[0, 1]`` (0=data, 1=noise). The
                    DiT forward converts this to embedding-time internally.
        condition : DINOv2 features (tensor / list / contexts dict).
        Returns   : model output in ``fwd_pred_type`` (default ``"flow"``). The
                    DiT already returns SMD ``flow``; no sign flip here.
        """
        contexts = self._to_contexts(condition)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(x_t, t, contexts, **fwd_kwargs)

        target = fwd_pred_type or self.net_pred_type
        if target != "flow":
            output = self.noise_scheduler.convert_model_output(
                x_t, output, t, src_pred_type="flow", target_pred_type=target
            )
        return output

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    def attach_lora(self) -> None:
        """Inject LoRA adapters into the DiT via peft.

        Targets attention projections (``to_q/to_k/to_v/out_proj`` in both
        self-attn ``attn1`` and cross-attn ``attn2``) AND FFN Linears:
        - Plain MLP blocks 0-14: ``fc1`` / ``fc2``
        - MoE blocks 15-20: ``net.0.proj`` / ``net.2`` in each expert + shared_expert

        The timestep/x embedders are intentionally left out. Call once, after
        pivot construction (deepcopy) and before FSDP wrap.
        """
        if not self.use_lora:
            return
        if self._lora_attached:
            raise RuntimeError("attach_lora() called twice on the same Hunyuan3DShape network")

        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
        )
        inject_adapter_in_model(lora_cfg, self.model)
        self._lora_attached = True
        if hasattr(self.model, "dtype"):
            self.model.to(self.model.dtype)
        log.info(
            f"Hunyuan3DShape: attached LoRA (r={self.lora_rank}, alpha={self.lora_alpha}, "
            f"targets={self.lora_target_modules})"
        )

    # ------------------------------------------------------------------
    # Inference / FSDP / bundle contract
    # ------------------------------------------------------------------
    def get_inference_module(self) -> nn.Module:
        return self.model

    def fully_shard(self, **kwargs):
        """Per-block FSDP2 sharding + activation checkpointing (mirrors TRELLIS)."""
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

        block_cls = type(self.model.blocks[0])
        apply_activation_checkpointing(self.model, check_fn=lambda m: isinstance(m, block_cls))
        log.info(
            f"Hunyuan3DShape: applied activation checkpointing to {len(self.model.blocks)} "
            f"blocks ({block_cls.__name__})"
        )
        for block in self.model.blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.model, **kwargs)
        log.info(f"Hunyuan3DShape: FSDP2 sharded {len(self.model.blocks)} blocks + root")

    @property
    def bundle_skip_prefixes(self) -> tuple[str, ...]:
        # The DINOv2 encoder + VAE are not part of the denoiser bundle.
        return ("image_cond_model", "image_processor")

    def get_noise_scheduler_snapshot(self) -> dict[str, Any]:
        return {
            "type": self.schedule_type,
            "shift": float(getattr(self.noise_scheduler, "shift", 1.0)),
        }


# ----------------------------------------------------------------------
# Sanity: build, forward, and verify the time/sign convention end-to-end.
#   python -m smd.networks.Hunyuan3DShape.network
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] Building tiny Hunyuan3DShape (no checkpoint)...")
    # Build a tiny model by hand to avoid the 7GB download in CI-style runs.
    from smd.networks.Hunyuan3DShape.dit import HunYuanDiTPlain

    net = Hunyuan3DShape(load_image_encoder=False, schedule_type="rf", shift=1.0)
    # Replace the (heavy) loaded DiT with a tiny one for a fast structural check,
    # unless HUNYUAN_FULL=1 is set (then keep the real 3B model).
    if os.environ.get("HUNYUAN_FULL", "0") != "1":
        net.model = HunYuanDiTPlain(
            input_size=16,
            in_channels=8,
            hidden_size=64,
            context_dim=32,
            depth=5,
            num_heads=4,
            qk_norm=True,
            qk_norm_type="rms",
            text_len=10,
            use_attention_pooling=False,
            use_pos_emb=False,
            qkv_bias=False,
            num_moe_layers=2,
            num_experts=4,
            moe_top_k=2,
        )
    net = net.to(device).eval()

    C = net.model.in_channels
    N = net.model.input_size
    L = net.model.text_len
    D = net.model.context_dim
    B = 2
    x = torch.randn(B, N, C, device=device)
    t = torch.rand(B, device=device)
    cond = torch.randn(B, L, D, device=device)

    print("[2] Forward (SMD convention) ...")
    with torch.no_grad():
        flow = net(x, t, cond)
    assert flow.shape == (B, N, C), flow.shape
    assert torch.isfinite(flow).all()
    print(f"    flow output {tuple(flow.shape)} finite OK")

    print("[3] Convention check: wrapper-flow == -(native u) and t_emb == 1 - t ...")
    with torch.no_grad():
        # Native path: feed embedding-time (1 - t) directly, expect native velocity u.
        u = net.model._forward_native(x, 1.0 - t, {"main": cond})
        # Public path returns SMD flow = -u.
        flow2 = net.model(x, t, {"main": cond})
    assert torch.allclose(flow2, -u, atol=1e-5), (flow2 - (-u)).abs().max()
    print("    flow == -u  ✓   (single-point conversion verified)")

    print("[4] x0 conversion via noise scheduler ...")
    with torch.no_grad():
        x0 = net(x, t, cond, fwd_pred_type="x0")
    assert x0.shape == (B, N, C) and torch.isfinite(x0).all()
    print(f"    x0 output {tuple(x0.shape)} finite OK")

    print("\nALL SANITY CHECKS PASSED")
