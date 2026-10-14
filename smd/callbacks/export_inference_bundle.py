# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save sampler metadata and complete native student weights alongside checkpoints.

Full fine-tuning writes ``weights.safetensors`` and ``bundle.json``. LoRA
training writes metadata referring to the trainable checkpoint; export and
fusion are performed separately with ``tools/export_lora.py`` and
``tools/weights.py``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import torch

from smd.callbacks.callback import Callback
from smd.utils.distributed import get_rank, synchronize
from smd.utils.state_dict_export import compute_inference_state_dict
import smd.utils.logging_utils as logger

if TYPE_CHECKING:
    from smd.methods import SMDModel


SCHEMA_VERSION = "1.0"
CAUSAL_WAN_SAMPLER_TYPE = "causal_wan_chunked_student"
CAUSAL_WAN_INFERENCE_PROTOCOL_VERSION = 1


class ExportInferenceBundle(Callback):
    """Write an inference-ready weights + metadata bundle next to each checkpoint.

    Args:
        suffix: directory suffix appended to the checkpoint path. The bundle
            will be written to ``<checkpoint_path><suffix>/``. Default
            ``".bundle"`` produces e.g. ``checkpoints/0000050.bundle/``.
        weights_filename: name of the safetensors file inside the bundle
            directory.
        metadata_filename: name of the JSON metadata file inside the bundle
            directory.
        target_dtype: dtype for the exported weights. Defaults to bfloat16.
    """

    def __init__(
        self,
        suffix: str = ".bundle",
        weights_filename: str = "weights.safetensors",
        metadata_filename: str = "bundle.json",
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        self.suffix = suffix
        self.weights_filename = weights_filename
        self.metadata_filename = metadata_filename
        self.target_dtype = target_dtype

    # ------------------------------------------------------------------ hooks

    def on_save_checkpoint_success(
        self,
        model: "SMDModel",
        iteration: int = 0,
        path: str = None,
    ) -> None:
        if path is None:
            raise RuntimeError(
                "ExportInferenceBundle: trainer passed path=None to "
                "on_save_checkpoint_success; cannot derive bundle output dir."
            )

        bundle_dir = path + self.suffix

        if model.is_lora_mode:
            self._export_lora_metadata_only(model, iteration, path, bundle_dir)
        else:
            self._export_full_bundle(model, iteration, bundle_dir)

    def _export_lora_metadata_only(
        self,
        model: "SMDModel",
        iteration: int,
        checkpoint_path: str,
        bundle_dir: str,
    ) -> None:
        """LoRA mode: write only bundle.json with checkpoint reference (no merge)."""
        if get_rank() == 0:
            os.makedirs(bundle_dir, exist_ok=True)
            meta_path = os.path.join(bundle_dir, self.metadata_filename)

            metadata = self._build_metadata(model, iteration)
            metadata["lora_mode"] = True
            metadata["checkpoint_path"] = checkpoint_path

            self._write_metadata(metadata, meta_path)

            logger.info(
                f"ExportInferenceBundle (LoRA): wrote metadata-only bundle to {meta_path} "
                f"(merge deferred to tools/weights.py)"
            )
        synchronize()

    def _export_full_bundle(
        self,
        model: "SMDModel",
        iteration: int,
        bundle_dir: str,
    ) -> None:
        """Non-LoRA mode: merge weights and write safetensors + metadata."""
        # All ranks must participate in compute_inference_state_dict so that
        # ``.full_tensor()`` all_gathers complete. Rank 0 then does the
        # actual safetensors+json write.
        inference_module = model.net.get_inference_module()
        merged = compute_inference_state_dict(
            inference_module,
            target_dtype=self.target_dtype,
            skip_key_prefixes=model.net.bundle_skip_prefixes,
        )

        if get_rank() == 0:
            os.makedirs(bundle_dir, exist_ok=True)
            weights_path = os.path.join(bundle_dir, self.weights_filename)
            meta_path = os.path.join(bundle_dir, self.metadata_filename)

            self._write_weights(merged, weights_path)
            metadata = self._build_metadata(model, iteration)
            self._write_metadata(metadata, meta_path)

            logger.info(
                f"ExportInferenceBundle: wrote {len(merged)} tensors to {weights_path} and metadata to {meta_path}"
            )

        # Release the (rank-0-only) full tensors before we sync, otherwise
        # idle ranks have already returned and rank 0 holds peak CPU RAM
        # longer than necessary.
        del merged
        synchronize()

    # --------------------------------------------------------------- helpers

    def _write_weights(self, state_dict: dict[str, torch.Tensor], dest: str) -> None:
        """Write ``state_dict`` to ``dest`` via safetensors."""
        try:
            from safetensors.torch import save
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ExportInferenceBundle: safetensors is required. pip install safetensors") from e

        # safetensors requires contiguous tensors; LoRA-merged tensors already
        # are, but be defensive against unusual buffer strides.
        contiguous_sd = {k: (v.contiguous() if not v.is_contiguous() else v) for k, v in state_dict.items()}
        with open(dest, 'wb') as stream:
            stream.write(save(contiguous_sd, metadata={'format': 'pt'}))

    def _write_metadata(self, metadata: dict[str, Any], dest: str) -> None:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=False)
            f.write("\n")

    def _build_metadata(self, model: "SMDModel", iteration: int) -> dict[str, Any]:
        """Snapshot the non-trained state required to reproduce sampling.

        All architecture- and method-specific knowledge is obtained via the
        contract methods on ``model.net`` (SMDNetwork) and ``model``
        (SMDModel).
        """
        net = model.net

        # network family — required for the inference-script dispatcher
        family = getattr(net, "network_family", None)
        if family is None:
            raise RuntimeError(
                "ExportInferenceBundle: net is missing the 'network_family' "
                'class attribute. Add e.g. `network_family: str = "flux"` '
                "to the network class."
            )

        model_id = getattr(net, "model_id", None)
        if model_id is None:
            raise RuntimeError(
                "ExportInferenceBundle: net has no .model_id; cannot point the inference script at a HuggingFace repo."
            )

        # method name — from the model class
        method = getattr(model, "method_name", None)
        if method is None:
            raise RuntimeError(
                "ExportInferenceBundle: model is missing 'method_name' class attribute. "
                'Add a method_name string to the model class.'
            )

        sampler = model.get_sampler_snapshot()
        if str(family) == "wan" and hasattr(model.config, "causal_transition_type"):
            sampler = dict(sampler)
            sampler.update(
                {
                    "type": CAUSAL_WAN_SAMPLER_TYPE,
                    "protocol_version": CAUSAL_WAN_INFERENCE_PROTOCOL_VERSION,
                    "chunk_size": int(net.chunk_size),
                    "total_num_frames": int(net.total_num_frames),
                    "transition_type": str(model.config.causal_transition_type),
                    "renoise_mode": str(model.config.causal_renoise_mode),
                    "cache_noise": float(model.config.causal_cache_noise),
                    "use_wan_official_sinusoidal": bool(net._use_wan_official_sinusoidal),
                }
            )

        schedule = dict(net.get_noise_scheduler_snapshot())
        schedule["dynamic_shift"] = bool(model.config.dynamic_shift)
        if model.config.dynamic_shift:
            schedule.update({name: getattr(model.config, "dynamic_shift_" + name)
                             for name in ("base_shift", "max_shift", "base_seq_len", "max_seq_len", "empirical_steps")})
            schedule["time_shift_type"] = model.config.time_shift_type
        else:
            schedule["shift"] = float(model.config.sample_t_cfg.shift)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "method": str(method),
            "network_family": str(family),
            "iteration": int(iteration),
            "model_id": str(model_id),
            "precision": str(self.target_dtype).replace("torch.", ""),
            "noise_scheduler": schedule,
            "sampler": sampler,
        }
        student_init_checkpoint = getattr(model.config, "student_init_checkpoint", None)
        if student_init_checkpoint is not None:
            metadata["student_init_checkpoint"] = str(student_init_checkpoint)
        return metadata


def _maybe_scalar(value: Any) -> Any:
    """Coerce torch scalars / numpy scalars to plain Python numbers for JSON.
    Passes None and already-JSON-safe types through unchanged."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"_maybe_scalar: expected 0-d/1-elem tensor, got shape {tuple(value.shape)}")
        return value.item()
    if isinstance(value, (int, float, bool, str)):
        return value
    # numpy etc — best-effort
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


__all__ = ["ExportInferenceBundle"]
