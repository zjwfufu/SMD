# SPDX-License-Identifier: Apache-2.0

"""Deep-Pivot backend for MiniMax H3's packed joint audio/video transformer."""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from smd.networks.MiniMaxH3.network import MiniMaxH3Latents
from smd.networks.MiniMaxH3 import inverse_patchify_video_latents, unpack_stereo_audio_rows
from smd.networks.MiniMaxH3.context_parallel import gather_sequence

from ..common import PivotBackend


@dataclass(frozen=True)
class MiniMaxH3PivotGrid:
    video: torch.Tensor
    audio: torch.Tensor


class MiniMaxH3Pivot(nn.Module):
    """Copied H3 tail with expanded video and audio output heads."""

    _smd_custom_fully_shard = True

    def __init__(
        self,
        transformer: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
    ) -> None:
        super().__init__()
        blocks = list(transformer.transformer_blocks)
        self.split_position = split_k if split_k >= 0 else len(blocks) + split_k
        if not 0 <= self.split_position < len(blocks):
            raise IndexError(f"MiniMax H3 split {split_k} is outside {len(blocks)} transformer blocks.")
        if proj_out_init_mode not in {"copy", "zero"}:
            raise ValueError("MiniMax H3 Pivot supports proj_out_init_mode='copy' or 'zero'.")

        self.n_grid = int(n_grid)
        self.blocks = nn.ModuleList(copy.deepcopy(blocks[self.split_position + 1 :]))
        self.norm_out = copy.deepcopy(transformer.norm_out)
        self.video_proj_out = self._expanded_head(transformer.proj_out, proj_out_init_mode)
        self.audio_proj_out = self._expanded_head(transformer.audio_proj_out, proj_out_init_mode)
        self._video_out_features = int(transformer.proj_out.out_features)
        self._audio_out_features = int(transformer.audio_proj_out.out_features)
        self._fsdp_checkpointing = False
        self.context_parallel_state = getattr(transformer, "context_parallel_state", None)

    def to(self, *args: Any, **kwargs: Any):
        from smd.networks.MiniMaxH3.precision import to_with_fp32_children

        return to_with_fp32_children(self, self, ("video_proj_out", "audio_proj_out"), *args, **kwargs)

    def _expanded_head(self, original: nn.Linear, init_mode: str) -> nn.Linear:
        head = nn.Linear(
            original.in_features,
            original.out_features * self.n_grid,
            bias=original.bias is not None,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        with torch.no_grad():
            if init_mode == "copy":
                head.weight.copy_(original.weight.repeat(self.n_grid, 1))
                if original.bias is not None:
                    head.bias.copy_(original.bias.repeat(self.n_grid))
            else:
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()
        return head

    def forward(self, hidden_states: torch.Tensor, hook_data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        temb = hook_data["temb"]
        adaln_indices = hook_data["adaln_indices"]
        rotary_emb = hook_data["rotary_emb"]
        attention_mask = hook_data.get("attention_mask")
        for block in self.blocks:
            use_cp_checkpoint = self.context_parallel_state is not None and self.context_parallel_state.enabled
            if self.training and torch.is_grad_enabled() and not self._fsdp_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    attention_mask,
                    use_reentrant=use_cp_checkpoint,
                )
            else:
                if use_cp_checkpoint:
                    block.gradient_checkpointing = True
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)

        hidden_states = self.norm_out(hidden_states, temb, hook_data["timestep_indices"])
        hidden_states = hidden_states.to(self.video_proj_out.weight.dtype)
        video = self.video_proj_out(hidden_states)
        audio = self.audio_proj_out(hidden_states)
        if self.context_parallel_state is not None and self.context_parallel_state.enabled:
            autograd = torch.is_grad_enabled() and hidden_states.requires_grad
            video = gather_sequence(video, self.context_parallel_state, autograd=autograd)
            audio = gather_sequence(audio, self.context_parallel_state, autograd=autograd)
        video = video.index_select(1, hook_data["video_indices"])
        audio = audio.index_select(1, hook_data["audio_indices"])
        return video, audio

    def fully_shard(self, **kwargs: Any) -> None:
        from torch.distributed._composable import checkpoint
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        self._fsdp_checkpointing = True
        self.video_proj_out.to(dtype=torch.float32)
        self.audio_proj_out.to(dtype=torch.float32)
        base_policy = kwargs.get("mp_policy")
        fp32_kwargs = dict(kwargs)
        fp32_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=getattr(base_policy, "reduce_dtype", None),
            output_dtype=None,
            cast_forward_inputs=False,
        )
        fully_shard(self.video_proj_out, **fp32_kwargs)
        fully_shard(self.audio_proj_out, **fp32_kwargs)
        for block in self.blocks:
            if self.context_parallel_state is None or not self.context_parallel_state.enabled:
                checkpoint(block)
            else:
                block.gradient_checkpointing = True
            fully_shard(block, **kwargs)
        if self.context_parallel_state is not None and self.context_parallel_state.enabled:
            fully_shard(self.norm_out, **kwargs)
        else:
            fully_shard(self, **kwargs)


class MiniMaxH3PivotBackend(PivotBackend):
    """Capture and replay the packed H3 transformer tail."""

    def get_transformer(self, net: nn.Module) -> nn.Module:
        return net.transformer

    def enumerate_blocks(self, net: nn.Module) -> list[nn.Module]:
        return list(self.get_transformer(net).transformer_blocks)

    def build_pivot(
        self,
        net: nn.Module,
        split_k: int,
        n_grid: int,
        proj_out_init_mode: str = "copy",
        proj_out_init_plan: list[str] | None = None,
    ) -> MiniMaxH3Pivot:
        if proj_out_init_plan is not None:
            raise ValueError("MiniMax H3 Pivot does not support per-slot output-head initialization plans.")
        return MiniMaxH3Pivot(self.get_transformer(net), split_k, n_grid, proj_out_init_mode)

    def register_hook(self, net: nn.Module, block: nn.Module, capture_store: dict[str, Any]) -> Any:
        transformer = self.get_transformer(net)
        original_forward = transformer.forward

        def wrapped(_self, *args, **kwargs):
            names = (
                "hidden_states",
                "audio_hidden_states",
                "encoder_hidden_states",
                "timestep",
                "timestep_indices",
                "token_tags",
                "position_ids",
                "video_indices",
                "audio_indices",
                "text_indices",
            )
            call = dict(zip(names, args))
            call.update(kwargs)
            for name in ("timestep_indices", "video_indices", "audio_indices"):
                capture_store[name] = call[name]
            condition_counts = getattr(net, "_active_condition_row_counts", (0, 0))
            capture_store["num_condition_video_rows"] = int(condition_counts[0])
            capture_store["num_condition_audio_rows"] = int(condition_counts[1])
            return original_forward(*args, **kwargs)

        def capture(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            def argument(name: str, position: int) -> Any:
                return kwargs[name] if name in kwargs else args[position]

            capture_store["hidden_states"] = output
            capture_store["temb"] = argument("temb", 1)
            adaln_indices = argument("adaln_indices", 2)
            capture_store["adaln_indices"] = adaln_indices
            capture_store["timestep_indices"] = torch.div(adaln_indices, 3, rounding_mode="floor")
            capture_store["rotary_emb"] = argument("rotary_emb", 3)
            capture_store["attention_mask"] = argument("attention_mask", 4)

        handle = block.register_forward_hook(capture, with_kwargs=True)
        transformer.forward = types.MethodType(wrapped, transformer)

        class CaptureHandle:
            def remove(self) -> None:
                handle.remove()
                transformer.forward = original_forward

        return CaptureHandle()

    def decode(self, pivot: MiniMaxH3Pivot, hook_data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        video, audio = pivot(hook_data["hidden_states"], hook_data)
        return (
            video[:, int(hook_data.get("num_condition_video_rows", 0)) :],
            audio[:, int(hook_data.get("num_condition_audio_rows", 0)) :],
        )

    def unpack_pair_grid(
        self,
        net: nn.Module,
        tokens: tuple[torch.Tensor, torch.Tensor],
        latents: MiniMaxH3Latents,
    ) -> MiniMaxH3PivotGrid:
        video, audio = tokens
        batch = latents.video.shape[0]
        audio_channels = latents.audio.shape[2]
        video_patch_dim = video.shape[-1] // self._grid_size(video, net.transformer.proj_out.out_features)
        n_grid = video.shape[-1] // video_patch_dim
        if audio.shape[-1] != n_grid * audio_channels:
            raise ValueError(
                f"MiniMax H3 expanded audio head {tuple(audio.shape)} is incompatible with n_grid={n_grid} "
                f"and latent channels={audio_channels}."
            )

        video_rows = video.reshape(batch, video.shape[1], n_grid, video_patch_dim).transpose(1, 2)
        video_rows = video_rows.reshape(batch * n_grid, video.shape[1], video_patch_dim)
        video_grid = inverse_patchify_video_latents(
            video_rows,
            patch_size=tuple(net.transformer.config.patch_size),
            num_frames=latents.video.shape[2],
            height=latents.video.shape[3],
            width=latents.video.shape[4],
        ).reshape(batch, n_grid, *latents.video.shape[1:])

        audio_rows = audio.reshape(batch, audio.shape[1], n_grid, audio_channels).transpose(1, 2)
        audio_rows = audio_rows.reshape(batch * n_grid, audio.shape[1], audio_channels)
        audio_grid = unpack_stereo_audio_rows(
            audio_rows,
            num_audio_latents=latents.audio.shape[3],
            audio_channels=latents.audio.shape[1],
        ).reshape(batch, n_grid, *latents.audio.shape[1:])
        return MiniMaxH3PivotGrid(video_grid, audio_grid)

    @staticmethod
    def _grid_size(tokens: torch.Tensor, base_out_features: int) -> int:
        if tokens.shape[-1] % base_out_features:
            raise ValueError(f"MiniMax H3 expanded output {tokens.shape[-1]} is not divisible by {base_out_features}.")
        return tokens.shape[-1] // base_out_features

    def unpack_to_spatial(self, net: nn.Module, tokens: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        del net, tokens, input_shape
        raise TypeError("MiniMax H3 has paired video/audio outputs; use unpack_pair_grid().")

    def rerun_output_layer(self, net: nn.Module, hook_data: dict[str, Any], split_k: int):
        transformer = self.get_transformer(net)
        position = split_k if split_k >= 0 else len(transformer.transformer_blocks) + split_k
        hidden_states = hook_data["hidden_states"]
        for block in transformer.transformer_blocks[position + 1 :]:
            hidden_states = block(
                hidden_states,
                hook_data["temb"],
                hook_data["adaln_indices"],
                hook_data["rotary_emb"],
                hook_data.get("attention_mask"),
            )
        hidden_states = transformer.norm_out(
            hidden_states,
            hook_data["temb"],
            hook_data["timestep_indices"],
        )
        context_parallel = getattr(transformer, "context_parallel_state", None)
        video = transformer.proj_out(hidden_states)
        audio = transformer.audio_proj_out(hidden_states)
        if context_parallel is not None and context_parallel.enabled:
            autograd = torch.is_grad_enabled() and hidden_states.requires_grad
            video = gather_sequence(video, context_parallel, autograd=autograd)
            audio = gather_sequence(audio, context_parallel, autograd=autograd)
        video = video.index_select(1, hook_data["video_indices"])
        audio = audio.index_select(1, hook_data["audio_indices"])
        return (
            video[:, int(hook_data.get("num_condition_video_rows", 0)) :],
            audio[:, int(hook_data.get("num_condition_audio_rows", 0)) :],
        )
