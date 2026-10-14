# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Set, Union, Tuple, Any, Dict
from abc import ABC, abstractmethod
import re

import torch

from smd.networks.noise_schedule import get_noise_schedule, NET_PRED_TYPES
import smd.utils.logging_utils as logger


class SMDNetwork(ABC, torch.nn.Module):
    """
    Abstract base class for SMD neural network architectures.

    This class provides a common interface for all neural network models used in the SMD framework
    for fast generative modeling. It defines the core methods that must be implemented by specific
    network architectures like EDM, EDM2, SD15, CogVideoX, etc.

    The class handles noise scheduling, prediction types, and provides abstract methods for
    forward passes and sampling operations.

    Attributes:
        net_pred_type (str): The prediction type of the network ('x0', 'eps', 'v', 'flow').
        schedule_type (str): The noise schedule type used by the network.
        noise_scheduler: The noise scheduler instance for handling diffusion timesteps.

    Args:
        net_pred_type (str, optional): The prediction type of the network.
            Must be one of ['x0', 'eps', 'v', 'flow']. Defaults to "x0".
        schedule_type (str, optional): The noise schedule type to use.
            Common types include 'edm', 'rf', 'sd'. Defaults to "edm".
        **net_kwargs: Additional keyword arguments passed to the noise scheduler.
    """

    @staticmethod
    def _is_in_meta_context() -> bool:
        """Check if we're currently in a torch.device('meta') context.

        This allows networks to detect when they're being instantiated inside
        a meta device context manager (used for FSDP memory-efficient loading).
        When in meta context, networks should use from_config instead of from_pretrained
        to avoid loading weights that will be broadcast from rank 0.

        Returns:
            bool: True if currently in a meta device context, False otherwise.
        """
        test_param = torch.nn.Parameter(torch.empty(0))
        return test_param.device.type == "meta"

    def __init__(
        self,
        net_pred_type: str = "x0",
        schedule_type: str = "edm",
        **net_kwargs,
    ):
        super().__init__()

        # Default prediction type - can be overridden by subclasses
        self.net_pred_type = net_pred_type
        self.schedule_type = schedule_type
        self._validate_net_pred_type(self.net_pred_type)
        self.set_noise_schedule(**net_kwargs)

    def _validate_net_pred_type(self, pred_type: str) -> None:
        """Validate that the network prediction type is supported.

        Args:
            pred_type (str): The prediction type to validate.

        Raises:
            ValueError: If the prediction type is not supported.
        """
        if pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported net_pred_type '{pred_type}'. Supported types are: {NET_PRED_TYPES}")

    def set_noise_schedule(self, schedule_type: Optional[str] = None, **noise_schedule_kwargs) -> None:
        """Set up the noise scheduler for the network.

        Args:
            schedule_type (Optional[str]): Type of noise schedule to use.
                If None, uses the current self.schedule_type.
            **noise_schedule_kwargs: Additional arguments passed to the noise scheduler.
        """
        if schedule_type is not None:
            self.schedule_type = schedule_type
        self.noise_scheduler = get_noise_schedule(self.schedule_type, **noise_schedule_kwargs)

    def reset_parameters(self):
        """Reset the parameters of the network.

        Subclasses should override this method to reinitialize their specific parameters,
        and call super().reset_parameters() to handle common components like the noise scheduler.
        """
        # Reinitialize noise scheduler (its _sigmas tensor is not a registered buffer)
        if hasattr(self, "noise_scheduler") and self.noise_scheduler is not None:
            self.set_noise_schedule()
            logger.debug("Reinitialized noise scheduler")

    @property
    def effective_lora_target_modules(self) -> str | list[str]:
        """Return the architecture's resolved PEFT target-module recipe."""
        targets = getattr(self, "lora_target_modules", None)
        if not targets:
            raise NotImplementedError(
                f"Network {self.__class__.__name__} does not expose an effective LoRA target-module recipe."
            )
        return targets

    def resolve_lora_targets_for_module(
        self,
        module: torch.nn.Module,
        *,
        recipe_prefix: str = "",
    ) -> list[str]:
        """Resolve the architecture recipe to exact module names inside ``module``."""
        recipe = self.effective_lora_target_modules
        matches: list[str] = []
        for name, _submodule in module.named_modules():
            if not name:
                continue
            recipe_name = f"{recipe_prefix}{name}"
            if isinstance(recipe, str):
                matched = re.fullmatch(recipe, recipe_name) is not None
            else:
                matched = any(recipe_name == target or recipe_name.endswith(f".{target}") for target in recipe)
            if matched:
                matches.append(name)
        if not matches:
            raise RuntimeError(
                f"LoRA recipe {recipe!r} matched no modules in {type(module).__name__} with prefix {recipe_prefix!r}."
            )
        return matches

    def attach_lora_to_module(
        self,
        module: torch.nn.Module,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: str | list[str] | None = None,
    ) -> torch.nn.Module:
        """Inject this architecture's LoRA recipe into an arbitrary compatible module."""
        from peft import LoraConfig, inject_adapter_in_model

        targets = self.effective_lora_target_modules if target_modules is None else target_modules
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=targets,
            lora_dropout=dropout,
            bias="none",
        )
        try:
            injected = inject_adapter_in_model(config, module)
        except ValueError as exc:
            raise RuntimeError(f"Cannot inject LoRA recipe {targets!r} into {type(module).__name__}.") from exc
        if not any("lora_" in name for name, _ in injected.named_parameters()):
            raise RuntimeError(f"LoRA recipe {targets!r} matched no parameters in {type(module).__name__}.")
        return injected

    def fully_shard(self, **kwargs):
        """Fully shard the network.

        Subclasses should override this method to shard their specific components.
        The default implementation raises NotImplementedError because the base class
        inherits from ABC, which causes issues with FSDP2's __class__ assignment.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement the fully_shard method. "
            "Use external sharding methods or implement this method in the subclass."
        )

    # ------------------------------------------------------------------
    # Bundle export contract
    # ------------------------------------------------------------------
    # These methods are called by ExportInferenceBundle to produce a
    # self-contained inference bundle.  Every concrete network that may
    # be used with the bundle exporter should override them.

    def get_inference_module(self) -> torch.nn.Module:
        """Return the sub-module whose state dict should be exported.

        For most architectures this is a distinct child attribute
        (e.g. ``self.transformer`` for Flux, ``self.unet`` for SD15).
        For architectures where the network itself IS the backbone
        (e.g. DiT), return ``self``.

        The default raises ``NotImplementedError`` so that forgetting to
        implement it produces a clear error instead of silently exporting
        the wrong thing.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement get_inference_module(). "
            "Add it to support inference-bundle export."
        )

    @property
    def bundle_skip_prefixes(self) -> Tuple[str, ...]:
        """FQN prefixes to omit from the exported state dict.

        After wrapper-segment stripping, any parameter whose cleaned FQN
        starts with one of these prefixes is dropped.  Use this to
        exclude SMD-only heads (e.g. ``logvar_linear``) that are not
        part of the upstream HuggingFace checkpoint.

        The default returns an empty tuple (nothing is skipped).
        """
        return ()

    def get_noise_scheduler_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict describing the noise scheduler.

        The dict MUST contain a ``"type"`` key so the inference-side
        dispatcher knows which scheduler to reconstruct.  All values
        must be plain Python scalars / lists / dicts (no tensors).

        The default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement "
            "get_noise_scheduler_snapshot(). Add it to support inference-bundle export."
        )

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the trained network.

        This method performs the sampling/inference process to generate new data from noise.
        It should implement the specific sampling algorithm for the network architecture.

        Note: This method is optional. Networks that don't support direct sampling
        can leave this unimplemented and use external sampling methods instead.

        Args:
            noise (torch.Tensor): Initial noise tensor to start sampling from.
                Shape should match the expected input format of the network.
            condition (Any): Conditioning information for guided generation.
                Can be text embeddings, class labels, or other structured conditioning data.
                Can be a single tensor, list of tensors, or dictionary mapping condition names to tensors.
                Defaults to None for unconditional generation.
            neg_condition (Any): Negative conditioning for classifier-free guidance. Defaults to None.
            guidance_scale (Optional[float], optional): Strength of guidance for conditional generation. None means no guidance.
                Higher values increase adherence to conditioning. Defaults to 5.0.
            num_steps (int, optional): Number of denoising steps to perform during sampling.
                More steps generally lead to higher quality but slower generation. Defaults to 50.
            **kwargs: Additional keyword arguments specific to the sampling implementation.

        Returns:
            torch.Tensor: Generated samples with the same spatial dimensions as latents
                but potentially different channel dimensions based on the network's output.

        Raises:
            NotImplementedError: If the network doesn't provide its own sampling implementation.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement the sample method. "
            "Use external sampling methods or implement this method in the subclass."
        )

    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of diffusion model.

        This method performs a single forward pass through the network, predicting the denoised
        output or noise based on the current noisy input and timestep.

        Args:
            x_t (torch.Tensor): The diffused/noisy data sample at timestep t.
                Shape should be [batch_size, channels, ...spatial_dims].
            t (torch.Tensor): The current timestep(s). Can be a scalar or tensor
                with shape [batch_size].
            condition (Any, optional): Optional conditioning information such as text embeddings, class labels,
                or attention masks. Can be a single tensor, list of tensors, or dictionary
                mapping condition names to tensors. Format depends on the specific network implementation.
                Defaults to None.
            r (Optional[torch.Tensor], optional): Additional timestep parameter,
                primarily used by mean flow methods. Defaults to None.
            return_features_early (bool, optional): If True, returns intermediate features
                as soon as they are computed, without completing the full forward pass.
                Defaults to False.
            feature_indices (Optional[Set[int]], optional): Set of layer indices to extract
                features from. If non-empty, features will be returned along with the output.
                Defaults to None (no feature extraction).
            return_logvar (bool, optional): If True, returns the log-variance estimate
                along with the main output. Defaults to False.
            fwd_pred_type (Optional[str], optional): Override the network's prediction type
                for this forward pass. Must be in ['x0', 'eps', 'v', 'flow'].
                None uses the network's default net_pred_type. Defaults to None.
            **fwd_kwargs: Additional keyword arguments specific to the network implementation.

        Returns:
            Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
                - If return_features_early=True: List of intermediate features
                - If feature_indices is non-empty: List containing [output, features]
                - If return_logvar=True: Tuple containing (output, logvar)
                - Otherwise: Single output tensor with predictions

        Raises:
            ValueError: If fwd_pred_type is provided but not supported.
            NotImplementedError: Must be implemented by concrete subclasses.
        """


class CausalSMDNetwork(SMDNetwork):
    """
    Abstract base class for causal (autoregressive) generative networks.

    This class extends SMDNetwork to support causal generation patterns,
    where outputs are generated sequentially with dependency on previous outputs.
    It is designed for models that need to maintain temporal causality and can
    benefit from chunked processing for memory efficiency.

    Key Features:
    - Chunked processing for long sequences to manage memory usage
    - KV cache management for attention-based models
    - Support for causal attention patterns
    - Abstract interface for cache management operations

    Common use cases include:
    - Autoregressive video generation (like CausalWan)
    - Sequential text generation
    - Time-series modeling with causal dependencies
    - Any model requiring temporal causality constraints

    Args:
        chunk_size (int): Size of chunks for processing long sequences.
            Smaller values use less memory but may be slower due to increased
            overhead. Typical values range from 1-16 depending on available
            GPU memory and sequence length. Defaults to 3.
        **net_kwargs: Additional keyword arguments passed to SMDNetwork,
            including net_pred_type, schedule_type, etc.

    Abstract Methods:
        clear_caches(): Must be implemented by subclasses to clear all internal
            caches (KV caches, attention caches, etc.). Should be called after
            generation sequences or when memory usage becomes too high.
    """

    def __init__(
        self,
        net_pred_type: str = "x0",
        schedule_type: str = "edm",
        chunk_size: int = 3,
        total_num_frames: int = 21,
        **net_kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **net_kwargs)
        self.chunk_size = chunk_size
        self.total_num_frames = total_num_frames

    @abstractmethod
    def clear_caches(self):
        """
        Clear all internal caches used for causal generation.

        This method must be implemented by subclasses to clear model-specific
        caches such as:
        - Key-Value (KV) caches from attention layers
        - Intermediate activation caches
        - Positional embedding caches
        - Any other memory-intensive cached data

        Should be called:
        - After each complete generation sequence
        - When switching between different generation tasks
        - When memory usage becomes too high
        - Before starting a new causal generation session
        """
