# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
#
# See licenses/detectron2/LICENSE for more details.
# -----------------------------------------------------------------------------

from collections import abc

from omegaconf import DictConfig, ListConfig
from dataclasses import is_dataclass
import torch
import contextlib

from smd.utils.registry import locate, _convert_target_to_string
import smd.utils.logging_utils as logger
from smd.utils import global_vars


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expands the input tensor `x` to have the same
    number of dimensions as the `target` tensor.

    Pads `x` with singleton dimensions on the end.

    # Example

    ```
    x = torch.ones(5)
    target = torch.ones(5, 10, 30, 1, 10)

    x = expand_like(x, target)
    print(x.shape) # <- [5, 1, 1, 1, 1]
    ```

    Args:
        x (torch.Tensor): The input tensor to expand.
        target (torch.Tensor): The target tensor whose shape length
            will be matched.

    Returns:
        torch.Tensor: The expanded tensor `x` with trailing singleton
            dimensions.
    """
    x = torch.atleast_1d(x)
    while len(x.shape) < len(target.shape):
        x = x[..., None]
    return x


def instantiate(cfg, *args, **kwargs):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.
    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments
    Returns:
        object instantiated by cfg
    """

    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            additional_kwargs = {}
            additional_kwargs.update(cfg)
            additional_kwargs.update(kwargs)
            return cls(*args, **additional_kwargs)
        except TypeError:
            logger.error(f"Error when instantiating {cls_name}!")
            raise
    return cfg  # return as-is if don't know what to do


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.

    Examples:
    ::
        from smd.utils import instantiate, LazyCall

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        if is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = _convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        return DictConfig(content=kwargs, flags={"allow_objects": True})


def set_global_vars(config: dict | DictConfig | None = None):
    config = config or {}
    # update all keys that exist in global_vars
    update_config = {k: v for k, v in config.items() if k in global_vars.__all__}
    logger.debug(f"Setting global variables {update_config}")
    global_vars.__dict__.update(update_config)
    # log keys that do not exist in global_vars
    ignore_keys = [k for k in config.keys() if k not in global_vars.__all__]
    if len(ignore_keys) > 0:
        logger.warning(f"Ignoring keys {ignore_keys} since they are not found in global_vars.")


@contextlib.contextmanager
def set_temp_global_vars(config):
    # Handle string "None" from command line parsing
    if config == "None":
        config = None
    original_global_vars = {k: global_vars.__dict__[k] for k in global_vars.__all__}
    try:
        set_global_vars(config)
        yield
    finally:
        logger.debug(f"Resetting global variables to {original_global_vars}")
        global_vars.__dict__.update(original_global_vars)
