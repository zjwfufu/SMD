# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import hashlib
from pathlib import Path
from typing import Any, Optional, List, Dict
import inspect
from copy import deepcopy

import attrs
import yaml
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

import importlib
import importlib.util
from dataclasses import fields as dataclass_fields
import attr
from dataclasses import is_dataclass
import smd.utils.logging_utils as logger


def import_config_from_python_file(config_file: str) -> Any:
    """
    Import a config from a python file.

    Args:
        config_file (str): The path to the python file.

    Returns:
        Any: The config object.
    """

    path = Path(config_file).expanduser().resolve()
    if path.suffix != ".py":
        raise ValueError(f"Config file must be a Python file with a .py extension. Received: {config_file}")

    if not path.is_file():
        raise FileNotFoundError(f"SMD config file ({config_file}) not found.")

    # Keep built-in package identities (and their relative imports) intact.
    source_root = Path(__file__).resolve().parents[2]
    if path.is_relative_to(source_root / "smd"):
        name = ".".join(path.relative_to(source_root).with_suffix("").parts)
        config = importlib.import_module(name)
    else:
        # External configs may share a basename. Give each package root its own
        # namespace so one experiment cannot accidentally import another's code.
        package_root = path.parent
        while (package_root / "__init__.py").is_file():
            package_root = package_root.parent
        namespace = "_smd_config_" + hashlib.sha256(str(package_root).encode()).hexdigest()[:16]
        if namespace not in sys.modules:
            import types

            package = types.ModuleType(namespace)
            package.__path__ = [str(package_root)]
            package.__package__ = namespace
            sys.modules[namespace] = package
        parent_parts = path.parent.relative_to(package_root).parts
        leaf = "_entry_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        name = ".".join((namespace, *parent_parts, leaf))
        parent = name.rpartition(".")[0]
        importlib.import_module(parent)
        spec = importlib.util.spec_from_file_location(name, path)
        config = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(name)
        sys.modules[name] = config
        try:
            spec.loader.exec_module(config)
        except BaseException:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
            raise

    if not callable(getattr(config, "create_config", None)):
        raise ValueError(f"SMD config file ({path}) must define create_config().")

    return config.create_config()


def config_from_dict(ref_instance: Any, kwargs: Any) -> Any:
    """
    Construct an instance of the same type as ref_instance using the provided dictionary or data or unstructured data

    Args:
        ref_instance: The reference instance to determine the type and fields when needed
        kwargs: A dictionary of keyword arguments to use for constructing the new instance or primitive data or unstructured data

    Returns:
        Any: A new instance of the same type as ref_instance constructed using the provided kwargs or the primitive data or unstructured data

    Raises:
        AssertionError: If the fields do not match or if extra keys are found.
        Exception: If there is an error constructing the new instance.
    """
    is_type = is_attrs_or_dataclass(ref_instance)
    if not is_type:
        return kwargs
    else:
        ref_fields = set(get_fields(ref_instance))
        assert isinstance(kwargs, dict) or isinstance(kwargs, DictConfig), "kwargs must be a dictionary or a DictConfig"
        keys = set(kwargs.keys())

        # ref_fields must equal to or include all keys
        extra_keys = keys - ref_fields
        assert ref_fields == keys or keys.issubset(ref_fields), (
            f"Fields mismatch: {ref_fields} != {keys}. Extra keys found: {extra_keys} \n \t when constructing {type(ref_instance)} with {keys}"
        )

        resolved_kwargs: Dict[str, Any] = {}
        for f in keys:
            resolved_kwargs[f] = config_from_dict(getattr(ref_instance, f), kwargs[f])
        try:
            new_instance = type(ref_instance)(**resolved_kwargs)
        except Exception as e:
            logger.error(f"Error when constructing {type(ref_instance)} with {resolved_kwargs}")
            logger.error(e)
            raise e
        return new_instance


def flatten_dict(nested_dict, parent_key="", sep=".", exclude_key="_target_"):
    """
    Flattens a nested dictionary by joining keys with a separator.

    Args:
        nested_dict (dict): The dictionary to flatten.
        parent_key (str, optional): The base key to prepend to flattened keys.
                                    Used in recursion. Defaults to ''.
        sep (str, optional): The separator to use between keys. Defaults to '.'.

    Returns:
        dict: A new, flattened dictionary.
    """
    items = []
    for key, value in nested_dict.items():
        if key == exclude_key:
            continue
        if value is None:
            continue

        # Create the new key by joining parent_key and key
        new_key = parent_key + sep + key if parent_key else key

        # If the value is a non-empty dictionary, recurse
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value)
        if isinstance(value, dict) and value:
            items.extend(flatten_dict(value, new_key, sep=sep, exclude_key=exclude_key).items())
        # If the value is not a dictionary, it's a leaf node
        else:
            items.append((new_key, value))

    return dict(items)


def override_config_with_opts(config: Any, opts: Optional[List[str]] = None) -> Any:
    """
    Override the config with the opts.

    Args:
        config (Any): The config object.
        opts (Dict[str, Any]): Dict for the overrides.

    Returns:
        Any: The config object.
    """

    # Convert Config object to a DictConfig object for Hydra

    if not isinstance(config, DictConfig):
        config_dict = attrs.asdict(config)
        config_dict = DictConfig(content=config_dict, flags={"allow_objects": True})
    else:
        config_dict = config

    # Use Hydra to handle overrides
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_dict)

    if opts is None:
        opts = []

    if len(opts) > 0 and opts[0] != "-":
        raise ValueError(f"opts must start with '-' to separate from other arguments. Got: {opts}")
    opts = opts[1:]
    with initialize(version_base=None):
        try:
            cfg = compose(config_name="config", overrides=opts)
        except Exception as e:
            raise ValueError(f"Failed to compose config with opts: {e}")

        OmegaConf.resolve(cfg)

    config = config_from_dict(config, cfg)

    return config


def override_config_with_yaml(config: Any, yaml_path: str) -> Any:
    """
    Override the config with the yaml file. **_target_ field is excluded**.
    """
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    yaml_dict = flatten_dict(yaml_dict)
    config_dict = flatten_dict(attrs.asdict(config))

    # Loose overriding: mismatched keys are dropped but loudly warned about
    # (silent drops historically caused doc/code drift to go unnoticed — e.g. pivot_n_grid).
    unknown_keys = sorted(set(yaml_dict.keys()) - set(config_dict.keys()))
    if unknown_keys:
        logger.warning(
            f"override_config_with_yaml({yaml_path!r}): dropping {len(unknown_keys)} unknown key(s) "
            f"not present in the target config schema: {unknown_keys}"
        )
    override_dict = {k: v for k, v in yaml_dict.items() if k in config_dict}

    opts = ["-"] + [f"{k}={v}" for k, v in override_dict.items()]
    return override_config_with_opts(config, opts=opts)


def is_attrs_or_dataclass(obj) -> bool:
    """
    Check if the object is an instance of an attrs class or a dataclass.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs class or a dataclass, False otherwise.
    """
    return is_dataclass(obj) or attr.has(type(obj))


def get_fields(obj):
    """
    Get the fields of an attrs class or a dataclass.

    Args:
        obj: The object to get fields from. Must be an instance of an attrs class or a dataclass.

    Returns:
        list: A list of field names.

    Raises:
        ValueError: If the object is neither an attrs class nor a dataclass.
    """
    if is_dataclass(obj):
        return [field.name for field in dataclass_fields(obj)]
    elif attr.has(type(obj)):
        return [field.name for field in attr.fields(type(obj))]
    else:
        raise ValueError("The object is neither an attrs class nor a dataclass.")


def serialize_config(
    config: Any,
    return_type: str = "dict",
    path: str | bytes | None = None,
    filename: str = "config.yaml",
    include_defaults: bool = False,
) -> Dict[str, Any] | str:
    """
    Serialize a config (BaseConfig or DictConfig) to various formats.

    Args:
        config: The config to serialize (BaseConfig attrs object or DictConfig).
        return_type: Output format - "dict" (plain dict), "yaml" (YAML string), or "file" (save to file).
        path: Directory path to save the file. Required if return_type is "file".
        filename: Name of the file to save. Only used if return_type is "file".
        include_defaults: If True, add default parameter values from _target_ classes.

    Returns:
        Dict[str, Any] if return_type is "dict"
        str (YAML) if return_type is "yaml" or "file"

    Raises:
        ValueError: If return_type is "file" but path is not provided.
    """
    if return_type == "file" and path is None:
        raise ValueError("path must be provided when return_type is 'file'")

    # Deep copy to avoid modifying original
    config = deepcopy(config)

    # Normalize to DictConfig with object support
    if not isinstance(config, DictConfig):
        config_dict = attrs.asdict(config)
        config_omegaconf = DictConfig(content=config_dict, flags={"allow_objects": True})
    else:
        config_omegaconf = config

    def is_serializable(item) -> bool:
        try:
            OmegaConf.to_yaml(item)
            return True
        except Exception:
            return False

    def get_default_params(cls_or_func):
        if callable(cls_or_func):
            signature = inspect.signature(cls_or_func)
        else:
            signature = inspect.signature(cls_or_func.__init__)
        params = signature.parameters
        return {name: param.default for name, param in params.items() if param.default is not inspect.Parameter.empty}

    def process_config(conf):
        if isinstance(conf, DictConfig):
            for key, value in conf.items():
                if isinstance(value, (DictConfig, ListConfig)):
                    # Optionally add default params from _target_ classes
                    if include_defaults:
                        try:
                            if "_target_" in value:
                                default_params = get_default_params(value["_target_"])
                                for default_key, default_v in default_params.items():
                                    if default_key not in value:
                                        value[default_key] = default_v
                        except Exception as e:
                            logger.error(f"Failed to add default argument values: {e}")
                    process_config(value)
                else:
                    if not is_serializable(value) and value is not None:
                        conf[key] = str(value)
        elif isinstance(conf, ListConfig):
            for i, item in enumerate(conf):
                if isinstance(item, (DictConfig, ListConfig)):
                    process_config(item)
                else:
                    if not is_serializable(item) and item is not None:
                        conf[i] = str(item)
        else:
            raise NotImplementedError("Input config must be a DictConfig or ListConfig.")
        return conf

    config_omegaconf = process_config(config_omegaconf)
    result_dict: Dict[str, Any] = OmegaConf.to_container(config_omegaconf, resolve=True)  # type: ignore

    if return_type == "dict":
        return result_dict

    # For yaml and file, convert to YAML string
    yaml_str = yaml.dump(result_dict, default_flow_style=False, sort_keys=True)

    if return_type == "file":
        os.makedirs(path, exist_ok=True)  # type: ignore
        with open(f"{path}/{filename}", "w") as f:
            f.write(yaml_str)
        logger.info(f"Config is saved at {path}/{filename}")

    return yaml_str
