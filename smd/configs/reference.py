"""Load a named Python experiment for inference and environment checks."""

from importlib import import_module


def load_experiment(name):
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError("Invalid experiment name")
    module = import_module(f"smd.configs.experiments.{name}")
    return module.create_config()
