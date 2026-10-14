# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 runtime; load transformer and decoders only when requested."""
from importlib import import_module
from .layout import *  # noqa: F403

_COMPONENTS = {
    "MiniMaxH3Transformer3DModel": ".transformer",
    "MiniMaxH3TransformerOutput": ".transformer",
    "MiniMaxH3Scheduler": ".scheduler",
    "AutoencoderKLMiniMaxH3": ".autoencoder_kl_minimax_h3",
    "AutoencoderKLMiniMaxH3Audio": ".autoencoder_kl_minimax_h3_audio",
}


def __getattr__(name):
    if name not in _COMPONENTS:
        raise AttributeError(name)
    value = getattr(import_module(_COMPONENTS[name], __name__), name)
    globals()[name] = value
    return value
