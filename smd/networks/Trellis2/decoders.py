"""TRELLIS.2 model loading — self-contained.

Loads decoder models (FlexiDualGridVaeDecoder, SparseUnetVaeDecoder) from
pretrained checkpoints.  All model classes are vendored locally.
"""

import json
import logging

from safetensors.torch import load_file

from .sc_vaes import (
    FlexiDualGridVaeDecoder,
    FlexiDualGridVaeEncoder,
    SparseUnetVaeDecoder,
    SparseUnetVaeEncoder,
)
from .sparse_structure import SparseStructureDecoder, SparseStructureFlowModel

log = logging.getLogger(__name__)

_CLASS_MAP = {
    "FlexiDualGridVaeDecoder": FlexiDualGridVaeDecoder,
    "FlexiDualGridVaeEncoder": FlexiDualGridVaeEncoder,
    "SparseUnetVaeDecoder": SparseUnetVaeDecoder,
    "SparseUnetVaeEncoder": SparseUnetVaeEncoder,
    "SparseStructureDecoder": SparseStructureDecoder,
    "SparseStructureFlowModel": SparseStructureFlowModel,
}


def from_pretrained(path, **kwargs):
    """Load a TRELLIS.2 model from a pretrained checkpoint.

    Args:
        path: Path stem (without .json/.safetensors extension).

    Returns:
        nn.Module: The loaded model.
    """
    config_path = f"{path}.json"
    weights_path = f"{path}.safetensors"

    with open(config_path) as f:
        cfg = json.load(f)

    class_name = cfg["name"]
    model_args = cfg.get("args", {})

    if class_name not in _CLASS_MAP:
        raise ValueError(f"Unknown model class '{class_name}'. Available: {list(_CLASS_MAP.keys())}")

    model_cls = _CLASS_MAP[class_name]
    model = model_cls(**model_args, **kwargs)

    state_dict = load_file(weights_path)
    load_info = model.load_state_dict(state_dict, strict=False)
    log.info(f"[trellis2.models] Loaded {class_name} from {weights_path}: {load_info}")

    return model


from smd.utils import logging_utils as logger

def _load_decoder(decoder_path: str):
    """Load a TRELLIS.2 decoder model on-demand.

    Args:
        decoder_path: Path stem (without .json/.safetensors extension).
    """
    decoder = from_pretrained(decoder_path)
    # Keep decoder's original mixed dtype (mostly fp16, from_latent float32).
    # Input must be float32 — decoder's from_latent handles the fp16 transition.
    decoder = decoder.cuda().eval()
    logger.info(
        f"[TrellisSample] loaded decoder from {decoder_path} ({sum(p.numel() for p in decoder.parameters()) / 1e6:.0f}M params)"
    )
    return decoder
