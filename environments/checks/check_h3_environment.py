# SPDX-License-Identifier: Apache-2.0
"""H3 import/API preflight; no weights, CUDA initialization or model forward."""
from __future__ import annotations

import inspect
from importlib.metadata import version
import json


def check_h3_environment():
    import torch
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
    from smd.networks.MiniMaxH3 import (
        AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel,
    )
    from smd.networks.MiniMaxH3.decoder import MiniMaxH3AVDecoder
    from smd.methods.simplified_distill.pivot.backends.minimax_h3 import MiniMaxH3PivotBackend

    required = {"mesh", "mp_policy", "offload_policy", "ignored_params"}
    missing = required - set(inspect.signature(fully_shard).parameters)
    if missing:
        raise RuntimeError(f"H3 FSDP2 requires fully_shard arguments: {sorted(missing)}")
    for cls in (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel):
        assert callable(cls.from_pretrained) and callable(cls.load_config)
    assert callable(MiniMaxH3AVDecoder.decode_joint)
    assert callable(MiniMaxH3PivotBackend.unpack_pair_grid)
    CPUOffloadPolicy(pin_memory=False)
    MixedPrecisionPolicy(param_dtype=torch.float32, cast_forward_inputs=False)
    return {"status": "imports_and_api_passed", "versions": {
        name: version(name) for name in ("torch", "diffusers", "transformers", "peft", "safetensors")},
        "model_weights_loaded": False, "model_forward_executed": False,
        "cuda_initialized": torch.cuda.is_initialized()}


if __name__ == "__main__":
    print(json.dumps(check_h3_environment(), indent=2))
