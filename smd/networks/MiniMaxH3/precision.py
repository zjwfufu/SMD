# SPDX-License-Identifier: Apache-2.0
"""Move H3 modules without rounding native FP32 parameters through BF16."""
from __future__ import annotations

import torch


def to_with_fp32_children(module, parent, names, *args, **kwargs):
    device, _dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    protected = {name: parent._modules[name] for name in names if parent._modules.get(name) is not None}
    try:
        for name in protected:
            parent._modules[name] = None
        torch.nn.Module.to(module, *args, **kwargs)
    finally:
        parent._modules.update(protected)
    for child in protected.values():
        options = dict(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if memory_format is not None:
            options["memory_format"] = memory_format
        child.to(**options)
    return module
