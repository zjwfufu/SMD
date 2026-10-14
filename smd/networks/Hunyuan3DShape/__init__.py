# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored, self-contained Hunyuan3D 2.1 Shape model (vec2set representation).

Nothing here imports the original ``hy3dshape`` package — all architecture is vendored
under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE (see per-file headers).
"""

from .network import Hunyuan3DShape

__all__ = ["Hunyuan3DShape"]
