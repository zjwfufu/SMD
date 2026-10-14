"""TRELLIS.2 sparse tensor infrastructure.

Core data structures (SparseTensor, VarLenTensor) and sparse attention
for the SLatFlowModel backbone. Separated from the model code because
these are also used by dataloaders, callbacks, and training utilities.
"""

from . import config  # noqa: F401 — initializes CONV/ATTN from env

from .basic import SparseTensor, VarLenTensor, sparse_cat, sparse_unbind, varlen_cat
from .linear import SparseLinear
from .nonlinearity import SparseGELU, SparseSiLU, SparseReLU
from .norm import SparseGroupNorm, SparseLayerNorm, SparseGroupNorm32, SparseLayerNorm32
from .attention import sparse_scaled_dot_product_attention
from .attention_module import SparseMultiHeadAttention
from .rope import SparseRotaryPositionEmbedder
from .conv import SparseConv3d, SparseInverseConv3d
from .spatial import SparseDownsample, SparseUpsample, SparseSpatial2Channel, SparseChannel2Spatial

__all__ = [
    "SparseTensor",
    "VarLenTensor",
    "sparse_cat",
    "sparse_unbind",
    "varlen_cat",
    "SparseLinear",
    "SparseGELU",
    "SparseSiLU",
    "SparseReLU",
    "SparseGroupNorm",
    "SparseLayerNorm",
    "SparseGroupNorm32",
    "SparseLayerNorm32",
    "sparse_scaled_dot_product_attention",
    "SparseMultiHeadAttention",
    "SparseRotaryPositionEmbedder",
]
