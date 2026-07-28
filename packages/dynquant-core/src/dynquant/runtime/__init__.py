"""Execution of quantized weights: backend selection and the packed Linear.

Split from :mod:`dynquant.quant`, which is about *producing* a checkpoint. This
package is about running one, and it is the only part of core that cares whether a
compiled extension is present.
"""

from __future__ import annotations

from .backend import (
    Backend,
    BackendStatus,
    available_backends,
    backend_report,
    resolve_backend,
)
from .linear import (
    DynQuantEmbedding,
    DynQuantLinear,
    PackReport,
    pack_model,
    packed_bytes,
)
from .ops import (
    active_backend,
    dequantize,
    embedding_lookup,
    gemv_max_rows,
    quantized_matmul,
    uses_compiled_kernels,
)

__all__ = [
    "Backend",
    "BackendStatus",
    "DynQuantEmbedding",
    "DynQuantLinear",
    "PackReport",
    "active_backend",
    "available_backends",
    "backend_report",
    "dequantize",
    "embedding_lookup",
    "gemv_max_rows",
    "pack_model",
    "packed_bytes",
    "quantized_matmul",
    "resolve_backend",
    "uses_compiled_kernels",
]
