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
from .experts import (
    DISPATCH_NAME,
    dynquant_experts_forward,
    register_experts_dispatch,
    use_dynquant_experts,
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
    warm_dispatch,
)

__all__ = [
    "DISPATCH_NAME",
    "Backend",
    "BackendStatus",
    "DynQuantEmbedding",
    "DynQuantLinear",
    "PackReport",
    "active_backend",
    "available_backends",
    "backend_report",
    "dequantize",
    "dynquant_experts_forward",
    "embedding_lookup",
    "gemv_max_rows",
    "pack_model",
    "packed_bytes",
    "quantized_matmul",
    "register_experts_dispatch",
    "resolve_backend",
    "use_dynquant_experts",
    "uses_compiled_kernels",
    "warm_dispatch",
]
