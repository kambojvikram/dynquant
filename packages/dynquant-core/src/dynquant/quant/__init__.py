"""Quantization: the numeric core and the on-disk representation.

* :mod:`dynquant.quant.pack` -- group-aligned uint32 bit packing. The layout
  contract every CUDA kernel is written against.
* :mod:`dynquant.quant.tensor` -- :class:`~dynquant.quant.tensor.QuantTensor`,
  a packed weight plus *every* piece of metadata needed to invert it. The
  research code stored no ``group_size`` and reverse-engineered it from tensor
  shapes at load time, which broke for padded and non-2-D weights.
* :mod:`dynquant.quant.grid` -- the per-group MSE-optimal clipping search.
* :mod:`dynquant.quant.quantizer` -- the driver that applies a bit map to a model.
"""

from __future__ import annotations

from .grid import CLIP_CANDIDATES, ClipSearchResult, quantize_with_search, search_clip_ratios
from .pack import (
    PER_ROW_GROUP_SIZE,
    RowGeometry,
    pack_nbit,
    row_geometry,
    unpack_nbit,
    words_per_group,
    words_per_row,
)
from .quantizer import LayerQuantResult, QuantizationReport, quantize_model
from .tensor import QuantLayout, QuantTensor

__all__ = [
    "CLIP_CANDIDATES",
    "PER_ROW_GROUP_SIZE",
    "ClipSearchResult",
    "LayerQuantResult",
    "QuantLayout",
    "QuantTensor",
    "QuantizationReport",
    "RowGeometry",
    "pack_nbit",
    "quantize_model",
    "quantize_with_search",
    "row_geometry",
    "search_clip_ratios",
    "unpack_nbit",
    "words_per_group",
    "words_per_row",
]
