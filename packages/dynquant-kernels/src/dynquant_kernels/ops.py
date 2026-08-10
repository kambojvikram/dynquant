"""Meta ("fake") implementations for the compiled operators.

Why these exist
---------------
A C++ operator registered only for CPU and CUDA is invisible to anything that
traces without executing: ``torch.compile``, ``torch.export``, meta-device model
construction, and FakeTensor-based memory planning. Without a meta kernel,
``torch.compile`` cannot infer the output shape of a quantized Linear, so it
inserts a graph break at every layer -- on a 40-layer model that is 40 breaks,
which costs more than the kernel saves.

A fake implementation computes only shapes, dtypes and devices. It must allocate
nothing and read no tensor data; ``FakeTensor`` has no data to read, and touching
it raises. The rule of thumb: if a line would need a value rather than a shape,
the schema needs to carry that value as an explicit argument instead.

Shape checks use ``torch._check`` rather than ``assert`` because the tracer
records them as guards, so a shape that violates one recompiles instead of
silently specialising on the first shape seen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

__all__: list[str] = []

# torch ships ``_check`` unannotated, so calling it directly makes every guard an
# untyped call in a strict-mode module. Binding it once through an annotated alias
# keeps the guards themselves type-checked.
_check: Callable[[bool, Callable[[], str]], None] = torch._check


@torch.library.register_fake("dynquant::probe_axpy")
def _probe_axpy_fake(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    _check(
        x.shape == y.shape,
        lambda: f"probe_axpy: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}",
    )
    _check(
        x.dtype == y.dtype,
        lambda: f"probe_axpy: dtype mismatch {x.dtype} vs {y.dtype}",
    )
    return torch.empty_like(x)


@torch.library.register_fake("dynquant::probe_reduce")
def _probe_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    # A 0-dim fp32 tensor regardless of input dtype: the reduction accumulates in
    # fp32 (see probe_cuda.cu), and the meta kernel has to agree or torch.compile
    # will generate a downstream cast that the real kernel does not need.
    return x.new_empty((), dtype=torch.float32)


@torch.library.register_fake("dynquant::probe_gemm")
def _probe_gemm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check(a.dim() == 2 and b.dim() == 2, lambda: "probe_gemm: expected 2-D tensors")
    _check(
        a.shape[1] == b.shape[0],
        lambda: f"probe_gemm: inner dims disagree {tuple(a.shape)} @ {tuple(b.shape)}",
    )
    return a.new_empty((a.shape[0], b.shape[1]))


# ---------------------------------------------------------------------------
# Quantized kernels
# ---------------------------------------------------------------------------
#
# The shape rules are short because the schema was designed so they could be:
# `in_features` and `group_values` arrive as integers instead of being recovered
# from `scales.numel() // rows`, so nothing here needs a tensor's contents. That is
# the same reason `geometry.h` gives for the C++ side -- one design decision paying
# off in two places.


@torch.library.register_fake("dynquant::dequant")
def _dequant_fake(
    packed: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor | None,
    bits: int,
    group_values: int,
    in_features: int,
) -> torch.Tensor:
    _check(packed.dim() == 2, lambda: f"dequant: packed must be 2-D, got {tuple(packed.shape)}")
    _check(scales.dim() == 2, lambda: f"dequant: scales must be 2-D, got {tuple(scales.shape)}")
    _check(
        packed.shape[0] == scales.shape[0],
        lambda: f"dequant: {packed.shape[0]} packed rows vs {scales.shape[0]} scale rows",
    )
    # Output dtype follows `scales`, not `packed`: packed is int32 bit patterns, and
    # the compute dtype of a quantized tensor is the dtype its scales are stored in.
    return scales.new_empty((packed.shape[0], in_features))


@torch.library.register_fake("dynquant::gemv")
def _gemv_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor | None,
    bits: int,
    group_values: int,
    in_features: int,
) -> torch.Tensor:
    _check(x.dim() == 2, lambda: f"gemv: x must be 2-D, got {tuple(x.shape)}")
    _check(packed.dim() == 2, lambda: f"gemv: packed must be 2-D, got {tuple(packed.shape)}")
    _check(
        x.shape[1] == in_features,
        lambda: f"gemv: x has {x.shape[1]} columns but in_features is {in_features}",
    )
    # No guard on `x.shape[0] <= GEMV_MAX_ROWS`. Under torch.compile that dimension
    # is usually dynamic, and asserting a bound on it would either specialise the
    # graph on the current batch or fail to trace at all. The runtime picks the
    # kernel before the call, so a traced graph only ever reaches this op with a
    # shape the kernel accepts.
    return x.new_empty((x.shape[0], packed.shape[0]))


@torch.library.register_fake("dynquant::moe_grouped_gemv")
def _moe_grouped_gemv_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor | None,
    seg_offsets: torch.Tensor,
    bits: int,
    group_values: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    # The shape rule reads `seg_offsets.shape`, never its values. That is not a
    # convenience -- it is the property the whole op is designed around. A grouped
    # MoE forward is traceable exactly to the extent that nothing downstream needs
    # to know where the router sent anything, and a meta kernel that had to read
    # the table would prove the op could not be traced.
    _check(x.dim() == 2, lambda: f"moe_grouped_gemv: x must be 2-D, got {tuple(x.shape)}")
    _check(
        x.shape[1] == in_features,
        lambda: f"moe_grouped_gemv: x has {x.shape[1]} columns but in_features is {in_features}",
    )
    _check(
        seg_offsets.dim() == 1,
        lambda: f"moe_grouped_gemv: seg_offsets must be 1-D, got {tuple(seg_offsets.shape)}",
    )
    _check(
        packed.shape[0] == (seg_offsets.shape[0] - 1) * out_features,
        lambda: (
            f"moe_grouped_gemv: a bank of {packed.shape[0]} rows is not "
            f"{seg_offsets.shape[0] - 1} experts of {out_features}"
        ),
    )
    return x.new_empty((x.shape[0], out_features))
