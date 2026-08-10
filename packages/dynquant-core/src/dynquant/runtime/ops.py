"""The dispatch layer: one function per operation, three possible backends.

Every quantized computation in DynQuant goes through this module, and nothing
above it knows whether a compiled kernel ran. That boundary is what makes
``DYNQUANT_BACKEND=torch`` a meaningful comparison rather than a different code
path -- the modules in :mod:`dynquant.runtime.linear` call the same four functions
either way.

The dispatch rule that matters
------------------------------
:func:`quantized_matmul` chooses between two implementations by activation-row
count, and the choice is not a heuristic:

* **M <= gemv_max_rows** -- decode. Arithmetic intensity is under 1 FLOP/byte, so
  the matmul takes exactly as long as it takes to read the weight. The packed GEMV
  reads 3 bits per weight instead of 16, and nothing dense is ever materialised.
  This is the only path on which DynQuant's VRAM claim is true.
* **M above it** -- prefill. The weight read is amortised over hundreds of rows and
  the operation is compute-bound, so the fastest thing available is fp16 tensor
  cores: dequantize into a workspace and call cuBLAS. Peak memory rises by one
  weight for the duration of the call, which is the honest cost of the fast path
  and is why the threshold is not simply "always dequantize".

The threshold is read from the loaded binary rather than assumed, because it is a
property of how the kernel was compiled (see ``DYNQUANT_GEMV_MAX_ROWS``).
"""

from __future__ import annotations

import torch

from dynquant.constants import GEMV_MAX_ROWS
from dynquant.errors import BackendUnavailableError
from dynquant.quant.tensor import QuantLayout, QuantTensor

from .backend import Backend, resolve_backend

__all__ = [
    "active_backend",
    "dequantize",
    "embedding_lookup",
    "gemv_max_rows",
    "grouped_quantized_matmul",
    "has_grouped_gemv",
    "quantized_matmul",
    "uses_compiled_kernels",
    "warm_dispatch",
]

# Three process-wide constants, cached in module globals rather than behind
# `lru_cache`, and the difference is `torch.compile`.
#
# All are decided by probing: which shared object imported, what the driver
# says, what number the binary was compiled with. Neither can change during a
# run. But `quantized_matmul` reads them on every call, and every call happens
# inside the model's forward -- so under `torch.compile` they are read while
# dynamo is tracing. Dynamo does not honour `lru_cache`; it traces the body,
# reaches `importlib.util.find_spec` in the CUDA probe, and raises
# `Unsupported`. vLLM compiles with `fullgraph`, so that is a hard failure at
# the first linear layer rather than a slow path.
#
# A module global that already holds a value is a constant to dynamo: it reads
# it, guards on it, and traces neither branch. The probe therefore has to have
# run before compilation, which is what `warm_dispatch` is for -- layer
# construction calls it, and layers are built before anything is traced.
#
# `torch.compiler.disable` would also stop the tracing, by breaking the graph at
# every quantized linear. That trades a startup call for a per-layer graph
# break, which under `fullgraph` is still an error.
_ACTIVE_BACKEND: Backend | None = None
_GEMV_MAX_ROWS: int | None = None
_HAS_GROUPED_GEMV: bool | None = None


def warm_dispatch() -> Backend:
    """Resolve the backend now, outside any compiled region. Returns it.

    Call from anywhere a quantized layer is constructed. Idempotent, and cheap
    after the first call -- it exists to control *when* the probe runs, not
    whether.

    Raises:
        BackendUnavailableError: If ``$DYNQUANT_BACKEND`` names a backend that is
            not usable. Deliberately at layer-construction time rather than at
            import: a process that merely imports DynQuant has not asked for a
            kernel yet, and failing there would make the package unimportable on
            a machine whose environment out-lived its wheel.
    """
    backend = active_backend()
    gemv_max_rows()
    # Same reason, one step further: the grouped path's probe touches the op table,
    # which dynamo cannot trace either, and an expert bank is constructed on the same
    # load pass as a linear -- so the answer is settled before the first forward.
    has_grouped_gemv()
    return backend


def active_backend() -> Backend:
    """The backend for this process. Resolved once; honours ``$DYNQUANT_BACKEND``.

    Cached because resolution imports a shared object and queries the driver, and
    because a backend that changed mid-run would make a benchmark meaningless.
    """
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = resolve_backend()
    return _ACTIVE_BACKEND


def uses_compiled_kernels(tensor_or_device: QuantTensor | torch.device | torch.Tensor) -> bool:
    """Whether the compiled path will actually be taken for this tensor.

    Two conditions, and both are needed. The CUDA backend being *available* says
    the wheel loaded; it says nothing about where a particular tensor lives. A
    quantized model held on CPU -- during ``from_pretrained``, or in a test -- must
    take the reference path even on a machine with working kernels.
    """
    if active_backend() is not Backend.CUDA:
        return False
    device = (
        tensor_or_device.device
        if isinstance(tensor_or_device, QuantTensor | torch.Tensor)
        else torch.device(tensor_or_device)
    )
    return device.type == "cuda"


def gemv_max_rows() -> int:
    """Largest M the packed GEMV accepts, asked of the binary that will run it.

    Falls back to :data:`dynquant.constants.GEMV_MAX_ROWS` when there is no
    extension -- the value is then unused for dispatch anyway, since the torch
    backend has only one implementation.
    """
    global _GEMV_MAX_ROWS
    if _GEMV_MAX_ROWS is None:
        _GEMV_MAX_ROWS = _probe_gemv_max_rows()
    return _GEMV_MAX_ROWS


def _probe_gemv_max_rows() -> int:
    if active_backend() is not Backend.CUDA:
        return GEMV_MAX_ROWS
    try:
        return int(torch.ops.dynquant.gemv_max_rows())
    except (AttributeError, RuntimeError):
        # An ABI-checked wheel should always have the op. Falling back rather than
        # raising keeps a diagnostic session usable; the constant is the same
        # number, held to the header by tests/test_abi.py.
        return GEMV_MAX_ROWS


def _kernel_args(
    qt: QuantTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int, int]:
    """Positional arguments shared by ``dequant`` and ``gemv``.

    ``effective_group`` rather than ``group_size``: the -1 per-row sentinel is
    resolved here, on the host, so no kernel contains a branch on grouping mode.
    """
    if qt.layout is not QuantLayout.LINEAR:
        raise NotImplementedError(
            f"the compiled kernels read {QuantLayout.LINEAR.value} layout; this tensor is "
            f"{qt.layout.value}. Tensor-core-shuffled weights arrive with the fused GEMM."
        )
    geom = qt.geometry
    return (qt.packed, qt.scales, qt.offsets, qt.bits, geom.effective_group, qt.in_features)


def dequantize(qt: QuantTensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Reconstruct the dense tensor, on whichever backend is active.

    The CUDA path is one kernel over the packed buffer; the torch path is
    :meth:`QuantTensor.dequantize`, which is the oracle it is tested against. Both
    accumulate in fp32 and both round once, so the two agree to within a single
    fp16 ulp -- see ``tests/test_kernels_parity.py``.
    """
    if not uses_compiled_kernels(qt):
        return qt.dequantize(dtype=dtype)
    packed, scales, offsets, bits, group_values, in_features = _kernel_args(qt)
    # Annotated because `torch.ops` resolves at runtime and is `Any` to a type
    # checker, so without this every tensor that came out of a kernel would silently
    # opt the rest of the function out of checking.
    dense: torch.Tensor = torch.ops.dynquant.dequant(
        packed, scales, offsets, bits, group_values, in_features
    )
    dense = dense.reshape(qt.logical_shape)
    return dense if dtype is None else dense.to(dtype)


def quantized_matmul(
    x: torch.Tensor, qt: QuantTensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ W.T + bias`` where ``W`` is stored packed.

    ``x`` may have any number of leading dimensions; only the last must equal
    ``in_features``. It is flattened to 2-D for the kernel and restored after,
    which costs nothing -- both views are contiguous.
    """
    if x.shape[-1] != qt.in_features:
        raise ValueError(
            f"activation has {x.shape[-1]} input features but the weight expects {qt.in_features}"
        )
    if len(qt.logical_shape) != 2:
        raise ValueError(
            f"quantized_matmul expects a 2-D weight; this tensor is {qt.logical_shape}. "
            f"Conv and gated state-space weights have their own entry points."
        )

    lead = x.shape[:-1]
    flat = x.reshape(-1, qt.in_features)
    # Under autocast the activation arrives in the autocast dtype while the bias is
    # still whatever the module was built with, and `addmm` rejects the pair rather
    # than promoting. The weight already follows `x`; the bias has to as well.
    if bias is not None and bias.dtype != flat.dtype:
        bias = bias.to(flat.dtype)

    if not uses_compiled_kernels(qt):
        weight = qt.dequantize(dtype=x.dtype)
        out = torch.nn.functional.linear(flat, weight, bias)
        return out.reshape(*lead, qt.num_rows)

    packed, scales, offsets, bits, group_values, in_features = _kernel_args(qt)
    if flat.shape[0] <= gemv_max_rows():
        # Decode. The weight is never materialised.
        out = torch.ops.dynquant.gemv(
            flat.contiguous(), packed, scales, offsets, bits, group_values, in_features
        )
        if bias is not None:
            out = out + bias
    else:
        # Prefill. One dense weight lives for the duration of the call; at this M
        # tensor cores are worth more than the bytes.
        weight = torch.ops.dynquant.dequant(
            packed, scales, offsets, bits, group_values, in_features
        )
        out = torch.nn.functional.linear(flat, weight.to(x.dtype), bias)
    return out.reshape(*lead, qt.num_rows)


def has_grouped_gemv() -> bool:
    """Whether the loaded binary carries ``moe_grouped_gemv``.

    Asked of the op table, not of the ABI number. ``moe_grouped_gemv`` arrived in
    ABI 3 and ``MIN_KERNEL_ABI_VERSION`` was deliberately *not* raised to match: an
    ABI-2 wheel serves every model it served before, just with the Python expert
    loop. Refusing to load it would trade a working slower path for no path at all.
    So the runtime asks the question whose answer it can act on -- does this
    operation exist -- rather than the one that only correlates with it.
    """
    global _HAS_GROUPED_GEMV
    if _HAS_GROUPED_GEMV is None:
        _HAS_GROUPED_GEMV = _probe_grouped_gemv()
    return _HAS_GROUPED_GEMV


def _probe_grouped_gemv() -> bool:
    if active_backend() is not Backend.CUDA:
        return False
    try:
        return torch.ops.dynquant.moe_grouped_gemv is not None
    except (AttributeError, RuntimeError):
        return False


def grouped_quantized_matmul(
    x: torch.Tensor,
    qt: QuantTensor,
    seg_offsets: torch.Tensor,
    *,
    out_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x[s:e] @ bank[e].T`` for every expert band, in one launch.

    ``x`` is the expert-sorted activation matrix and ``seg_offsets`` is an ``[E + 1]``
    int32 tensor **on device** giving each expert's band. That the offsets stay on
    device is the entire point, and it is not a micro-optimisation: the Python loop
    this replaces has to know its bounds on the host, which costs a ``bincount`` ->
    ``.tolist()`` synchronization per bank per layer -- 44 per token on a 22-layer,
    two-bank model -- and makes ``torch.compile(fullgraph=True)`` and CUDA-graph
    capture impossible. The launch geometry here is derived from *shapes* alone, so
    an empty expert costs a two-int block launch instead of a sync to discover it is
    empty.

    The arithmetic is not the win. Each expert's GEMV is the same work either way,
    and this is deliberately not a tensor-core GEMM -- prefill still dequantizes and
    calls cuBLASLt, exactly as :func:`quantized_matmul` does above its row threshold.

    Rows past ``seg_offsets[-1]`` are the expert-parallel sentinels. They are left at
    zero because the output is allocated zeroed, which matches what the reference
    path writes and not what ``grouped_mm`` writes -- the grouped kernel leaves them
    uninitialised and relies on a later mask, and one NaN reaching the next
    projection survives a mask that only covers the last one.
    """
    if x.shape[-1] != qt.in_features:
        raise ValueError(
            f"activation has {x.shape[-1]} input features but the bank expects {qt.in_features}"
        )
    packed, scales, offsets, bits, group_values, in_features = _kernel_args(qt)
    out: torch.Tensor = torch.ops.dynquant.moe_grouped_gemv(
        x.contiguous(),
        packed,
        scales,
        offsets,
        seg_offsets,
        bits,
        group_values,
        in_features,
        out_features,
    )
    return out if bias is None else out + bias


def embedding_lookup(qt: QuantTensor, ids: torch.Tensor) -> torch.Tensor:
    """Gather rows of a packed embedding table without unpacking the table.

    ``index_select`` on the packed buffer is exact and cheap: a row's words are
    contiguous, so gathering packed rows is the same memory pattern as gathering
    dense ones, on a third of the bytes. Only the gathered rows are then
    dequantized -- for a 151k-row table and a 2k-token batch that is about 1% of
    the work of dequantizing the table, which is what lets a tied embedding stay
    packed in VRAM for the whole run instead of only on disk.

    No separate kernel: this is ``index_select`` composed with ``dequant``, and
    composing them is why neither needed to know about the other.
    """
    if len(qt.logical_shape) != 2:
        raise ValueError(f"embedding table must be 2-D, got {qt.logical_shape}")
    flat_ids = ids.reshape(-1)
    if not uses_compiled_kernels(qt):
        return qt.dequantize()[flat_ids].reshape(*ids.shape, qt.in_features)

    packed, scales, offsets, bits, group_values, in_features = _kernel_args(qt)
    rows = packed.index_select(0, flat_ids)
    meta_scales = scales.index_select(0, flat_ids)
    meta_offsets = None if offsets is None else offsets.index_select(0, flat_ids)
    dense: torch.Tensor = torch.ops.dynquant.dequant(
        rows, meta_scales, meta_offsets, bits, group_values, in_features
    )
    return dense.reshape(*ids.shape, in_features)


def require_compiled_kernels(what: str) -> None:
    """Raise unless the compiled backend is active.

    For call sites where falling back would invalidate the thing being measured --
    a VRAM benchmark on the torch backend reports fp16 usage and calls it
    quantized.
    """
    backend = active_backend()
    if backend is not Backend.CUDA:
        raise BackendUnavailableError(
            "cuda",
            f"{what} requires the compiled kernels, but the active backend is {backend.value}",
            remedy="pip install dynquant-kernels, then run `dynquant doctor`.",
        )
