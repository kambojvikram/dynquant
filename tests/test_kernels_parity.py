"""The compiled kernels against the torch reference, on real geometries.

The oracle is :meth:`QuantTensor.dequantize`, which is deliberately the slow,
readable implementation in ``dynquant/quant/pack.py``. Everything here compares
against it rather than against a second fast path, because two fast paths that
agree can still both be wrong about the layout in the same way -- which is the
exact failure the research code shipped with.

What makes this test worth its runtime is the *choice of shapes*. A parity test on
`[256, 512]` at 4-bit passes on almost any wrong implementation, because every
size divides everything. The cases below are the ones where the layout has room to
be wrong: 3-bit (values straddle words), per-row grouping (the ragged tail), an
``in_features`` that is not a multiple of the group size (the zero pad), and a
single row (the tail-warp path in the GEMV).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dynquant.constants import PER_ROW_GROUP_SIZE
from dynquant.quant.grid import quantize_with_search
from dynquant.quant.tensor import QuantTensor

kernels = pytest.importorskip("dynquant_kernels")

pytestmark = pytest.mark.skipif(
    not kernels.is_available(),
    reason="compiled extension unavailable",
)

# (out_features, in_features, group_size) -- see the module docstring for why these.
#
# The GEMV has two kernels behind one op: a vectorized fast path for geometries
# whose groups tile ``in_features`` and whose words tile a 128-bit load, and a
# general path for everything else. Which one a case takes is a property of its
# geometry *and its bit width*, so the list is chosen to put both under test at
# every width rather than to name them:
#
#   (128, 512, 128)   vectorized at all four widths
#   (70, 2048, 128)   vectorized, and the only case where the chunk loop runs more
#                     than one iteration per lane -- plus a ragged 2-row tail warp
#   (64, 320, 128)    general: 320 values do not fill three 128-value groups
#   (16, 128, 32)     general at 2/3-bit (a 32-value group is 2 and 3 words, neither
#                     a whole 128-bit load), vectorized at 4/8-bit
#   the per-row pair  general: a row that rounds up to a whole word has a tail
GEOMETRIES = [
    (128, 512, 128),  # the ordinary case
    (70, 2048, 128),  # transformer-sized K; multi-iteration chunk loop, ragged tail
    (64, 320, 128),  # in_features not a multiple of the group: one padded group
    (33, 256, 128),  # rows not a multiple of the warp tiling
    (16, 128, 32),  # smallest legal group
    (48, 100, PER_ROW_GROUP_SIZE),  # per-row, ragged tail, in_features % 32 != 0
    (7, 4, PER_ROW_GROUP_SIZE),  # Mamba conv1d: fewer values than a 3-bit block
]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _quantized(
    out_features: int,
    in_features: int,
    group_size: int,
    bits: int,
    device: str,
    dtype: torch.dtype,
    *,
    symmetric: bool = False,
) -> tuple[QuantTensor, torch.Tensor]:
    torch.manual_seed(bits * 1000 + in_features)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    quantized, _ = quantize_with_search(
        dense,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        compute_dtype=dtype,
    )
    return quantized.to(device), dense


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize(("out_features", "in_features", "group_size"), GEOMETRIES)
def test_dequant_matches_the_reference(device, bits, out_features, in_features, group_size):
    """Bit-exact, not approximately equal.

    Both sides decode the same integers and apply the same ``fma`` in fp32, then
    round once to the storage dtype. There is no reordering and no accumulation, so
    a difference of even one ulp means the two disagree about a value's bits -- a
    tolerance here would hide exactly what the test is for.
    """
    quantized, _ = _quantized(out_features, in_features, group_size, bits, device, torch.float16)
    geom = quantized.geometry

    got = torch.ops.dynquant.dequant(
        quantized.packed,
        quantized.scales,
        quantized.offsets,
        bits,
        geom.effective_group,
        in_features,
    )
    expected = quantized.dequantize()
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dequant_handles_every_compute_dtype(device, dtype):
    """Exact at fp16 and bf16; one ulp at fp32.

    The kernel computes ``fmaf(code, scale, offset)`` -- one rounding -- while the
    torch reference multiplies and then adds, which is two. When the result is
    rounded down to fp16 or bf16 the difference is far below the storage step and
    both land on the same bits. At fp32 the storage *is* the accumulator, so the
    kernel's extra half-ulp of accuracy becomes visible. Demanding equality there
    would mean making the reference less accurate to match, which is backwards.
    """
    quantized, _ = _quantized(64, 256, 128, 3, device, dtype)
    geom = quantized.geometry
    got = torch.ops.dynquant.dequant(
        quantized.packed, quantized.scales, quantized.offsets, 3, geom.effective_group, 256
    )
    tolerance = {"rtol": 1e-6, "atol": 1e-6} if dtype is torch.float32 else {"rtol": 0, "atol": 0}
    torch.testing.assert_close(got, quantized.dequantize(), **tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_dequant_accepts_a_tensor_with_no_offsets(device, bits):
    """``offsets=None`` is a real input, so the null branch needs its own coverage.

    The encoder always emits offsets -- symmetric mode pins the additive term to the
    grid centre rather than dropping it, and a constant group folds its value into
    the offset with ``scale = 0`` regardless of mode. ``None`` is what a checkpoint
    whose rows were exactly representable without one carries (see
    ``QuantTensor.offsets``), and the kernels have a separate code path for it. The
    tensor is therefore built here rather than asked of the encoder: waiting for the
    encoder to produce one would leave that path untested at every width.
    """
    quantized, _ = _quantized(64, 256, 128, bits, device, torch.float16)
    stripped = replace(quantized, offsets=None, symmetric=True)
    geom = stripped.geometry

    got = torch.ops.dynquant.dequant(
        stripped.packed, stripped.scales, None, bits, geom.effective_group, 256
    )
    torch.testing.assert_close(got, stripped.dequantize(), rtol=0, atol=0)


# --------------------------------------------------------------------------
# GEMV
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize(("out_features", "in_features", "group_size"), GEOMETRIES)
def test_gemv_matches_dequant_then_matmul(device, m, bits, out_features, in_features, group_size):
    """Tolerance here, unlike above, because this one accumulates.

    The kernel sums a row in a fixed ``__shfl_down`` tree while the reference sums
    it in cuBLAS's order, so the two differ by floating-point reassociation. That is
    a real difference and it is bounded by the accumulation width -- both are fp32 --
    not by the weight precision, which is why the tolerance does not vary with bits.
    """
    quantized, _ = _quantized(out_features, in_features, group_size, bits, device, torch.float16)
    geom = quantized.geometry
    torch.manual_seed(m)
    x = torch.randn(m, in_features, dtype=torch.float16, device=device)

    got = torch.ops.dynquant.gemv(
        x,
        quantized.packed,
        quantized.scales,
        quantized.offsets,
        bits,
        geom.effective_group,
        in_features,
    )
    weight = quantized.dequantize(dtype=torch.float32)
    expected = (x.to(torch.float32) @ weight.t()).to(torch.float16)

    assert got.shape == (m, out_features)
    assert got.dtype == torch.float16
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="determinism is a GPU property here")
def test_gemv_is_deterministic():
    """Same input, bit-identical output, every time.

    The kernel uses no atomics and no split-K precisely so this holds. It is not a
    nicety: an eval whose logits shift run to run cannot be compared against a
    baseline at the margins that matter, and a non-deterministic kernel makes every
    regression test flaky in a way that gets it deleted rather than fixed.
    """
    quantized, _ = _quantized(512, 1024, 128, 3, "cuda", torch.float16)
    geom = quantized.geometry
    x = torch.randn(4, 1024, dtype=torch.float16, device="cuda")
    args = (
        x,
        quantized.packed,
        quantized.scales,
        quantized.offsets,
        3,
        geom.effective_group,
        1024,
    )
    first = torch.ops.dynquant.gemv(*args)
    for _ in range(8):
        torch.testing.assert_close(torch.ops.dynquant.gemv(*args), first, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="both paths are CUDA kernels")
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_the_two_gemv_paths_agree(tmp_path, bits):
    """The fast path is an optimization, and this is what makes that checkable.

    Every other GEMV test compares one kernel against the torch oracle at a
    tolerance wide enough to admit fp32 reassociation. That tolerance is also wide
    enough to admit a fast path that is subtly wrong in a way the oracle comparison
    happens to survive -- a dropped tail chunk on a shape where the tail carries
    little weight, say. Comparing the two kernels to *each other* on the same input
    closes that: they decode the same integers with the same scales and differ only
    in the order they sum a row, so they must agree to accumulation noise and not to
    anything looser.

    It runs in a subprocess because the path is chosen by an environment variable
    read once into a function-local static -- deliberately, so that the choice
    cannot change between two calls within one process and make a benchmark
    meaningless.
    """
    import os
    import subprocess
    import sys
    import textwrap

    saved = tmp_path / "general.pt"
    script = textwrap.dedent(
        f"""
        import torch
        import dynquant_kernels
        assert dynquant_kernels.is_available()
        from dynquant.quant.grid import quantize_with_search

        torch.manual_seed(7)
        dense = torch.randn(260, 2048)
        q, _ = quantize_with_search(
            dense, bits={bits}, group_size=128, symmetric=False,
            compute_dtype=torch.float16,
        )
        q = q.to("cuda")
        torch.manual_seed(11)
        x = torch.randn(4, 2048, dtype=torch.float16, device="cuda")
        y = torch.ops.dynquant.gemv(
            x, q.packed, q.scales, q.offsets, {bits}, q.geometry.effective_group, 2048
        )
        torch.save(y.cpu(), {str(saved)!r})
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "DYNQUANT_GEMV_SCALAR": "1"},
    )

    torch.manual_seed(7)
    dense = torch.randn(260, 2048)
    quantized, _ = quantize_with_search(
        dense, bits=bits, group_size=128, symmetric=False, compute_dtype=torch.float16
    )
    quantized = quantized.to("cuda")
    torch.manual_seed(11)
    x = torch.randn(4, 2048, dtype=torch.float16, device="cuda")
    vectorized = torch.ops.dynquant.gemv(
        x,
        quantized.packed,
        quantized.scales,
        quantized.offsets,
        bits,
        quantized.geometry.effective_group,
        2048,
    )

    general = torch.load(saved).cuda()
    torch.testing.assert_close(vectorized, general, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_gemv_refuses_more_rows_than_it_was_built_for(device):
    """The limit is enforced identically on both backends.

    An op whose contract depends on the device is an op whose tests pass on the CPU
    runner and fail in production.
    """
    quantized, _ = _quantized(64, 256, 128, 4, device, torch.float16)
    geom = quantized.geometry
    limit = int(torch.ops.dynquant.gemv_max_rows())
    x = torch.randn(limit + 1, 256, dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError, match="exceeds the kernel's limit"):
        torch.ops.dynquant.gemv(
            x, quantized.packed, quantized.scales, quantized.offsets, 4, geom.effective_group, 256
        )


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_unresolved_per_row_sentinel_is_rejected(device):
    """-1 must be resolved by the caller, and passing it through must not be
    interpreted as anything. A kernel that treated it as a size would index with a
    negative stride and read whatever is behind the buffer."""
    quantized, _ = _quantized(48, 100, PER_ROW_GROUP_SIZE, 3, device, torch.float16)
    with pytest.raises(RuntimeError, match="resolved positive count"):
        torch.ops.dynquant.dequant(
            quantized.packed, quantized.scales, quantized.offsets, 3, PER_ROW_GROUP_SIZE, 100
        )


@pytest.mark.parametrize("device", DEVICES)
def test_mismatched_scales_are_rejected_rather_than_reinterpreted(device):
    quantized, _ = _quantized(64, 256, 128, 4, device, torch.float16)
    geom = quantized.geometry
    with pytest.raises(RuntimeError):
        torch.ops.dynquant.dequant(
            quantized.packed,
            quantized.scales[:-1],  # one row short
            None if quantized.offsets is None else quantized.offsets[:-1],
            4,
            geom.effective_group,
            256,
        )
