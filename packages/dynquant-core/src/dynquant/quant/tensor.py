"""``QuantTensor`` -- a packed weight plus every piece of metadata needed to invert it.

Why this class carries so many explicit fields
----------------------------------------------
The research code stored ``bits``, ``shape``, ``packed``, ``scale``, ``zero`` and
nothing else. ``group_size`` was never written down, so the loader guessed it
(``dynquant_paper/quantizer.py:587-602``)::

    if scale.numel() > qt.shape[0]:
        num_groups = scale.numel() // out_features
        if in_features % num_groups == 0:   group_size = in_features // num_groups
        elif (in_features + 127) // 128 == num_groups:  group_size = 128
        else:                               group_size = in_features // num_groups

That inference is wrong whenever ``in_features`` was padded (the ``elif`` guesses
128 and the first branch can silently pick a *different* divisor), and it cannot
run at all for a weight that is not 2-D -- a stacked MoE expert tensor
``[E, out, in]`` or a Mamba ``conv1d``. Worse, being wrong here does not raise:
it dequantizes at the wrong stride and yields a tensor of the right shape full of
wrong numbers.

So: nothing about the encoding is ever re-derived. Every parameter needed to
reconstruct the dense tensor is stored, validated on load, and version-tagged.

The affine convention
---------------------
Dequantization is a single fused multiply-add per value::

    w  ~=  q * scale + offset            with q an unsigned integer in [0, 2**bits)

``offset`` is an unconstrained *float* additive term -- for the asymmetric case,
simply the group minimum. There is no integer zero-point anywhere in the format,
and that is a deliberate departure from GPTQ and AWQ:

* The kernel inner loop becomes one ``__hfma2`` per value pair, with no integer
  subtract and no second unpacking pass over a packed zero-point table.
* Packed integer zero-points need their own word alignment along the *group*
  axis, and ``num_groups`` is frequently not a multiple of 32 (5120/128 = 40
  groups, and 40 x 3 bits = 120 bits is not a whole number of words). That is an
  entire class of alignment bug that simply does not arise here.
* An integer zero-point is only invertible if code ``zero_point`` lands inside
  ``[0, qmax]``, which forces every group's range to be widened to include zero.
  For a group whose values all share a sign that is pure loss -- widening
  ``[0.50, 0.52]`` to ``[0, 0.52]`` inflates the range 26x and discards better
  than four bits of resolution. A float offset needs no such widening, so this
  encoder is the plain min/max affine quantizer with nothing given away.

The one thing given up is that exact zero is not guaranteed to be a grid point: a
weight of ``0.0`` reconstructs within half a step, like every other value. Nothing
in the format depends on it being exact -- an all-zero group folds through the
constant path (``scale = 0``), and the zero-filled pad region contributes nothing
because the kernel predicates activation loads past ``in_features`` to zero rather
than relying on the weights there being zero.

Symmetric quantization is not a separate code path -- it is the constrained case
``offset == -2**(bits-1) * scale``, which is exactly what the supplement's 4-bit
embedding path did by hand (``zero`` fixed at 8, ``scale = max_abs / 7``).

Uniform bit-width per tensor
----------------------------
A ``QuantTensor`` has *one* bit-width. Fused projections that deserve different
precision per projection -- Phi-4's ``qkv_proj`` and ``gate_up_proj`` -- are
represented as several ``QuantTensor`` shards over disjoint row ranges
(:attr:`row_offset`), whose outputs the runtime concatenates. That keeps every
CUDA kernel templated on a single compile-time ``BITS`` with no per-row
branching, which is the whole reason the inner loop can be unrolled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from dynquant.constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE, QUANT_TENSOR_SUFFIXES
from dynquant.errors import PackingError

from .pack import (
    PER_ROW_GROUP_SIZE,
    RowGeometry,
    pack_nbit,
    row_geometry,
    unpack_nbit,
)

__all__ = ["QuantLayout", "QuantTensor"]

_EPS = 1e-12


class QuantLayout(str, Enum):
    """Physical arrangement of the packed words.

    ``LINEAR``
        Row-major, groups consecutive within a row:
        ``packed[r, g * words_per_group + w]``. Readable, and what the GEMV
        decode kernels want -- a thread block streams one row's groups as
        consecutive ``uint4`` loads.

    ``TC_SHUFFLED``
        Values permuted at *quantization time* into the register order that
        ``mma.sync`` fragments expect, so the tensor-core prefill kernel can load
        straight from global memory into fragments without a shared-memory
        transpose. Semantically identical, byte-order different.

    This field exists in the format from day one even though the tensor-core path
    lands in a later phase. Adding it later would mean either a format break or a
    heuristic that guesses the layout from the shape -- and guessing metadata from
    shapes is the exact mistake this class was written to eliminate.
    """

    LINEAR = "linear"
    TC_SHUFFLED = "tc_shuffled"


def storage_dtype(weight: torch.Tensor) -> torch.dtype:
    """The dtype scales and offsets are stored in: 16-bit, unless the weight already is.

    This is a *format* decision, not a numerical one, and it is settled by
    :mod:`dynquant.allocate.budget`: ``metadata_bits: int = 16``, "an fp16 scale and an
    fp16 offset per 128". Every quoted bit-width in this project is computed against that
    constant, so a weight whose scales were stored in fp32 would occupy 0.25 bits/weight
    more than its own manifest claims at group 128 -- and the manifest is the number the
    method is judged on.

    It has to be *one* function, which is the part that was learned the hard way. There
    were three copies: the packer returned the weight's own dtype, the exporter returned
    fp16 unless the weight was already half, and the in-place encoder passed nothing and
    took :meth:`QuantTensor.from_dense`'s own fallback, which was the exporter's rule. On
    fp16 and bf16 all three agree -- so on every model anyone ships they agree -- and the
    disagreement waited for an fp32 test model, where it surfaced as a packed bank sitting
    0.0082 from the in-place encoder that was supposed to be its yardstick: one 4-bit
    step, and 16% of what quantizing the model moved in the first place.

    The objection to fp16 metadata is that :meth:`QuantTensor.dequantize` returns the
    scale dtype when it is not told otherwise, so an fp32 model would find fp16 weights in
    its graph. It does not hold for a Linear: :func:`dynquant.runtime.ops.quantized_matmul`
    dequantizes to ``x.dtype``, following the activation. It holds for exactly one caller,
    :meth:`dynquant.runtime.linear.DynQuantExpertBank.__getitem__`, which has no activation
    to follow -- so that one is told the dtype at construction instead of inferring it here.
    """
    return weight.dtype if weight.dtype in (torch.float16, torch.bfloat16) else torch.float16


@dataclass(slots=True)
class QuantTensor:
    """A group-wise affine-quantized 2-D matrix in packed form.

    Attributes:
        packed: int32 words, shape ``[num_rows, num_groups * words_per_group]``.
            Bit patterns are unsigned; int32 is a storage convention (see
            :mod:`dynquant.quant.pack`).
        scales: Multiplicative term, shape ``[num_rows, num_groups]``.
        offsets: Additive term, same shape as ``scales``. ``None`` only when the
            tensor is exactly representable without one (all-constant rows).
        bits: 2, 3, 4 or 8. Uniform across the whole tensor.
        group_size: Values per quantization group along the input dimension, or
            :data:`~dynquant.constants.PER_ROW_GROUP_SIZE` (-1) for a single group
            spanning the row. The sentinel is stored verbatim, never resolved to
            ``in_features``: it is the only record that this tensor is exempt from
            the 32-value alignment rule, and a reader that sees the resolved width
            cannot distinguish it from a misaligned explicit group size.
        in_features: True reduction-dimension length, *before* padding. Kept so
            the kernel knows where real data stops.
        logical_shape: Shape of the dense tensor this came from, which may have
            rank > 2 (stacked MoE experts). Rows are the flattened leading dims.
        row_offset: Where this shard's rows begin in the parent module's output
            space. Non-zero only for shards of a fused projection.
        layout: See :class:`QuantLayout`.
    """

    packed: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor | None
    bits: int
    group_size: int
    in_features: int
    logical_shape: tuple[int, ...]
    row_offset: int = 0
    layout: QuantLayout = QuantLayout.LINEAR
    symmetric: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived geometry --------------------------------------------------

    @property
    def num_rows(self) -> int:
        return int(self.packed.shape[0])

    @property
    def geometry(self) -> RowGeometry:
        """All derived sizes, from the one resolver the encoder also used.

        Every size below delegates here rather than recomputing, so
        :meth:`validate` cannot arrive at a different answer than
        :meth:`from_dense` did -- which is exactly how per-row tensors came to be
        rejected by the validator that was supposed to accept them.
        """
        return row_geometry(self.bits, self.group_size, self.in_features)

    @property
    def padded_in_features(self) -> int:
        return self.geometry.padded_in_features

    @property
    def num_groups(self) -> int:
        return self.geometry.num_groups

    @property
    def words_per_group(self) -> int:
        return self.geometry.words_per_group

    @property
    def is_per_row(self) -> bool:
        """One group spans the whole reduction dimension (embeddings, conv1d)."""
        return self.group_size == PER_ROW_GROUP_SIZE

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.scales.dtype

    # -- honest size accounting -------------------------------------------

    @property
    def nbytes(self) -> int:
        """Total bytes on disk/in VRAM, **including** scales and offsets.

        The supplement reported ``target_avg_bits`` counting packed weights only,
        so a "3-bit" checkpoint at group size 128 actually occupied 3.25 bits per
        weight once metadata was included -- and the gap widens at smaller group
        sizes. Budgeting against this property instead means the number in the
        manifest is the number on the filesystem.
        """
        total = self.packed.numel() * self.packed.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.offsets is not None:
            total += self.offsets.numel() * self.offsets.element_size()
        return total

    @property
    def num_dense_elements(self) -> int:
        n = 1
        for dim in self.logical_shape:
            n *= dim
        return n

    @property
    def bits_per_weight(self) -> float:
        """Effective bits per *dense* element, metadata amortised in."""
        return self.nbytes * 8.0 / max(self.num_dense_elements, 1)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dense(
        cls,
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int = DEFAULT_GROUP_SIZE,
        symmetric: bool = False,
        compute_dtype: torch.dtype | None = None,
        row_offset: int = 0,
        clip_ratio: float | torch.Tensor = 1.0,
    ) -> QuantTensor:
        """Quantize a dense tensor with plain group-wise min/max.

        This is the **reference** encoder: straightforward, exact, and the oracle
        that the CUDA quantizer and every kernel are validated against. The
        MSE-optimal clipping search and the fused Thrust/CUB implementation live
        in :mod:`dynquant.quant.quantizer` (phase 5) and must agree with this to
        within documented tolerances.

        Args:
            weight: Dense tensor, rank >= 2. Leading dims fold into rows, so a
                stacked expert weight ``[E, out, in]`` becomes ``[E * out, in]``
                and each expert keeps its own scales.
            clip_ratio: Shrink each group's range by this factor before
                quantizing. ``1.0`` is pure min/max. Values below 1 trade
                outlier fidelity for finer resolution on the bulk -- the axis the
                phase-5 grid search optimises over. A tensor of shape
                ``[num_rows, num_groups]`` sets the ratio per group, which is what
                the search returns: the best trade-off is a property of a group's
                own distribution, and one ratio for a whole tensor forces groups
                with a single outlier and groups with none to share a compromise.
        """
        if bits not in BIT_OPTIONS:
            raise PackingError(f"bits={bits} not supported; expected one of {list(BIT_OPTIONS)}")
        if weight.ndim < 2:
            raise PackingError(
                f"expected a tensor of rank >= 2 to quantize, got shape {tuple(weight.shape)}"
            )
        if isinstance(clip_ratio, torch.Tensor):
            if not bool(((clip_ratio > 0.0) & (clip_ratio <= 1.0)).all()):
                raise PackingError("every clip_ratio must be in (0, 1]")
        elif not 0.0 < clip_ratio <= 1.0:
            raise PackingError(f"clip_ratio must be in (0, 1], got {clip_ratio}")

        logical_shape = tuple(weight.shape)
        in_features = logical_shape[-1]
        geom = row_geometry(bits, group_size, in_features)
        group_size = geom.group_size  # PER_ROW_GROUP_SIZE stays the sentinel
        compute_dtype = compute_dtype or storage_dtype(weight)

        rows = weight.reshape(-1, in_features)
        padded = geom.padded_in_features
        if padded != in_features:
            # Zero-fill: the pad participates in its group's min/max like any
            # other zero, and contributes nothing at inference because the kernel
            # predicates activation loads past `in_features` to zero.
            rows = torch.nn.functional.pad(rows, (0, padded - in_features))

        num_rows = rows.shape[0]
        num_groups = geom.num_groups
        grouped = rows.reshape(num_rows, num_groups, geom.effective_group).to(torch.float32)

        qmax = (1 << bits) - 1

        # Normalised once, so the two branches below never re-test the type. `None`
        # means "no clipping at all" and keeps the untouched arithmetic path: at
        # ratio 1.0 the general formula is an algebraic no-op but not a *float*
        # no-op, and a one-ulp shift in every group's bounds is a gratuitous change
        # to the output of the default encoder.
        clip: torch.Tensor | None
        if isinstance(clip_ratio, torch.Tensor):
            clip = clip_ratio.to(device=grouped.device, dtype=torch.float32)
            if clip.shape != (num_rows, num_groups):
                raise PackingError(
                    f"clip_ratio tensor has shape {tuple(clip.shape)}, "
                    f"expected {(num_rows, num_groups)} (one ratio per group)"
                )
        elif clip_ratio < 1.0:
            clip = torch.full((num_rows, num_groups), clip_ratio, device=grouped.device)
        else:
            clip = None

        # Constant-group detection uses the *raw* range, before clipping narrows it
        # and before either branch derives a scale: a group of all 0.75 has a zero
        # range and folds exactly into `scale = 0, offset = 0.75`, which is cheaper
        # *and* more accurate than spending the whole code space on one value.
        constant = grouped.amax(dim=-1) <= grouped.amin(dim=-1)

        if symmetric:
            # The additive term is pinned to the grid centre, so the grid is
            # symmetric about zero and the scale is set by the larger side.
            max_abs = grouped.abs().amax(dim=-1)
            if clip is not None:
                max_abs = max_abs * clip
            scale = max_abs / max(float(1 << (bits - 1)) - 1.0, 1.0)
            base = -float(1 << (bits - 1)) * scale
        else:
            wmin = grouped.amin(dim=-1)
            wmax = grouped.amax(dim=-1)

            if clip is not None:
                centre = 0.5 * (wmin + wmax)
                half = 0.5 * (wmax - wmin) * clip
                wmin, wmax = centre - half, centre + half

            # The additive term is the group minimum, unmodified. Note what is
            # *absent*: GPTQ and AWQ widen every group's range to include zero,
            # because they store an integer zero-point and code `zero_point` must
            # land inside `[0, qmax]` for the affine map to be invertible. That
            # widening is pure loss for a one-signed group -- for a group spanning
            # [0.50, 0.52] it inflates the range 26x and throws away better than
            # four bits of resolution.
            #
            # Storing a float offset removes the constraint entirely, so this is
            # the textbook min/max affine quantizer with no zero-point at all. The
            # price is that exact zero is no longer guaranteed to be a grid point;
            # a weight of 0 reconstructs within half a step like any other value.
            # Nothing in the format needs it to be exact: an all-zero group folds
            # through the constant path above, and the zero-filled pad region
            # contributes nothing because the kernel predicates activation loads
            # past `in_features` to zero rather than relying on zero weights.
            scale = (wmax - wmin) / qmax
            base = wmin

        # A constant group is representable exactly: emit code 0 and fold the
        # constant into the offset with scale 0. The supplement carried a separate
        # `is_constant`/`constant_value` pair for this; folding it into the affine
        # map removes a branch from every kernel.
        degenerate = (scale <= _EPS) | constant
        safe_scale = torch.where(degenerate, torch.ones_like(scale), scale)

        # Round scale and offset to storage precision *before* deriving the codes,
        # so the encoder solves against exactly the two numbers the kernel will
        # read. Rounding afterwards leaves a systematic per-group bias of up to one
        # storage-dtype ulp of the group range -- ~12% of a quantization step at
        # 8 bits, where steps are small enough for that to show up in the error.
        stored_scale = safe_scale.to(compute_dtype).to(torch.float32)
        offset = torch.where(degenerate, torch.zeros_like(base), base)
        offset = offset.to(compute_dtype).to(torch.float32)

        codes = torch.round((grouped - offset.unsqueeze(-1)) / stored_scale.unsqueeze(-1))
        codes = codes.clamp_(0, qmax).to(torch.uint8)
        codes = torch.where(degenerate.unsqueeze(-1), torch.zeros_like(codes), codes)

        if degenerate.any():
            const_value = grouped[..., 0]
            offset = torch.where(degenerate, const_value, offset)
            stored_scale = torch.where(degenerate, torch.zeros_like(stored_scale), stored_scale)

        packed = pack_nbit(codes.reshape(num_rows, padded), bits, validate=False)
        return cls(
            packed=packed,
            scales=stored_scale.to(compute_dtype),
            offsets=offset.to(compute_dtype),
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            logical_shape=logical_shape,
            row_offset=row_offset,
            layout=QuantLayout.LINEAR,
            symmetric=symmetric,
        )

    @classmethod
    def from_codes(
        cls,
        codes: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        *,
        bits: int,
        group_size: int = DEFAULT_GROUP_SIZE,
        symmetric: bool = False,
        compute_dtype: torch.dtype | None = None,
        row_offset: int = 0,
        logical_shape: tuple[int, ...] | None = None,
    ) -> QuantTensor:
        """Adopt a grid someone else fitted, instead of fitting one.

        :meth:`from_dense` takes float weights and chooses the affine map. This takes
        the map as given and only packs. The difference matters when the weights being
        imported are *already* on a grid -- a GPTQ or AWQ checkpoint, or anything else
        whose scales were fitted by a different algorithm -- because re-deriving the map
        from the dequantized values is not the identity. Min/max recovers the original
        scale only when a group still occupies both ends of its code range, and after
        GPTQ's error compensation or AWQ's clipping search many groups do not. Where they
        do not, the re-fitted step is narrower than the original, the original levels fall
        between the new ones, and the weights move. Not by much -- a fraction of a step --
        but enough that the published checkpoint is no longer the one that was scored,
        which is the single thing a published baseline has to be.

        Importing a zero-point convention: the ecosystem's ``scale * (q - zero)`` is this
        format's ``scale * code + offset`` with ``code = q`` and ``offset = -scale * zero``.
        The translation belongs at the call site, so that the one place that knows about a
        foreign convention is the one reading the foreign checkpoint. It preserves the grid
        exactly -- same codes, same levels -- but not the last bit of the reconstruction,
        because ``s * q + (-s * z)`` and ``s * (q - z)`` round differently. That gap is
        ~1e-7 relative, orders below both the quantization step and the storage rounding.

        Args:
            codes: Integer codes in ``[0, 2**bits - 1]``, shaped like the dense tensor
                they encode (rank >= 2, leading dims fold into rows). Unpadded.
            scales: ``[num_rows, num_groups]``. A zero scale is the constant-group
                encoding and is preserved, not repaired.
            offsets: ``[num_rows, num_groups]``, the additive term.
            logical_shape: Overrides the shape recorded in the result. Only for callers
                whose codes arrive already folded to 2-D but which describe a stacked
                expert bank; the folded row count must agree.
        """
        if bits not in BIT_OPTIONS:
            raise PackingError(f"bits={bits} not supported; expected one of {list(BIT_OPTIONS)}")
        if codes.ndim < 2:
            raise PackingError(f"expected codes of rank >= 2, got shape {tuple(codes.shape)}")
        if not torch.is_floating_point(codes):
            code_min, code_max = int(codes.min()), int(codes.max())
        else:
            raise PackingError(
                f"codes must be an integer tensor, got {codes.dtype}. Rounding them here "
                "would hide a caller that passed dequantized weights by mistake"
            )
        qmax = (1 << bits) - 1
        if code_min < 0 or code_max > qmax:
            raise PackingError(
                f"codes span [{code_min}, {code_max}], outside the {bits}-bit range "
                f"[0, {qmax}] -- the grid and the width disagree"
            )

        shape = tuple(codes.shape) if logical_shape is None else tuple(logical_shape)
        in_features = shape[-1]
        if in_features != codes.shape[-1]:
            raise PackingError(
                f"logical_shape {shape} ends in {in_features} but codes are {codes.shape[-1]} wide"
            )
        geom = row_geometry(bits, group_size, in_features)
        group_size = geom.group_size  # PER_ROW_GROUP_SIZE stays the sentinel
        compute_dtype = compute_dtype or storage_dtype(scales)

        rows = codes.reshape(-1, in_features)
        num_rows, num_groups = rows.shape[0], geom.num_groups
        if int(codes.numel()) != num_rows * in_features:
            raise PackingError(f"codes {tuple(codes.shape)} do not fold to rows of {in_features}")
        expected = (num_rows, num_groups)
        for name, term in (("scales", scales), ("offsets", offsets)):
            if tuple(term.shape) != expected:
                raise PackingError(f"{name} shape {tuple(term.shape)} != expected {expected}")

        padded = geom.padded_in_features
        if padded != in_features:
            # The pad value is arbitrary and deliberately recorded as such: unlike
            # `from_dense`, where the zero fill participates in its group's min/max, the
            # scales here are the caller's and nothing recomputes them, and the kernel
            # predicates activation loads past `in_features` to zero so the codes are
            # never read. A mutation that pads with `qmax` instead reddens no test, and
            # should not -- zero is chosen only so two encodings of the same tensor
            # produce the same words.
            rows = torch.nn.functional.pad(rows, (0, padded - in_features))

        packed = pack_nbit(rows.to(torch.uint8), bits, validate=False)
        return cls(
            packed=packed,
            scales=scales.to(compute_dtype),
            offsets=offsets.to(compute_dtype),
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            logical_shape=shape,
            row_offset=row_offset,
            layout=QuantLayout.LINEAR,
            symmetric=symmetric,
        )

    # -- inversion ---------------------------------------------------------

    def dequantize(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Reconstruct the dense tensor. Reference implementation.

        Accumulates in fp32 regardless of storage dtype, matching what the CUDA
        kernels do, so parity tests compare like with like.
        """
        if self.layout is not QuantLayout.LINEAR:
            raise PackingError(
                f"dequantize() expects {QuantLayout.LINEAR.value} layout; this tensor is "
                f"{self.layout.value}. Convert with QuantTensor.to_layout() first."
            )
        geom = self.geometry
        codes = unpack_nbit(self.packed, self.bits, geom.padded_in_features)
        # `effective_group`, not `group_size`: the latter is -1 when per-row, and
        # reshape would silently *infer* that dimension. It happens to infer the
        # right value, which is worse than it failing -- the next reader would
        # take it for a deliberate use of -1 as a wildcard.
        grouped = codes.reshape(self.num_rows, geom.num_groups, geom.effective_group)
        grouped = grouped.to(torch.float32)
        scale = self.scales.to(torch.float32).unsqueeze(-1)
        dense = grouped * scale
        if self.offsets is not None:
            dense = dense + self.offsets.to(torch.float32).unsqueeze(-1)
        dense = dense.reshape(self.num_rows, geom.padded_in_features)
        dense = dense[:, : self.in_features]
        out_dtype = dtype or self.compute_dtype
        return dense.reshape(self.logical_shape).to(out_dtype)

    def rows(self, start: int, stop: int) -> QuantTensor:
        """A contiguous band of rows, sharing this tensor's storage.

        Rows are the flattened leading dimensions, so on a batched expert bank
        ``[E, out, in]`` the band ``[e * out, (e + 1) * out)`` is expert ``e``, and
        nothing is copied to get it: ``packed``, ``scales`` and ``offsets`` are all
        row-major with the row as dimension 0, so each slice is a view. That is the
        whole reason a packed bank is cheaper at runtime than a dequantized one --
        one expert's worth of fp16 exists at a time instead of the bank's.

        ``row_offset`` stays 0 rather than becoming ``start``. That field means where
        a shard begins in *its module's output space*, which is a fused projection's
        question. An expert's outputs are its own, numbered from zero, and recording
        the bank position there would tell a serving planner that expert 3 of a
        4096-row bank occupies output rows 12288..16383 of something.
        """
        rows = self.num_rows
        if not 0 <= start < stop <= rows:
            raise PackingError(
                f"row band [{start}, {stop}) does not fall inside this tensor's {rows} rows"
            )
        return QuantTensor(
            packed=self.packed[start:stop],
            scales=self.scales[start:stop],
            offsets=None if self.offsets is None else self.offsets[start:stop],
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            logical_shape=(stop - start, self.in_features),
            row_offset=0,
            layout=self.layout,
            symmetric=self.symmetric,
        )

    def quantization_error(self, reference: torch.Tensor) -> dict[str, float]:
        """Relative Frobenius error and max absolute error against a dense tensor.

        Recorded per layer in the manifest so a bad allocation is visible in the
        checkpoint itself rather than only in a downstream eval run.
        """
        ref = reference.reshape(self.logical_shape).to(torch.float32)
        err = self.dequantize(dtype=torch.float32) - ref
        ref_norm = torch.linalg.vector_norm(ref).item()
        return {
            "rel_fro": torch.linalg.vector_norm(err).item() / max(ref_norm, _EPS),
            "max_abs": err.abs().max().item(),
        }

    # -- movement ----------------------------------------------------------

    def to(self, device: torch.device | str | None = None, **kwargs: Any) -> QuantTensor:
        """Move packed buffers. Never dequantizes."""

        return QuantTensor(
            packed=self.packed.to(device=device),  # dtype must stay int32
            scales=self.scales.to(device=device, **kwargs),
            # `offsets` is None for a symmetric tensor and has to stay None: an
            # all-zero offsets buffer would be numerically identical but would cost
            # a second buffer per row and make `symmetric` unverifiable from the
            # tensor alone.
            offsets=None if self.offsets is None else self.offsets.to(device=device, **kwargs),
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            logical_shape=self.logical_shape,
            row_offset=self.row_offset,
            layout=self.layout,
            symmetric=self.symmetric,
            extra=dict(self.extra),
        )

    # -- serialisation -----------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable description. Goes into the manifest verbatim."""
        meta: dict[str, Any] = {
            "bits": self.bits,
            "group_size": self.group_size,
            "in_features": self.in_features,
            "logical_shape": list(self.logical_shape),
            "num_rows": self.num_rows,
            "num_groups": self.num_groups,
            "words_per_group": self.words_per_group,
            "row_offset": self.row_offset,
            "layout": self.layout.value,
            "symmetric": self.symmetric,
            "has_offsets": self.offsets is not None,
            "compute_dtype": str(self.compute_dtype).removeprefix("torch."),
            "nbytes": self.nbytes,
            "bits_per_weight": round(self.bits_per_weight, 6),
        }
        if self.extra:
            meta["extra"] = self.extra
        return meta

    def state_dict(self, prefix: str) -> dict[str, torch.Tensor]:
        """Tensors to write into safetensors, keyed ``<prefix>.<suffix>``."""
        out = {
            f"{prefix}.{QUANT_TENSOR_SUFFIXES['packed']}": self.packed,
            f"{prefix}.{QUANT_TENSOR_SUFFIXES['scale']}": self.scales,
        }
        if self.offsets is not None:
            out[f"{prefix}.{QUANT_TENSOR_SUFFIXES['offset']}"] = self.offsets
        return out

    @classmethod
    def from_state_dict(
        cls,
        prefix: str,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> QuantTensor:
        """Rebuild from safetensors entries plus its manifest metadata.

        Validates every geometric invariant. A checkpoint that has been truncated,
        re-sharded by an unaware tool, or written by a mismatched version fails
        here with a specific message instead of decoding to noise.
        """
        packed_key = f"{prefix}.{QUANT_TENSOR_SUFFIXES['packed']}"
        scale_key = f"{prefix}.{QUANT_TENSOR_SUFFIXES['scale']}"
        offset_key = f"{prefix}.{QUANT_TENSOR_SUFFIXES['offset']}"
        try:
            packed = tensors[packed_key]
            scales = tensors[scale_key]
        except KeyError as exc:
            raise PackingError(
                f"quantized tensor {prefix!r} is incomplete: missing {exc.args[0]!r}"
            ) from exc
        offsets = tensors.get(offset_key)
        if metadata.get("has_offsets", True) and offsets is None:
            raise PackingError(
                f"{prefix!r} metadata says it has offsets but {offset_key!r} is absent"
            )

        qt = cls(
            packed=packed,
            scales=scales,
            offsets=offsets,
            bits=int(metadata["bits"]),
            group_size=int(metadata["group_size"]),
            in_features=int(metadata["in_features"]),
            logical_shape=tuple(metadata["logical_shape"]),
            row_offset=int(metadata.get("row_offset", 0)),
            layout=QuantLayout(metadata.get("layout", QuantLayout.LINEAR.value)),
            symmetric=bool(metadata.get("symmetric", False)),
            extra=dict(metadata.get("extra", {})),
        )
        qt.validate()
        return qt

    @classmethod
    def empty(
        cls,
        *,
        bits: int,
        group_size: int,
        in_features: int,
        logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        symmetric: bool = False,
        has_offsets: bool = True,
        row_offset: int = 0,
        layout: QuantLayout = QuantLayout.LINEAR,
    ) -> QuantTensor:
        """Correctly shaped buffers with undefined contents, for a load path to fill.

        A checkpoint loader needs the *shape* of every packed tensor before it has any
        of the values: ``transformers`` builds the module tree first and then copies
        tensors into it by name. Deriving those shapes at the call site would be a
        third implementation of the geometry -- after :meth:`from_dense` and
        :meth:`validate` -- and the one place this project has repeatedly been bitten
        is a second copy of a rule that agrees until it doesn't. So the shapes come
        from :func:`~dynquant.quant.pack.row_geometry`, the same resolver the encoder
        used, and the result is a tensor that :meth:`validate` accepts.

        The contents are ``torch.empty``: uninitialised, not zeroed. A loader that
        fails to fill one of these has a bug, and zeros would hide it behind a model
        that runs and answers badly. Garbage does not survive a comparison against
        the reference and zeros can.
        """
        geom = row_geometry(bits, group_size, in_features)
        num_rows = 1
        for extent in logical_shape[:-1]:
            num_rows *= int(extent)
        return cls(
            packed=torch.empty((num_rows, geom.words_per_row), dtype=torch.int32, device=device),
            scales=torch.empty((num_rows, geom.num_groups), dtype=dtype, device=device),
            offsets=(
                torch.empty((num_rows, geom.num_groups), dtype=dtype, device=device)
                if has_offsets
                else None
            ),
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            logical_shape=tuple(int(extent) for extent in logical_shape),
            row_offset=row_offset,
            layout=layout,
            symmetric=symmetric,
        )

    def validate(self) -> None:
        """Assert every geometric and dtype invariant. Cheap; always run on load."""
        if self.bits not in BIT_OPTIONS:
            raise PackingError(f"{self.bits}-bit is not a supported width")
        geom = self.geometry  # raises PackingError on an invalid group size
        if self.packed.dtype is not torch.int32:
            raise PackingError(
                f"packed words must be int32 (unsigned bit patterns), got {self.packed.dtype}"
            )
        if self.packed.ndim != 2:
            raise PackingError(f"packed must be 2-D [rows, words], got {tuple(self.packed.shape)}")

        expected_words = geom.words_per_row
        if self.packed.shape[1] != expected_words:
            raise PackingError(
                f"{self.bits}-bit, group_size={self.group_size}, in_features="
                f"{self.in_features} (padded to {geom.padded_in_features}) implies "
                f"{expected_words} words per row, but packed has {self.packed.shape[1]}. "
                f"The checkpoint is inconsistent -- do not use it."
            )
        expected_scale_shape = (self.num_rows, geom.num_groups)
        if tuple(self.scales.shape) != expected_scale_shape:
            raise PackingError(
                f"scales shape {tuple(self.scales.shape)} != expected {expected_scale_shape}"
            )
        if self.offsets is not None and tuple(self.offsets.shape) != expected_scale_shape:
            raise PackingError(
                f"offsets shape {tuple(self.offsets.shape)} != expected {expected_scale_shape}"
            )
        folded_rows = self.num_dense_elements // max(self.in_features, 1)
        if folded_rows != self.num_rows:
            raise PackingError(
                f"logical_shape {self.logical_shape} folds to {folded_rows} rows but packed "
                f"has {self.num_rows}"
            )

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        shard = f", rows@{self.row_offset}" if self.row_offset else ""
        return (
            f"QuantTensor({list(self.logical_shape)}, {self.bits}-bit, g={self.group_size}, "
            f"{self.bits_per_weight:.3f} bpw effective, {self.layout.value}{shard})"
        )


def _json_default(obj: Any) -> Any:  # pragma: no cover -- manifest writing helper
    if isinstance(obj, torch.dtype):
        return str(obj).removeprefix("torch.")
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"{type(obj)!r} is not JSON-serialisable")


def dumps_metadata(meta: dict[str, Any]) -> str:
    """Serialise manifest metadata deterministically (sorted keys, no NaN)."""
    return json.dumps(meta, sort_keys=True, allow_nan=False, default=_json_default)
