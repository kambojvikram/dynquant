"""Group-aligned uint32 bit packing -- the precondition for every fast kernel.

The problem with the research layout
------------------------------------
``dynquant_paper/quantizer.py`` flattened the entire weight matrix and *then*
packed it, so a 3-bit quantization group did not begin at a predictable bit
offset: group ``g`` started at bit ``g * 128 * 3 = g * 384``, which is word-aligned
only because 384 is divisible by 32 -- but the *rows* were not, since a row of
5120 values is 15360 bits = 480 words, and the flattening ran straight across the
row boundary. A thread block asked to load "row 7, group 12" had to compute a
bit offset that depended on the entire preceding matrix, and every value could
straddle a word boundary at a different shift. No coalesced load, no constant
shift, no unrolled inner loop -- which is why the paper ships no kernels.

The layout here
---------------
Values are packed **per row, per group**, into ``uint32`` words, LSB-first::

    packed[r, g * words_per_group + w]   holds group g's w-th word of row r

The single invariant that makes this work is ``group_size % 32 == 0``. With that,
``group_size * bits`` is a multiple of 32 for every ``bits`` in {2,3,4,8}, so a
group occupies a whole number of words and the next group starts on a word
boundary. A block loading a group reads ``group_size * bits / 32`` consecutive
words -- 8 words at 2-bit, 12 at 3-bit, 16 at 4-bit, 32 at 8-bit for
``group_size=128`` -- as ``uint4`` vector loads, and every value's shift is a
compile-time constant.

3-bit falls out of the same rule rather than needing a special case: 32 values
occupy exactly 3 words (96 bits), and exactly two of those 32 values straddle a
word boundary (indices 10 and 21). The generic code below handles straddling
uniformly, so there is no separate 3-bit path to keep in sync.

There is exactly one exemption, ``group_size == PER_ROW_GROUP_SIZE`` (-1): a
single group spanning the whole row, used for embeddings and for tiny trailing
dimensions like Mamba's ``conv1d.weight`` of shape ``[channels, 1, 4]``. The
alignment rule exists to keep the *next* group on a word boundary, and per-row
there is no next group, so the row simply rounds up to a whole number of words
and the final word may carry unused high bits. :func:`row_geometry` is the single
place that resolves either mode into sizes.

Storage dtype
-------------
Packed words are stored as **int32**, not ``torch.uint32``. The bit pattern is
identical (two's-complement reinterpretation) and the kernel reads the pointer as
``uint32*``, but int32 has full operator coverage across torch versions,
safetensors and every Hub tool, whereas ``torch.uint32`` still lacks bitwise ops
on several supported torch builds. GPTQ, AWQ and Marlin all made the same choice,
so checkpoints stay legible to existing tooling.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

from dynquant.constants import BIT_OPTIONS, GROUP_SIZE_ALIGNMENT, PER_ROW_GROUP_SIZE
from dynquant.errors import PackingError

__all__ = [
    "MASK32",
    "PER_ROW_GROUP_SIZE",
    "RowGeometry",
    "checked_group_size",
    "pack_nbit",
    "padded_in_features",
    "row_geometry",
    "stored_bits",
    "unpack_nbit",
    "words_per_group",
    "words_per_row",
]

MASK32 = 0xFFFFFFFF
_WORD_BITS = 32


# --------------------------------------------------------------------------
# Shape arithmetic
# --------------------------------------------------------------------------


def checked_group_size(group_size: int, in_features: int, bits: int) -> int:
    """Validate a group size, returning it unchanged.

    Rejects anything that would break word alignment, because the failure mode
    otherwise is a kernel that reads at the wrong shift and produces plausible
    but wrong numbers -- far worse than an exception.

    :data:`PER_ROW_GROUP_SIZE` is returned **as the sentinel**, not resolved to
    ``in_features``. Resolving it here is what made the exemption below
    unenforceable: the resolved value went into the tensor and into the manifest,
    and every later check re-ran this function on ``100`` rather than on ``-1``,
    took the alignment branch, and rejected a tensor this module had just
    produced. The sentinel is the only thing that records "alignment does not
    apply", so it has to survive into the format.
    """
    if bits not in BIT_OPTIONS:
        raise PackingError(f"bits={bits} not supported; expected one of {list(BIT_OPTIONS)}")
    if group_size == PER_ROW_GROUP_SIZE:
        # Per-row: the whole reduction dimension is one group. Exempt from the
        # alignment rule because there is nothing to align *to* -- the single
        # group starts at word 0 and the row is padded up to a whole number of
        # words, so no group ever begins mid-word. This is the path for
        # embeddings (symmetric per-row) and for tiny trailing dimensions like
        # Mamba's `conv1d.weight` of shape [channels, 1, 4], where demanding a
        # multiple of 32 would reject a legitimate tensor outright.
        if in_features <= 0:
            raise PackingError(
                f"in_features must be positive for per-row grouping, got {in_features}"
            )
        return PER_ROW_GROUP_SIZE
    if group_size <= 0:
        raise PackingError(f"group_size must be positive or -1 (per-row), got {group_size}")
    if group_size % GROUP_SIZE_ALIGNMENT != 0:
        raise PackingError(
            f"group_size={group_size} must be a multiple of {GROUP_SIZE_ALIGNMENT} so that every "
            f"group starts on a uint32 word boundary for all of {list(BIT_OPTIONS)}-bit. "
            f"Nearest valid values: {group_size // GROUP_SIZE_ALIGNMENT * GROUP_SIZE_ALIGNMENT} "
            f"or {(group_size // GROUP_SIZE_ALIGNMENT + 1) * GROUP_SIZE_ALIGNMENT}."
        )
    return group_size


def padded_in_features(in_features: int, group_size: int) -> int:
    """Round the reduction dimension up to a whole number of groups.

    The pad region is zero-filled in the weight *before* quantization, so it
    participates in the group's min/max like any other zero. It contributes
    nothing to the output because the kernel predicates its activation loads
    past ``in_features`` to zero -- the tail block pays one comparison.

    Per-row grouping pads by **zero values**: the single group is exactly
    ``in_features`` wide and the row instead rounds up to a whole number of
    *words*, leaving at most 31 unused bits in the final word (see
    :func:`row_geometry`). Padding values here instead would fold real zeros into
    the group's min/max -- for a Mamba ``conv1d`` of 4 positive taps that means 4
    invented zeros in a group of 8, widening the range to include zero and
    throwing away resolution for nothing.
    """
    if group_size == PER_ROW_GROUP_SIZE:
        return in_features
    return -(-in_features // group_size) * group_size


def words_per_group(bits: int, group_size: int) -> int:
    """uint32 words occupied by one quantization group."""
    total_bits = group_size * bits
    if total_bits % _WORD_BITS != 0:
        raise PackingError(
            f"group_size={group_size} at {bits}-bit occupies {total_bits} bits, "
            f"not a whole number of 32-bit words"
        )
    return total_bits // _WORD_BITS


def words_per_row(bits: int, num_values: int) -> int:
    """uint32 words needed for ``num_values`` consecutive quantized values.

    Rounds up; a trailing partial word is zero-padded. Group-aligned callers
    never hit the partial case, but per-row grouping and side tables (per-row bit
    widths, expert masks) do.
    """
    return -(-num_values * bits // _WORD_BITS)


class RowGeometry(NamedTuple):
    """Every derived size for one row of a packed tensor.

    Both grouping modes resolve through here, and that is the point. The encoder
    used to compute the padded width and group count inline while
    :meth:`~dynquant.quant.tensor.QuantTensor.validate` recomputed them from the
    stored metadata by a different route; when the two routes disagreed -- which
    they did for every per-row tensor whose ``in_features`` was not a multiple of
    32 -- the encoder produced a tensor the validator refused to load. One
    resolver means a disagreement is not expressible.
    """

    group_size: int
    """As stored, so possibly :data:`PER_ROW_GROUP_SIZE`."""

    effective_group: int
    """Values per group for arithmetic: ``in_features`` when per-row."""

    padded_in_features: int
    num_groups: int

    words_per_group: int
    """Words one group occupies. Exact for aligned groups; for per-row this
    rounds up, so the final word may carry unused high bits."""

    @property
    def words_per_row(self) -> int:
        return self.num_groups * self.words_per_group

    @property
    def is_per_row(self) -> bool:
        return self.group_size == PER_ROW_GROUP_SIZE


def row_geometry(bits: int, group_size: int, in_features: int) -> RowGeometry:
    """Resolve the packed geometry of a row. Validates as a side effect."""
    group_size = checked_group_size(group_size, in_features, bits)
    padded = padded_in_features(in_features, group_size)

    if group_size == PER_ROW_GROUP_SIZE:
        # One group covering the row. `words_per_row` rounds up rather than
        # demanding `in_features * bits % 32 == 0`, which is legitimate here and
        # nowhere else: there is no group after this one whose start could be
        # knocked off a word boundary by the slack.
        return RowGeometry(
            group_size=PER_ROW_GROUP_SIZE,
            effective_group=in_features,
            padded_in_features=padded,
            num_groups=1,
            words_per_group=words_per_row(bits, in_features),
        )

    return RowGeometry(
        group_size=group_size,
        effective_group=group_size,
        padded_in_features=padded,
        num_groups=padded // group_size,
        words_per_group=words_per_group(bits, group_size),
    )


def stored_bits(
    rows: int,
    in_features: int,
    bits: int,
    *,
    group_size: int,
    symmetric: bool = False,
    metadata_bits: int = 16,
) -> int:
    """Bits a packed tensor of ``rows x in_features`` actually occupies.

    The arithmetic counterpart of :attr:`~dynquant.quant.tensor.QuantTensor.nbytes`,
    for callers that must price a tensor before quantizing it -- the allocator, which
    compares widths it will never all materialise. The two must agree exactly, and a
    test asserts they do at every real phase-3 shape.

    Counting ``rows * in_features * bits`` instead is wrong wherever ``in_features``
    is not a whole number of groups, because :func:`padded_in_features` zero-fills
    the tail group *before* quantization and those pad values are packed like any
    other. The undercount is 1.1% for Gemma-3's vision ``fc2`` (4304 -> 4352) and
    4.8x for its ``patch_embedding``, whose ``[1152, 3, 14, 14]`` folds to 14 columns
    and pads to 128. A budget built on the smaller number targets a size the writer
    cannot hit.
    """
    if rows <= 0 or in_features <= 0:
        return 0
    geo = row_geometry(bits, group_size, in_features)
    terms = 1 if symmetric else 2  # a scale, plus an offset when asymmetric
    return rows * (geo.words_per_row * _WORD_BITS + geo.num_groups * metadata_bits * terms)


# --------------------------------------------------------------------------
# Bit-offset tables
# --------------------------------------------------------------------------


class _BitLayout(NamedTuple):
    """Precomputed per-value word/shift indices for one (bits, num_values).

    Building these costs a few microseconds and they are reused for every row of
    every layer, so they are cached. All tensors are 1-D of length
    ``num_values``.
    """

    lo_word: torch.Tensor  # int64: word holding the value's low bits
    lo_shift: torch.Tensor  # int64: bit offset within that word
    lo_bits: torch.Tensor  # int64: how many bits fit before the word ends
    hi_word: torch.Tensor  # int64: word holding the remainder (== lo_word if none)
    lo_mask: torch.Tensor  # int64: (1 << lo_bits) - 1
    hi_mask: torch.Tensor  # int64: (1 << (bits - lo_bits)) - 1, zero if no straddle
    num_words: int


@functools.lru_cache(maxsize=64)
def _layout_cpu(bits: int, num_values: int) -> _BitLayout:
    idx = torch.arange(num_values, dtype=torch.int64)
    bit_offset = idx * bits
    num_words = words_per_row(bits, num_values)

    lo_word = bit_offset // _WORD_BITS
    lo_shift = bit_offset % _WORD_BITS
    # Bits of this value that fit in lo_word before it ends.
    lo_bits = torch.clamp(_WORD_BITS - lo_shift, max=bits)
    hi_bits = bits - lo_bits

    # Values that do not straddle point hi_word at their own word; their high
    # contribution is `q >> bits`, which is 0 because q < 2**bits. That makes the
    # straddling case branch-free instead of a separate masked path.
    hi_word = torch.where(hi_bits > 0, lo_word + 1, lo_word)
    hi_word = torch.clamp(hi_word, max=num_words - 1)

    return _BitLayout(
        lo_word=lo_word,
        lo_shift=lo_shift,
        lo_bits=lo_bits,
        hi_word=hi_word,
        lo_mask=(1 << lo_bits) - 1,
        hi_mask=(1 << hi_bits) - 1,
        num_words=num_words,
    )


@functools.lru_cache(maxsize=256)
def _layout(bits: int, num_values: int, device: torch.device) -> _BitLayout:
    base = _layout_cpu(bits, num_values)
    if device.type == "cpu":
        return base
    # Field by field rather than `*(t.to(device) for t in base[:-1])`: the slice
    # form silently depends on `num_words` being last, so adding a field would move
    # a tensor into the int slot and produce a confusing shape error far from here.
    return _BitLayout(
        lo_word=base.lo_word.to(device, non_blocking=True),
        lo_shift=base.lo_shift.to(device, non_blocking=True),
        lo_bits=base.lo_bits.to(device, non_blocking=True),
        hi_word=base.hi_word.to(device, non_blocking=True),
        lo_mask=base.lo_mask.to(device, non_blocking=True),
        hi_mask=base.hi_mask.to(device, non_blocking=True),
        num_words=base.num_words,
    )


# --------------------------------------------------------------------------
# Pack / unpack
# --------------------------------------------------------------------------


def pack_nbit(q: torch.Tensor, bits: int, *, validate: bool = True) -> torch.Tensor:
    """Pack unsigned ``bits``-wide codes along the last dimension into int32 words.

    Args:
        q: Integer codes in ``[0, 2**bits)``. Any integer dtype, any leading
            dims. The last dimension is the one packed.
        bits: 2, 3, 4 or 8.
        validate: Check the value range. Costs one reduction; leave on outside
            of hot loops -- an out-of-range code corrupts its *neighbours* in the
            same word, so it must never pass silently.

    Returns:
        int32 tensor of shape ``(*q.shape[:-1], words_per_row(bits, n))``.

    The packing is a sum rather than an OR because each value owns a disjoint bit
    range within a word, which makes ``index_add_`` (vectorized, one kernel)
    equivalent to a scatter-OR that torch does not provide.
    """
    if bits not in BIT_OPTIONS:
        raise PackingError(f"bits={bits} not supported; expected one of {list(BIT_OPTIONS)}")
    if q.ndim == 0:
        raise PackingError("cannot pack a 0-dim tensor")

    num_values = q.shape[-1]
    if num_values == 0:
        return torch.zeros((*q.shape[:-1], 0), dtype=torch.int32, device=q.device)

    qi = q.to(torch.int64)
    if validate:
        lo, hi = int(qi.min()), int(qi.max())
        if lo < 0 or hi >= (1 << bits):
            raise PackingError(
                f"codes out of range for {bits}-bit packing: found [{lo}, {hi}], "
                f"need [0, {(1 << bits) - 1}]. An out-of-range code silently "
                f"corrupts the other values sharing its 32-bit word."
            )

    lay = _layout(bits, num_values, q.device)
    words = torch.zeros((*q.shape[:-1], lay.num_words), dtype=torch.int64, device=q.device)
    words.index_add_(-1, lay.lo_word, qi << lay.lo_shift)
    words.index_add_(-1, lay.hi_word, qi >> lay.lo_bits)
    return _to_int32(words)


def unpack_nbit(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    """Inverse of :func:`pack_nbit`.

    Returns uint8 codes of shape ``(*packed.shape[:-1], num_values)``. This is
    the reference implementation the CUDA dequant kernels are tested against;
    it must stay exact and readable rather than fast.
    """
    if bits not in BIT_OPTIONS:
        raise PackingError(f"bits={bits} not supported; expected one of {list(BIT_OPTIONS)}")
    expected = words_per_row(bits, num_values)
    if packed.shape[-1] != expected:
        raise PackingError(
            f"packed last dim is {packed.shape[-1]}, expected {expected} words for "
            f"{num_values} values at {bits}-bit"
        )
    if num_values == 0:
        return torch.zeros((*packed.shape[:-1], 0), dtype=torch.uint8, device=packed.device)

    lay = _layout(bits, num_values, packed.device)
    w = packed.to(torch.int64) & MASK32
    lo = (w.index_select(-1, lay.lo_word) >> lay.lo_shift) & lay.lo_mask
    hi = (w.index_select(-1, lay.hi_word) & lay.hi_mask) << lay.lo_bits
    return (lo | hi).to(torch.uint8)


def _to_int32(words: torch.Tensor) -> torch.Tensor:
    """Reinterpret int64 word values in ``[0, 2**32)`` as int32 bit patterns.

    Done explicitly rather than by relying on ``.to(torch.int32)`` truncation,
    whose behaviour on out-of-range input is not guaranteed across backends.
    """
    words = words & MASK32
    return torch.where(words >= (1 << 31), words - (1 << 32), words).to(torch.int32)
