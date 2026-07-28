"""Bit packing: bijection, word arithmetic, group independence.

Packing is the one place in DynQuant where a bug is completely silent. A wrong
shift does not raise -- it returns a tensor of the right shape full of wrong
numbers, and every downstream metric degrades a little without anything pointing
at the cause. So the invariants are tested exhaustively rather than
representatively.
"""

from __future__ import annotations

import pytest

from dynquant.constants import BIT_OPTIONS, GROUP_SIZE_ALIGNMENT, PER_ROW_GROUP_SIZE
from dynquant.errors import PackingError
from dynquant.quant.pack import (
    checked_group_size,
    pack_nbit,
    padded_in_features,
    row_geometry,
    unpack_nbit,
    words_per_group,
    words_per_row,
)

# Sizes chosen to break alignment assumptions: prime, off-by-one either side of a
# word boundary, and 1 (a single value straddling nothing).
AWKWARD_SIZES = [1, 2, 3, 7, 8, 31, 32, 33, 37, 96, 127, 128, 129, 1023]


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("n", AWKWARD_SIZES)
def test_roundtrip_is_bijective(torch_seeded, bits, n):
    torch = torch_seeded
    q = torch.randint(0, 2**bits, (5, n), dtype=torch.uint8)
    packed = pack_nbit(q, bits)
    assert torch.equal(unpack_nbit(packed, bits, n), q)


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_roundtrip_rank3(torch_seeded, bits):
    """Stacked MoE expert weights arrive as ``[E, out, in]``."""
    torch = torch_seeded
    q = torch.randint(0, 2**bits, (3, 4, 128), dtype=torch.uint8)
    packed = pack_nbit(q, bits)
    assert packed.shape[:-1] == q.shape[:-1]
    assert torch.equal(unpack_nbit(packed, bits, 128), q)


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("fill", ["min", "max"])
def test_extremal_codes_survive(torch_seeded, bits, fill):
    """All-zeros and all-max exercise every bit position at once.

    All-max is the one that catches sign-extension bugs: the top bit of the top
    word is set, and int32 storage makes that a negative number.
    """
    torch = torch_seeded
    value = 0 if fill == "min" else 2**bits - 1
    q = torch.full((2, 256), value, dtype=torch.uint8)
    assert torch.equal(unpack_nbit(pack_nbit(q, bits), bits, 256), q)


@pytest.mark.parametrize("bits", [b for b in BIT_OPTIONS if b < 8])
def test_out_of_range_code_is_rejected(torch_seeded, bits):
    """A code that does not fit corrupts its word-mates, so it must not pass.

    Only checked below 8 bits: a uint8 code physically cannot exceed the 8-bit
    range, so there is nothing to reject there.
    """
    torch = torch_seeded
    q = torch.zeros((1, 64), dtype=torch.uint8)
    q[0, 5] = 2**bits  # one too large
    with pytest.raises(PackingError, match="range"):
        pack_nbit(q, bits)


@pytest.mark.parametrize(
    ("bits", "expected_words"),
    [(2, 8), (3, 12), (4, 16), (8, 32)],
)
def test_word_count_for_default_group(bits, expected_words):
    """A 128-value group must occupy a whole number of 32-bit words."""
    assert words_per_group(bits, 128) == expected_words
    assert expected_words * 32 == 128 * bits


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_every_aligned_group_is_word_exact(bits, group_size):
    """The invariant the kernels depend on: no group ever starts mid-word."""
    assert group_size % GROUP_SIZE_ALIGNMENT == 0
    assert words_per_group(bits, group_size) * 32 == group_size * bits


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_group_independence(torch_seeded, bits):
    """Rewriting one group must not touch any other group's words.

    This is what lets a kernel load exactly the words for the groups it needs.
    If packing leaked across boundaries, a tiled kernel would have to read
    neighbouring data to decode its own tile.
    """
    torch = torch_seeded
    group_size = 128
    q = torch.randint(0, 2**bits, (1, group_size * 4), dtype=torch.uint8)
    base = pack_nbit(q, bits)

    wpg = words_per_group(bits, group_size)
    q2 = q.clone()
    q2[0, group_size * 2 : group_size * 3] = torch.randint(
        0, 2**bits, (group_size,), dtype=torch.uint8
    )
    changed = pack_nbit(q2, bits)

    touched = slice(wpg * 2, wpg * 3)
    untouched_before = slice(0, wpg * 2)
    untouched_after = slice(wpg * 3, wpg * 4)
    assert torch.equal(base[:, untouched_before], changed[:, untouched_before])
    assert torch.equal(base[:, untouched_after], changed[:, untouched_after])
    # sanity: the group we did change actually changed
    assert not torch.equal(base[:, touched], changed[:, touched])


def test_three_bit_straddlers_are_where_theory_says(torch_seeded):
    """3-bit is the only width whose values cross word boundaries.

    32 values x 3 bits = 96 bits = exactly 3 words, so within each 3-word block
    the values at indices 10 and 21 straddle (bits 30-32 and 63-65). Any other
    answer means the layout drifted from the format spec, which would break every
    3-bit kernel while leaving the round-trip test green.
    """
    torch = torch_seeded
    n = 32
    straddlers = []
    for i in range(n):
        q = torch.zeros((1, n), dtype=torch.uint8)
        q[0, i] = 7
        words = pack_nbit(q, 3)
        if int((words != 0).sum()) > 1:
            straddlers.append(i)
    assert straddlers == [10, 21]


@pytest.mark.parametrize(
    ("in_features", "group_size", "expected"),
    [
        (1024, 128, 1024),
        (300, 128, 384),
        (128, 128, 128),
        (1, 128, 128),
        (129, 32, 160),
        # Per-row pads no values at all -- the row rounds up to whole words instead.
        (100, PER_ROW_GROUP_SIZE, 100),
        (4, PER_ROW_GROUP_SIZE, 4),
    ],
)
def test_padding_rounds_up_to_group(in_features, group_size, expected):
    assert padded_in_features(in_features, group_size) == expected


@pytest.mark.parametrize("group_size", [0, 31, 48, 100, 127, -5])
def test_misaligned_group_size_is_rejected(group_size):
    with pytest.raises(PackingError):
        checked_group_size(group_size, 1024, 4)


def test_per_row_group_size_is_allowed():
    """``-1`` means one group spanning the whole row (the embedding path)."""
    assert checked_group_size(PER_ROW_GROUP_SIZE, 1024, 4) == PER_ROW_GROUP_SIZE


def test_per_row_sentinel_is_never_resolved_away():
    """The sentinel must survive, not become ``in_features``.

    It is the only record that this tensor is exempt from the 32-alignment rule.
    Storing the resolved width instead made the exemption unenforceable: every
    later check re-ran ``checked_group_size`` on the resolved value, took the
    alignment branch, and rejected a tensor the encoder had just produced.
    """
    assert checked_group_size(PER_ROW_GROUP_SIZE, 100, 4) == PER_ROW_GROUP_SIZE
    geom = row_geometry(4, PER_ROW_GROUP_SIZE, 100)
    assert geom.group_size == PER_ROW_GROUP_SIZE
    assert geom.is_per_row
    assert geom.effective_group == 100  # resolved only for arithmetic


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("in_features", [1, 4, 31, 100, 1024, 5120])
def test_per_row_geometry_is_self_consistent(bits, in_features):
    """Per-row rounds up to whole *words*, never to whole values.

    ``in_features`` here is deliberately mostly non-multiples of 32 -- including
    4, the Mamba ``conv1d`` kernel width that ``checked_group_size`` names as its
    motivating case and that the aligned rule would reject outright.
    """
    geom = row_geometry(bits, PER_ROW_GROUP_SIZE, in_features)
    assert geom.num_groups == 1
    assert geom.padded_in_features == in_features, "per-row must not pad values"
    assert geom.words_per_group == words_per_row(bits, in_features)
    assert geom.words_per_row == geom.words_per_group
    # Enough bits to hold every value, and no more than one word of slack.
    assert geom.words_per_row * 32 >= in_features * bits
    assert (geom.words_per_row - 1) * 32 < in_features * bits


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("group_size", [32, 128])
def test_aligned_geometry_agrees_with_the_standalone_helpers(bits, group_size):
    """``row_geometry`` must not become a second, divergent source of sizes."""
    geom = row_geometry(bits, group_size, 300)
    assert geom.padded_in_features == padded_in_features(300, group_size)
    assert geom.words_per_group == words_per_group(bits, group_size)
    assert geom.num_groups == geom.padded_in_features // group_size
    assert not geom.is_per_row
    assert geom.effective_group == group_size


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_words_per_row_matches_bit_budget(bits):
    for n in AWKWARD_SIZES:
        words = words_per_row(bits, n)
        assert words * 32 >= n * bits, "layout must not lose bits"
        assert (words - 1) * 32 < n * bits, "layout must not waste a whole word"
