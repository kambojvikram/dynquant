"""QuantTensor: error vs theory, honest accounting, explicit metadata.

The error tests compare against :func:`conftest.expected_rel_fro`, which predicts
the error from the dense weight with plain torch. A quantizer that silently
mis-strides its groups produces error far above prediction; a test that
accidentally measures nothing produces error far below it. Both fail.
"""

from __future__ import annotations

import pytest

from dynquant.constants import BIT_OPTIONS, PER_ROW_GROUP_SIZE
from dynquant.errors import PackingError
from dynquant.quant.tensor import QuantLayout, QuantTensor

from _oracle import assert_error_matches_theory  # type: ignore[import-not-found]


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_error_matches_quantization_theory(weight, bits, group_size):
    qt = QuantTensor.from_dense(weight, bits=bits, group_size=group_size)
    measured = qt.quantization_error(weight)["rel_fro"]
    assert_error_matches_theory(measured, weight, bits, group_size)


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_wider_is_strictly_better(weight, bits):
    """Each extra bit must roughly halve the error. Catches a stuck grid."""
    qt = QuantTensor.from_dense(weight, bits=bits, group_size=128)
    err = qt.quantization_error(weight)["rel_fro"]
    if bits == max(BIT_OPTIONS):
        return
    wider = min(b for b in BIT_OPTIONS if b > bits)
    err_wider = QuantTensor.from_dense(weight, bits=wider, group_size=128).quantization_error(
        weight
    )["rel_fro"]
    assert err_wider < err, f"{wider}-bit ({err_wider:.5f}) not better than {bits}-bit ({err:.5f})"


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_bits_per_weight_is_honest(weight, bits, group_size):
    """The reported cost must include scales and offsets.

    The budget the user asks for is an on-disk budget. If ``bits_per_weight``
    reported the nominal width, a "3-bit" model would land at 3.25 bits and the
    manifest would disagree with ``os.path.getsize``.
    """
    qt = QuantTensor.from_dense(weight, bits=bits, group_size=group_size)
    # two fp16 values (scale, offset) per group, amortised over group_size weights
    expected = bits + 32.0 / group_size
    assert qt.bits_per_weight == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_nbytes_matches_actual_tensor_storage(weight, bits):
    qt = QuantTensor.from_dense(weight, bits=bits, group_size=128)
    actual = qt.packed.numel() * qt.packed.element_size() + qt.scales.numel() * (
        qt.scales.element_size()
    )
    if qt.offsets is not None:
        actual += qt.offsets.numel() * qt.offsets.element_size()
    assert qt.nbytes == actual


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_constant_groups_are_exact(torch_seeded, bits):
    """A constant group folds into the offset with scale 0 -- zero error.

    Worth pinning: it removes the research code's separate ``is_constant`` /
    ``constant_value`` special case, which every kernel would otherwise have to
    branch on.
    """
    torch = torch_seeded
    w = torch.zeros(4, 256, dtype=torch.float16)
    w[1, :] = 0.75
    w[2, :128] = -2.0
    qt = QuantTensor.from_dense(w, bits=bits, group_size=128)
    recon = qt.dequantize(dtype=torch.float32)
    assert torch.equal(recon, w.float())


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_zeros_are_interior_grid_points(torch_seeded, bits):
    """A zero weight must round like any other interior value -- never be clipped.

    Zero is *not* bit-exact in this format and deliberately so: ``offset`` is an
    unconstrained float, so the grid is not anchored to zero. That is what lets a
    one-signed group keep its full resolution (see the next test), and it costs
    only that a reconstructed zero lands within half a step instead of on zero.

    What must not happen is a zero being *clipped* to a group endpoint, which is
    what an out-of-range offset would cause and which would show up here as an
    error of many steps rather than a fraction of one. Half a step plus a storage
    ulp is the whole budget.
    """
    torch = torch_seeded
    w = torch.randn(8, 256, dtype=torch.float16) * 0.02
    w[:, ::4] = 0.0
    qt = QuantTensor.from_dense(w, bits=bits, group_size=128)
    recon = qt.dequantize(dtype=torch.float32)

    zeros = recon[:, ::4].abs().max().item()
    step = (qt.scales.to(torch.float32).abs().max()).item()
    assert zeros <= 0.51 * step, f"zeros reconstructed at {zeros:.3e}, {zeros / step:.3f} of a step"


def test_all_zero_group_is_exactly_zero(torch_seeded):
    """The case where exactness *does* matter reaches it through the constant fold.

    A fully-zero group -- a dead expert, a masked head, the padded tail of a
    non-multiple-of-group_size row -- is constant, so it folds to ``scale = 0``,
    ``offset = 0`` and reconstructs bit-exactly. That is why the interior-rounding
    of individual zeros above is acceptable.
    """
    torch = torch_seeded
    w = torch.randn(4, 256, dtype=torch.float16) * 0.02
    w[1, :128] = 0.0  # one whole group
    for bits in BIT_OPTIONS:
        recon = QuantTensor.from_dense(w, bits=bits, group_size=128).dequantize(dtype=torch.float32)
        assert torch.equal(recon[1, :128], torch.zeros(128)), f"{bits}-bit"


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_one_signed_group_does_not_waste_range_on_zero(torch_seeded, bits):
    """A group with no zeros in it must not have its range widened to include one.

    GPTQ and AWQ widen unconditionally because an integer zero-point is only
    invertible when it lands in ``[0, qmax]``. For a group spanning [0.50, 0.52]
    that inflates the range 26x and throws away more than four bits of resolution.
    The float offset here has no such constraint, so the measured error must still
    track theory -- which is computed from the group's *own* range.
    """
    torch = torch_seeded
    w = torch.rand(8, 256, dtype=torch.float16) * 0.02 + 0.5  # strictly positive
    qt = QuantTensor.from_dense(w, bits=bits, group_size=128)
    assert_error_matches_theory(qt.quantization_error(w)["rel_fro"], w, bits, 128, band=0.35)


@pytest.mark.parametrize("in_features", [300, 127, 1000])
def test_padded_in_features(torch_seeded, in_features):
    torch = torch_seeded
    w = torch.randn(8, in_features, dtype=torch.float16) * 0.02
    qt = QuantTensor.from_dense(w, bits=4, group_size=128)
    assert qt.in_features == in_features
    assert qt.padded_in_features % 128 == 0
    assert qt.padded_in_features >= in_features
    assert qt.dequantize().shape == w.shape
    # padding must not inflate the error beyond theory
    assert_error_matches_theory(qt.quantization_error(w)["rel_fro"], w, 4, 128)


def test_single_column_weight_survives(torch_seeded):
    """``in_features == 1`` pads to a full group of 127 zeros.

    Degenerate enough that the error oracle has nothing to predict -- the one
    real value per row is a range endpoint and quantizes exactly -- so only the
    geometry and the reconstruction are checked.
    """
    torch = torch_seeded
    w = torch.randn(8, 1, dtype=torch.float16) * 0.02
    qt = QuantTensor.from_dense(w, bits=4, group_size=128)
    assert qt.in_features == 1
    assert qt.padded_in_features == 128
    assert qt.dequantize().shape == w.shape
    assert qt.quantization_error(w)["rel_fro"] < 1e-3


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_stacked_experts_fold_to_rows(torch_seeded, bits):
    """``[E, out, in]`` expert stacks quantize as ``E*out`` rows and restore shape."""
    torch = torch_seeded
    we = torch.randn(4, 64, 512, dtype=torch.float16) * 0.02
    qt = QuantTensor.from_dense(we, bits=bits, group_size=128)
    assert qt.num_rows == 4 * 64
    assert qt.logical_shape == (4, 64, 512)
    assert qt.dequantize().shape == we.shape
    assert_error_matches_theory(qt.quantization_error(we)["rel_fro"], we, bits, 128)


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_conv1d_shape_survives(torch_seeded, bits):
    """Mamba's ``conv1d.weight`` is ``[channels, 1, kernel]`` -- ndim 3, tiny last dim.

    The research quantizer guessed ``group_size`` from ``scale.numel() //
    out_features``, which is meaningless here. Explicit metadata is the fix.

    Parametrised over every width on purpose. ``4 * bits`` is a whole number of
    32-bit words only at 8-bit, so a single-width test passed while 2-, 3- and
    4-bit raised out of ``words_per_group`` -- on the exact shape
    ``checked_group_size`` cites as the reason per-row grouping exists.
    """
    torch = torch_seeded
    w = torch.randn(512, 1, 4, dtype=torch.float16) * 0.1
    qt = QuantTensor.from_dense(w, bits=bits, group_size=PER_ROW_GROUP_SIZE)
    qt.validate()
    assert qt.dequantize().shape == w.shape
    assert qt.group_size == PER_ROW_GROUP_SIZE, "the sentinel must not be resolved away"
    assert qt.is_per_row
    assert qt.num_groups == 1
    assert qt.geometry.effective_group == 4


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("in_features", [4, 31, 100, 1024])
def test_per_row_tensors_round_trip_at_unaligned_widths(torch_seeded, bits, in_features):
    """The regression guard for the resolved-sentinel bug.

    ``from_dense`` used to store ``group_size = in_features``, so ``validate()``
    re-ran the alignment check against e.g. ``100`` and rejected it, and
    ``words_per_group`` raised for any ``in_features * bits`` that was not a
    multiple of 32. The encoder therefore produced per-row tensors that could not
    be loaded back -- silently, since nothing in the suite validated one.
    """
    torch = torch_seeded
    w = torch.randn(16, in_features, dtype=torch.float16) * 0.1
    qt = QuantTensor.from_dense(w, bits=bits, group_size=PER_ROW_GROUP_SIZE)
    qt.validate()

    restored = QuantTensor.from_state_dict("w", qt.state_dict("w"), qt.metadata())
    assert restored.group_size == PER_ROW_GROUP_SIZE
    assert restored.metadata() == qt.metadata()
    assert torch.equal(restored.dequantize(), qt.dequantize())
    assert restored.dequantize().shape == w.shape


def test_symmetric_mode_produces_no_offset(weight):
    qt = QuantTensor.from_dense(weight, bits=4, group_size=128, symmetric=True)
    assert qt.symmetric
    err = qt.quantization_error(weight)["rel_fro"]
    asym = QuantTensor.from_dense(weight, bits=4, group_size=128).quantization_error(weight)
    # symmetric throws away half the codes for a zero-mean weight, so it is worse
    # -- but only modestly, and it must still be finite and sane.
    assert 0.0 < err < 4 * asym["rel_fro"]


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_state_dict_roundtrip_preserves_metadata(weight, bits):
    """group_size must be *stored*, never re-derived (the bug-9 guard)."""
    qt = QuantTensor.from_dense(weight, bits=bits, group_size=64)
    prefix = "model.layers.0.mlp.down_proj"
    restored = QuantTensor.from_state_dict(prefix, qt.state_dict(prefix), qt.metadata())

    assert restored.group_size == 64
    assert restored.bits == bits
    assert restored.in_features == qt.in_features
    assert restored.logical_shape == qt.logical_shape
    assert restored.layout is QuantLayout.LINEAR
    assert restored.symmetric == qt.symmetric
    import torch

    assert torch.equal(restored.dequantize(), qt.dequantize())


def test_state_dict_keys_are_namespaced(weight):
    qt = QuantTensor.from_dense(weight, bits=4, group_size=128)
    keys = set(qt.state_dict("layer"))
    assert keys == {"layer.qweight", "layer.scales", "layer.offsets"}


def test_metadata_is_json_serialisable(weight):
    import json

    qt = QuantTensor.from_dense(weight, bits=3, group_size=128)
    assert json.loads(json.dumps(qt.metadata(), default=str)) is not None


def test_validate_catches_truncated_packed(weight):
    qt = QuantTensor.from_dense(weight, bits=4, group_size=128)
    qt.validate()  # the healthy case must not raise

    bad = QuantTensor(
        packed=qt.packed[:, :-1].contiguous(),
        scales=qt.scales,
        offsets=qt.offsets,
        bits=qt.bits,
        group_size=qt.group_size,
        in_features=qt.in_features,
        logical_shape=qt.logical_shape,
    )
    with pytest.raises(PackingError):
        bad.validate()


def test_validate_catches_scale_shape_mismatch(weight):
    qt = QuantTensor.from_dense(weight, bits=4, group_size=128)
    bad = QuantTensor(
        packed=qt.packed,
        scales=qt.scales[:, :-1].contiguous(),
        offsets=qt.offsets,
        bits=qt.bits,
        group_size=qt.group_size,
        in_features=qt.in_features,
        logical_shape=qt.logical_shape,
    )
    with pytest.raises(PackingError):
        bad.validate()


@pytest.mark.parametrize("bits", [1, 5, 6, 16, 0, -1])
def test_unsupported_bit_width_is_rejected(weight, bits):
    """Only widths every kernel is templated over may be produced."""
    with pytest.raises((PackingError, ValueError, KeyError)):
        QuantTensor.from_dense(weight, bits=bits, group_size=128)


def test_row_offset_records_shard_position(weight):
    """Fused projections split into row shards; each must know where it sits."""
    qt = QuantTensor.from_dense(weight[:128], bits=4, group_size=128, row_offset=0)
    hi = QuantTensor.from_dense(weight[128:], bits=3, group_size=128, row_offset=128)
    assert qt.row_offset == 0
    assert hi.row_offset == 128
    assert qt.bits != hi.bits, "shards of one projection may differ in width"


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_clip_ratio_is_a_real_knob(weight, bits):
    """``clip_ratio`` must change the encoding, and 1.0 must be the identity.

    Whether clipping *helps* is a property of the weight and the width, not an
    invariant: at 8 bits the grid already resolves the bulk, so discarding range
    can only hurt, and the phase-5 grid search is expected to pick 1.0 there. The
    thing to pin is that the knob is wired up and that its neutral value is
    neutral -- a silently ignored clip_ratio would make the whole search a no-op
    that still reported an "optimal" alpha.
    """
    base = QuantTensor.from_dense(weight, bits=bits, group_size=128)
    identity = QuantTensor.from_dense(weight, bits=bits, group_size=128, clip_ratio=1.0)
    clipped = QuantTensor.from_dense(weight, bits=bits, group_size=128, clip_ratio=0.85)

    import torch

    assert torch.equal(identity.packed, base.packed), "clip_ratio=1.0 must be the identity"
    assert not torch.equal(clipped.packed, base.packed), "clip_ratio had no effect"
    assert clipped.quantization_error(weight)["rel_fro"] < 1.0


# -- adopting a foreign grid -------------------------------------------------


def _foreign_grid(rows: int, in_features: int, group_size: int, bits: int, *, occupies: tuple):
    """Codes on a grid whose occupied range is ``occupies``, with per-group scale/offset.

    ``occupies`` is the inclusive code range the values actually use. A foreign quantizer
    that clipped, or that moved weights after fixing its scale, leaves groups that do not
    reach both ends -- which is the case this constructor exists for.
    """
    import torch

    groups = in_features // group_size
    low, high = occupies
    generator = torch.Generator().manual_seed(11)
    codes = torch.randint(
        low, high + 1, (rows, in_features), generator=generator, dtype=torch.int64
    )
    # Pin the extremes of the occupied band so the band is exactly `occupies`, not a
    # sample of it -- otherwise the test's premise depends on the seed.
    codes[:, 0] = low
    codes[:, 1] = high
    scales = torch.rand(rows, groups, generator=generator) * 0.02 + 0.01
    offsets = torch.rand(rows, groups, generator=generator) * 0.5 - 0.25
    return codes, scales.to(torch.float32), offsets.to(torch.float32)


def test_from_codes_reproduces_the_grid_it_was_given():
    import torch

    bits, group_size, rows, in_features = 4, 32, 6, 128
    codes, scales, offsets = _foreign_grid(rows, in_features, group_size, bits, occupies=(0, 15))
    qt = QuantTensor.from_codes(
        codes, scales, offsets, bits=bits, group_size=group_size, compute_dtype=torch.float32
    )
    qt.validate()
    expected = scales.repeat_interleave(group_size, dim=1) * codes.to(
        torch.float32
    ) + offsets.repeat_interleave(group_size, dim=1)
    assert torch.equal(qt.dequantize(dtype=torch.float32), expected)


def test_refitting_a_partly_occupied_grid_moves_the_weights_and_from_codes_does_not():
    """The whole reason this constructor exists, stated as a test.

    A group that no longer occupies both ends of its code range -- what GPTQ's error
    compensation and AWQ's clipping search both leave behind -- has a min/max range
    narrower than the grid that produced it. Re-deriving the map from the dequantized
    values therefore lands on a *finer* step, the original levels fall between the new
    ones, and the weights move. ``from_codes`` is exact because it never re-derives.

    If someone replaces ``from_codes`` with a ``from_dense`` call, this goes red.
    """
    import torch

    bits, group_size, rows, in_features = 4, 32, 6, 128
    codes, scales, offsets = _foreign_grid(rows, in_features, group_size, bits, occupies=(3, 12))
    dense = scales.repeat_interleave(group_size, dim=1) * codes.to(
        torch.float32
    ) + offsets.repeat_interleave(group_size, dim=1)

    carried = QuantTensor.from_codes(
        codes, scales, offsets, bits=bits, group_size=group_size, compute_dtype=torch.float32
    )
    refitted = QuantTensor.from_dense(
        dense, bits=bits, group_size=group_size, compute_dtype=torch.float32
    )

    assert torch.equal(carried.dequantize(dtype=torch.float32), dense)
    assert not torch.equal(refitted.dequantize(dtype=torch.float32), dense)
    # And the drift is real rather than a last-ulp artifact: a partly-occupied 4-bit band
    # re-fitted onto 16 levels misplaces most of them by a sizeable fraction of a step.
    step = scales.repeat_interleave(group_size, dim=1)
    drift = (refitted.dequantize(dtype=torch.float32) - dense).abs() / step
    assert drift.max() > 0.1, f"expected a visible re-fit drift, got {drift.max():.4f} steps"


def test_a_zero_point_convention_translates_onto_the_same_grid():
    """``scale * (q - zero)`` is ``scale * code + offset`` with ``offset = -scale * zero``.

    Exact as a grid, not as a bit pattern. The codes carried are the codes given -- that is
    what determines the bytes on disk, and it is checked exactly. The reconstruction differs
    in the last place because ``s * q + (-s * z)`` and ``s * (q - z)`` round differently in
    fp32, which is a property of the arithmetic and not of the import: ~1e-7 relative,
    against a quantization step of ~6e-2 here, and smaller than the bf16 rounding both
    conventions undergo when the checkpoint is written.
    """
    import torch

    bits, group_size, rows, in_features = 4, 32, 4, 64
    groups = in_features // group_size
    generator = torch.Generator().manual_seed(5)
    q = torch.randint(0, 1 << bits, (rows, in_features), generator=generator, dtype=torch.int64)
    scale = (torch.rand(rows, groups, generator=generator) * 0.02 + 0.01).to(torch.float32)
    zero = torch.randint(0, 1 << bits, (rows, groups), generator=generator).to(torch.float32)

    qt = QuantTensor.from_codes(
        q, scale, -scale * zero, bits=bits, group_size=group_size, compute_dtype=torch.float32
    )
    foreign = scale.repeat_interleave(group_size, dim=1) * (
        q.to(torch.float32) - zero.repeat_interleave(group_size, dim=1)
    )
    from dynquant.quant.pack import unpack_nbit

    recovered = unpack_nbit(qt.packed, bits, in_features).reshape(rows, in_features)
    assert torch.equal(recovered.to(torch.int64), q), "the codes on disk are not the codes given"

    got = qt.dequantize(dtype=torch.float32)
    step = scale.repeat_interleave(group_size, dim=1)
    assert ((got - foreign).abs() / step).max() < 1e-5


def test_from_codes_pads_a_ragged_row_without_touching_the_given_scales():
    import torch

    bits, group_size, rows, in_features = 4, 32, 3, 100  # 100 is not a multiple of 32
    groups = -(-in_features // group_size)
    generator = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 1 << bits, (rows, in_features), generator=generator, dtype=torch.int64)
    scales = torch.ones(rows, groups, dtype=torch.float32)
    offsets = torch.zeros(rows, groups, dtype=torch.float32)

    qt = QuantTensor.from_codes(
        codes, scales, offsets, bits=bits, group_size=group_size, compute_dtype=torch.float32
    )
    qt.validate()
    assert qt.in_features == in_features
    assert torch.equal(qt.dequantize(dtype=torch.float32), codes.to(torch.float32))


def test_from_codes_records_a_stacked_bank_shape():
    import torch

    bits, group_size, experts, out_features, in_features = 4, 32, 3, 8, 64
    codes = torch.zeros(experts, out_features, in_features, dtype=torch.int64)
    rows, groups = experts * out_features, in_features // group_size
    qt = QuantTensor.from_codes(
        codes,
        torch.ones(rows, groups),
        torch.zeros(rows, groups),
        bits=bits,
        group_size=group_size,
    )
    qt.validate()
    assert qt.logical_shape == (experts, out_features, in_features)
    assert qt.num_rows == rows


def test_from_codes_refuses_codes_outside_the_width():
    import torch

    with pytest.raises(PackingError, match="outside the 3-bit range"):
        QuantTensor.from_codes(
            torch.tensor([[0, 8]], dtype=torch.int64),
            torch.ones(1, 1),
            torch.zeros(1, 1),
            bits=3,
            group_size=PER_ROW_GROUP_SIZE,
        )


def test_from_codes_refuses_float_codes():
    """Passing dequantized weights here is the mistake worth an exception, not a round."""
    import torch

    with pytest.raises(PackingError, match="must be an integer tensor"):
        QuantTensor.from_codes(
            torch.tensor([[0.0, 1.0]]),
            torch.ones(1, 1),
            torch.zeros(1, 1),
            bits=4,
            group_size=PER_ROW_GROUP_SIZE,
        )


def test_from_codes_refuses_scales_that_do_not_match_the_grouping():
    import torch

    with pytest.raises(PackingError, match="scales shape"):
        QuantTensor.from_codes(
            torch.zeros(4, 64, dtype=torch.int64),
            torch.ones(4, 1),  # 64 wide at group 32 is two groups, not one
            torch.zeros(4, 2),
            bits=4,
            group_size=32,
        )
