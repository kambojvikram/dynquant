"""The clipping search and the bit-map driver.

The load-bearing property here is that the search cannot make things worse. It
scores candidates with the real encoder, so its answer is only meaningful if pure
min/max is one of the candidates it is allowed to pick -- otherwise a tensor with no
outliers gets clipped for nothing and every layer pays a little accuracy that no
measurement afterwards can attribute back to this step.
"""

from __future__ import annotations

import pytest
import torch

from dynquant.errors import DynQuantError
from dynquant.quant.grid import CLIP_CANDIDATES, quantize_with_search, search_clip_ratios
from dynquant.quant.quantizer import quantize_model
from dynquant.quant.tensor import QuantTensor


def _outlier_weight(rows: int = 4, cols: int = 256, seed: int = 0) -> torch.Tensor:
    """Well-behaved bulk plus one large outlier per group -- the case clipping is for."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator) * 0.02
    weight[:, ::128] = 3.0  # one dominant value in every group of 128
    return weight


# --------------------------------------------------------------------------
# Per-group clip ratios in the encoder
# --------------------------------------------------------------------------


def test_a_scalar_ratio_of_one_is_untouched_min_max() -> None:
    """The default path must not change when per-group support is added.

    At ratio 1.0 the clip formula is an algebraic no-op but not a float no-op, so
    routing the default through it would shift every group's bounds by an ulp.
    """
    weight = _outlier_weight()
    plain = QuantTensor.from_dense(weight, bits=4, group_size=128)
    explicit = QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=1.0)
    assert torch.equal(plain.dequantize(), explicit.dequantize())


def test_a_uniform_ratio_tensor_matches_the_scalar() -> None:
    weight = _outlier_weight()
    rows, groups = 4, 2
    scalar = QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=0.9)
    per_group = QuantTensor.from_dense(
        weight, bits=4, group_size=128, clip_ratio=torch.full((rows, groups), 0.9)
    )
    assert torch.equal(scalar.dequantize(), per_group.dequantize())


def test_per_group_ratios_act_independently() -> None:
    """Two groups given different ratios must not influence each other."""
    weight = _outlier_weight()
    mixed = torch.tensor([[1.0, 0.8]] * 4)
    combined = QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=mixed).dequantize()

    left = QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=1.0).dequantize()
    right = QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=0.8).dequantize()

    assert torch.equal(combined[:, :128], left[:, :128])
    assert torch.equal(combined[:, 128:], right[:, 128:])


def test_a_wrongly_shaped_ratio_tensor_is_rejected() -> None:
    weight = _outlier_weight()
    with pytest.raises(Exception, match="one ratio per group"):
        QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=torch.ones(4, 7))


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_ratios_outside_the_unit_interval_are_rejected(ratio: float) -> None:
    weight = _outlier_weight()
    with pytest.raises(Exception, match="clip_ratio"):
        QuantTensor.from_dense(weight, bits=4, group_size=128, clip_ratio=ratio)


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_the_search_never_loses_to_pure_min_max(bits: int) -> None:
    """The whole point: 1.0 is a candidate, so the result is at worst a tie."""
    weight = _outlier_weight()
    result = search_clip_ratios(weight, bits=bits, group_size=128)
    assert torch.all(result.best_mse <= result.baseline_mse + 1e-9)
    assert result.improvement >= 0.0


def test_the_search_finds_clipping_worth_doing_on_outlier_data() -> None:
    """A group whose range is set by one value at 150x the bulk should clip."""
    weight = _outlier_weight()
    result = search_clip_ratios(weight, bits=3, group_size=128)
    assert result.clipped_fraction > 0.0, "no group preferred clipping on outlier-dominated data"
    assert result.improvement > 0.0


def test_clean_data_mostly_declines_to_clip() -> None:
    """Without outliers, shrinking the range is a straight loss and must be refused."""
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(4, 256, generator=generator)
    result = search_clip_ratios(weight, bits=8, group_size=128)
    assert result.clipped_fraction < 0.5


def test_the_returned_ratios_reproduce_the_reported_error() -> None:
    """Re-encoding with the winners must give back exactly the error that won.

    If these disagree, the search is optimising one encoder and the checkpoint is
    written by another -- which is the defect this search was rewritten to avoid.
    """
    weight = _outlier_weight()
    quantized, result = quantize_with_search(weight, bits=3, group_size=128)
    recon = quantized.dequantize(dtype=torch.float32)
    actual = ((weight.to(torch.float32) - recon) ** 2).reshape(4, 2, 128).sum(dim=-1)
    assert torch.allclose(actual, result.best_mse, rtol=1e-5, atol=1e-8)


def test_a_single_candidate_degenerates_to_that_ratio() -> None:
    weight = _outlier_weight()
    result = search_clip_ratios(weight, bits=4, group_size=128, candidates=[0.85])
    assert torch.all(result.ratios == 0.85)


def test_an_empty_candidate_list_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        search_clip_ratios(_outlier_weight(), bits=4, candidates=[])


def test_the_supplements_grid_is_preserved() -> None:
    """``paper-3.15`` reproduces published numbers, so the grid is part of the contract."""
    assert CLIP_CANDIDATES == (1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80)


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


class _Tiny(torch.nn.Module):
    def __init__(self, tie: bool = False) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(256, 128)
        self.fc = torch.nn.Linear(128, 256, bias=False)
        self.head = torch.nn.Linear(128, 256, bias=False)
        if tie:
            self.head.weight = self.embed.weight


def test_the_driver_quantizes_every_named_module() -> None:
    model = _Tiny()
    report = quantize_model(model, {"fc": 4, "head": 8}, group_size=128)
    assert set(report.layers) == {"fc", "head"}
    assert report.layers["fc"].bits == 4
    assert all(r.relative_error > 0 for r in report.layers.values())


def test_more_bits_means_less_error() -> None:
    """The direction of the whole method. If this inverts, nothing downstream means
    anything, and it would still produce a plausible-looking results table."""
    errors = {}
    for bits in (2, 3, 4, 8):
        model = _Tiny()
        torch.nn.init.normal_(model.fc.weight, std=0.02)
        report = quantize_model(model, {"fc": bits}, group_size=128)
        errors[bits] = report.layers["fc"].relative_error
    assert errors[2] > errors[3] > errors[4] > errors[8]


def test_in_place_writes_the_dequantized_values_back() -> None:
    """Exactly the encoder's reconstruction, not something close to it.

    Accuracy measured on an in-place model is only the accuracy of the packed
    checkpoint if these are the same numbers to the last bit.
    """
    model = _Tiny()
    before = model.fc.weight.detach().clone()
    quantize_model(model, {"fc": 3}, group_size=128)
    after = model.fc.weight.detach()
    assert not torch.equal(before, after)

    quantized, _ = quantize_with_search(before, bits=3, group_size=128)
    expected = quantized.dequantize(dtype=torch.float32).to(before.dtype)
    assert torch.equal(after, expected)


def test_in_place_false_leaves_the_model_alone() -> None:
    model = _Tiny()
    before = model.fc.weight.detach().clone()
    report = quantize_model(model, {"fc": 3}, group_size=128, in_place=False)
    assert torch.equal(before, model.fc.weight.detach())
    assert report.layers["fc"].relative_error > 0


def test_a_tied_weight_is_quantized_once_and_seen_everywhere() -> None:
    """Quantizing under one name must reach the alias without a second pass."""
    model = _Tiny(tie=True)
    torch.nn.init.normal_(model.embed.weight, std=0.02)
    assert model.head.weight is model.embed.weight

    report = quantize_model(model, {"embed": 3}, group_size=128)
    assert set(report.layers) == {"embed"}
    assert model.head.weight is model.embed.weight, "copy_ was replaced by assignment; tie broken"
    assert torch.equal(model.head.weight.detach(), model.embed.weight.detach())


def test_re_encoding_at_the_same_width_changes_nothing() -> None:
    """Idempotence, which bounds how bad a duplicated bit-map entry can be.

    The reconstructed values already sit exactly on the grid, so min/max recovers the
    same range and ``round`` the same codes. Worth pinning: it is the reason a tied
    pair listed twice at one width is merely redundant rather than twice as damaged,
    and it would stop being true the moment the encoder gained a non-idempotent step
    such as error feedback.
    """
    model = _Tiny()
    torch.nn.init.normal_(model.fc.weight, std=0.02)
    quantize_model(model, {"fc": 3}, group_size=128)
    once = model.fc.weight.detach().clone()
    quantize_model(model, {"fc": 3}, group_size=128)
    assert torch.equal(once, model.fc.weight.detach())


def test_conflicting_widths_on_a_tied_pair_are_order_dependent() -> None:
    """The actual hazard behind listing one representative per tied group.

    Nothing here can detect it -- both names resolve, both encode, and the survivor
    is whichever the sorted walk reached last. The manifest would then price the
    tensor at one width while the weights carry another, so this is a documented
    property of the driver and a constraint on its callers, not a bug to fix here.
    """
    model = _Tiny(tie=True)
    torch.nn.init.normal_(model.embed.weight, std=0.02)
    before = model.embed.weight.detach().clone()

    report = quantize_model(model, {"embed": 8, "head": 2}, group_size=128)
    assert set(report.layers) == {"embed", "head"}

    # "head" sorts after "embed", so 2 bits is what survives on the shared storage.
    at_two = QuantTensor.from_dense(before, bits=2, group_size=128)
    assert report.layers["head"].bits == 2
    damage = float(torch.linalg.vector_norm(model.embed.weight.detach() - before))
    reference = float(torch.linalg.vector_norm(at_two.dequantize().to(before.dtype) - before))
    assert damage > reference / 2, "the 8-bit pass, not the 2-bit one, survived"


def test_an_unknown_name_is_an_error_not_a_skip() -> None:
    """A silent skip leaves that tensor at fp16 and the checkpoint overshoots its
    target with nothing in the output explaining why."""
    model = _Tiny()
    with pytest.raises(DynQuantError, match="not a module"):
        quantize_model(model, {"nope": 4})


def test_a_module_without_a_weight_is_an_error() -> None:
    model = _Tiny()
    model.act = torch.nn.ReLU()
    with pytest.raises(DynQuantError, match="no weight"):
        quantize_model(model, {"act": 4})


def test_progress_is_reported_for_every_module() -> None:
    model = _Tiny()
    seen: list[tuple[int, int]] = []
    quantize_model(model, {"fc": 4, "head": 4}, progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]


def test_the_report_ranks_the_worst_layers_first() -> None:
    model = _Tiny()
    torch.nn.init.normal_(model.fc.weight, std=0.02)
    torch.nn.init.normal_(model.head.weight, std=0.02)
    report = quantize_model(model, {"fc": 2, "head": 8}, group_size=128)
    assert report.worst(1)[0].name == "fc"
    assert "worst layers" in report.summary()
