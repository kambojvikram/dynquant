"""The deep clip grid and the ``E[x_c^2]``-weighted clip objective.

Two changes to the encoder, measured together on Qwen3.5-2B / CaseHOLD and separable
only because each was run as a byte-identical A/B. They are tested in one file
because their failure modes are the same failure mode seen from two sides: a
checkpoint priced against one objective and encoded against another is the right
size and simply worse, and nothing downstream can detect it.

**The grid.** ``CLIP_CANDIDATES`` stops at 0.80. That is a defensible floor at 4 bits
and cannot be one at 2 -- with four levels the MSE-optimal shrink for anything
resembling a Gaussian group is far tighter, so every 2-bit group returns the floor
and reports it as the winner. :func:`test_the_shipped_grid_is_floor_bound_at_two_bits`
is that defect written down: on clean Gaussian data the shipped grid picks 0.80 for
*every* group, minimum and mean alike, which is what a search that never found an
interior optimum looks like. The extension is inert at 4 bits, and
:func:`test_the_deep_grid_is_inert_at_four_bits` pins that, because a grid change that
perturbed the widths it was not aimed at would silently invalidate ``paper-3.15``.

**The objective.** A group's error reaches the loss through
``sum_c E[x_c^2] (W - Q)^2_rc``, so a clip chosen on unweighted MSE can lower total
error while raising the part of it that matters -- measured on Qwen3.5-2B, extending
the grid cut ``k_proj``'s 2-bit reconstruction error by 16% while its Gauss-Newton
sensitivity *rose* 6.7%. The tests below fix the two properties that make the weighted
objective safe to adopt: a weight vector of ones reproduces the unweighted search
bit-for-bit, so the default path cannot move; and the ratios really do minimise the
weighted error rather than the plain one.

**Where each pays is not interchangeable.** At a byte-identical 740,724,736 B the
deep grid was worth +0.09 points through the encoder alone and +0.67 once the
allocator was re-priced with it; the weighted objective was worth +0.41 through the
encoder and nothing further through re-pricing (-0.09, p = 0.53). A change that
merely *scales* sensitivity cancels in the ratios an allocator reads and pays in the
encoder; one that *distorts* those ratios pays in the allocator. Hence
``quantize_model``'s all-or-nothing coverage rule, which is the last group of tests
here.
"""

from __future__ import annotations

import pytest
import torch

from dynquant.errors import DynQuantError
from dynquant.quant.grid import (
    CLIP_CANDIDATES,
    DEEP_CLIP_CANDIDATES,
    quantize_with_search,
    search_clip_ratios,
)
from dynquant.quant.quantizer import quantize_model

GROUP = 128


def _clean(rows: int = 8, cols: int = 512, seed: int = 1) -> torch.Tensor:
    """No outliers, so the clip optimum is set by the level count alone."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator)


def _outlier(rows: int = 4, cols: int = 256, seed: int = 0) -> torch.Tensor:
    """One dominant value in the first channel of every group of 128."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator) * 0.02
    weight[:, ::GROUP] = 3.0
    return weight


# --------------------------------------------------------------------------
# The deep grid
# --------------------------------------------------------------------------


def test_the_deep_grid_extends_the_shipped_one_rather_than_replacing_it() -> None:
    """``paper-3.15`` reproduces published numbers, so the original prefix is contract.

    Extension rather than replacement is also what makes the comparison meaningful:
    every ratio the shipped grid could pick is still available, so the deep grid can
    only win where the shipped one had no candidate to offer.
    """
    assert DEEP_CLIP_CANDIDATES[: len(CLIP_CANDIDATES)] == CLIP_CANDIDATES
    assert min(CLIP_CANDIDATES) == 0.80
    assert min(DEEP_CLIP_CANDIDATES) == 0.40
    assert list(DEEP_CLIP_CANDIDATES) == sorted(DEEP_CLIP_CANDIDATES, reverse=True)


def test_the_shipped_grid_is_floor_bound_at_two_bits() -> None:
    """The defect, stated as a measurement rather than an argument.

    A search whose answer is its own boundary for *every* group has not found an
    optimum; it has run out of candidates. That is invisible from the outside --
    ``search_clip_ratios`` reports 0.80 as the winner either way -- which is why it
    survived to be measured end to end rather than caught here first.
    """
    weight = _clean()
    shipped = search_clip_ratios(weight, bits=2, group_size=GROUP)
    assert float(shipped.ratios.max()) == pytest.approx(0.80)
    assert float(shipped.ratios.min()) == pytest.approx(0.80)

    deep = search_clip_ratios(weight, bits=2, group_size=GROUP, candidates=DEEP_CLIP_CANDIDATES)
    assert float(deep.ratios.max()) < 0.80
    assert float(deep.best_mse.sum()) < float(shipped.best_mse.sum())


def test_the_deep_grid_is_inert_at_four_bits() -> None:
    """It must change nothing where the original floor was already right.

    If the extension moved 4-bit groups it would be a general re-tuning of the
    encoder wearing a 2-bit justification, and every 4-bit number in the campaign
    would need re-measuring.
    """
    weight = _clean()
    shipped = search_clip_ratios(weight, bits=4, group_size=GROUP)
    deep = search_clip_ratios(weight, bits=4, group_size=GROUP, candidates=DEEP_CLIP_CANDIDATES)
    assert torch.equal(shipped.ratios, deep.ratios)
    assert torch.equal(shipped.best_mse, deep.best_mse)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_the_deep_grid_never_loses(bits: int) -> None:
    """1.0 is still a candidate, so a wider grid is at worst a tie at every width."""
    weight = _outlier()
    shipped = search_clip_ratios(weight, bits=bits, group_size=GROUP)
    deep = search_clip_ratios(weight, bits=bits, group_size=GROUP, candidates=DEEP_CLIP_CANDIDATES)
    assert torch.all(deep.best_mse <= shipped.best_mse + 1e-9)


# --------------------------------------------------------------------------
# The weighted objective
# --------------------------------------------------------------------------


def test_a_weight_of_ones_is_the_unweighted_search_exactly() -> None:
    """Bit-for-bit, not approximately.

    The docstring tells callers with no usable moments to pass a vector of ones
    rather than ``None``, so that advice has to be free. Multiplying float32 error by
    1.0 is exact, and if it ever stops being the same code path this is what notices.
    """
    weight = _outlier()
    plain = search_clip_ratios(weight, bits=3, group_size=GROUP)
    ones = search_clip_ratios(
        weight, bits=3, group_size=GROUP, channel_weight=torch.ones(weight.shape[1])
    )
    assert torch.equal(plain.ratios, ones.ratios)
    assert torch.equal(plain.best_mse, ones.best_mse)
    assert torch.equal(plain.baseline_mse, ones.baseline_mse)


def test_an_expensive_channel_stops_the_group_clipping() -> None:
    """The mechanism, isolated: the same tensor, two verdicts, set only by the moments.

    The outlier sits in the first channel of every group. Told that channel carries
    the activations, clipping it is no longer a bargain and the search declines
    entirely -- 1.0 in every group, against 0.96-0.98 unweighted.
    """
    weight = _outlier()
    plain = search_clip_ratios(weight, bits=3, group_size=GROUP)
    assert float(plain.ratios.max()) < 1.0

    loud = torch.ones(weight.shape[1])
    loud[:: weight.shape[1] // 2] = 1e6  # the outlier channel of each group
    weighted = search_clip_ratios(weight, bits=3, group_size=GROUP, channel_weight=loud)
    assert torch.all(weighted.ratios == 1.0)


def test_the_ratios_minimise_the_weighted_error_and_report_it() -> None:
    """Re-encoding with the winners reproduces the error that won, weighting included.

    The unweighted version of this test exists because the search once optimised one
    encoder while the checkpoint was written by another. The same hazard applies to
    the objective: a search that reports weighted error but selects on plain MSE
    would pass every other test in this file.
    """
    weight = _outlier()
    channel = torch.rand(weight.shape[1], generator=torch.Generator().manual_seed(3)) + 0.05

    quantized, result = quantize_with_search(
        weight, bits=3, group_size=GROUP, channel_weight=channel
    )
    recon = quantized.dequantize(dtype=torch.float32)
    error = (weight.to(torch.float32) - recon) ** 2 * channel.unsqueeze(0)
    groups = weight.shape[1] // GROUP
    actual = error.reshape(weight.shape[0], groups, GROUP).sum(dim=-1)
    assert torch.allclose(actual, result.best_mse, rtol=1e-5, atol=1e-8)


def test_the_two_objectives_disagree_about_which_ratio_wins() -> None:
    """If they always agreed the argument would be free and would also be pointless."""
    weight = _outlier()
    channel = torch.ones(weight.shape[1])
    channel[:: weight.shape[1] // 2] = 500.0
    plain = search_clip_ratios(weight, bits=2, group_size=GROUP, candidates=DEEP_CLIP_CANDIDATES)
    weighted = search_clip_ratios(
        weight,
        bits=2,
        group_size=GROUP,
        candidates=DEEP_CLIP_CANDIDATES,
        channel_weight=channel,
    )
    assert not torch.equal(plain.ratios, weighted.ratios)


def test_a_short_weight_vector_is_padded_not_broadcast() -> None:
    """Padded columns are zero in both operands, so their weight cannot matter.

    The encoder pads ``in_features`` up to the group size. A weight vector sized to
    the real width has to survive that without being silently recycled across
    channels, which is what a broadcast would do.
    """
    weight = _outlier(cols=GROUP + 8)  # 136 -> padded to 256
    channel = torch.ones(weight.shape[1])
    plain = search_clip_ratios(weight, bits=3, group_size=GROUP)
    ones = search_clip_ratios(weight, bits=3, group_size=GROUP, channel_weight=channel)
    assert torch.equal(plain.ratios, ones.ratios)


# --------------------------------------------------------------------------
# The driver's coverage rule
# --------------------------------------------------------------------------


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(GROUP, 64, bias=False)
        self.head = torch.nn.Linear(GROUP, 64, bias=False)


@pytest.fixture
def tiny() -> _Tiny:
    """One instance, reused across both arms of a comparison.

    ``in_place=False`` does not touch the model, so the two arms must differ only in
    the objective. Building a fresh ``_Tiny`` per arm would give them different random
    weights, and the comparison would measure the initialisation instead.
    """
    return _Tiny()


def test_partial_coverage_is_refused_with_the_names_that_are_missing(tiny) -> None:
    """All-or-nothing, and the message has to say what to do about it.

    A module encoded on a different objective than it was priced with is mispriced in
    a way nothing downstream can detect: the checkpoint is the right size and simply
    worse. Filling the gap silently -- with ones, say -- would be that defect
    implemented as a convenience.
    """
    with pytest.raises(DynQuantError, match="channel_weights is missing"):
        quantize_model(
            tiny,
            {"fc": 4, "head": 4},
            group_size=GROUP,
            channel_weights={"fc": torch.ones(GROUP)},
        )


def test_a_wrongly_sized_weight_vector_is_refused(tiny) -> None:
    """Length is checked against the module, not against the first module seen."""
    with pytest.raises(DynQuantError, match="input channels"):
        quantize_model(
            tiny,
            {"fc": 4},
            group_size=GROUP,
            channel_weights={"fc": torch.ones(GROUP + 1)},
        )


def test_ones_everywhere_reproduces_the_unweighted_checkpoint(tiny) -> None:
    """The opt-in must be genuinely opt-in: supplying the neutral weight changes nothing."""
    bits = {"fc": 3, "head": 4}
    plain = quantize_model(tiny, bits, group_size=GROUP, in_place=False)
    ones = quantize_model(
        tiny,
        bits,
        group_size=GROUP,
        in_place=False,
        channel_weights={name: torch.ones(GROUP) for name in bits},
    )
    for name in bits:
        assert plain.layers[name].relative_error == pytest.approx(ones.layers[name].relative_error)


def test_the_weighted_objective_changes_the_checkpoint(tiny) -> None:
    """The end-to-end version of the point, through the driver rather than the search."""
    bits = {"fc": 2}
    loud = torch.ones(GROUP)
    loud[:8] = 400.0
    plain = quantize_model(tiny, bits, group_size=GROUP, in_place=False)
    weighted = quantize_model(
        tiny,
        bits,
        group_size=GROUP,
        in_place=False,
        candidates=DEEP_CLIP_CANDIDATES,
        channel_weights={"fc": loud},
    )
    assert plain.layers["fc"].relative_error != pytest.approx(weighted.layers["fc"].relative_error)
