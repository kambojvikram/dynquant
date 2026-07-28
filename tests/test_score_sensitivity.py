"""Cardinal sensitivity -- estimated ``dLoss`` per module per width.

Why this exists at all is a measured result, not a preference. 187 modules of a
fine-tuned Qwen3.5-2B were quantized to 3 bits and the model then given one module
back at 4 bits, one at a time, with the task loss read off a fixed batch each time.
Allocating a 3.125-bit budget by the quantity this module computes recovered **85.5%**
of the uniform-3-bit damage; by the shipped rank-product score, **28.5%** -- inside
the band spanned by five random allocations of the same budget, and below both
"upgrade the biggest tensors" (52.0%) and the role-aware control (65.6%). Dropping
the label-dependent factor and pricing by weight error alone came in at **-29.5%**:
worse than not spending the bits.

So the tests here are about the two ways that number can be quietly wrong.

**The arithmetic.** ``sum_rc E[delta_r^2] E[x_c^2] (W - Q_b(W))_rc^2`` is checked
against the same sum formed the direct way, including under row chunking, because the
chunked path is the one that runs on anything real and is the one that can silently
drop a block.

**The refusals.** A module with no moments, or with moments on the wrong axis, must
land in :attr:`SensitivityTable.unestimable` rather than acquire a fabricated number.
A missing entry read as zero sensitivity makes that module the *first* thing the
allocator downgrades, which is the opposite of what "we did not measure this" means.
"""

from __future__ import annotations

import pytest
import torch
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.graph.classify import classify_model
from dynquant.graph.roles import ModuleRole
from dynquant.quant.grid import quantize_with_search
from dynquant.score import sensitivity as sens_mod
from dynquant.score.sensitivity import (
    SensitivityTable,
    estimate_sensitivity,
    module_weights,
    weight_only_sensitivity,
)
from dynquant.signals.moments import ChannelMoments

GROUP = 32


@pytest.fixture(scope="module")
def model():
    return Qwen3_5ForCausalLM(tie=True)


@pytest.fixture(scope="module")
def graph(model):
    return classify_model(model)


@pytest.fixture(scope="module")
def weights(model):
    return module_weights(model)


@pytest.fixture(scope="module")
def moments(graph, weights):
    return _moments_for(graph, weights)


@pytest.fixture(scope="module")
def table34(graph, moments, weights):
    return estimate_sensitivity(graph, moments, weights, bit_options=(3, 4), group_size=GROUP)


@pytest.fixture(scope="module")
def control34(graph, weights):
    return weight_only_sensitivity(graph, weights, bit_options=(3, 4), group_size=GROUP)


def _moments_for(graph, weights, *, seed: int = 0) -> ChannelMoments:
    """Plausible positive moments for every 2-D quantizable weight."""
    generator = torch.Generator().manual_seed(seed)
    moments = ChannelMoments()
    for info in graph.quantizable():
        weight = weights.get(info.name)
        if weight is None or weight.ndim != 2 or info.role is ModuleRole.EMBEDDING:
            continue
        rows, cols = weight.shape
        moments.input_sq[info.name] = torch.rand(cols, generator=generator) + 0.1
        moments.output_grad_sq[info.name] = torch.rand(rows, generator=generator) + 0.1
        moments.observations[info.name] = 8
    return moments


def _oracle(weight: torch.Tensor, x2: torch.Tensor, d2: torch.Tensor, bits: int) -> float:
    """The formula, formed the expensive way: full [out, in] weighted error matrix."""
    quantized, _ = quantize_with_search(weight, bits=bits, group_size=GROUP)
    error = (weight.float() - quantized.dequantize(dtype=torch.float32)) ** 2
    weighted = error * x2.unsqueeze(0) * d2.unsqueeze(1)
    return float(weighted.sum())


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_the_estimate_is_the_gauss_newton_sum() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 96)
    x2 = torch.rand(96) + 0.1
    d2 = torch.rand(64) + 0.1

    for bits in (2, 3, 4, 8):
        got = sens_mod._module_sensitivity(
            weight, x2, d2, bits=bits, group_size=GROUP, symmetric=False
        )
        assert got == pytest.approx(_oracle(weight, x2, d2, bits), rel=1e-5)


def test_row_chunking_does_not_change_the_answer(monkeypatch) -> None:
    """Groups run along the *input* axis, so no group straddles a row boundary and
    chunking is exact rather than approximate. If that stopped being true this test
    is what would notice."""
    torch.manual_seed(0)
    weight = torch.randn(64, 96)
    x2 = torch.rand(96) + 0.1
    d2 = torch.rand(64) + 0.1

    whole = sens_mod._module_sensitivity(weight, x2, d2, bits=3, group_size=GROUP, symmetric=False)
    monkeypatch.setattr(sens_mod, "_ROW_CHUNK_BYTES", 96 * 4 * 7)  # 7 rows per chunk
    chunked = sens_mod._module_sensitivity(
        weight, x2, d2, bits=3, group_size=GROUP, symmetric=False
    )
    assert chunked == pytest.approx(whole, rel=1e-5)


def test_more_bits_costs_less_loss() -> None:
    """Across the whole ladder, on one tensor, so the check is over widths rather
    than over modules -- the ordering the allocator's move pricing subtracts."""
    torch.manual_seed(0)
    weight = torch.randn(48, 96)
    x2 = torch.rand(96) + 0.1
    d2 = torch.rand(48) + 0.1

    ladder = [
        sens_mod._module_sensitivity(weight, x2, d2, bits=bits, group_size=GROUP, symmetric=False)
        for bits in (2, 3, 4, 8)
    ]
    assert ladder == sorted(ladder, reverse=True)
    assert all(value >= 0.0 for value in ladder)


def test_no_module_is_priced_negative(table34) -> None:
    assert table34.values
    for name, widths in table34.values.items():
        assert widths[3] >= widths[4] >= 0.0, name


def test_gain_is_the_difference_between_two_widths(table34) -> None:
    table = table34
    name = next(iter(table.values))
    assert table.gain(name, 3, 4) == pytest.approx(table.values[name][3] - table.values[name][4])
    assert table.gain(name, 3, 4) > 0.0


def test_the_weighting_actually_changes_the_ordering() -> None:
    """Two tensors with identical quantization error, different channel moments.

    If the moments were being dropped or broadcast the wrong way, both would price
    the same and the estimator would be an expensive way to compute ``w_err`` --
    which measured -29.5% on the allocation benchmark, i.e. actively harmful.
    """
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    flat = torch.ones(64)
    loud = torch.ones(64)
    loud[:8] = 50.0

    quiet_price = sens_mod._module_sensitivity(
        weight, flat, torch.ones(32), bits=3, group_size=GROUP, symmetric=False
    )
    loud_price = sens_mod._module_sensitivity(
        weight, loud, torch.ones(32), bits=3, group_size=GROUP, symmetric=False
    )
    assert loud_price > quiet_price * 1.5


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


def test_a_module_with_no_moments_is_reported_not_scored(graph, moments, weights) -> None:
    victim = next(i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN)
    holed = ChannelMoments(
        input_sq={k: v for k, v in moments.input_sq.items() if k != victim},
        output_grad_sq={k: v for k, v in moments.output_grad_sq.items() if k != victim},
    )

    table = estimate_sensitivity(graph, holed, weights, bit_options=(3,), group_size=GROUP)
    assert victim in table.unestimable
    assert victim not in table.values
    assert table.gain(victim, 3, 4) is None


def test_a_transposed_moment_is_refused_rather_than_broadcast(graph, moments, weights) -> None:
    """On a square weight a swapped axis pair is numerically silent. It is refused on
    shape, before any arithmetic, precisely because the arithmetic would succeed."""
    victim = next(i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_UP)
    rows, cols = weights[victim].shape
    assert rows != cols, "fixture changed; this test needs a rectangular weight"
    swapped = ChannelMoments(
        input_sq=dict(moments.input_sq), output_grad_sq=dict(moments.output_grad_sq)
    )
    swapped.input_sq[victim] = moments.output_grad_sq[victim]
    swapped.output_grad_sq[victim] = moments.input_sq[victim]

    table = estimate_sensitivity(graph, swapped, weights, bit_options=(3,), group_size=GROUP)
    assert victim in table.unestimable


def test_a_tied_embedding_is_priced_from_its_head(graph, moments, weights) -> None:
    """The tied ``embed_tokens``/``lm_head`` tensor is 27% of Qwen3.5-2B.

    Only the head produces a pairing in the stored tensor's orientation -- an
    embedding's input is a token id -- and the graph's representative for the pair is
    whichever ``named_modules`` reached first, which is the embedding. Without alias
    resolution the single largest tensor in the model goes unestimable for a reason
    that is an accident of iteration order.
    """
    tied = [i for i in graph.modules.values() if i.tied_to is not None]
    assert tied, "fixture changed; this test is about weight tying"

    head = tied[0]
    representative = head.tied_to
    assert representative is not None
    assert head.name not in {i.name for i in graph.quantizable()}, (
        "the follower is not quantized in its own right; only the representative is, "
        "which is exactly why the alias lookup has to exist"
    )
    rows, cols = weights[representative].shape
    aliased = ChannelMoments(
        input_sq={**moments.input_sq, head.name: torch.rand(cols) + 0.1},
        output_grad_sq={**moments.output_grad_sq, head.name: torch.rand(rows) + 0.1},
    )

    table = estimate_sensitivity(graph, aliased, weights, bit_options=(3,), group_size=GROUP)
    assert representative in table.values
    assert representative not in table.unestimable


def test_absolute_is_false_because_the_estimate_is_first_order(table34) -> None:
    """The sum of 187 individually-measured disturbances was +0.089 against an actual
    all-3-bit cost of +0.047. The table is an ordering with proportions, and the flag
    is what keeps a caller from printing it as a predicted loss."""
    assert table34.absolute is False


def test_gain_refuses_unknown_names_and_widths() -> None:
    table = SensitivityTable(values={"a": {3: 2.0, 4: 1.0}})
    assert table.gain("a", 3, 4) == pytest.approx(1.0)
    assert table.gain("a", 3, 2) is None
    assert table.gain("nope", 3, 4) is None


# --------------------------------------------------------------------------
# The control
# --------------------------------------------------------------------------


def test_weight_only_sensitivity_needs_no_moments(control34) -> None:
    """The honest calibration-free baseline, kept so the claim that data beats no
    data can be re-run rather than cited."""
    table = control34
    assert table.values
    for widths in table.values.values():
        assert widths[3] > widths[4] >= 0.0


def test_weight_only_and_gauss_newton_disagree(table34, control34) -> None:
    """They must not be the same ordering, or there was nothing to collect."""
    data, control = table34, control34

    shared = sorted(set(data.values) & set(control.values))
    assert len(shared) > 4
    by_data = sorted(shared, key=lambda n: -(data.gain(n, 3, 4) or 0.0))
    by_control = sorted(shared, key=lambda n: -(control.gain(n, 3, 4) or 0.0))
    assert by_data != by_control


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def test_module_weights_is_keyed_the_way_the_graph_is(graph, weights) -> None:
    """A PEFT-wrapped model and the merged checkpoint it becomes must produce the
    same dictionary, or the estimator silently prices nothing."""
    for info in graph.quantizable():
        assert info.name in weights, info.name
        assert tuple(weights[info.name].shape) == tuple(info.shape)
