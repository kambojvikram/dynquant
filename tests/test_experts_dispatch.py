"""The experts dispatch a packed bank can serve without leaving the grouped path.

Until :mod:`dynquant.runtime.experts` existed, loading a packed LFM2 moved the model to
``eager``, and the tests for that move only ever checked that it *happened*. What it cost
went unmeasured until the campaign put a number on it: at LFM2.5-8B-A1B's MoE geometry --
hidden 2048, 32 experts, top-4 -- ``eager`` and the default ``grouped_mm`` disagree by 0.62
logits against a 4.13 maximum and on **1.95% of argmax tokens**, while the new dispatch is
bit-identical to ``grouped_mm`` (``experiments/phase4/probe_experts_dispatch.py``, four
layers, 256 tokens, bf16). On the real 24-layer model the same disagreement is 1.24%.

That measurement is what these tests protect, and it is worth being precise about the
division of labour. The probe establishes *that* the two agree on a real
``Lfm2MoeForCausalLM``; it needs transformers 5.14+ and a machine with the time to build a
2048-wide model, so it cannot run in CI. These tests pin the properties that make the
agreement true -- above all that the k contributions are reduced once rather than
rounded into a bf16 output k times, which is the whole mechanism -- on structures small
enough to reason about exactly.

Local transformers is 4.53.2 and has no ``ALL_EXPERTS_FUNCTIONS`` to register into, so
:func:`use_dynquant_experts` here exercises its fallback. That is not a gap to apologise
for: the fallback is a supported configuration and had no test before this file. The
registration path is driven against a stand-in for the interface, which is the honest way
to test a branch whose real dependency the test machine does not have.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from dynquant.runtime.experts import (  # noqa: E402
    DISPATCH_NAME,
    dynquant_experts_forward,
    register_experts_dispatch,
    use_dynquant_experts,
)

TINY = 2.0**-9
"""Small enough that ``1.0 + TINY`` rounds back to ``1.0`` in bfloat16.

bfloat16 carries eight significand bits, so a value in ``[1, 2)`` resolves to ``2**-7`` and
``TINY`` is a quarter of that. Three of them survive being added to each other in a wider
accumulator and reach one full ulp; each one added separately to 1.0 in bf16 disappears.
That gap is the construction the reduction test is built on.
"""


class _Experts(nn.Module):
    """The attributes ``use_experts_implementation`` injects, and nothing else.

    Deliberately not a transformers class. The dispatch's contract is the
    :class:`~dynquant.runtime.experts.ExpertsModule` protocol, and a fixture that
    satisfies exactly that -- no more -- is what proves the protocol is the contract. A
    real ``Lfm2MoeExperts`` would also pass, and would hide any accidental reliance on
    something only it provides.
    """

    def __init__(
        self,
        up: Any,
        down: Any,
        *,
        has_gate: bool = False,
        is_transposed: bool = False,
        num_experts: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts if num_experts is not None else up.shape[0])
        self.has_gate = has_gate
        self.has_bias = False
        self.is_transposed = is_transposed
        if has_gate:
            self.gate_up_proj = up
        else:
            self.up_proj = up
        self.down_proj = down

    def act_fn(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return F.silu(gate) * up


def _scalar_experts(values: list[float], dtype: torch.dtype) -> _Experts:
    """Experts over a one-wide hidden state, expert ``e`` multiplying by ``values[e]``.

    Width one is the point. Every projection is a 1x1 matmul, so no rounding can enter
    from the GEMM and the only arithmetic left in the function is the reduction -- which
    is the thing under test. A wider fixture would measure the two together and could not
    say which one moved.
    """
    count = len(values)
    up = nn.Parameter(torch.ones(count, 1, 1, dtype=dtype), requires_grad=False)
    down = torch.tensor(values, dtype=dtype).reshape(count, 1, 1)
    return _Experts(up, nn.Parameter(down, requires_grad=False))


def test_the_k_contributions_are_summed_once_not_rounded_k_times() -> None:
    """One reduction over the k axis, not an accumulation that rounds at every step.

    This is the entire mechanism by which the new dispatch matches ``grouped_mm`` and
    ``eager`` does not, so it gets the most exact test in the file -- and the mechanism is
    narrower than "a different order". ``torch.sum`` over a bf16 axis accumulates in an
    fp32 accumulator and rounds **once** at the end; ``eager`` adds each expert's
    contribution into a bf16 output tensor as it walks the experts, so it rounds to bf16
    **k times**. Order alone is nearly harmless; the repeated rounding is not.

    The construction makes the difference maximal and exactly predictable. Expert 0
    contributes 1.0 and experts 1-3 contribute ``TINY`` each. Accumulating in bf16 from
    the largest first, every ``TINY`` falls off the bottom of the significand and the
    answer stays 1.0. Summing once, the three ``TINY`` s add to ``2**-7`` in the fp32
    accumulator -- exactly one bf16 ulp at this magnitude -- and the answer is 1.0078125.

    The second assertion guards the first. If a future torch gave ``sum`` a bf16
    accumulator, the two would agree and the real assertion would pass while testing
    nothing; that should fail loudly rather than quietly go vacuous.

    Turns red when: the reduction becomes incremental, ``view(...).sum(1)`` is replaced by
    an ``index_add`` into a bf16 buffer, or the cast to ``hidden_states.dtype`` is hoisted
    above the sum -- each of which reintroduces the k roundings this dispatch exists to
    avoid, and with them the 1.95% argmax gap.
    """
    big, tiny = 1.0, TINY
    experts = _scalar_experts([big, tiny, tiny, tiny], torch.bfloat16)

    hidden = torch.ones(1, 1, dtype=torch.bfloat16)
    index = torch.tensor([[0, 1, 2, 3]])
    weights = torch.ones(1, 4, dtype=torch.bfloat16)

    got = dynquant_experts_forward(experts, hidden, index, weights)

    summed_once = torch.tensor([[big, tiny, tiny, tiny]], dtype=torch.bfloat16).sum(dim=1)
    rounded_each_step = torch.zeros(1, dtype=torch.bfloat16)
    for value in (big, tiny, tiny, tiny):
        rounded_each_step = rounded_each_step + torch.tensor([value], dtype=torch.bfloat16)

    assert not torch.equal(summed_once, rounded_each_step), (
        "the fixture no longer distinguishes one reduction from k roundings, so the "
        f"assertion below would pass vacuously: both give {summed_once.tolist()}"
    )
    assert torch.equal(got.reshape(-1), summed_once)
    assert got.item() == pytest.approx(1.0078125)


def test_an_expert_no_token_reached_is_never_read() -> None:
    """Skipping an empty segment is a cost claim, and cost claims need a counter.

    ``_grouped_linear_packed`` skips ``start == stop``, which on a packed bank is the
    difference between dequantizing four experts and dequantizing thirty-two. Nothing
    about the *output* changes if the skip is removed, so only a count can see it.

    Turns red when: the ``start == stop`` continue is dropped, or the loop is rewritten to
    dequantize the bank whole before selecting -- which would be correct, silent, and cost
    LFM2.5-8B-A1B 336 MiB of transient fp16 per layer.
    """
    reads: list[int] = []

    class _Counting:
        def __init__(self, dense: torch.Tensor) -> None:
            self._dense = dense
            self.shape = tuple(dense.shape)

        def __getitem__(self, expert: int) -> torch.Tensor:
            reads.append(int(expert))
            return self._dense[expert]

    dense = torch.ones(8, 1, 1)
    experts = _Experts(_Counting(dense), _Counting(dense), num_experts=8)

    # Two tokens, both routed to experts 2 and 5. Six of the eight are never touched.
    hidden = torch.ones(2, 1)
    index = torch.tensor([[2, 5], [5, 2]])
    dynquant_experts_forward(experts, hidden, index, torch.ones(2, 2))

    assert sorted(set(reads)) == [2, 5]
    # Once per expert per projection, not once per token: the segment is the unit.
    assert len(reads) == 4


def test_routing_past_the_expert_count_contributes_nothing() -> None:
    """Expert-parallel sentinels must leave the output unchanged, not merely finite.

    Under EP a rank sees routing indices for experts it does not hold, encoded as ids at
    or past ``num_experts``. The grouped kernel leaves those rows uninitialised and relies
    on a post-mask; this implementation zeroes them going in as well, because a Python
    loop that skipped them would hand ``torch.empty``'s contents to the next projection
    and one NaN there outlives a mask applied at the end.

    Turns red when: either mask is dropped, or the ``clamp`` is removed so a sentinel id
    indexes past the end of the bank.
    """
    experts = _scalar_experts([2.0, 3.0], torch.float32)
    hidden = torch.ones(2, 1)
    weights = torch.ones(2, 2)

    real = dynquant_experts_forward(experts, hidden, torch.tensor([[0, 1], [1, 0]]), weights)
    # Second slot of each token routed to an expert this rank does not hold.
    held = dynquant_experts_forward(experts, hidden, torch.tensor([[0, 7], [1, 9]]), weights)

    assert torch.equal(real.reshape(-1), torch.tensor([5.0, 5.0]))
    assert torch.equal(held.reshape(-1), torch.tensor([2.0, 3.0]))


def test_a_packed_bank_answers_the_dispatch_like_a_dense_one() -> None:
    """The substitution the whole module is built on: ``bank[e]`` where a slice was.

    Everything else here runs on dense parameters, which would let the dispatch be
    perfectly correct and still useless -- a packed bank is the only reason it exists.

    Turns red when: ``_expert_weight`` stops going through ``__getitem__``, or
    ``_out_features`` reads ``.shape`` on a bank instead of ``logical_shape`` (a packed
    bank's ``.shape`` is the flattened word count, so the output would be the wrong width
    rather than merely inaccurate).
    """
    from dynquant.quant.device import quantize_tensor
    from dynquant.runtime.linear import DynQuantExpertBank

    torch.manual_seed(0)
    up = torch.randn(4, 96, 64).half() * 0.05
    down = torch.randn(4, 64, 96).half() * 0.05

    dense = _Experts(nn.Parameter(up, requires_grad=False), nn.Parameter(down, requires_grad=False))
    packed = _Experts(
        DynQuantExpertBank(quantize_tensor(up, bits=8, device=None)[0], out_dtype=torch.float16),
        DynQuantExpertBank(quantize_tensor(down, bits=8, device=None)[0], out_dtype=torch.float16),
    )

    hidden = (torch.randn(6, 64) * 0.1).half()
    index = torch.tensor([[0, 3], [1, 2], [3, 3], [2, 0], [1, 1], [0, 2]])
    weights = torch.full((6, 2), 0.5, dtype=torch.float16)

    want = dynquant_experts_forward(dense, hidden, index, weights)
    got = dynquant_experts_forward(packed, hidden, index, weights)

    assert got.shape == want.shape == (6, 64)
    assert torch.isfinite(got).all()
    # 8 bits over a 0.05-scaled normal: close, and nowhere near equal, which is what
    # distinguishes "the bank was read" from "the bank was ignored and zeros returned".
    assert (got - want).abs().max() < 2e-2
    assert (got - want).abs().max() > 0.0


def test_a_gated_projection_splits_before_the_second_matmul() -> None:
    """``has_gate`` routes through ``_apply_gate``, on the transposed layout LFM2 uses.

    LFM2 is gated, so the ungated fixtures above -- chosen because they isolate the
    reduction -- exercise the branch the real model does not take. This covers the other
    one, and the transposed layout with it, since ``is_transposed`` selects between
    ``x @ w`` and ``F.linear`` and reading it wrong is a silent transpose.

    Every width here is distinct -- hidden 24, ``moe_intermediate`` 10, so ``gate_up`` is
    ``[3, 24, 20]`` and ``down`` is ``[3, 10, 24]`` -- and that is load-bearing rather than
    incidental. An earlier version of this test used hidden 32 with ``moe_intermediate``
    16, which makes ``gate_up`` square, and a mutation swapping the two ends of
    :func:`_out_features` left it green: there was nothing for the transpose to get wrong.
    Only the *first* projection consults ``_out_features`` at all -- the down projection is
    handed ``out_features=hidden_dim`` directly -- so ``gate_up`` is the one shape that has
    to be rectangular for the branch to be observable.

    The expectation is a full recomputation rather than a shape-and-finite smoke check.
    Asserting the output is the right size and not NaN would pass with the gate applied in
    the wrong place, which is the specific claim in the function's name.

    Turns red when: the gate is applied after the down projection instead of between the
    two, ``is_transposed`` stops reaching ``_grouped_linear_packed``, or ``_out_features``
    reads the wrong end of the bank.
    """
    torch.manual_seed(0)
    hidden_dim, inter = 24, 10
    gate_up = nn.Parameter(torch.randn(3, hidden_dim, 2 * inter) * 0.1, requires_grad=False)
    down = nn.Parameter(torch.randn(3, inter, hidden_dim) * 0.1, requires_grad=False)
    experts = _Experts(gate_up, down, has_gate=True, is_transposed=True)
    hidden = torch.randn(4, hidden_dim) * 0.1
    index = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]])
    weights = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])

    got = dynquant_experts_forward(experts, hidden, index, weights)

    # One token, one slot at a time, with the gate explicitly between the two matmuls.
    want = torch.zeros_like(hidden)
    for token in range(hidden.size(0)):
        for slot in range(index.size(1)):
            expert = int(index[token, slot])
            projected = hidden[token] @ gate_up[expert]
            gate, up = projected.chunk(2, dim=-1)
            want[token] += (F.silu(gate) * up) @ down[expert] * weights[token, slot]

    assert got.shape == (4, hidden_dim)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------
# Selecting the dispatch, and what happens where there is nothing to select
# --------------------------------------------------------------------------


class _Interface:
    """``ExpertsInterface`` reduced to the two operations registration uses."""

    def __init__(self) -> None:
        self.registered: dict[str, Any] = {}

    def __contains__(self, name: object) -> bool:
        return name in self.registered

    def register(self, name: str, fn: Any) -> None:
        self.registered[name] = fn


class _Config:
    def __init__(self, implementation: str | None) -> None:
        if implementation is not None:
            self._experts_implementation = implementation


class _Model(nn.Module):
    def __init__(self, implementation: str | None = "grouped_mm") -> None:
        super().__init__()
        self.config = _Config(implementation)
        self.moves: list[str] = []

    def set_experts_implementation(self, implementation: str) -> None:
        self.moves.append(implementation)
        self.config._experts_implementation = implementation


@pytest.fixture
def moe_interface(monkeypatch: pytest.MonkeyPatch) -> _Interface:
    """Stand in for ``transformers.integrations.moe`` on a transformers that lacks it.

    The import inside ``register_experts_dispatch`` is resolved through ``sys.modules``,
    so inserting a module there is enough and no transformers internals are touched.
    """
    import types

    interface = _Interface()
    module = types.ModuleType("transformers.integrations.moe")
    module.ALL_EXPERTS_FUNCTIONS = interface  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers.integrations.moe", module)
    return interface


def test_registration_is_idempotent_and_installs_the_forward(moe_interface: _Interface) -> None:
    """One entry, the real function, however many times a load path asks for it.

    Turns red when: registration overwrites an existing entry (a second call would then
    replace a dispatch a user had deliberately swapped), or registers a wrapper instead of
    the function the tests above exercise.
    """
    assert register_experts_dispatch() is True
    assert moe_interface.registered[DISPATCH_NAME] is dynquant_experts_forward

    sentinel = object()
    moe_interface.registered[DISPATCH_NAME] = sentinel
    assert register_experts_dispatch() is True
    assert moe_interface.registered[DISPATCH_NAME] is sentinel


def test_a_model_with_the_interface_lands_on_dynquant(moe_interface: _Interface) -> None:
    """The point of the whole module: the packed model never goes to ``eager``.

    Turns red when: the preference order flips, or the setter is bypassed in favour of
    writing the config directly -- transformers models do bookkeeping in
    ``set_experts_implementation`` that a raw attribute write skips.
    """
    model = _Model("grouped_mm")

    assert use_dynquant_experts(model) == "grouped_mm"

    assert model.moves == [DISPATCH_NAME]
    assert model.config._experts_implementation == DISPATCH_NAME
    assert DISPATCH_NAME in moe_interface


def test_it_falls_back_to_eager_where_there_is_nothing_to_register_into(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older transformers still gets a model that runs, just not a comparable one.

    This is the branch that actually executes on the test machine's transformers 4.53.2,
    and the configuration every packed LFM2 ran under before this module existed. It has
    to keep working, because "the dispatch could not be registered" must degrade to a
    slower model rather than to a crash on the first forward.

    Turns red when: the ``ImportError`` guard is narrowed or removed, or the fallback
    starts returning ``None`` -- which would tell a caller nothing moved when a model was
    in fact moved to a dispatch with a 1.95% argmax disagreement.
    """
    import types

    empty = types.ModuleType("transformers.integrations.moe")
    monkeypatch.setitem(sys.modules, "transformers.integrations.moe", empty)

    model = _Model("grouped_mm")
    assert use_dynquant_experts(model) == "grouped_mm"
    assert model.config._experts_implementation == "eager"


def test_nothing_to_move_is_reported_as_nothing_moved(moe_interface: _Interface) -> None:
    """A dense model, and one already on the dispatch, are both ``None``.

    Turns red when: a model already on ``dynquant`` is moved again (harmless but reported
    as a move, which would put a false line in every pack report), or a config without the
    attribute raises instead of returning.
    """
    assert use_dynquant_experts(_Model(None)) is None
    assert use_dynquant_experts(_Model(DISPATCH_NAME)) is None

    bare = nn.Linear(4, 4)
    assert use_dynquant_experts(bare) is None


def test_the_pack_report_names_the_dispatch_it_landed_on() -> None:
    """Origin and destination are two fields because the destination is negotiated.

    ``use_dynquant_experts`` prefers ``dynquant`` and settles for ``eager``, and those are
    numerically different models. A report naming only what it moved *from* leaves the
    reader unable to tell which one they got.

    Turns red when: the summary line hardcodes a destination again, as it did when
    ``eager`` was the only one available.
    """
    from dynquant.runtime.linear import PackReport

    report = PackReport()
    report.modules["block.experts.up_proj"] = (4, 4096, 1024)
    report.experts_implementation = "grouped_mm"
    report.experts_dispatch = DISPATCH_NAME

    summary = report.summary()
    assert "moved from 'grouped_mm' to 'dynquant'" in summary

    report.experts_dispatch = "eager"
    assert "moved from 'grouped_mm' to 'eager'" in report.summary()
