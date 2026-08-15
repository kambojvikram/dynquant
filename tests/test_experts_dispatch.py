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
        up_bias: torch.Tensor | None = None,
        down_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts if num_experts is not None else up.shape[0])
        self.has_gate = has_gate
        # Per-expert, ``[E, out]``, gathered to per-row by the dispatch. LFM2 has none;
        # GPT-OSS does, and it is the one argument to the fused op that has no analogue
        # in the loop's per-expert slicing -- so a fixture that never sets it leaves the
        # fused path's only unshared line untested.
        self.has_bias = up_bias is not None
        self.is_transposed = is_transposed
        if has_gate:
            self.gate_up_proj = up
            self.gate_up_proj_bias = up_bias
        else:
            self.up_proj = up
            self.up_proj_bias = up_bias
        self.down_proj = down
        self.down_proj_bias = down_bias

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


# ---------------------------------------------------------------------------
# The fused grouped path (P8)
#
# What these can and cannot reach. The kernel itself is CUDA and its numerical
# parity belongs in ``tests/test_kernels_parity.py``, which skips without a built
# wheel. What is testable here -- and what actually broke twice while it was being
# written -- is everything between the dispatch and the launch: what the segment
# table is, which calls are allowed to take it, and whether the caller still
# synchronizes. A kernel that is bit-perfect and reached with the wrong band is a
# silently wrong model, and none of that failure lives in the ``.cu`` file.
# ---------------------------------------------------------------------------


def test_the_segment_table_is_device_resident_int32_with_sentinels_dropped() -> None:
    """``[E + 1]`` int32, and nothing in it was read on the host.

    Three properties, each load-bearing for a different reason. **int32** because the
    kernel does two ``__ldg`` loads per block against that type and a wider table would
    read halves of neighbouring entries. **A tensor, not a list**, because the whole
    saving of the fused path is that this never leaves the device. And **sentinels
    dropped**, which is older than P8 but is now the thing a device table has to keep
    getting right unassisted: an expert-parallel id lands past the end and the
    ``[:num_experts]`` slice discards it, so the last offset counts the pairs this rank
    holds and every sentinel row sits beyond it.

    Turns red when: the table goes back to a list (the annotation would still say
    Tensor), widens to int64 to match the counter it is built from, or starts counting
    clamped ids -- which would widen one band and displace every band after it.

    Says nothing about whether the *construction* reads a value on the host, which is a
    separate property with a separate cost and its own two tests below.
    """
    from dynquant.runtime.experts import _segment_offsets

    # Four real experts and two sentinels at 4 and 5, already sorted as the dispatch
    # sorts them.
    ids = torch.tensor([0, 0, 0, 1, 3, 3, 4, 5])
    seg = _segment_offsets(ids, num_experts=4)

    assert isinstance(seg, torch.Tensor)
    assert seg.dtype is torch.int32
    assert seg.shape == (5,)
    # 3 in expert 0, 1 in expert 1, none in 2, 2 in 3, and the two sentinels excluded.
    assert seg.tolist() == [0, 3, 4, 4, 6]


def test_each_thing_the_kernel_cannot_do_falls_back_instead_of_raising() -> None:
    """Four refusals, and every one of them is a supported configuration.

    This is the test that keeps the fused path from becoming a requirement. A dense
    bank in a partly-packed model, a transposed bank, an ABI-2 wheel, and an fp32 model
    over fp16 scales all reach ``_grouped_linear_packed``, and each has a correct answer
    that is not the kernel's. ``_fusable`` returning ``None`` is how they get it.

    Turns red when: any of the four starts returning the tensor -- for the transposed
    case that is not a crash but a silent read of the wrong expert's rows, at the right
    shape.
    """
    from dynquant.quant.device import quantize_tensor
    from dynquant.runtime import ops as ops_module
    from dynquant.runtime.experts import _fusable
    from dynquant.runtime.linear import DynQuantExpertBank

    torch.manual_seed(0)
    bank = DynQuantExpertBank(
        quantize_tensor(torch.randn(3, 16, 8).half() * 0.05, bits=4, device=None)[0],
        out_dtype=torch.float16,
    )
    x = torch.zeros(5, 8, dtype=torch.float16)

    saved = (ops_module.has_grouped_gemv, ops_module.uses_compiled_kernels)
    ops_module.has_grouped_gemv = lambda: True  # type: ignore[assignment]
    ops_module.uses_compiled_kernels = lambda _t: True  # type: ignore[assignment]
    try:
        # The configuration everything else is a deviation from. Compared by buffer
        # address: `weight_qt` rebuilds its view from the module's buffers on every
        # read, so two calls are equal and never identical.
        fused = _fusable(x, bank, is_transposed=False)
        assert fused is not None
        assert fused.packed.data_ptr() == bank.weight_qt.packed.data_ptr()

        # 1. transposed: the flattening the row arithmetic assumes is the other one.
        assert _fusable(x, bank, is_transposed=True) is None

        # 2. a dense bank in a mixed model: nothing packed to point at.
        dense = nn.Parameter(torch.randn(3, 16, 8).half(), requires_grad=False)
        assert _fusable(x, dense, is_transposed=False) is None

        # 3. fp32 activations over fp16 scales: one scalar type, both operands.
        assert _fusable(x.float(), bank, is_transposed=False) is None

        # 4. an ABI-2 wheel, which is a wheel this runtime deliberately still loads.
        ops_module.has_grouped_gemv = lambda: False  # type: ignore[assignment]
        assert _fusable(x, bank, is_transposed=False) is None
    finally:
        ops_module.has_grouped_gemv, ops_module.uses_compiled_kernels = saved


class _StandInGroupedGemv:
    """The compiled op's contract, in torch, checking what it was handed.

    Not a mock. It asserts that every buffer it received is the one the bank actually
    holds -- by address, not by value -- and then computes the documented answer. So a
    test built on it fails both when the arithmetic is wrong and when the marshalling
    is: passing ``scales`` where ``offsets`` belongs, or the flattened row count where
    ``out_features`` belongs, produces the right shape and the wrong model, and neither
    would be visible from the output alone at these tolerances.

    Reading ``seg`` with ``.tolist()`` here is not the sync the fused path exists to
    remove. This stands in for a device kernel that reads the same two values with two
    loads; what matters is that the *caller* did not read them, which is what
    :func:`test_the_fused_path_agrees_with_the_loop_and_stops_synchronizing` counts.
    """

    def __init__(self, qt: Any, out_features: int) -> None:
        self.qt = qt
        self.out_features = out_features
        self.calls = 0

    def __call__(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor | None,
        seg: torch.Tensor,
        bits: int,
        group_values: int,
        in_features: int,
        out_features: int,
    ) -> torch.Tensor:
        assert packed.data_ptr() == self.qt.packed.data_ptr()
        assert scales.data_ptr() == self.qt.scales.data_ptr()
        if self.qt.offsets is None:
            assert offsets is None
        else:
            # By address. An earlier version asserted only that it was non-None, and a
            # mutation passing `scales` in this slot survived: same shape, same dtype,
            # plausible numbers, a different model.
            assert offsets is not None
            assert offsets.data_ptr() == self.qt.offsets.data_ptr()
        assert bits == self.qt.bits
        assert in_features == self.qt.in_features
        assert out_features == self.out_features
        assert seg.dtype is torch.int32 and seg.device == packed.device
        assert packed.shape[0] == (seg.numel() - 1) * out_features
        self.calls += 1

        # Flattened, because that is how the kernel addresses it: a bank is one
        # ``[E * out, in]`` buffer and expert e is a row band of it.
        dense = self.qt.dequantize(dtype=x.dtype).reshape(-1, in_features)
        out = x.new_zeros((x.shape[0], out_features))
        bounds = seg.tolist()
        for expert in range(len(bounds) - 1):
            start, stop = bounds[expert], bounds[expert + 1]
            if start == stop:
                continue
            band = dense[expert * out_features : (expert + 1) * out_features]
            out[start:stop] = F.linear(x[start:stop], band)
        return out


def test_the_fused_path_agrees_with_the_loop_and_stops_synchronizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same tokens, same answer, and two host reads per layer become none.

    The agreement half is what makes the fused path shippable: a packed LFM2 scored
    through the loop and served through the kernel has to be one model, or the panel's
    accuracy numbers describe something nobody can download. It is asserted as exact
    equality, not a tolerance -- both paths dequantize the same codes with the same
    scales and call the same ``F.linear``, so any difference at all is a difference in
    *which* codes, not in rounding.

    The banks carry biases, which is the one line the two paths do not share: the loop
    adds it in place after walking the experts and the fused path adds it to the kernel's
    return. Without a biased fixture that addition can be deleted outright and every
    other assertion here still holds.

    The counting half is what the path is *for*. ``.tolist()`` on the segment table is a
    device synchronization, and the loop takes one per bank per layer -- 44 per token on
    the 22-layer, two-bank model this campaign quantized. Counting them is the only way
    to state that as a property rather than a hope, because removing the sync changes no
    output: a version that computed the list and then ignored it would pass every other
    test in this file.

    Turns red when: the caller goes back to building the host list eagerly, the two
    banks stop sharing one table, or the fused branch stops being taken at all -- the
    call counter catches the last one, which an agreement test alone cannot, since a
    fused path that silently never runs agrees with the loop perfectly.
    """
    from dynquant.quant.device import quantize_tensor
    from dynquant.runtime import ops as ops_module
    from dynquant.runtime.linear import DynQuantExpertBank

    torch.manual_seed(0)
    up = torch.randn(4, 48, 32).half() * 0.05
    down = torch.randn(4, 32, 48).half() * 0.05
    up_bank = DynQuantExpertBank(
        quantize_tensor(up, bits=4, device=None)[0], out_dtype=torch.float16
    )
    down_bank = DynQuantExpertBank(
        quantize_tensor(down, bits=4, device=None)[0], out_dtype=torch.float16
    )
    experts = _Experts(
        up_bank,
        down_bank,
        up_bias=torch.randn(4, 48).half() * 0.1,
        down_bias=torch.randn(4, 32).half() * 0.1,
    )

    hidden = (torch.randn(6, 32) * 0.1).half()
    # Two shapes the bands have to survive, both present on purpose. **Expert 2 is
    # unreached** -- the band whose start equals its stop, where the loop skips and the
    # kernel's block returns immediately. And **ids 4 and 5 are expert-parallel
    # sentinels**, which sort to the tail and must be counted out of every band rather
    # than folded into one: fold them into a low bin and the bands, being offsets into a
    # sorted array, displace every real row after it.
    index = torch.tensor([[0, 3], [1, 4], [3, 3], [0, 0], [1, 5], [0, 3]])
    weights = torch.full((6, 2), 0.5, dtype=torch.float16)

    original_tolist = torch.Tensor.tolist
    seen: list[int] = []

    def counting_tolist(self: torch.Tensor) -> Any:
        # Only the segment table is of interest; the stand-in's own read is one of
        # these, so it is subtracted by construction -- it is not installed yet.
        if self.dtype is torch.int32 and self.ndim == 1:
            seen.append(self.numel())
        return original_tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)

    want = dynquant_experts_forward(experts, hidden, index, weights)
    assert seen == [5, 5], f"the loop should read the table once per bank, got {seen}"
    assert torch.isfinite(want).all()

    seen.clear()
    up_op = _StandInGroupedGemv(up_bank.weight_qt, 48)
    down_op = _StandInGroupedGemv(down_bank.weight_qt, 32)
    dispatch = {
        up_bank.weight_qt.packed.data_ptr(): up_op,
        down_bank.weight_qt.packed.data_ptr(): down_op,
    }

    def route(x: torch.Tensor, packed: torch.Tensor, *rest: Any) -> torch.Tensor:
        return dispatch[packed.data_ptr()](x, packed, *rest)

    monkeypatch.setattr(ops_module, "has_grouped_gemv", lambda: True)
    monkeypatch.setattr(ops_module, "uses_compiled_kernels", lambda _t: True)
    monkeypatch.setattr(torch.ops.dynquant, "moe_grouped_gemv", route, raising=False)

    got = dynquant_experts_forward(experts, hidden, index, weights)

    assert up_op.calls == 1 and down_op.calls == 1
    # The stand-in's own reads are the only ones left, and it reads the table it was
    # handed -- so two, not four. The caller took none.
    assert seen == [5, 5], f"the fused path added a host read of the table: {seen}"
    assert torch.equal(got, want)


def test_a_sentinel_never_widens_a_band_it_sorts_after() -> None:
    """The bands index a sorted array, so one over-wide band steals the next one's rows.

    Distinct from :func:`test_routing_past_the_expert_count_contributes_nothing`, which
    checks that a sentinel *contributes* nothing -- it is masked twice over, so that
    holds even when the table is wrong. This checks the second thing a sentinel must not
    do: occupy a slot. Fold one into a low bin and the cumulative sum shifts every band
    after it, and the rows it displaces are real tokens that then get some other expert's
    weight, at the right shape and with no mask to catch them.

    Three experts and one sentinel is the smallest fixture that can show it. Ids
    ``[0, 1, 2, 4]`` sort with the sentinel last; fold it in anywhere below expert 2 and
    expert 2's single row moves into the band before it. Token 1 is then multiplied by
    3.0 instead of 5.0 -- finite, plausible, and wrong.

    Turns red when: the segment table is built from clamped, wrapped, or otherwise
    in-range-forced ids instead of the raw ones the ``[:num_experts]`` slice discards.
    An A/B between the fused and loop paths cannot see this, because both read the same
    table; only a known answer can.
    """
    experts = _scalar_experts([2.0, 3.0, 5.0], torch.float32)
    hidden = torch.ones(2, 1)
    index = torch.tensor([[0, 1], [2, 4]])

    out = dynquant_experts_forward(experts, hidden, index, torch.ones(2, 2))

    assert torch.equal(out.reshape(-1), torch.tensor([5.0, 5.0]))


def test_the_segment_table_is_built_without_reading_a_value_on_the_host() -> None:
    """No op in ``_segment_offsets`` reads the tensor's contents to decide anything.

    This is the property that had a docstring and no test, and the gap was not
    academic: the claim said "``bincount`` and ``cumsum`` are both shape-determined",
    and ``torch.bincount`` sizes its output from ``input.max()``, read on the host.
    ``minlength`` raises the floor on that size but does not remove the read -- measured
    both ways in ``experiments/phase4/graph_capture_probe.py``, which is what found it,
    because a host read changes no output and so no value assertion can see it.

    Checked through the dispatcher rather than by monkeypatching ``torch.bincount``, so
    a rewrite that reaches the same kernel by another spelling is still caught.
    ``_local_scalar_dense`` is on the list for the same reason under a different name:
    it is what ``.item()`` lowers to, and a shape computed from it is a fence whatever
    the Python looked like.

    Turns red when: the counting goes back to ``bincount``, or anything else in here
    starts sizing a tensor from a value. Its consequence is the test below, which needs
    a GPU; this one runs everywhere and names the same defect.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    from dynquant.runtime.experts import _segment_offsets

    banned = {"aten::bincount", "aten::_local_scalar_dense", "aten::item", "aten::nonzero"}
    seen: list[str] = []

    class _Record(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None):  # type: ignore[no-untyped-def]
            # The *schema* name, ``aten::bincount``, not ``str(func)``, which is the
            # overload's ``aten.bincount.default``. A first draft of this test compared
            # against the schema spelling while recording the overload spelling, so it
            # passed with ``bincount`` restored -- caught only by putting the defect
            # back and watching for red.
            schema = getattr(func, "_schema", None)
            seen.append(getattr(schema, "name", None) or str(func))
            return func(*args, **(kwargs or {}))

    ids = torch.tensor([0, 0, 1, 3, 4, 5])
    with _Record():
        seg = _segment_offsets(ids, num_experts=4)

    assert seg.tolist() == [0, 2, 3, 3, 4]
    offenders = sorted({name for name in seen if name in banned})
    assert not offenders, f"{offenders} read a value on the host; seen: {sorted(set(seen))}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_the_segment_table_captures_into_a_cuda_graph() -> None:
    """The consequence of the test above, stated as the thing P8's gate actually asks.

    A forward free of host reads is capturable; a forward with one is not, and CUDA
    refuses it at the offending op rather than degrading. So capture is the sharpest
    available assertion that no fence survived, and it is sharper than counting ops
    because it does not need to know in advance what to count -- this is how the
    ``bincount`` read was found after the ``.tolist()`` read had been removed and the
    path was believed clean.

    Replay is checked against a *fresh* input, because that is the failure this cannot
    afford to miss. A graph that captured stale pointers replays without error and
    returns the previous call's answer, which a timing-only check reads as a speedup.

    Deliberately scoped to ``_segment_offsets`` and not to the whole dispatch: the loop
    path genuinely cannot be captured -- it reads its bounds with ``.tolist()`` -- so a
    forward-level capture test would pass or fail on whether the compiled kernel happens
    to be installed, which is a different question from this one.
    """
    ids = torch.tensor([0, 0, 1, 3, 4, 5], device="cuda")

    from dynquant.runtime.experts import _segment_offsets

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _segment_offsets(ids, num_experts=4)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _segment_offsets(ids, num_experts=4)

    ids.copy_(torch.tensor([0, 1, 1, 1, 2, 4], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    assert captured.tolist() == [0, 1, 4, 5, 5]
