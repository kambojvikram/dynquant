"""Signal collection on batched MoE expert banks.

Every MoE family on ``transformers`` 5.x -- qwen3_moe, mixtral, olmoe, gpt_oss, lfm2_moe --
stores its experts as 3-D ``nn.Parameter`` tensors on one module rather than as per-expert
``nn.Linear``. The tracker used to refuse them outright, and on a sparse model that is most of
the checkpoint: measured on ``LiquidAI/LFM2.5-8B-A1B``, 44 refused tensors carrying 7.751 B of
8.468 B parameters, so 88.4% of the model reached the allocator with no signal and was widened
by role floors alone. The graph classified those tensors and the quantizer could pack them; only
the measurement could not -- which is the kind of gap that yields a complete results table in
which the method's distinguishing mechanism touched a ninth of the bytes.

The bank stubs here are *runnable*, unlike the ones in ``test_graph_experts.py``, and that is the
point: the question these tests exist to answer is which activations cross the module boundary,
and a stub with no forward cannot be asked.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dynquant.signals import SignalTracker, TrackerConfig

HIDDEN = 16
INTER = 8
EXPERTS = 4
TOKENS = 6

GATE_UP = "experts.gate_up_proj"
DOWN = "experts.down_proj"


class _Config:
    """What ``classify_model`` and ``bank_orientation`` read off the model."""

    def __init__(self) -> None:
        self.model_type = "stub_moe"
        self.hidden_size = HIDDEN
        self.moe_intermediate_size = INTER
        self.intermediate_size = INTER
        self.num_experts = EXPERTS
        self.vocab_size = 32


class _StubExperts(nn.Module):
    """A bank shaped and applied the way ``Lfm2MoeExperts`` is.

    Two matmuls with a non-linearity between them is the whole of what matters. ``gate``, ``up``
    and the post-activation product are *locals*: no module hook can see them, which is why
    ``down_proj``'s input and ``gate_up_proj``'s output gradient are unavailable at any price and
    the two tensors have to be measured from opposite sides.

    The class name ends in ``Experts`` because :func:`is_expert_container` tests for that suffix
    as well as for 3-D parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(EXPERTS, 2 * INTER, HIDDEN) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INTER) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        for expert in range(EXPERTS):
            gate, up = F.linear(x, self.gate_up_proj[expert]).chunk(2, dim=-1)
            out = out + F.linear(F.silu(gate) * up, self.down_proj[expert])
        return out


class _MoeModel(nn.Module):
    """``pre -> probe -> experts -> post``.

    ``probe`` and ``post`` exist to give the tracker a plain ``nn.Linear`` view of the two
    activations that cross the bank's boundary: under ``saliency_source="input"``, ``probe``
    measures what enters the bank and ``post`` measures what leaves it. That turns "does the bank
    read the activation it owns?" into an exact equality against the tracker's own arithmetic,
    rather than a value this test would have to reimplement -- including the ``eps`` inside the
    square root, which lives nowhere a test can import it from.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.pre = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.probe = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.experts = _StubExperts()
        self.post = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.pre(x)
        self.probe(hidden)
        return self.post(self.experts(hidden))


def _frozen_model() -> _MoeModel:
    """Experts frozen, which is the case that matters.

    Under LoRA -- and LoRA cannot adapt a 3-D parameter, so this is what every PEFT run on an MoE
    looks like -- the expert tensors arrive with ``requires_grad=False``. A test built on freshly
    constructed ``nn.Parameter``s would have them requiring gradients already, and would pass
    without the tracker ever having to enable anything.
    """
    torch.manual_seed(0)
    model = _MoeModel()
    for param in model.experts.parameters():
        param.requires_grad_(False)
    return model


def _batch() -> Tensor:
    torch.manual_seed(1)
    return torch.randn(TOKENS, HIDDEN)


def _step(model: _MoeModel, tracker: SignalTracker, x: Tensor) -> None:
    model(x).square().sum().backward()
    tracker.on_optimizer_step()


def test_expert_banks_are_refused_unless_the_flag_asks_for_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Off is the default, and the default is a memory decision, not a correctness one.

    Measuring a bank means ``requires_grad_(True)`` on the expert tensors, which buys one
    gradient buffer for the whole expert mass -- 15.5 GB in bfloat16 on LFM2.5-8B-A1B, against
    17 GB of weights. Flipping this default would turn a working LoRA run on an MoE into an OOM
    on upgrade, so the flag is the user's to set. What the refusal owes in return is a warning
    that says how much is unmeasured and how to lift it; a refusal that says neither reads as
    "impossible" rather than "not asked for", which is how 88% of a model stays unmeasured.

    Turns red when the default flips, when the refusal stops being recorded per tensor (which
    would under-report the unmeasured share by half), or when the warning stops naming the flag.
    """
    assert TrackerConfig().measure_expert_banks is False

    with caplog.at_level(logging.WARNING):
        tracker = SignalTracker(_frozen_model(), TrackerConfig())

    assert GATE_UP not in tracker.tracked_names
    assert DOWN not in tracker.tracked_names
    assert {GATE_UP, DOWN} <= set(tracker.skipped)

    warnings = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "measure_expert_banks" in warnings, "the refusal does not say how to lift it"
    expert_params = EXPERTS * (2 * INTER * HIDDEN + HIDDEN * INTER)
    assert f"{expert_params:,}" in warnings, "the refusal does not say how much is unmeasured"


def test_each_bank_tensor_is_tracked_under_the_name_the_graph_gives_it() -> None:
    """One entry per 3-D tensor, keyed to join against the graph.

    The quantizer looks a stats entry up by the name
    :func:`~dynquant.graph.classify.classify_model` produced for the weight. A bank measured as a
    single entry named after the *module* would be a stats file that matches neither of the two
    tensors it has to widen -- and matches them silently, since a missing entry is a neutral
    score rather than an error.

    Turns red when the key stops being ``<bank module>.<parameter>``, or when the two tensors are
    collapsed into one entry.
    """
    from dynquant.graph.classify import classify_model

    model = _frozen_model()
    tracker = SignalTracker(model, TrackerConfig(measure_expert_banks=True))
    assert {GATE_UP, DOWN} <= set(tracker.tracked_names)
    assert not ({GATE_UP, DOWN} & set(tracker.skipped))

    graph_names = {info.name for info in classify_model(model)}
    assert {GATE_UP, DOWN} <= graph_names, "the tracker and the graph disagree on the key"

    layers = tracker.snapshot().layers
    assert layers[GATE_UP].param_count == EXPERTS * 2 * INTER * HIDDEN
    assert layers[DOWN].param_count == EXPERTS * HIDDEN * INTER


def test_the_two_tensors_measure_the_two_activations_that_cross_the_boundary() -> None:
    """``gate_up_proj`` reads the bank's input; ``down_proj`` reads its output.

    Asserted as exact equality against the same two activations measured through ordinary
    ``nn.Linear`` modules in the same run, so the expected values come from the tracker's own
    saliency arithmetic rather than from a second implementation of it.

    Turns red when a bank tensor is given the wrong side -- the failure with no symptom, since
    either side produces a well-formed number of a plausible magnitude for every module in the
    model.
    """
    model = _frozen_model()
    config = TrackerConfig(measure_expert_banks=True, saliency_source="input")
    with SignalTracker(model, config) as tracker:
        _step(model, tracker, _batch())
        layers = tracker.snapshot().layers

    assert layers[GATE_UP].activation_rms_ema == pytest.approx(
        layers["probe"].activation_rms_ema, rel=1e-12
    )
    assert layers[DOWN].activation_rms_ema == pytest.approx(
        layers["post"].activation_rms_ema, rel=1e-12
    )
    # And the two are different numbers, so the pair of assertions above is not satisfied by
    # everything having collapsed onto one activation.
    assert layers[GATE_UP].activation_rms_ema != layers[DOWN].activation_rms_ema


def test_the_bank_sides_are_pinned_and_do_not_follow_saliency_source() -> None:
    """``saliency_source`` is a choice about 2-D modules, which have both sides available.

    A bank does not: of the four activations its forward touches, two are locals. So each tensor
    is measured from the one boundary activation it owns whatever the config says, and flipping
    the config must move neither number.

    Turns red when a bank starts honouring ``saliency_source``, at which point one of the two
    tensors is being measured against an activation that belongs to the other.
    """
    banks: list[tuple[float, float]] = []
    twodee: list[float] = []
    for source in ("input", "output"):
        model = _frozen_model()
        config = TrackerConfig(measure_expert_banks=True, saliency_source=source)
        with SignalTracker(model, config) as tracker:
            _step(model, tracker, _batch())
            layers = tracker.snapshot().layers
        banks.append((layers[GATE_UP].activation_rms_ema, layers[DOWN].activation_rms_ema))
        twodee.append(layers["post"].activation_rms_ema)

    assert banks[0] == banks[1], "bank saliency moved with saliency_source"
    assert twodee[0] != twodee[1], "saliency_source had no effect at all -- the test is inert"


@pytest.mark.parametrize("estimator", ["outer_exact", "param", "lowrank"])
def test_plasticity_on_a_bank_is_the_parameter_gradient_whatever_the_estimator(
    estimator: str,
) -> None:
    """A bank has no 2-D boundary to reconstruct ``dW`` across and no adapter factors.

    So the estimator choice -- which is entirely about how to reach a frozen 2-D weight -- must
    not decide whether a bank gets measured at all. Under every mode the gradient autograd wrote
    to the parameter is both present and exact, which is more than any of the three give a
    Linear.

    Turns red when the bank collector is folded into the estimator branch, which would leave
    banks unmeasured under the default ``outer_exact``.
    """
    model = _frozen_model()
    config = TrackerConfig(measure_expert_banks=True, grad_estimator=estimator)
    with SignalTracker(model, config) as tracker:
        _step(model, tracker, _batch())
        layers = tracker.snapshot().layers

    for name in (GATE_UP, DOWN):
        assert layers[name].grad_norm_count == 1, f"{estimator}: {name} unmeasured"
        assert layers[name].grad_norm_mean > 0.0, f"{estimator}: {name} measured as zero"


def test_bank_gradients_are_released_so_plasticity_is_not_a_running_sum() -> None:
    """Nothing else will zero these gradients, because no optimizer owns them.

    ``optimizer.zero_grad()`` walks the parameters it was given, and the expert tensors are
    measured rather than trained -- so a tracker that reads ``.grad`` without releasing it reads
    an accumulation. Two identical steps then produce a second norm twice the first, and
    plasticity, being a *variance* of gradient norms, reports the drift as signal: a smooth,
    well-populated column that ranks experts by how late in the run they were touched.

    Turns red when ``param.grad = None`` leaves ``_collect_bank_grads``. Two identical batches
    have to give variance zero, not variance.
    """
    model = _frozen_model()
    x = _batch()

    with SignalTracker(model, TrackerConfig(measure_expert_banks=True)) as tracker:
        _step(model, tracker, x)
        assert model.experts.gate_up_proj.grad is None, "gradient not released after collection"
        first = tracker.snapshot().layers[GATE_UP].grad_norm_mean
        _step(model, tracker, x)
        stats = tracker.snapshot().layers[GATE_UP]

    assert stats.grad_norm_count == 2
    assert stats.grad_norm_mean == pytest.approx(first, rel=1e-9)
    assert stats.grad_norm_var == pytest.approx(0.0, abs=1e-10)


def test_a_bank_fires_one_observation_per_forward_and_not_one_per_tensor() -> None:
    """Two entries share one module, so the forward hook is registered once.

    ``forward_calls`` is the ``k`` in the saliency EMA's ``1 - beta**k`` debias, so counting a
    forward pass twice does not merely double a counter -- it debiases the activation signal by
    the wrong exponent.

    Turns red when ``attach`` stops de-duplicating hook registration by module id.
    """
    model = _frozen_model()
    with SignalTracker(model, TrackerConfig(measure_expert_banks=True)) as tracker:
        _step(model, tracker, _batch())
        layers = tracker.snapshot().layers

    assert layers[GATE_UP].forward_calls == 1
    assert layers[DOWN].forward_calls == 1
    assert layers["post"].forward_calls == 1


def test_detach_gives_back_the_gradient_buffers_it_asked_for() -> None:
    """The tracker enabled the gradients, so the tracker has to undo it.

    A detached tracker that left the expert mass requiring gradients would keep paying for a
    buffer the size of the experts for the rest of the run, with nothing reading it and no
    optimizer to release it.

    Turns red when either half of the restore goes away -- the ``requires_grad_(False)`` or the
    ``grad = None``.
    """
    model = _frozen_model()
    tracker = SignalTracker(model, TrackerConfig(measure_expert_banks=True))

    assert not model.experts.gate_up_proj.requires_grad, "enabled before attach"
    tracker.attach()
    assert model.experts.gate_up_proj.requires_grad
    model(_batch()).square().sum().backward()
    tracker.detach()

    for param in model.experts.parameters():
        assert not param.requires_grad
        assert param.grad is None


def test_a_bank_tensor_with_an_unplaceable_name_is_refused_rather_than_guessed() -> None:
    """Which side of the matrix the residual stream sits on is not guessable.

    A tensor whose name says nothing about direction cannot be assigned a boundary activation,
    and assigning one anyway would measure it against whichever activation happened to be at the
    other end of the bank -- a number for every module and a signal for none.

    Turns red when an unrecognised 3-D expert tensor is given a side by default.
    """

    class _OddExperts(_StubExperts):
        def __init__(self) -> None:
            super().__init__()
            self.mystery_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INTER))

    model = _frozen_model()
    model.experts = _OddExperts()
    for param in model.experts.parameters():
        param.requires_grad_(False)

    tracker = SignalTracker(model, TrackerConfig(measure_expert_banks=True))
    assert "experts.mystery_proj" in tracker.skipped
    assert "experts.mystery_proj" not in tracker.tracked_names
    assert {GATE_UP, DOWN} <= set(tracker.tracked_names), "the placeable tensors still track"
