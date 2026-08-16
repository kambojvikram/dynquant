"""Batched MoE expert banks: discovery, orientation, and refusing to guess.

These tests build the expert bank *by hand* rather than through ``AutoConfig`` +
``from_config``, and that is the point. The layout under test -- one module holding the
whole expert bank as 3-D parameters -- arrived in ``transformers`` 5.x. The existing
architecture-matrix tests build MoE models through the library, so on a 4.x install
they construct the old per-expert ``nn.Linear`` layout and pass without ever touching
this code path. That is exactly how the defect survived: the tests and the library had
drifted apart while both stayed green.

Hand-built stubs pin the layout the *format* has to handle, independent of whichever
``transformers`` happens to be installed. The shapes and the class names here are the
ones measured on transformers 5.14.1.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dynquant.errors import DynQuantError
from dynquant.graph.classify import _bank_config, classify_model
from dynquant.graph.experts import (
    IN_OUT,
    OUT_IN,
    UNKNOWN,
    bank_orientation,
    batched_expert_params,
    is_expert_container,
    owning_configs,
)
from dynquant.graph.roles import ModuleRole
from dynquant.quant.quantizer import quantize_model

HIDDEN = 64
INTER = 32
EXPERTS = 4


class Config:
    """The three fields orientation needs, plus what classification reads."""

    def __init__(self, **kwargs: object) -> None:
        self.model_type = "stub_moe"
        self.hidden_size = HIDDEN
        self.moe_intermediate_size = INTER
        self.intermediate_size = INTER
        self.num_experts = EXPERTS
        self.vocab_size = 128
        self.__dict__.update(kwargs)


class Qwen3MoeExperts(nn.Module):
    """``[E, out, in]`` -- input axis last. qwen3_moe, qwen2_moe, mixtral, olmoe."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(EXPERTS, 2 * INTER, HIDDEN))
        self.down_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INTER))


class GptOssExperts(nn.Module):
    """``[E, in, out]`` -- input axis in the middle. Plus rank-2 biases."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(EXPERTS, INTER, HIDDEN))
        self.gate_up_proj_bias = nn.Parameter(torch.randn(EXPERTS, 2 * INTER))
        self.down_proj_bias = nn.Parameter(torch.randn(EXPERTS, HIDDEN))


class Block(nn.Module):
    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.gate = nn.Linear(HIDDEN, EXPERTS, bias=False)
        self.experts = experts


class Layer(nn.Module):
    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)  # type: ignore[assignment]
        self.mlp = Block(experts)


class Model(nn.Module):
    def __init__(self, experts_cls: type[nn.Module], **config: object) -> None:
        super().__init__()
        self.config = Config(**config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(experts_cls())])  # type: ignore[assignment]
        self.lm_head = nn.Linear(HIDDEN, self.config.vocab_size, bias=False)


def bank_of(model: nn.Module) -> nn.Module:
    return model.get_submodule("model.layers.0.mlp.experts")


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_a_bank_is_recognised_but_a_plain_container_is_not() -> None:
    assert is_expert_container(bank_of(Model(Qwen3MoeExperts)))
    assert not is_expert_container(nn.Linear(4, 4))
    assert not is_expert_container(nn.ModuleList([nn.Linear(4, 4)]))


def test_a_class_named_experts_without_3d_parameters_is_not_a_bank() -> None:
    """The old per-expert layout is also spelled ``...Experts``.

    Class name alone would claim it, then find nothing to quantize and report a bank
    with no tensors -- worse than not claiming it, because the caller stops looking.
    """

    class OldStyleExperts(nn.ModuleList):
        pass

    assert not is_expert_container(OldStyleExperts([nn.Linear(4, 4)]))


def test_biases_are_not_offered_for_quantization() -> None:
    """gpt_oss carries per-expert biases beside the weights.

    They are excluded by rank, and the name test behind that is belt-and-braces. A
    bias is a rounding error of the parameter count and lands straight on the residual
    stream, so quantizing it is cost without benefit.
    """
    names = {name for name, _ in batched_expert_params(bank_of(Model(GptOssExperts)))}
    assert names == {"gate_up_proj", "down_proj"}


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------


def test_orientation_is_read_from_the_config_not_assumed() -> None:
    qwen = Model(Qwen3MoeExperts)
    gpt = Model(GptOssExperts)
    assert bank_orientation(bank_of(qwen), qwen.config) == OUT_IN
    assert bank_orientation(bank_of(gpt), gpt.config) == IN_OUT


def test_orientation_is_settled_by_down_proj_when_gate_up_is_ambiguous() -> None:
    """``hidden == 2 * inter`` makes a fused gate_up square, and square is undecidable.

    The bank still has an answer because ``down_proj`` beside it is not square, and
    orientation is a property of the family rather than of one tensor. Without that,
    every model whose expert width happens to be half its hidden size -- not a rare
    ratio -- would be refused for no reason.
    """

    class SquareExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # hidden 64, inter 32: gate_up is [E, 64, 64], which fits both readings.
            self.gate_up_proj = nn.Parameter(torch.randn(EXPERTS, 2 * INTER, HIDDEN))
            self.down_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INTER))

    model = Model(SquareExperts)
    assert HIDDEN == 2 * INTER, "this test needs the degenerate ratio to be present"
    assert bank_orientation(bank_of(model), model.config) == OUT_IN


def test_orientation_without_config_dimensions_is_unknown_not_a_guess() -> None:
    model = Model(Qwen3MoeExperts)
    stripped = Config()
    del stripped.hidden_size
    del stripped.moe_intermediate_size
    del stripped.intermediate_size
    assert bank_orientation(bank_of(model), stripped) == UNKNOWN


def test_an_unrecognised_parameter_name_is_unknown() -> None:
    """A shape alone cannot say which axis is the input; only the name can.

    ``[E, 64, 32]`` is ``[out, in]`` for a down-projection and ``[in, out]`` for an
    up-projection. Refusing an unknown name is what keeps that ambiguity from being
    resolved by coin flip.
    """

    class Odd(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mystery_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INTER))

    class OddExperts(Odd):
        pass

    model = Model(OddExperts)
    assert bank_orientation(bank_of(model), model.config) == UNKNOWN


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_the_graph_accounts_for_the_expert_parameters() -> None:
    """The defect was silent, so the assertion is on the *parameter share*.

    Before the fix the bank contributed nothing and the graph still looked healthy:
    every module it did find was classified correctly. On a 128-expert model the
    missing fraction is ~91%, and the only symptom would have been a checkpoint that
    came out far larger than its target.
    """
    model = Model(Qwen3MoeExperts)
    graph = classify_model(model)
    total = sum(p.numel() for p in model.parameters())
    seen = sum(info.num_params for info in graph)
    expert_params = sum(p.numel() for _, p in batched_expert_params(bank_of(model)))

    assert expert_params > 0
    assert seen == total, f"graph accounts for {seen}/{total} parameters"


def test_expert_tensors_get_moe_roles_not_the_router_role() -> None:
    """The trap: ``weight.shape[0]`` on a 3-D expert tensor is the *expert count*.

    Structural inference calls a module a router when its output width equals
    ``num_experts`` and it has an ``experts`` sibling. Both hold for every batched
    expert tensor -- the bank is its own sibling -- so running these through the
    structural path would floor the entire expert bank at 8 bits with full confidence.
    """
    graph = classify_model(Model(Qwen3MoeExperts))
    roles = {name: info.role for name, info in graph.modules.items()}

    assert roles["model.layers.0.mlp.experts.gate_up_proj"] == ModuleRole.MOE_EXPERT_GATE_UP
    assert roles["model.layers.0.mlp.experts.down_proj"] == ModuleRole.MOE_EXPERT_DOWN
    assert roles["model.layers.0.mlp.gate"] == ModuleRole.MOE_ROUTER


def test_an_override_still_wins_on_an_expert_tensor() -> None:
    graph = classify_model(
        Model(Qwen3MoeExperts),
        overrides={"*.experts.down_proj": "moe.expert.up"},
    )
    info = graph.modules["model.layers.0.mlp.experts.down_proj"]
    assert info.role == ModuleRole.MOE_EXPERT_UP
    assert info.source == "override"


def test_a_reversed_bank_is_skipped_with_a_reason_not_quantized_wrongly() -> None:
    """gpt_oss stores ``[E, in, out]`` and the encoder groups along the last axis.

    Grouping the wrong way still round-trips and still reports a plausible
    reconstruction error -- the scales simply average over output channels instead of
    input ones. There is no later symptom, so this has to be caught at classification
    or not at all.
    """
    graph = classify_model(Model(GptOssExperts))

    assert "model.layers.0.mlp.experts.down_proj" not in graph.modules
    entry = graph.skipped["model.layers.0.mlp.experts.down_proj"]
    assert "input axis is not last" in entry.reason
    # The whole bank, priced dense. A refusal here is not a rounding error: on an
    # LFM2-class MoE the expert banks are 91.5% of the quantizable parameters, so a
    # refused bank that never reaches the denominator misprices most of the model.
    assert entry.num_params == EXPERTS * INTER * HIDDEN
    assert entry.num_params <= graph.skipped_params()


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------


def test_the_quantizer_resolves_a_name_that_addresses_a_parameter() -> None:
    """``get_submodule`` raises for ``...experts.gate_up_proj``: it is a Parameter.

    That name is what the graph, the stats file and the state dict all use, so the
    lookup widens rather than the name narrowing.
    """
    model = Model(Qwen3MoeExperts)
    before = bank_of(model).gate_up_proj.detach().clone()

    report = quantize_model(
        model,
        {"model.layers.0.mlp.experts.gate_up_proj": 4},
        in_place=True,
    )

    result = report.layers["model.layers.0.mlp.experts.gate_up_proj"]
    assert result.num_params == EXPERTS * 2 * INTER * HIDDEN
    assert not torch.equal(before, bank_of(model).gate_up_proj)
    # Group-128 4-bit on Gaussian weights lands near step/sqrt(12); anything far off
    # means the tensor was grouped along the wrong axis or reshaped incorrectly.
    assert 0.02 < result.relative_error < 0.25


def test_quantizing_the_bank_itself_says_what_to_name_instead() -> None:
    model = Model(Qwen3MoeExperts)
    with pytest.raises(DynQuantError, match="no weight to quantize"):
        quantize_model(model, {"model.layers.0.mlp.experts": 4}, in_place=True)


def test_a_missing_parameter_on_a_real_bank_names_the_whole_path() -> None:
    """The bank resolving is not enough -- the leaf has to exist on it.

    Worth its own test because this is the near-miss case: a typo'd or stale
    parameter name lands on a bank that does very much exist, so a lookup that
    stopped at the parent would have found something and gone on.
    """
    model = Model(Qwen3MoeExperts)
    with pytest.raises(DynQuantError, match=r"experts\.no_such_proj.*not a module or a parameter"):
        quantize_model(model, {"model.layers.0.mlp.experts.no_such_proj": 4}, in_place=True)


def test_a_name_that_resolves_to_nothing_still_raises() -> None:
    """A bit map naming a module this model does not have is a divergence, not a skip.

    Skipping would leave the tensor at fp16 and produce a checkpoint over its target
    with nothing in the report saying why -- which is the same silent-oversize failure
    the batched-expert defect caused.
    """
    model = Model(Qwen3MoeExperts)
    with pytest.raises(DynQuantError, match="not a module or a parameter"):
        quantize_model(model, {"model.layers.0.mlp.nowhere.no_such_proj": 4}, in_place=True)


def test_a_name_that_resolves_to_something_unquantizable_says_what_it_found() -> None:
    """Distinct from the missing-name case, and it wants a distinct message.

    A name that resolves to a 1-D tensor is a bit map pointing at something that
    was never a weight -- a bias, a scale, a norm gain. Telling the reader what was
    actually found there is the difference between fixing the bit map and hunting
    for a module that is sitting right where the name says it is.
    """
    model = Model(Qwen3MoeExperts)
    bank_of(model).register_parameter("router_scale", nn.Parameter(torch.randn(EXPERTS)))
    with pytest.raises(DynQuantError, match="no quantizable tensor called 'router_scale'"):
        quantize_model(model, {"model.layers.0.mlp.experts.router_scale": 4}, in_place=True)


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def test_the_tracker_refuses_expert_banks_loudly_rather_than_silently() -> None:
    """Unmeasured has to be distinguishable from measured-and-uninteresting.

    Neither signal is observable at the bank's boundary: the forward hook sees
    ``gate_up_proj``'s input but not ``down_proj``'s, and the backward hook sees
    ``down_proj``'s output gradient but not ``gate_up_proj``'s. A boundary hook would
    give each tensor one right signal and one wrong one, and the score multiplies the
    two ranks -- so the result would be a plausible bit map with no anomaly to notice.
    """
    from dynquant.signals.tracker import SignalTracker

    tracker = SignalTracker(Model(Qwen3MoeExperts))
    skipped = tracker.skipped

    for leaf in ("gate_up_proj", "down_proj"):
        reason = skipped[f"model.layers.0.mlp.experts.{leaf}"]
        assert "not separable" in reason
    assert "model.layers.0.mlp.gate" in tracker.tracked_names


# --------------------------------------------------------------------------
# Composite models: whose widths describe this bank
# --------------------------------------------------------------------------
#
# Qwen3-Omni is two full MoE models in one checkpoint. The top-level
# `Qwen3OmniMoeConfig` holds no `hidden_size` and no `moe_intermediate_size` at all --
# each tower keeps its own in `<tower>_config.text_config`, and the two disagree
# (Thinker 2048/768, Talker 1024/384). The classifier's one-level `text_config`
# descent does not reach them, so orientation was undecidable for every bank in the
# model: 90.8% of the parameters refused for want of a number that was in the file the
# whole time.
#
# Shapes here are scaled down but keep the property that matters -- two towers with
# different, mutually incompatible widths.

THINKER_HIDDEN, THINKER_INTER = 64, 32
TALKER_HIDDEN, TALKER_INTER = 32, 8


class TowerTextConfig:
    """The config that actually holds a tower's dimensions."""

    def __init__(self, hidden: int, inter: int) -> None:
        self.model_type = "stub_omni_text"
        self.hidden_size = hidden
        self.moe_intermediate_size = inter
        self.num_experts = EXPERTS
        self.vocab_size = 128


class TowerConfig:
    """Wraps the text config, as ``Qwen3OmniMoeThinkerConfig`` does. No widths of its own."""

    def __init__(self, text_config: TowerTextConfig) -> None:
        self.model_type = "stub_omni_tower"
        self.text_config = text_config


class OuterConfig:
    """The composite config: names its towers, holds no dimensions."""

    def __init__(self, thinker: TowerConfig, talker: TowerConfig) -> None:
        self.model_type = "stub_omni"
        self.thinker_config = thinker
        self.talker_config = talker


class TowerExperts(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(EXPERTS, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.randn(EXPERTS, hidden, inter))


class Tower(nn.Module):
    """One tower: its own wrapper config, and an inner text model carrying the real one."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        text = TowerTextConfig(hidden, inter)
        self.config = TowerConfig(text)
        self.model = nn.Module()
        self.model.config = text  # type: ignore[assignment]
        block = nn.Module()
        block.experts = TowerExperts(hidden, inter)  # type: ignore[assignment]
        layer = nn.Module()
        layer.mlp = block  # type: ignore[assignment]
        self.model.layers = nn.ModuleList([layer])  # type: ignore[assignment]


class Composite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thinker = Tower(THINKER_HIDDEN, THINKER_INTER)
        self.talker = Tower(TALKER_HIDDEN, TALKER_INTER)
        self.config = OuterConfig(self.thinker.config, self.talker.config)


def _bank(model: nn.Module, tower: str) -> nn.Module:
    return model.get_submodule(f"{tower}.model.layers.0.mlp.experts")


def test_the_outer_config_of_a_composite_model_decides_nothing() -> None:
    """The state before the fix, pinned so the fix has something to be a fix of."""
    model = Composite()
    assert bank_orientation(_bank(model, "thinker"), model.config) == UNKNOWN
    assert bank_orientation(_bank(model, "talker"), model.config) == UNKNOWN
    # The one-level descent the classifier relies on elsewhere is not there either.
    assert getattr(model.config, "text_config", None) is None


def test_each_towers_bank_resolves_through_that_towers_own_config() -> None:
    model = Composite()
    modules = dict(model.named_modules())
    for tower in ("thinker", "talker"):
        raw = f"{tower}.model.layers.0.mlp.experts"
        resolved = _bank_config(modules[raw], modules, raw, model.config)
        assert bank_orientation(modules[raw], resolved) == OUT_IN, tower


def test_the_wrong_towers_config_refuses_rather_than_guessing() -> None:
    """Why walking outward from the bank is safe even though it tries several configs.

    The Thinker's widths against a Talker bank do not produce a plausible wrong axis;
    they produce :data:`UNKNOWN`, because orientation is decided by matching config
    dimensions against the tensor's actual shape. A candidate from the wrong tower
    disqualifies itself. Measured on the real model in both directions too.
    """
    model = Composite()
    thinker_text = model.thinker.config.text_config
    talker_text = model.talker.config.text_config
    assert bank_orientation(_bank(model, "talker"), thinker_text) == UNKNOWN
    assert bank_orientation(_bank(model, "thinker"), talker_text) == UNKNOWN


def test_owning_configs_offers_the_nearest_enclosing_config_first() -> None:
    model = Composite()
    modules = dict(model.named_modules())
    candidates = owning_configs(modules, "thinker.model.layers.0.mlp.experts")
    assert candidates, "no ancestor config found"
    assert candidates[0] is model.thinker.config.text_config
    assert all(c is not model.talker.config.text_config for c in candidates), (
        "the Talker is not an ancestor of a Thinker bank and must never be a candidate"
    )


def test_a_composite_models_expert_tensors_all_reach_the_graph() -> None:
    graph = classify_model(Composite())
    for tower in ("thinker", "talker"):
        for param in ("gate_up_proj", "down_proj"):
            name = f"{tower}.model.layers.0.mlp.experts.{param}"
            assert name in graph.modules, f"{name} missing from the graph"
            assert name not in graph.skipped, f"{name} was refused"


def test_a_flat_model_still_resolves_through_the_config_it_always_did() -> None:
    """The no-regression guard: the search only runs for a bank already being refused.

    ``_bank_config`` returns ``default`` unchanged whenever ``default`` decides, so a
    model that classifies today classifies through exactly the same config object
    today -- identity, not equality.
    """
    model = Model(Qwen3MoeExperts)
    modules = dict(model.named_modules())
    raw = "model.layers.0.mlp.experts"
    assert _bank_config(modules[raw], modules, raw, model.config) is model.config


def test_a_module_that_is_not_a_bank_short_circuits() -> None:
    """Containers outnumber banks; none of them should trigger an ancestor walk."""
    model = Model(Qwen3MoeExperts)
    modules = dict(model.named_modules())
    sentinel = object()
    assert _bank_config(modules["model.layers.0"], modules, "model.layers.0", sentinel) is sentinel


def test_an_undecidable_bank_is_still_refused_after_the_search() -> None:
    """The fallback turns refusals into answers or leaves them standing -- never a guess."""

    class Odd(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mystery_proj = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INTER))

    class OddExperts(Odd):
        pass

    model = Model(OddExperts)
    modules = dict(model.named_modules())
    raw = "model.layers.0.mlp.experts"
    resolved = _bank_config(modules[raw], modules, raw, model.config)
    assert bank_orientation(modules[raw], resolved) == UNKNOWN
