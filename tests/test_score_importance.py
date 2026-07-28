"""Combining the two signals into one importance number.

This is the method. Everything else is plumbing around it, and everything
downstream consumes only its output -- so a defect here does not surface as an
error, it surfaces as a bit map that looks entirely reasonable and is driven by the
wrong ordering.
"""

from __future__ import annotations

import pytest
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.graph.classify import classify_model
from dynquant.graph.roles import ModuleRole
from dynquant.score.importance import (
    COMBINE_MODES,
    NEUTRAL_RANK,
    ScoreConfig,
    score_modules,
)
from dynquant.signals.schema import LayerStats, StatsFile


@pytest.fixture(scope="module")
def graph():
    return classify_model(Qwen3_5ForCausalLM(tie=True))


def _neutral_score(combine: str) -> float:
    """What a module with nothing measured about it must score, per mode.

    Written out rather than read back from the implementation, so a change to how
    the ranks are combined has to restate what "no information" means instead of
    silently redefining it.
    """
    return NEUTRAL_RANK if combine == "plasticity" else NEUTRAL_RANK**2


def _stats(graph, *, skip: set[str] | None = None, calls: int = 4) -> StatsFile:
    """A stats file where every module has a distinct, strictly positive signal."""
    skip = skip or set()
    stats = StatsFile()
    for index, info in enumerate(graph.quantizable()):
        if info.name in skip:
            continue
        stats.layers[info.name] = LayerStats(
            name=info.name,
            activation_rms_ema=1.0 + index,
            grad_norm_count=8,
            grad_norm_mean=1.0,
            grad_norm_var=1.0 + index,
            forward_calls=calls,
        )
    return stats


def test_no_measured_module_scores_zero(graph) -> None:
    """The invariant the whole allocation rests on.

    The score multiplies two percentile ranks, and the allocator prices damage as
    ``score * params * Δerror`` -- so a score of exactly zero makes a module free to
    cut to the 2-bit minimum, no matter how large it is or how loose the budget. The
    obvious rank formula ``(rank - 1) / (n - 1)`` hands exactly that to the lowest
    member of every role, in each of the two signals independently. On Qwen3.5-2B
    that was 20 of 187 modules at 2 bits at a *4-bit* target, with reconstruction
    errors up to 0.61 relative, and a model that scored 20.8% against 65.1%.

    Nothing in the bit map looked wrong. The histogram was plausible, the target was
    hit to four decimals, and the floor-violation report listed the breaches without
    suggesting any of them were unmotivated.
    """
    report = score_modules(graph, _stats(graph))
    assert report.modules
    for scored in report.modules.values():
        assert scored.score > 0.0, f"{scored.name} scored zero ({scored.explain})"


def test_a_single_member_role_is_neutral_not_maximal(graph) -> None:
    """EMBEDDING has one instance, so its ranks come from a group of size one.

    Returning 1.0 there gives the largest tensor in the model the maximum possible
    score by construction. It is the tensor one would most want protected, which is
    exactly what makes the artifact hard to notice: the allocation it produces looks
    like good judgement rather than an arithmetic accident.
    """
    solo = [i for i in graph.quantizable() if i.role is ModuleRole.EMBEDDING]
    assert len(solo) == 1, "fixture changed; this test is about single-member roles"

    report = score_modules(graph, _stats(graph))
    scored = report.modules[solo[0].name]
    assert scored.saliency_rank == NEUTRAL_RANK
    assert scored.plasticity_rank == NEUTRAL_RANK
    assert scored.score == pytest.approx(NEUTRAL_RANK)

    legacy = score_modules(graph, _stats(graph), ScoreConfig(combine="rank_product"))
    assert legacy.modules[solo[0].name].score == pytest.approx(NEUTRAL_RANK**2)


def test_ranking_is_within_role_by_default(graph) -> None:
    """Roles have systematically different magnitudes, so a global ranking sorts
    mostly by role and squeezes the within-role question into the noise."""
    stats = _stats(graph)
    within = score_modules(graph, stats, ScoreConfig(rank_within_role=True))
    globally = score_modules(graph, stats, ScoreConfig(rank_within_role=False))

    role = ModuleRole.MLP_DOWN
    members = [i.name for i in graph.quantizable() if i.role is role]
    assert len(members) > 2

    # Within role, this role's members must span most of the interval. Globally they
    # occupy whatever narrow band their magnitudes put them in.
    spread = max(within.modules[n].saliency_rank for n in members) - min(
        within.modules[n].saliency_rank for n in members
    )
    global_spread = max(globally.modules[n].saliency_rank for n in members) - min(
        globally.modules[n].saliency_rank for n in members
    )
    assert spread > global_spread


def test_higher_signal_ranks_higher(graph) -> None:
    """Direction check. An inverted comparator produces a complete, plausible map
    that protects precisely the modules that needed it least."""
    members = [i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN]
    stats = _stats(graph)
    ordered = sorted(members, key=lambda n: stats.layers[n].activation_rms_ema)

    report = score_modules(graph, stats)
    ranks = [report.modules[n].saliency_rank for n in ordered]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize("combine", COMBINE_MODES)
def test_an_unmeasured_module_is_neutral_not_zero(graph, combine) -> None:
    """ "Not measured" and "measured, unimportant" must not produce the same score.

    Conflating them decides, on a sparse MoE, whether a rarely-routed expert is cut
    to 2 bits on the strength of evidence that was never collected.
    """
    victim = next(i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN)
    config = ScoreConfig(combine=combine)
    report = score_modules(graph, _stats(graph, skip={victim}), config)

    assert victim in report.missing_stats
    scored = report.modules[victim]
    assert scored.score == pytest.approx(_neutral_score(combine))
    assert not scored.exercised
    assert "UNEXERCISED" in scored.explain


@pytest.mark.parametrize("combine", COMBINE_MODES)
def test_a_module_with_no_forward_calls_counts_as_unexercised(graph, combine) -> None:
    graph_modules = list(graph.quantizable())
    stats = _stats(graph)
    victim = graph_modules[0].name
    stats.layers[victim] = LayerStats(
        name=victim, activation_rms_ema=99.0, grad_norm_count=8, forward_calls=0
    )

    report = score_modules(graph, stats, ScoreConfig(combine=combine))
    assert victim in report.unexercised
    assert report.modules[victim].score == pytest.approx(_neutral_score(combine))


def test_a_module_with_no_gradient_samples_counts_as_unexercised(graph) -> None:
    """A forward pass without a backward pass is half a measurement. Ranking the
    activation side alone would let a frozen module compete with a trained one."""
    stats = _stats(graph)
    victim = next(iter(stats.layers))
    stats.layers[victim] = LayerStats(
        name=victim, activation_rms_ema=99.0, grad_norm_count=0, forward_calls=4
    )

    report = score_modules(graph, stats)
    assert victim in report.unexercised


def test_the_graph_decides_which_modules_exist(graph) -> None:
    """A stale name in the stats file must not add a module, and a missing one must
    not drop one -- a dropped module does not fail, it silently keeps whatever width
    the quantizer defaults to."""
    stats = _stats(graph)
    stats.layers["model.layers.0.mlp.does_not_exist"] = LayerStats(
        name="model.layers.0.mlp.does_not_exist",
        activation_rms_ema=1e9,
        grad_norm_count=8,
        grad_norm_var=1e9,
        forward_calls=4,
    )

    report = score_modules(graph, stats)
    assert set(report.modules) == {i.name for i in graph.quantizable()}


def test_coherence_is_off_by_default_and_multiplies_when_on(graph) -> None:
    stats = _stats(graph)
    for index, layer in enumerate(stats.layers.values()):
        layer.coherence_ema = 0.1 + 0.001 * index

    off = score_modules(graph, stats, ScoreConfig(use_coherence=False))
    on = score_modules(graph, stats, ScoreConfig(use_coherence=True))

    name = next(iter(stats.layers))
    assert off.modules[name].coherence_rank is None
    assert on.modules[name].coherence_rank is not None
    assert on.modules[name].score < off.modules[name].score
