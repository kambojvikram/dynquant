"""Budget accounting and bit allocation.

The load-bearing test in this file is
:func:`test_scores_change_the_allocation_at_a_three_bit_target`. The supplement's
allocator returned the floor map whenever floors exceeded the budget, which is
every target below about 3.5 bits -- so at the paper's own headline 3-bit setting
the importance scores had no effect on the result at all, and the ablation table
cannot have been produced by the allocator as shipped. That is the defect this
phase exists to fix, and it is a defect that leaves no trace: the bit map is
plausible, the model works, and the method simply is not running.

The rest of the file guards the accounting, because a budget that ignores scales
and offsets produces a "3-bit" checkpoint that is 3.25 bits on disk.
"""

from __future__ import annotations

import pytest
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.allocate.budget import Budget, module_stored_bits, parse_size
from dynquant.allocate.knapsack import InfeasibleTargetError, allocate_bits
from dynquant.allocate.policy import AllocationPolicy
from dynquant.graph.classify import classify_model
from dynquant.graph.roles import UNQUANTIZED_FLOOR, ModuleRole
from dynquant.score.sensitivity import SensitivityTable


@pytest.fixture(scope="module")
def graph():
    return classify_model(Qwen3_5ForCausalLM(tie=True))


def _flat_scores(graph, value: float = 0.5) -> dict[str, float]:
    return {info.name: value for info in graph.quantizable()}


# --------------------------------------------------------------------------
# Size parsing and accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1GiB", 1024**3),
        ("1GB", 1000**3),
        ("6.5GiB", int(6.5 * 1024**3)),
        ("700MB", 700 * 1000**2),
        ("512", 512),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


def test_gib_and_gb_stay_distinct() -> None:
    """The 7% between them is wider than the gap between adjacent bit maps."""
    assert parse_size("1GiB") != parse_size("1GB")


def test_stored_bits_include_scales_and_offsets(graph) -> None:
    """A 3-bit tensor at group 128 occupies 3.25 bits per weight, not 3.

    The supplement budgeted the payload only, so its "3-bit" checkpoints were
    consistently larger than advertised -- and the gap grows as the group shrinks.
    """
    info = next(i for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN)

    counted = module_stored_bits(info, 3, group_size=128, symmetric=False)
    assert counted > info.num_params * 3
    assert counted / info.num_params == pytest.approx(3.25, abs=0.01)


def test_symmetric_pays_half_the_metadata(graph) -> None:
    """Symmetric stores a scale per group; asymmetric stores a scale and an offset."""
    info = next(i for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN)
    asymmetric = module_stored_bits(info, 3, group_size=128, symmetric=False)
    symmetric = module_stored_bits(info, 3, group_size=128, symmetric=True)
    payload = info.num_params * 3
    assert (asymmetric - payload) == pytest.approx(2 * (symmetric - payload))


def test_a_smaller_group_costs_more_metadata(graph) -> None:
    """Group size is a quality/size dial, and the budget has to see both ends."""
    info = next(i for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN)
    coarse = module_stored_bits(info, 3, group_size=128)
    fine = module_stored_bits(info, 3, group_size=32)
    assert fine > coarse
    assert fine / info.num_params == pytest.approx(4.0, abs=0.01)  # 3 + 32/32


def test_unquantized_tensors_cost_compute_dtype(graph) -> None:
    """No groups, no scales -- just the weights at the compute dtype."""
    info = next(iter(graph.quantizable()))
    assert module_stored_bits(info, UNQUANTIZED_FLOOR) == info.num_params * UNQUANTIZED_FLOOR


# --------------------------------------------------------------------------
# Budget resolution
# --------------------------------------------------------------------------


def test_the_three_target_forms_agree(graph) -> None:
    """``target_ratio=0.25`` of fp16 is 4 bits is a size; all three must land together."""
    by_bits = Budget.from_target(graph, target_bits=4.0)
    by_ratio = Budget.from_target(graph, target_ratio=0.25)
    assert by_bits.total_bits == pytest.approx(by_ratio.total_bits)

    by_size = Budget.from_target(graph, target_size=int(by_bits.total_bits // 8))
    assert by_size.total_bits == pytest.approx(by_bits.total_bits, rel=1e-6)


def test_exactly_one_target_is_required(graph) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Budget.from_target(graph, target_bits=3.0, target_ratio=0.2)
    with pytest.raises(ValueError, match="exactly one"):
        Budget.from_target(graph)


def test_the_denominator_counts_every_parameter(graph) -> None:
    """Including the tensors that stay at compute dtype.

    Their bits are in the numerator, so leaving their params out of the
    denominator inflates the reported average -- the model looks like it is at
    more bits than it is.
    """
    budget = Budget.from_target(graph, target_bits=3.0)
    assert budget.denominator == graph.total_params()
    assert budget.denominator > graph.unique_params() - graph.unquantized_params()


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def test_allocation_hits_the_target(graph) -> None:
    scores = _flat_scores(graph)
    for target in (3.0, 3.5, 4.0, 5.0):
        budget = Budget.from_target(graph, target_bits=target)
        result = allocate_bits(graph, scores, budget)
        assert result.average_bits <= target + 1e-9, f"{target}: overspent"
        assert result.average_bits == pytest.approx(target, abs=0.02), f"{target}: underspent"


def test_every_quantizable_module_gets_a_width(graph) -> None:
    result = allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=4.0))
    assert set(result.bits) == {info.name for info in graph.quantizable()}


def test_scores_change_the_allocation_at_a_three_bit_target(graph) -> None:
    """The bug-4 regression guard.

    At 3.0 bits this model's floors are unsatisfiable -- they demand about 4.77
    average bits. The supplement's allocator detected exactly this condition and
    returned the floor map, so two completely different importance rankings
    produced byte-identical checkpoints and the method silently stopped running.

    Here the allocator downgrades below the soft floors by lowest damage, so the
    ranking decides *which* tensors give up precision. Reversing the ranking must
    therefore produce a different map.
    """
    names = [info.name for info in graph.quantizable()]
    ascending = {name: i / len(names) for i, name in enumerate(names)}
    descending = {name: 1.0 - i / len(names) for i, name in enumerate(names)}

    budget = Budget.from_target(graph, target_bits=3.0)
    first = allocate_bits(graph, ascending, budget)
    second = allocate_bits(graph, descending, budget)

    assert first.bits != second.bits, "importance scores had no effect on the allocation"
    differing = sum(1 for name in names if first.bits[name] != second.bits[name])
    assert differing > len(names) // 10, (
        f"only {differing}/{len(names)} modules moved; the scores are barely biting"
    )
    # And both still respect the budget they were given.
    for result in (first, second):
        assert result.average_bits <= 3.0 + 1e-9


def test_a_high_score_buys_more_bits_than_a_low_one(graph) -> None:
    """The direction of the effect, not just its presence."""
    peers = [i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN]
    assert len(peers) >= 4

    scores = _flat_scores(graph, 0.01)
    scores[peers[0]] = 1.0  # one clear favourite among identical siblings

    result = allocate_bits(graph, scores, Budget.from_target(graph, target_bits=3.0))
    assert result.bits[peers[0]] >= max(result.bits[name] for name in peers[1:])


def test_structural_floors_survive_an_impossible_budget(graph) -> None:
    """Soft floors bend; correctness constraints do not.

    A decay projection at 2 bits does not make the model slightly worse -- error
    in a recurrence coefficient compounds along the sequence rather than
    averaging out. The budget is not allowed to buy savings there.
    """
    policy = AllocationPolicy()
    result = allocate_bits(
        graph, _flat_scores(graph), Budget.from_target(graph, target_bits=2.6), policy
    )
    for info in graph.quantizable():
        if policy.is_structural(info.role, info.tied_roles):
            assert result.bits[info.name] >= policy.floor_for(info.role, info.tied_roles), (
                f"{info.name} ({info.role.value}) was cut below its structural floor"
            )


def test_the_tied_embedding_takes_the_lm_head_floor(graph) -> None:
    """One tensor, two roles, and the stricter one wins.

    ``embed_tokens`` and ``lm_head`` are the same 26% of this model, and the
    allocator only ever reaches the tensor under the embedding's name. The
    embedding's own floor is lower than the head's, so without strictest-wins the
    tensor that produces every logit would be sized as if it were only an input
    lookup table.
    """
    policy = AllocationPolicy()
    shared = graph["model.embed_tokens"]
    assert shared.tied_roles == (ModuleRole.LM_HEAD,)

    alone = policy.floor_for(shared.role)
    tied = policy.floor_for(shared.role, shared.tied_roles)
    assert tied == 8
    assert tied > alone, "the tie made no difference; the strict role was ignored"

    # And a budget with room honours it rather than treating it as advisory.
    result = allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=8.0))
    assert result.bits[shared.name] >= 8


def test_the_biggest_tensor_is_not_the_cheapest_thing_to_cut(graph) -> None:
    """Damage is extensive, so size must not act as a discount.

    Pricing a downgrade as ``score * Δerror`` while its saving scales with
    ``num_params`` leaves the ratio ``score * Δerror / num_params``, which makes the
    largest tensor in the model the cheapest place to find bits. On a tied model
    that is the embedding -- a quarter of the parameters, feeding both the residual
    stream and every logit -- and it would be cut to the minimum width before a
    single small projection gave up anything. Nothing about the resulting map looks
    wrong: the histogram is sensible, the target is hit, and the model is ruined.
    """
    biggest = max(graph.quantizable(), key=lambda i: i.num_params)
    assert biggest.name == "model.embed_tokens"
    peers = [i for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN]

    result = allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=3.0))
    assert result.bits[biggest.name] >= min(result.bits[i.name] for i in peers), (
        f"the largest tensor ({biggest.num_params:,} params) was cut to "
        f"{result.bits[biggest.name]}b while smaller peers kept more precision"
    )


def test_a_zero_score_is_cut_to_the_minimum_for_free(graph) -> None:
    """Documents why the ranker must never emit an exact zero.

    Damage is priced as ``score * params * Δerror``, so a score of 0 makes every cut
    free and the module goes straight to the 2-bit minimum -- regardless of how loose
    the budget is or how large the tensor. That is correct behaviour for a score that
    genuinely means "worthless", and catastrophic for one that reached zero as an
    artifact of the rank formula. On Qwen3.5-2B it put exactly 20 of 187 modules at 2
    bits at a 4-bit target, two per role, with reconstruction errors up to 0.61
    relative; the model scored 20.8% against 65.1% unquantized.

    The fix is upstream, in :func:`~dynquant.score.ranks.percentile_ranks`, which now
    maps to the open interval. This test pins the consequence so the two stay
    connected: if the allocator is ever changed to treat a zero score as merely
    lowest-priority, the ranker's open interval stops being load-bearing and the
    comment explaining it becomes wrong.

    The target is 4.0 because that is where the mechanism acts, and it is the target the
    incident above happened at. Under a loose target the downgrade pass never runs, so
    nothing is cut and a zero score shows up only as being upgraded last -- which is not
    a stable observable. The upgrade pass accepts a zero-value move when no priced move
    can afford the slack that is left, so whether the victim finishes below its peers
    depends on the size of that remainder, and the remainder moves with any change to
    budget accounting. Pricing the tensors the graph refuses moved it by 0.09% of the
    budget and flipped this assertion while changing nothing about how a zero is
    treated. At 4.0 the victim is cut through its own floor to the minimum and is the
    only module in the model there, which is the fact worth pinning.
    """
    scores = _flat_scores(graph)
    victim = next(i for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN)
    scores[victim.name] = 0.0

    result = allocate_bits(graph, scores, Budget.from_target(graph, target_bits=4.0))
    peers = [
        i.name
        for i in graph.quantizable()
        if i.role is ModuleRole.MLP_DOWN and i.name != victim.name
    ]
    assert result.bits[victim.name] == 2, "a free cut should go all the way to the minimum"
    assert result.bits[victim.name] < victim.floor_bits, "through its floor to get there"
    assert result.histogram()[2] == 1, "and nothing else in the model should be down here"
    assert result.bits[victim.name] < min(result.bits[n] for n in peers)


def test_floor_violations_are_reported_not_hidden(graph) -> None:
    """A breached floor is a real quality decision and has to be visible."""
    result = allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=3.0))
    assert result.violations, "3.0 bits cannot fit this model's floors; expected violations"
    for violation in result.violations:
        assert violation.assigned_bits < violation.floor_bits
        assert violation.name in result.bits
    assert "floors breached" in result.summary()


def test_a_generous_budget_breaches_nothing(graph) -> None:
    result = allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=8.0))
    assert result.violations == ()
    for info in graph.quantizable():
        assert result.bits[info.name] >= info.floor_bits


def test_an_impossible_target_raises_rather_than_missing_it(graph) -> None:
    """Below the structural floors there is no allocation, and saying so beats
    emitting a checkpoint that quietly misses the size the caller asked for."""
    with pytest.raises(InfeasibleTargetError, match="structural floors"):
        allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=1.5))


def test_hard_floors_reproduce_the_papers_dead_end(graph) -> None:
    """``soft_floors=False`` is the compat path, and it must still be the old bug.

    The ``paper-3.15`` preset has to reproduce published numbers, defects included,
    so this asserts the broken behaviour survives *behind the flag* -- overshooting
    the budget rather than allocating under it.
    """
    policy = AllocationPolicy(soft_floors=False)
    result = allocate_bits(
        graph, _flat_scores(graph), Budget.from_target(graph, target_bits=3.0), policy
    )
    assert result.violations == ()
    assert result.average_bits > 3.0, "hard floors at a 3-bit target must overshoot"


def test_allocation_is_deterministic(graph) -> None:
    """Same inputs, same map -- otherwise a rerun silently produces a different
    checkpoint and no comparison across runs means anything."""
    scores = {info.name: (hash(info.name) % 1000) / 1000 for info in graph.quantizable()}
    budget = Budget.from_target(graph, target_bits=3.2)
    assert allocate_bits(graph, scores, budget).bits == allocate_bits(graph, scores, budget).bits


def test_unscored_modules_are_not_silently_dropped(graph) -> None:
    """A module absent from the stats file still needs a width."""
    partial = _flat_scores(graph)
    victim = next(iter(partial))
    del partial[victim]
    result = allocate_bits(graph, partial, Budget.from_target(graph, target_bits=4.0))
    assert victim in result.bits


# --------------------------------------------------------------------------
# Pricing the move with a measured sensitivity instead of a rank
# --------------------------------------------------------------------------


def _sensitivity(graph, favourite: str, *, bits=(2, 3, 4, 8)) -> SensitivityTable:
    """A table where one module is expensive to quantize and its peers are cheap.

    Values fall with width the way a real table's do, so the *move* price -- the
    difference between two adjacent entries, which is what the knapsack reads -- is
    positive everywhere and largest for the favourite. They also scale with parameter
    count, because a real entry is a sum over every element of the weight; a table
    that did not would make a 500M-parameter tensor as cheap to cut as a 4k one.
    """
    table = SensitivityTable(bit_options=tuple(bits))
    for info in graph.quantizable():
        weight = 100.0 if info.name == favourite else 1.0
        table.values[info.name] = {b: weight * info.num_params * 4.0**-b for b in bits}
    return table


def test_a_measured_sensitivity_outranks_the_score_that_disagrees_with_it(graph) -> None:
    """The whole point of collecting moments.

    On the model this was built for, allocating a 3.125-bit budget by measured
    sensitivity recovered 85.5% of the uniform-3-bit damage against 28.5% for the
    rank-product score -- and 28.5% is inside the band spanned by random allocations
    of the same budget. So when a table is present it must win, not merely be
    consulted.
    """
    peers = [i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN]
    assert len(peers) >= 4

    scores = _flat_scores(graph, 0.5)
    scores[peers[0]] = 0.001  # the score says this one is the least important
    budget = Budget.from_target(graph, target_bits=3.0)

    without = allocate_bits(graph, scores, budget)
    with_table = allocate_bits(graph, scores, budget, sensitivity=_sensitivity(graph, peers[0]))

    assert with_table.bits[peers[0]] > without.bits[peers[0]]
    assert with_table.bits[peers[0]] >= max(with_table.bits[name] for name in peers[1:])


def _hole(graph, table: SensitivityTable, role: ModuleRole) -> list[str]:
    """Remove a role from the table the way an uncollectable module leaves it."""
    orphans = [i.name for i in graph.quantizable() if i.role is role]
    for name in orphans:
        del table.values[name]
    table.unestimable = tuple(orphans)
    return orphans


def test_an_unestimable_module_still_gets_priced(graph) -> None:
    """A module missing from the table must fall back to its score, not to zero.

    Zero sensitivity reads as "measured, costs nothing", which makes an unmeasured
    module the *first* thing the allocator takes bits from -- the exact inversion of
    what a missing measurement means. Embeddings land here on every real model,
    because a token id is not a feature axis.

    Half the orphans are scored well above the population and half well below, so the
    assertion is about the *ordering* the fallback produces rather than about a
    constant, which a calibration is free to absorb and should.
    """
    table = _sensitivity(graph, favourite="")
    orphans = _hole(graph, table, ModuleRole.MLP_UP)
    assert len(orphans) >= 4

    loud, quiet = orphans[::2], orphans[1::2]
    scores = _flat_scores(graph, 0.5)
    for name in loud:
        scores[name] = 1.0
    for name in quiet:
        scores[name] = 0.05

    result = allocate_bits(
        graph, scores, Budget.from_target(graph, target_bits=3.0), sensitivity=table
    )
    assert min(result.bits[n] for n in loud) > max(result.bits[n] for n in quiet)


def test_the_fallback_is_calibrated_against_the_rung_not_the_floor(graph) -> None:
    """The two populations sit at different floors, and must still be comparable.

    MLP up/down floor at 3 bits while attention floors at 4, so "next step from
    here" means 3->4 for one group and 4->8 for the other -- a factor of three in the
    width span alone. Calibrating on the raw step values charges that factor twice
    and drives the unmeasured group under every measured peer of the same size, on no
    evidence. The check is that an orphan scored at the population's own level lands
    at the same width as the measured module it is a structural twin of.
    """
    table = _sensitivity(graph, favourite="")
    orphans = _hole(graph, table, ModuleRole.MLP_UP)
    twins = [i.name for i in graph.quantizable() if i.role is ModuleRole.MLP_DOWN]
    assert twins and orphans

    result = allocate_bits(
        graph,
        _flat_scores(graph, 0.5),
        Budget.from_target(graph, target_bits=3.0),
        sensitivity=table,
    )
    assert min(result.bits[n] for n in orphans) >= min(result.bits[n] for n in twins)


def test_a_partial_table_still_hits_the_budget(graph) -> None:
    """Loss-unit prices and score-proxy prices share one heap, so they have to be
    calibrated against each other or the covered half wins every comparison by unit
    choice alone. Whatever the calibration does, the target still has to be met."""
    table = _sensitivity(graph, favourite="")
    keep = [i.name for i in graph.quantizable()][::2]
    table.values = {name: table.values[name] for name in keep if name in table.values}
    table.unestimable = tuple(i.name for i in graph.quantizable() if i.name not in table.values)

    budget = Budget.from_target(graph, target_bits=3.4)
    result = allocate_bits(graph, _flat_scores(graph), budget, sensitivity=table)
    assert result.average_bits <= 3.4 + 1e-9
    assert set(result.bits) == {i.name for i in graph.quantizable()}


# --------------------------------------------------------------------------
# Which price decided the widths
# --------------------------------------------------------------------------


def test_the_map_records_how_much_of_itself_the_proxy_priced(graph) -> None:
    """The split has to survive onto the artifact, in parameters as well as modules.

    It was reachable only as an INFO log line, and the campaign that needed it runs
    the allocator as a subprocess at the default level and keeps the saved map. On a
    batched-expert MoE the Gauss-Newton estimate is unavailable for every expert
    bank by construction, which on the model this was written for is 44 of 133
    modules and 91.5% of the quantizable parameters: a count that reads as a footnote
    and a mass that is the whole result. So both are asserted, and the share is
    asserted to be the larger of the two -- a map that reported only the count would
    pass a test that only checked the count.
    """
    table = _sensitivity(graph, favourite="")
    orphans = _hole(graph, table, ModuleRole.MLP_DOWN)
    result = allocate_bits(
        graph,
        _flat_scores(graph),
        Budget.from_target(graph, target_bits=3.4),
        sensitivity=table,
    )

    pricing = result.pricing
    assert pricing is not None
    assert pricing.proxied_modules == len(orphans)
    assert pricing.measured_modules == len(list(graph.quantizable())) - len(orphans)
    assert pricing.proxied_params == sum(graph[name].num_params for name in orphans)
    assert pricing.scale is not None and pricing.scale > 0.0
    modules_share = pricing.proxied_modules / (pricing.proxied_modules + pricing.measured_modules)
    assert pricing.proxied_share > modules_share, "MLP down is the mass, not the count"


def test_a_fully_measured_map_says_so_rather_than_saying_nothing(graph) -> None:
    """Absence of a pricing record must not be how "all measured" is expressed.

    ``None`` already means "this path never priced anything" -- a hard-floor map, or
    one written before the field existed. If a fully measured run also produced
    ``None`` the reader could not tell a complete table from a missing one, and the
    complete case is the one worth being able to prove.
    """
    result = allocate_bits(
        graph,
        _flat_scores(graph),
        Budget.from_target(graph, target_bits=3.4),
        sensitivity=_sensitivity(graph, favourite=""),
    )
    pricing = result.pricing
    assert pricing is not None
    assert pricing.proxied_modules == 0
    assert pricing.proxied_share == 0.0
    assert pricing.scale == 1.0
    assert "measured" in pricing.summary()


def test_no_common_scale_is_recorded_as_none_not_as_one(graph) -> None:
    """The case where the proxy-priced modules' order is arbitrary must be legible.

    When no positive proxy price exists there is nothing to calibrate against and
    the multiplier stays 1.0 internally -- an arbitrary number that happens to look
    deliberate. Writing ``scale: 1.0`` onto the artifact would launder that into a
    calibration that never happened, so the field carries ``None`` and the summary
    says the order is arbitrary.
    """
    table = _sensitivity(graph, favourite="")
    orphans = _hole(graph, table, ModuleRole.MLP_DOWN)
    scores = _flat_scores(graph)
    for name in orphans:
        scores[name] = 0.0

    result = allocate_bits(
        graph, scores, Budget.from_target(graph, target_bits=3.4), sensitivity=table
    )
    pricing = result.pricing
    assert pricing is not None
    assert pricing.proxied_modules == len(orphans)
    assert pricing.scale is None
    assert "arbitrary" in pricing.summary()


def test_a_run_with_no_moments_at_all_is_recorded_as_such(graph) -> None:
    """ "Nothing was measured" is a finding, not an absence, and it is not "uncalibrated".

    With no table every module takes the same proxy formula, so the heap is internally
    consistent and the order it produces means something -- which is why the scale is
    1.0 here and not ``None``. ``None`` is reserved for the mixed case that could not
    be reconciled. Collapsing the two would put the most-trustworthy ordering and the
    least-trustworthy one under the same marker on the artifact.
    """
    result = allocate_bits(graph, _flat_scores(graph), Budget.from_target(graph, target_bits=3.4))
    pricing = result.pricing
    assert pricing is not None
    assert pricing.measured_modules == 0
    assert pricing.proxied_share == 1.0
    assert pricing.scale == 1.0
    assert "arbitrary" not in pricing.summary()
