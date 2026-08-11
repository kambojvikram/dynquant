"""The control arm, and the ways a control arm quietly stops being one.

A null that is slightly wrong is worse than no null, because it produces a number
in the same units as the result and gets subtracted from it. Three failures would
each do that here and none of them would raise:

**The null does not null.** A permutation with many fixed points, or one that
skips the sensitivity table and leaves the measured half of the pricing intact,
still moves the widths a little and gets reported as "the signal is worth 3
points" when the signal was mostly still there. So the shuffle is checked to be a
real permutation, to carry ``dL`` with the score rather than beside it, and to
report its own fixed points instead of letting the caller assume there were none.

**The null changes something else too.** If nulling moved a role, relaxed a floor,
or landed on a different byte total, then the arm differs from the real one in two
ways and the subtraction means nothing. Those are asserted directly against the
real arm's map at the same budget.

**The null is not labelled.** This is the one that survives into a paper. A bit
map allocated from shuffled scores is a normal-looking bit map; its provenance
says ``sensitivity`` unless something makes it say otherwise, and six weeks later
it is a checkpoint on a Hub with a number attached. So the label is asserted where
it actually has to appear -- inside the written JSON -- and not only on the object
that produced it.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.allocate.budget import Budget
from dynquant.allocate.knapsack import allocate_bits
from dynquant.allocate.policy import AllocationPolicy
from dynquant.errors import DynQuantError
from dynquant.graph.classify import classify_model
from dynquant.score.null import (
    NULL_MODES,
    STOCHASTIC_NULL_MODES,
    apply_null,
    uses_seed,
)
from dynquant.score.sensitivity import SensitivityTable


@pytest.fixture(scope="module")
def graph():
    return classify_model(Qwen3_5ForCausalLM(tie=True))


@pytest.fixture(scope="module")
def scores(graph):
    """Distinct per module, so a permutation is visible rather than a no-op.

    Ranked scores in a real run are near-distinct for the same reason, and a
    fixture of equal values would let a broken permutation pass every assertion
    in this file.
    """
    return {info.name: float(i + 1) for i, info in enumerate(graph.quantizable())}


@pytest.fixture(scope="module")
def sensitivity(graph, scores):
    """A table over the first half of the modules, so the boundary is crossable.

    Half rather than all: this campaign's own model measures 89 of 133 modules
    and proxies the rest, and a table covering everything would never exercise
    ``estimability_changed``.
    """
    names = [info.name for info in graph.quantizable()]
    covered = names[: len(names) // 2]
    return SensitivityTable(
        values={name: {b: scores[name] * (4.0**-b) for b in (2, 3, 4, 8)} for name in covered},
        unestimable=tuple(names[len(names) // 2 :]),
    )


def _by_role(graph) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for info in graph.quantizable():
        out.setdefault(info.role.value, []).append(info.name)
    return out


# --------------------------------------------------------------------------
# It nulls
# --------------------------------------------------------------------------


def test_shuffle_is_a_permutation_within_each_role(graph, scores, sensitivity) -> None:
    """Same numbers, same roles, different owners -- the definition of the arm.

    Turns red if the shuffle ever becomes a resample, a global permutation, or a
    partial one. A global permutation would be the subtle version: still a
    permutation, still passes a total-multiset check, and it hands an embedding's
    score to a router on a model where the roles differ in shape by three orders
    of magnitude, so the arm would measure that scores are not transferable
    across roles and report it as the signal's contribution.
    """
    nulled, _, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)

    assert set(nulled) == set(scores)
    for role, members in _by_role(graph).items():
        before = Counter(scores[name] for name in members)
        after = Counter(nulled[name] for name in members)
        assert after == before, role


def test_shuffle_actually_moves_most_of_them(graph, scores, sensitivity) -> None:
    """A permutation that fixes everything is a null arm that measures nothing."""
    _, _, report = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)

    assert report.moved + report.fixed == report.modules
    # A uniform random permutation fixes one point per role in expectation, so a
    # bound well under half is generous and still catches an identity map.
    assert report.moved > report.modules * 0.6


def test_the_sensitivity_row_travels_with_the_score(graph, scores, sensitivity) -> None:
    """Both halves of the driving quantity move together, or the null is partial.

    91.5% of this campaign's model is priced from the score and the rest from
    measured ``dL``. Permuting only the score would leave the measured half of the
    pricing pointing at the right modules, and the arm would understate the
    signal by whatever that half is worth.
    """
    nulled, table, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)
    assert table is not None

    for name, widths in table.values.items():
        # The fixture ties every row to its own score, so the row a module ends up
        # with must be the row belonging to whoever donated its score.
        assert widths[4] == pytest.approx(nulled[name] * (4.0**-4))


def test_unestimable_is_recomputed_not_carried(graph, scores, sensitivity) -> None:
    """It is the complement of ``values``; a stale copy sends the caller to a score
    for a module that now has a measured row, and vice versa."""
    _, table, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)
    assert table is not None

    names = {info.name for info in graph.quantizable()}
    assert set(table.values) | set(table.unestimable) == names
    assert not set(table.values) & set(table.unestimable)


def test_crossing_the_pricing_boundary_is_counted(graph, scores, sensitivity) -> None:
    """Reported rather than absorbed: it is a second thing the null changed."""
    _, _, report = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)
    assert report.estimability_changed > 0
    assert str(report.estimability_changed) in report.summary()


def test_uniform_removes_the_ordering_and_the_table(graph, scores, sensitivity) -> None:
    """There is no measured sensitivity without the fine-tune, so a signal-free arm
    cannot keep the table -- dropping it is the arm, not a second change."""
    nulled, table, report = apply_null(graph, scores, sensitivity, mode="uniform", seed=7)

    assert set(nulled.values()) == {1.0}
    assert table is None
    assert report.seed is None
    assert report.label == "null:uniform"


# --------------------------------------------------------------------------
# It nulls reproducibly
# --------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_arm(graph, scores, sensitivity) -> None:
    a, _, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=3)
    b, _, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=3)
    assert a == b


def test_a_different_seed_gives_a_different_arm(graph, scores, sensitivity) -> None:
    """One seed is one draw from the null. A report quoting a single shuffled arm
    is quoting a sample of size one, and this is what makes a second one cheap."""
    a, _, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)
    b, _, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=1)
    assert a != b


def test_the_permutation_does_not_depend_on_graph_order(graph, scores, sensitivity) -> None:
    """Seeded off a sorted list, so the arm reproduces from the map alone.

    ``graph.quantizable()`` yields in module-tree order. Seeding a permutation off
    an order this module does not own makes the control reproducible only for as
    long as nobody reorders a model's submodules -- and the reproduction would
    fail silently, as a different map at the same seed.
    """
    reversed_scores = dict(reversed(list(scores.items())))
    a, _, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)
    b, _, _ = apply_null(graph, reversed_scores, sensitivity, mode="shuffle", seed=0)
    assert a == b


def test_a_singleton_role_is_named_not_silently_fixed(graph, scores, sensitivity) -> None:
    """A role with one member permutes to itself, and the arm still holds that
    module at whatever the signal chose. On a tied embedding that is not a
    rounding detail, so it is named rather than folded into ``fixed``."""
    singletons = {role for role, members in _by_role(graph).items() if len(members) == 1}
    _, _, report = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)

    assert set(report.singleton_roles) == singletons
    if singletons:
        assert "could not be permuted" in report.summary()


def test_an_unknown_mode_refuses(graph, scores) -> None:
    with pytest.raises(DynQuantError, match="unknown null mode"):
        apply_null(graph, scores, None, mode="scramble")


def test_no_stats_still_nulls(graph) -> None:
    """A module the score never saw scores 0, and the donor's 0 must arrive as 0
    rather than as a KeyError -- the null runs before the allocator's own
    missing-name default and must not pre-empt it."""
    nulled, _, _ = apply_null(graph, {}, None, mode="shuffle", seed=0)
    assert set(nulled.values()) == {0.0}


# --------------------------------------------------------------------------
# It nulls and nothing else
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", NULL_MODES)
def test_the_null_arm_lands_on_the_same_budget(graph, scores, sensitivity, mode) -> None:
    """The comparison is at matched bytes or it is not a comparison.

    Same graph, same floors, same policy, same target: only the widths may move.
    A null that also shifted the byte total would be confounded with size, which
    is the one confound this whole panel is built to exclude.
    """
    policy = AllocationPolicy(group_size=128)
    budget = Budget.from_target(graph, target_bits=3.0, group_size=128)

    real = allocate_bits(graph, scores, budget, policy, sensitivity=sensitivity)
    null_scores, null_table, _ = apply_null(graph, scores, sensitivity, mode=mode, seed=0)
    null = allocate_bits(graph, null_scores, budget, policy, sensitivity=null_table)

    assert null.budget_bits == real.budget_bits
    assert null.denominator == real.denominator
    assert null.allocated_bits <= budget.total_bits
    assert set(null.bits) == set(real.bits)


@pytest.mark.parametrize("mode", NULL_MODES)
def test_the_null_arm_respects_the_same_floors(graph, scores, sensitivity, mode) -> None:
    """Role structure is the thing under test, so the null must not relax it.

    If the control breached floors the real arm respected, its losses would be
    attributable to the breach rather than to the missing signal.
    """
    policy = AllocationPolicy(group_size=128)
    budget = Budget.from_target(graph, target_bits=3.0, group_size=128)

    null_scores, null_table, _ = apply_null(graph, scores, sensitivity, mode=mode, seed=0)
    null = allocate_bits(graph, null_scores, budget, policy, sensitivity=null_table)

    floors = {
        info.name: policy.floor_for(info.role, info.tied_roles) for info in graph.quantizable()
    }
    for violation in null.violations:
        assert violation.floor_bits == floors[violation.name]
    structural = [
        info.name
        for info in graph.quantizable()
        if policy.is_structural(info.role, info.tied_roles)
    ]
    for name in structural:
        assert null.bits[name] >= floors[name]


def test_the_null_arm_is_not_the_real_arm(graph, scores, sensitivity) -> None:
    """The arm has to be capable of a different answer, or it is not a control.

    An assertion that looks tautological and is not: with soft floors off, or a
    budget the floors already satisfy, the allocator returns the floor map for
    every score and a null arm scores identically for a reason that has nothing
    to do with the signal. That is the supplement's original defect, and this is
    where it would show up in the control rather than in the result.
    """
    policy = AllocationPolicy(group_size=128)
    budget = Budget.from_target(graph, target_bits=3.0, group_size=128)

    real = allocate_bits(graph, scores, budget, policy, sensitivity=sensitivity)
    null_scores, null_table, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)
    null = allocate_bits(graph, null_scores, budget, policy, sensitivity=null_table)

    assert null.bits != real.bits


# --------------------------------------------------------------------------
# It says so
# --------------------------------------------------------------------------


def test_the_label_reaches_the_written_map(graph, scores, sensitivity, tmp_path) -> None:
    """Asserted in the JSON and not on the object that made it.

    A control arm's whole risk is being read later as a result. Later means from
    the file, by someone without the run log, so the file is where the label has
    to be -- and an `allocator` field reading `sensitivity` on a map allocated
    from shuffled scores is how a control becomes a headline.
    """
    from dynquant.commands import _shared

    policy = AllocationPolicy(group_size=128)
    budget = Budget.from_target(graph, target_bits=3.0, group_size=128)
    null_scores, null_table, report = apply_null(graph, scores, sensitivity, mode="shuffle", seed=4)
    bit_map = allocate_bits(graph, null_scores, budget, policy, sensitivity=null_table)

    written = _shared.write_bit_maps(
        tmp_path / "dqnull_3b.json",
        {"3.00": bit_map},
        model="m",
        stats="s",
        allocator=f"sensitivity+{report.label}",
        group_size=128,
        extra={"score_null": report.as_dict()},
    )
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["allocator"] == "sensitivity+null:shuffle(seed=4)"
    assert payload["score_null"]["mode"] == "shuffle"
    assert payload["score_null"]["seed"] == 4
    assert payload["score_null"]["moved"] == report.moved


def test_a_real_arm_carries_no_null_block(graph, scores, sensitivity) -> None:
    """Absence must mean absence: a reader keying on `score_null` to decide whether
    a map is a control gets nothing on a real one, not a mode of `none`."""
    from dynquant.commands._shared import AllocationInputs

    inputs = AllocationInputs(
        graph, scores, policy=AllocationPolicy(group_size=128), sensitivity=sensitivity
    )
    assert inputs.null_report is None
    assert inputs.allocator == "sensitivity"


@pytest.mark.parametrize(
    ("mode", "sens", "expected"),
    [
        ("shuffle", True, "sensitivity+null:shuffle(seed=0)"),
        ("shuffle", False, "rank_product+null:shuffle(seed=0)"),
        ("uniform", True, "null:uniform"),
        ("uniform", False, "null:uniform"),
    ],
)
def test_the_allocator_field_says_which_null_over_which_pricing(
    graph, scores, sensitivity, mode, sens, expected
) -> None:
    """Both halves, because both change what the map means.

    ``uniform`` drops the table, so naming a pricing that no longer ran would be
    the same defect pointed the other way.
    """
    from dynquant.commands._shared import AllocationInputs

    table = sensitivity if sens else None
    null_scores, null_table, report = apply_null(graph, scores, table, mode=mode, seed=0)
    inputs = AllocationInputs(
        graph,
        null_scores,
        policy=AllocationPolicy(group_size=128),
        sensitivity=null_table,
        null_report=report,
    )
    assert inputs.allocator == expected


def test_a_seed_names_an_arm_only_when_the_mode_actually_draws() -> None:
    """One function owns "does the seed matter here", and both readers ask it.

    `NullReport.label` decides whether the seed belongs in the allocator string, and a
    caller planning arms decides whether two seeds are two arms or one arm named twice.
    Answered independently, the two go out of step in the direction that costs a
    measurement: a deterministic mode given two seeds plans two arms writing to one
    record, and a stochastic one given two seeds plans one arm that silently keeps
    whichever draw ran last.

    Turns red when: a second caller starts answering it with `mode == "uniform"`.
    """
    assert set(STOCHASTIC_NULL_MODES) <= set(NULL_MODES)
    assert [mode for mode in NULL_MODES if uses_seed(mode)] == ["shuffle"]


def test_a_deterministic_null_records_no_seed_however_it_was_called(graph, scores) -> None:
    """`uniform` ignores the seed, so a report carrying one would be reporting a choice.

    The seed reaches `apply_null` from a flag that is shared across every mode in one
    invocation, so `--score-null uniform --null-seed 7` is not a user asking for a seeded
    uniform -- it is the shuffle's seed arriving at an arm that has no use for it. A report
    that stored 7 would put it in the manifest and in the allocator string, and two
    identical arms would look like two configurations.

    Turns red when: the report starts storing the seed it was passed rather than the seed
    that applied.
    """
    _, table, report = apply_null(graph, scores, None, mode="uniform", seed=7)

    assert table is None
    assert report.seed is None
    assert report.label == "null:uniform"
