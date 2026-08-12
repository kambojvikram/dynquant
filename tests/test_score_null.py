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
import torch
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.allocate.budget import Budget
from dynquant.allocate.knapsack import allocate_bits
from dynquant.allocate.policy import AllocationPolicy
from dynquant.errors import DynQuantError
from dynquant.graph.classify import classify_model
from dynquant.score.null import (
    NULL_LADDER,
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
        ("flat", True, "sensitivity+null:flat(seed=0)"),
        ("flat", False, "rank_product+null:flat(seed=0)"),
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


def test_a_seed_names_an_arm_only_when_the_mode_actually_draws(graph, scores, sensitivity) -> None:
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
    # Derived from behaviour, not from either tuple. An earlier version of this pinned
    # the deterministic side to a literal `["uniform"]` on the reasoning that the list
    # gains no members -- then `table` arrived and the fix was to edit the literal, which
    # is the shape of an assertion that has stopped asserting. What is actually claimed
    # is that `uses_seed` answers a question about the code: does a second seed produce
    # a second arm? So ask the code. This needs no edit when a mode is added and catches
    # a mode filed on the wrong side of either tuple.
    for mode in NULL_MODES:
        first = apply_null(graph, scores, sensitivity, mode=mode, seed=0)[:2]
        second = apply_null(graph, scores, sensitivity, mode=mode, seed=1)[:2]
        assert uses_seed(mode) == (first != second), mode


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


# --------------------------------------------------------------------------
# The third null: the score goes, the pricing stays
# --------------------------------------------------------------------------


def test_flat_differs_from_shuffle_in_the_score_channel_and_in_nothing_else(
    graph, scores, sensitivity
) -> None:
    """The rung between two arms prices what separates them, so one thing may separate them.

    `flat` exists to split the large step between a permuted arm and a signal-free one into
    what the score bought and what the measured table bought. That split is only a split if
    `flat` is `shuffle` with the score removed and every other input byte-identical -- same
    permutation, same seed, same table. Rebuilt from a fresh `Random` or a different donor
    map it would differ in two things at once, and the rung would price both while reading
    as though it priced one.

    Turns red when: `flat` stops sharing the permutation `shuffle` drew at the same seed.
    """
    flat_scores, flat_table, _ = apply_null(graph, scores, sensitivity, mode="flat", seed=0)
    shuffled, shuffled_table, _ = apply_null(graph, scores, sensitivity, mode="shuffle", seed=0)

    assert set(flat_scores) == set(shuffled)
    assert set(flat_scores.values()) == {1.0}
    assert flat_scores != shuffled, "the fixture's scores are not already constant"
    assert flat_table == shuffled_table


def test_the_nulls_are_nested_in_the_order_they_are_declared(graph, scores, sensitivity) -> None:
    """A ladder over these modes partitions the margin only if each removes the last's more.

    Asserted as the property rather than as the tuple: what a chain built over `NULL_LADDER`
    needs is that mode k+1 removes everything mode k removed, and a test that pinned the
    tuple would go green on a reordering that broke exactly that. Two inputs reach the
    allocator, so nesting is visible in two columns -- the score loses its ordering, then
    the table goes -- and a mode that gave one back while taking the other is a mode whose
    rung is a difference rather than a step.

    Turns red when: a mode is inserted at a rank it does not belong at.
    """
    ordering, tables = [], []
    for mode in NULL_LADDER:
        null_scores, null_table, _ = apply_null(graph, scores, sensitivity, mode=mode, seed=0)
        ordering.append(len(set(null_scores.values())) > 1)
        tables.append(null_table is not None)

    # Once a column goes it stays gone: `sorted(reverse=True)` on the booleans is the
    # nesting, and any other pattern is a mode that put something back.
    assert ordering == sorted(ordering, reverse=True), NULL_LADDER
    assert tables == sorted(tables, reverse=True), NULL_LADDER
    assert (ordering, tables) != ([], []), "there is at least one mode to check"


def test_flat_is_capable_of_a_different_map_from_either_of_its_neighbours(
    graph, scores, sensitivity
) -> None:
    """A rung between two arms that always allocate alike is a guaranteed zero.

    The same argument the `shuffle` arm gets: an arm incapable of a different answer is not
    a control, and a decomposition row that could only ever print +0.00 would read as a
    measurement of the thing it never varied. `flat` has two neighbours on the ladder and
    has to be able to differ from both -- from `shuffle` because the proxied modules lose
    their ordering, from `uniform` because the measured ones keep their prices.

    Turns red when: the score stops reaching the allocator, or the table stops doing.
    """
    policy = AllocationPolicy(group_size=128)
    budget = Budget.from_target(graph, target_bits=3.0, group_size=128)

    maps = {}
    for mode in ("shuffle", "flat", "uniform"):
        null_scores, null_table, _ = apply_null(graph, scores, sensitivity, mode=mode, seed=0)
        maps[mode] = allocate_bits(graph, null_scores, budget, policy, sensitivity=null_table).bits

    assert maps["flat"] != maps["shuffle"]
    assert maps["flat"] != maps["uniform"]


def test_the_table_mode_passes_the_measured_table_through_untouched(
    graph, scores, sensitivity
) -> None:
    """The one mode that isolates a single channel, and identity is what makes it one.

    `flat` was reached for as "the pricing with a constant score" and is not: it shares a
    drawn permutation with `shuffle`, so its table is permuted and the rung below it prices
    a permuted table against no table. This mode is the clean contrast, and it is only clean
    if the table the allocator sees is the *same object* the real arm sees -- a faithful
    rebuild is still a second edit, and a second edit is the thing that made `flat` unable
    to answer this.

    Turns red when: the table is rebuilt, filtered or permuted on the way through.
    """
    null_scores, null_table, report = apply_null(graph, scores, sensitivity, mode="table")

    assert null_table is sensitivity
    assert set(null_scores) == {info.name for info in graph.quantizable()}
    assert set(null_scores.values()) == {1.0}
    assert report.moved == 0
    assert report.fixed == report.modules
    assert report.estimability_changed == 0
    assert report.seed is None


def test_the_table_mode_is_capable_of_a_different_map_from_the_real_arm(
    graph, scores, sensitivity
) -> None:
    """An arm that cannot allocate differently is not a control.

    The same argument every other null gets. This one is worth making explicitly because
    the mode keeps the more informative of the two inputs: if the knapsack priced every
    module from its measured row the score would be dead weight, the map would be
    identical by construction, and the rung would print +0.00 as a fact about the code
    rather than about the signal. It differs because modules the moments never reached are
    priced `score x params x error-curve`.

    Turns red when: the score stops reaching the allocator at all.
    """
    policy = AllocationPolicy(group_size=128)
    budget = Budget.from_target(graph, target_bits=3.0, group_size=128)

    real = allocate_bits(graph, scores, budget, policy, sensitivity=sensitivity).bits
    null_scores, null_table, _ = apply_null(graph, scores, sensitivity, mode="table")
    nulled = allocate_bits(graph, null_scores, budget, policy, sensitivity=null_table).bits

    assert real != nulled


def test_the_ladder_is_a_subset_of_the_modes_and_says_so() -> None:
    """Two tuples, two facts, and the one that must not drift is which is which.

    `NULL_MODES` is every mode the CLI accepts; `NULL_LADDER` is the smaller claim that a
    chain over it partitions a margin. A mode added to the ladder without earning it turns
    a decomposition into a sum of overlapping differences that still adds up, which is the
    failure that prints a clean-looking table and means nothing.

    Turns red when: a mode joins the ladder, or the ladder stops being ordered as declared.
    """
    assert set(NULL_LADDER) <= set(NULL_MODES)
    assert "table" not in NULL_LADDER
    # A contiguous run, so `NULL_MODES.index` -- which is what the panel sorts controls by
    # -- never interleaves a non-rung between two rungs of a ladder it is displaying.
    positions = [NULL_MODES.index(mode) for mode in NULL_LADDER]
    assert positions == sorted(positions)
    assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_the_table_report_says_nothing_moved_and_why(graph, scores, sensitivity) -> None:
    """`moved == 0` reads as a failed null everywhere else in this file.

    For a permutation it means the draw left the signal in place and the arm is overclaiming.
    Here it means nothing was supposed to move. The summary is the only place a reader meets
    that distinction, so it has to make it rather than reporting the zero and leaving it.

    Turns red when: the summary falls through to the permutation wording.
    """
    _, _, report = apply_null(graph, scores, sensitivity, mode="table")
    summary = report.summary()

    assert "not permuted" in summary
    assert "by construction" in summary
    assert "not a rung" in summary
    assert "another module" not in summary
    assert report.label == "null:table"


def test_the_flat_report_says_the_table_survived_the_score(graph, scores, sensitivity) -> None:
    """Its summary is the only place a reader learns this arm still has measured prices.

    `flat` is the one mode whose name does not describe what it kept. Summarised with the
    `shuffle` wording it would read as an arm whose scores were permuted -- they were not,
    they were erased -- and summarised with the `uniform` wording it would claim no
    sensitivity table was consulted, which is the opposite of the arm's purpose.

    Turns red when: a third mode reuses another mode's summary.
    """
    _, _, report = apply_null(graph, scores, sensitivity, mode="flat", seed=2)
    summary = report.summary()

    assert report.label == "null:flat(seed=2)"
    assert report.seed == 2
    assert "scores 1.0" in summary
    assert "permuted within role" in summary
    assert "no sensitivity table" not in summary


# --------------------------------------------------------------------------
# The draw itself, pinned
# --------------------------------------------------------------------------


#: Wide enough that every module pays for its own scales, and no wider.
#:
#: ``is_quantizable`` refuses a tensor whose scales and zeros cost more than the weights
#: they price, so an 8-wide fixture classifies into thirty roles and allocates none of
#: them -- a graph that looks right and yields nothing to permute.
_PINNED_HIDDEN, _PINNED_INTERMEDIATE, _PINNED_VOCAB = 128, 256, 512


class _PinnedCfg:
    """Seven roles of four modules each, with names this test owns."""

    model_type = "llama"
    num_hidden_layers = 4
    num_attention_heads = 2
    num_key_value_heads = 2
    hidden_size = _PINNED_HIDDEN
    intermediate_size = _PINNED_INTERMEDIATE
    vocab_size = _PINNED_VOCAB
    tie_word_embeddings = False


class _PinnedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _PinnedCfg()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(_PINNED_VOCAB, _PINNED_HIDDEN)
        self.model.layers = torch.nn.ModuleList(
            _PinnedLayer() for _ in range(_PinnedCfg.num_hidden_layers)
        )
        self.model.norm = torch.nn.LayerNorm(_PINNED_HIDDEN)
        self.lm_head = torch.nn.Linear(_PINNED_HIDDEN, _PINNED_VOCAB, bias=False)


class LlamaAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, torch.nn.Linear(_PINNED_HIDDEN, _PINNED_HIDDEN, bias=False))


class LlamaMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(_PINNED_HIDDEN, _PINNED_INTERMEDIATE, bias=False)
        self.up_proj = torch.nn.Linear(_PINNED_HIDDEN, _PINNED_INTERMEDIATE, bias=False)
        self.down_proj = torch.nn.Linear(_PINNED_INTERMEDIATE, _PINNED_HIDDEN, bias=False)


class _PinnedLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = LlamaAttention()
        self.mlp = LlamaMLP()


#: What ``seed`` draws, recorded once and never recomputed from the code under test.
#:
#: Regenerating this dict from a failing run is the one repair that is never right: the
#: whole point of it is that the mapping is the same today as it was when an arm was
#: banked, and a golden anyone re-bakes on red asserts only that the code agrees with
#: itself. If it goes red, either the change is wrong or every banked control arm has to
#: be re-run and re-labelled.
PINNED_DRAW_SEED_0 = {
    "lm_head": "lm_head",
    "model.embed_tokens": "model.embed_tokens",
    "model.layers.0.mlp.down_proj": "model.layers.2.mlp.down_proj",
    "model.layers.0.mlp.gate_proj": "model.layers.3.mlp.gate_proj",
    "model.layers.0.mlp.up_proj": "model.layers.2.mlp.up_proj",
    "model.layers.0.self_attn.k_proj": "model.layers.2.self_attn.k_proj",
    "model.layers.0.self_attn.o_proj": "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn.q_proj": "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj": "model.layers.1.self_attn.v_proj",
    "model.layers.1.mlp.down_proj": "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.gate_proj": "model.layers.1.mlp.gate_proj",
    "model.layers.1.mlp.up_proj": "model.layers.0.mlp.up_proj",
    "model.layers.1.self_attn.k_proj": "model.layers.0.self_attn.k_proj",
    "model.layers.1.self_attn.o_proj": "model.layers.1.self_attn.o_proj",
    "model.layers.1.self_attn.q_proj": "model.layers.2.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj": "model.layers.0.self_attn.v_proj",
    "model.layers.2.mlp.down_proj": "model.layers.3.mlp.down_proj",
    "model.layers.2.mlp.gate_proj": "model.layers.2.mlp.gate_proj",
    "model.layers.2.mlp.up_proj": "model.layers.3.mlp.up_proj",
    "model.layers.2.self_attn.k_proj": "model.layers.1.self_attn.k_proj",
    "model.layers.2.self_attn.o_proj": "model.layers.3.self_attn.o_proj",
    "model.layers.2.self_attn.q_proj": "model.layers.1.self_attn.q_proj",
    "model.layers.2.self_attn.v_proj": "model.layers.3.self_attn.v_proj",
    "model.layers.3.mlp.down_proj": "model.layers.1.mlp.down_proj",
    "model.layers.3.mlp.gate_proj": "model.layers.0.mlp.gate_proj",
    "model.layers.3.mlp.up_proj": "model.layers.1.mlp.up_proj",
    "model.layers.3.self_attn.k_proj": "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.o_proj": "model.layers.2.self_attn.o_proj",
    "model.layers.3.self_attn.q_proj": "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.v_proj": "model.layers.2.self_attn.v_proj",
}


def test_the_permutation_a_seed_draws_is_pinned_across_versions() -> None:
    """Arms banked weeks apart are one sample only if the draw never moved.

    Every other determinism test here runs both sides in one process, so all of them
    stay green through a change that alters which module donates to which -- they
    compare the new code against itself. The property a multi-seed control family needs
    is stronger and cannot be expressed that way: `dq_3b_shufs1` was allocated under one
    commit and `shufs2` under another, and if the draw moved in between, the spread over
    the three is not a permutation spread. Nothing raises when that happens. Both arms
    allocate, both hit the byte anchor, and the report carries a number with no meaning.

    The graph is built here rather than taken from the shared fixture, so a module added
    to the campaign's model fixture cannot redden a test about the random number stream.

    Turns red when: the draw at a fixed seed changes -- a different shuffle call, a
    different sort key, a different grouping, or a different source of the role list.
    """
    graph = classify_model(_PinnedModel())
    names = [info.name for info in graph.quantizable()]
    module_scores = {name: float(i + 1) for i, name in enumerate(names)}
    lookup = {value: key for key, value in module_scores.items()}

    null_scores, _, _ = apply_null(graph, module_scores, None, mode="shuffle", seed=0)
    donors = {name: lookup[value] for name, value in null_scores.items()}

    # The whole map, not a sample of it: a subset would go green on a change that
    # moved only the modules it happened not to name.
    assert donors == PINNED_DRAW_SEED_0
    assert sum(name != donor for name, donor in donors.items()) == 21
