"""The arm that says how much of a margin belongs to the signal.

A panel row reading "DynQuant 79.89% against a uniform 3-bit recipe's 60.76% at
matched bytes" is a fact about two checkpoints and a claim about three things at
once: that role-aware floors matter, that spending the saved bytes by measured
damage matters, and that the *fine-tuning signal* -- plasticity times saliency,
and the channel moments the same hook collects -- picked the right modules. The
first two need no signal at all. A reader is entitled to ask which of the three
the nineteen points came from, and a bit map cannot answer it: every allocation
this package emits has a signal in it by construction.

So the answer has to be another arm. Keep the graph, the roles, the floors, the
budget, the byte accounting, the knapsack and the quantizer exactly as they were,
remove *only* the correspondence between a module and what the fine-tune said
about it, and re-run at the same anchor. Whatever the real arm has left over that
arm is the signal's share; whatever both have over a uniform recipe is the
allocator's structure.

The nulls, because the margin has more than one thing in it
-----------------------------------------------------------

``shuffle`` permutes the driving quantity **within role**, under a seed. Every
score and every measured ``dL`` row still exists, still has its magnitude, and is
still priced by the same code -- it has simply moved to a different module of the
same role. The population the allocator sees is identical in distribution and
carries no information about which module it describes. This is the strict null:
it changes one thing.

Within role and not globally, for a reason that decides whether the arm is fair.
A global permutation hands an embedding's number to a router, and the roles on
this architecture differ in shape by three orders of magnitude, so the resulting
map would be wrecked by a units mismatch that has nothing to do with the signal --
and the arm would read as a large signal contribution when it is really measuring
that scores are not transferable across roles. Within a role the members are
near-identical in shape (twenty-two copies of one projection, one per layer), so
the swap is between comparable numbers and the only thing destroyed is which
layer.

``flat`` keeps that same permutation and additionally sets every score to
1.0. The sensitivity table is still there, still permuted, still priced by the
same code; what is gone is the score's magnitude -- the plasticity-times-saliency
number the knapsack falls back on for every module the channel moments could not
price. It sits between the other two by construction: it removes everything
``shuffle`` removes and the ordering besides, and ``uniform`` removes everything
it removes and the table as well. That nesting is the point of it. The rung from
``shuffle`` down to ``flat`` is what the score channel is worth and the rung from
``flat`` down to ``uniform`` is what the measured channel is worth, so the single
large step between a permuted arm and a signal-free one splits into ranking and
pricing instead of standing as one number that could be either.

Worth being exact about which modules that rung can move. The knapsack prices a
module from its measured ``dL`` row when it has one and from ``score x params x
error-curve`` when it does not, so flattening the score changes only the widths
of modules the moments never reached -- on a batched-expert architecture that is
most of the parameters -- together with the scale that puts the proxy price on
the table's units. A ``shuffle``-to-``flat`` rung at zero would therefore be a
finding about the fallback, saying the proxied modules were allocated no better
than by size alone; it would not say the signal was worthless, because the rung
below it never asked the score anything.

``table`` sets every score to 1.0 and leaves the measured sensitivity table
exactly as the real arm received it -- not permuted, not rebuilt. It is the
only mode that isolates a single channel cleanly, and it exists because the
``shuffle``-to-``flat`` rung does not: those two share one drawn permutation, so
the table under ``flat`` is a *permuted* table and the rung below it prices a
permuted table against no table rather than the real one against no table. Read
against the real arm, ``table`` is the score channel and nothing else.

It is not on the ``shuffle`` ladder, and it is not off the ladder either: the
nesting over these modes is a partial order, not a line. ``shuffle`` and ``table``
are the incomparable pair -- one keeps every magnitude and destroys both
correspondences, the other keeps one correspondence exactly and destroys every
magnitude -- so no chain holds both, and a rung between them would be a difference
rather than a step. Both sit above ``flat``, which removes what either removes and
more, and ``flat`` sits above ``uniform``. So there are two chains, and
:data:`NULL_CHAINS` names them.

The ``table`` chain is the one to prefer when the question is what each channel is
worth. Against the real arm its rungs are single-channel contrasts -- real to
``table`` moves the score and nothing else, ``table`` to ``uniform`` moves the
measured table and nothing else, and the two sum to the whole signal in two steps.
Every middle rung of the ``shuffle`` chain moves two things at once.

``uniform`` gives every module the same score and consults no sensitivity table
at all. There is no such thing as measured sensitivity without the fine-tune --
``dL`` is built from ``E[x^2]`` and ``E[delta^2]`` -- so a genuinely signal-free
arm cannot have one, and dropping it is not a second change smuggled in beside
the first. What is left is the allocator running on role floors, parameter counts
and the universal error curve: precisely what a method with no training-time hook
could compute. It is the weaker null and the more useful floor, because it is the
honest alternative a reader will propose.

Neither is a *worse* allocator on purpose. ``uniform`` in particular is a real
strategy that a real person would ship, and if it lands within noise of the full
arm then that is the finding, stated in the direction the measurement points.

What this refuses to do quietly
-------------------------------

A null arm that is not labelled as one is worse than no null arm: it is a bit map
that looks like every other bit map, with provenance saying ``sensitivity``, and
downstream it becomes a headline. So :class:`NullReport` carries a label that
travels into the ``allocator`` field of every written map, and the report names
the roles it could not move -- a role with one member permutes to itself, and
saying "shuffled" over an arm where forty of a hundred modules kept their own
number would overstate what was removed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dynquant.graph.classify import ModelGraph
    from dynquant.score.sensitivity import SensitivityTable

__all__ = [
    "NULL_CHAINS",
    "NULL_MODES",
    "STOCHASTIC_NULL_MODES",
    "NullReport",
    "apply_null",
    "uses_seed",
]

#: The nulls, in increasing order of how much they remove.
#:
#: Increasing, and asserted to be: a ladder built over these is only a partition of the
#: margin if each mode removes everything the one before it removed, and the order is
#: read by name from here rather than restated wherever a chain gets built.
NULL_MODES = ("table", "shuffle", "flat", "uniform")

#: Every chain a decomposition may be built over, each in the order that makes it one.
#:
#: :data:`NULL_MODES` is every mode the CLI accepts, ordered for display; this is the
#: smaller claim, that within a chain each mode removes everything the one before it
#: removed. Plural because the nesting is a partial order: ``shuffle`` and ``table``
#: remove incomparable things, so no chain holds both, and subtracting one from the
#: other prices two changes at once and adds up anyway. Separate names from
#: :data:`NULL_MODES` because the two facts go stale independently -- adding a mode
#: extends the registry and must not silently extend a chain.
NULL_CHAINS = (
    ("shuffle", "flat", "uniform"),
    ("table", "flat", "uniform"),
)

#: The subset that draws, so that a seed names a different arm rather than the same one.
#:
#: ``flat`` is in it even though its scores are constant: the permutation it applies to
#: the sensitivity table is drawn, so two seeds are two arms and each needs its own record.
#: ``table`` is not, for the same reason ``uniform`` is not -- it draws nothing.
STOCHASTIC_NULL_MODES = ("shuffle", "flat")


def uses_seed(mode: str) -> bool:
    """Whether a seed distinguishes two runs of ``mode``.

    One function answers this for the whole package. ``NullReport`` uses it to decide
    whether the seed belongs in the allocator string and whether to record one at all;
    a caller planning arms uses it to decide whether two seeds are two arms or one arm
    named twice. Every caller that answered it with its own ``mode == "uniform"`` would
    be another copy of the registry, and the copy that goes stale is the one that names
    files -- a deterministic mode given two seeds plans two arms that write to one
    record, and a stochastic one given two seeds plans one arm that silently keeps the
    second draw.
    """
    return mode in STOCHASTIC_NULL_MODES


@dataclass(frozen=True, slots=True)
class NullReport:
    """What the null actually did, in the terms a reader would challenge it on."""

    mode: str
    seed: int | None
    modules: int
    """Quantizable modules the null was applied over."""

    moved: int
    """Modules whose driving quantity came from a different module.

    The number that says how strong the null is. A permutation with many fixed
    points has left the signal partly in place, and an arm described as
    "shuffled" would be overclaiming by exactly that much."""

    fixed: int
    """Modules a same-role permutation happened to leave alone."""

    singleton_roles: tuple[str, ...]
    """Roles with one member, which permute to themselves by construction.

    Named rather than counted, because on a tied embedding this is not a rounding
    detail: it means the arm still holds that module at whatever the signal chose
    for it, and the role is usually one of the expensive ones."""

    estimability_changed: int
    """Modules that gained or lost a measured ``dL`` row in the swap.

    Nonzero means a role mixes measured and proxied modules, so the permutation
    moved a module across the pricing boundary as well as within the role. That
    is not wrong -- the boundary is a property of the moments, which the null is
    entitled to scramble -- but it is a second thing changing and it is reported
    rather than absorbed."""

    @property
    def label(self) -> str:
        """The suffix that goes in every artifact's ``allocator`` field."""
        if not uses_seed(self.mode):
            return f"null:{self.mode}"
        return f"null:{self.mode}(seed={self.seed})"

    def summary(self) -> str:
        if self.mode == "uniform":
            return (
                f"null arm: every one of {self.modules} modules scores 1.0 and no "
                "sensitivity table is consulted. Widths come from role floors, "
                "parameter counts and the universal error curve -- no fine-tuning "
                "signal reaches the allocator."
            )
        if self.mode == "table":
            return (
                f"null arm: every one of {self.modules} modules scores 1.0, and the "
                "measured sensitivity table is passed through exactly as the real arm "
                "received it -- not permuted, not rebuilt. Nothing moved, so `moved` is "
                "0 by construction rather than by a weak draw: the one thing removed is "
                "the score's magnitude, and every module the moments could price keeps "
                "its own measured row. Subtracts cleanly from the real arm and from "
                "`uniform`; never from `shuffle`, which removes something else "
                "entirely."
            )
        if self.mode == "flat":
            lines = [
                f"null arm: every one of {self.modules} modules scores 1.0, and the "
                f"measured sensitivity table is kept but permuted within role, seed "
                f"{self.seed}. {self.moved}/{self.modules} modules took another "
                f"module's row, {self.fixed} kept their own. Every module the moments "
                "could not price is left to role floors, parameter counts and the "
                "universal error curve."
            ]
        else:
            lines = [
                f"null arm: driving quantity permuted within role, seed {self.seed}. "
                f"{self.moved}/{self.modules} modules took another module's number, "
                f"{self.fixed} kept their own."
            ]
        if self.singleton_roles:
            lines.append(
                f"  {len(self.singleton_roles)} role(s) have a single member and "
                f"could not be permuted: {', '.join(self.singleton_roles)}"
            )
        if self.estimability_changed:
            lines.append(
                f"  {self.estimability_changed} module(s) crossed the "
                "measured/proxied boundary in the swap"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "label": self.label,
            "modules": self.modules,
            "moved": self.moved,
            "fixed": self.fixed,
            "singleton_roles": list(self.singleton_roles),
            "estimability_changed": self.estimability_changed,
        }


def apply_null(
    graph: ModelGraph,
    scores: Mapping[str, float],
    sensitivity: SensitivityTable | None,
    *,
    mode: str,
    seed: int = 0,
) -> tuple[dict[str, float], SensitivityTable | None, NullReport]:
    """Strip the fine-tune's information out of the allocator's inputs.

    Returns the same two objects the allocator takes, so the caller substitutes
    them and changes nothing else. The graph is read but never modified: roles,
    floors and shapes are the structure under test and must survive the null
    intact.

    Args:
        mode: ``"shuffle"`` to permute the driving quantity within role,
            ``"flat"`` to permute it and drop the score's magnitude as well,
            ``"uniform"`` to remove the signal entirely -- nested in that order,
            ``"table"`` drops the score's magnitude and keeps the measured table
            untouched, which makes it the cleanest contrast against the real arm.
            :data:`NULL_CHAINS` says which sequences nest; ``"shuffle"`` and
            ``"table"`` are incomparable and never chain. See the module docstring
            for which question each one answers.
        seed: Fixed so the arm is reproducible and so a second seed is a
            deliberate act. Ignored by ``"uniform"``, which is deterministic.
    """
    from dynquant.errors import DynQuantError

    if mode not in NULL_MODES:
        raise DynQuantError(
            f"unknown null mode {mode!r}; expected one of {', '.join(NULL_MODES)}. "
            "`shuffle` permutes the driving quantity within role and `uniform` "
            "removes it, and they answer different questions -- see "
            "dynquant.score.null. NULL_CHAINS names which sequences of them chain "
            "into a partition; `shuffle` and `table` are in no chain together."
        )

    names = [info.name for info in graph.quantizable()]
    if mode == "table":
        # The table goes through by identity, not rebuilt from `names`: this arm has to
        # differ from the real one in the score channel and in nothing else, and a
        # rebuild would be a second edit no matter how faithful it looked.
        return (
            dict.fromkeys(names, 1.0),
            sensitivity,
            NullReport(
                mode=mode,
                seed=seed if uses_seed(mode) else None,
                modules=len(names),
                moved=0,
                fixed=len(names),
                singleton_roles=(),
                estimability_changed=0,
            ),
        )
    if mode == "uniform":
        return (
            dict.fromkeys(names, 1.0),
            None,
            NullReport(
                mode=mode,
                seed=seed if uses_seed(mode) else None,
                modules=len(names),
                moved=len(names),
                fixed=0,
                singleton_roles=(),
                estimability_changed=0,
            ),
        )

    by_role: dict[str, list[str]] = defaultdict(list)
    for info in graph.quantizable():
        # `.value` rather than the enum, so the report reads in the same vocabulary
        # as the map's own `violations` block and a reader can match them up.
        by_role[info.role.value].append(info.name)

    rng = random.Random(seed)
    donor: dict[str, str] = {}
    singletons: list[str] = []
    for role, members in sorted(by_role.items()):
        if len(members) == 1:
            singletons.append(role)
        # Sorted before shuffling: `graph.quantizable()` order is the module tree's,
        # which is stable today, but seeding a permutation off an order this file
        # does not own makes the arm reproducible only by accident.
        ordered = sorted(members)
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        donor.update(zip(ordered, shuffled, strict=True))

    # `flat` keeps the permutation and drops the magnitudes. The table built below is
    # the permuted one either way, so the two arms differ in the score channel and in
    # nothing else -- which is the only thing that makes the rung between them a price
    # for that channel rather than for that channel plus whatever else moved.
    null_scores = (
        dict.fromkeys(names, 1.0)
        if mode == "flat"
        else {name: float(scores.get(donor[name], 0.0)) for name in names}
    )

    null_table: SensitivityTable | None = None
    changed = 0
    if sensitivity is not None:
        from dataclasses import replace

        values = {
            name: dict(sensitivity.values[donor[name]])
            for name in names
            if donor[name] in sensitivity.values
        }
        changed = sum(
            1
            for name in names
            if (name in sensitivity.values) != (donor[name] in sensitivity.values)
        )
        # `unestimable` is recomputed rather than permuted: it is the complement of
        # `values` by definition, and a stale copy would tell the caller to fall
        # back to a score for a module that now has a row.
        null_table = replace(
            sensitivity,
            values=values,
            unestimable=tuple(name for name in names if name not in values),
        )

    fixed = sum(1 for name in names if donor[name] == name)
    return (
        null_scores,
        null_table,
        NullReport(
            mode=mode,
            seed=seed if uses_seed(mode) else None,
            modules=len(names),
            moved=len(names) - fixed,
            fixed=fixed,
            singleton_roles=tuple(sorted(singletons)),
            estimability_changed=changed,
        ),
    )
