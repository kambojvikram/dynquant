"""Percentile ranking.

Ranks are the only thing the allocator sees of the collected signals, so a defect
here is invisible downstream: the bit map still looks reasonable, it is just driven
by the wrong ordering. The cases below are the ones where a naive implementation
gives an answer that looks fine.
"""

from __future__ import annotations

import math

from dynquant.score.ranks import percentile_ranks


def test_ranks_are_evenly_spaced_inside_the_unit_interval() -> None:
    ranks = percentile_ranks({"a": 1.0, "b": 2.0, "c": 3.0})
    assert ranks == {"a": 1 / 6, "b": 0.5, "c": 5 / 6}


def test_no_finite_value_ever_ranks_zero() -> None:
    """The defect this formula exists to prevent.

    The importance score is a product of two ranks, so one exact zero zeroes the
    score, and a module with score zero costs the allocator nothing to destroy: it
    goes to the 2-bit minimum and the report shows no reason why. Being the *least*
    important member of a role is a measurement; being worthless is not.
    """
    for size in (2, 3, 5, 20, 187):
        ranks = percentile_ranks({f"m{i}": float(i) for i in range(size)})
        assert min(ranks.values()) > 0.0, f"a finite value ranked 0 at n={size}"
        assert max(ranks.values()) < 1.0, f"a finite value ranked 1 at n={size}"


def test_empty_and_singleton() -> None:
    assert percentile_ranks({}) == {}
    # Neither 0.0 nor 1.0. A role with one member -- EMBEDDING on most
    # architectures -- would otherwise be handed the extreme score by construction
    # rather than by measurement, and the output cannot tell the two apart.
    assert percentile_ranks({"only": 42.0}) == {"only": 0.5}


def test_ties_share_the_average_rank() -> None:
    """Whole groups legitimately tie -- every unrouted expert has RMS exactly 0.

    Splitting them by iteration order would hand one arbitrary member a better
    rank than its identical siblings, and the allocator would spend real bits
    acting on that arbitrary choice.
    """
    ranks = percentile_ranks({"a": 5.0, "b": 5.0, "c": 5.0, "d": 9.0})
    assert ranks["a"] == ranks["b"] == ranks["c"]
    assert ranks["d"] > ranks["a"]


def test_all_equal_collapses_to_one_rank() -> None:
    ranks = percentile_ranks(dict.fromkeys("abcd", 7.0))
    assert len(set(ranks.values())) == 1
    assert next(iter(ranks.values())) == 0.5


def test_ordering_is_preserved() -> None:
    values = {"a": -3.0, "b": 0.0, "c": 0.001, "d": 7.0, "e": 1e9}
    ranks = percentile_ranks(values)
    assert sorted(ranks, key=lambda k: ranks[k]) == sorted(values, key=lambda k: values[k])


def test_result_is_independent_of_input_order() -> None:
    """Otherwise a dict rebuild silently changes the bit map."""
    forward = percentile_ranks({"a": 1.0, "b": 1.0, "c": 2.0})
    backward = percentile_ranks({"c": 2.0, "b": 1.0, "a": 1.0})
    assert forward == backward


def test_non_finite_values_rank_at_the_bottom() -> None:
    """A NaN is a failed measurement, not a high one.

    ``nan`` compares false against everything, so it lands wherever the sort
    happens to leave it. Left alone it can be handed a top rank and pull precision
    onto a module whose signal was never successfully collected. Zero is the right
    rank here -- and the only place zero is right, because it is the only case where
    nothing was measured at all.
    """
    ranks = percentile_ranks({"good": 1.0, "broken": math.nan, "big": 100.0})
    assert ranks["broken"] == 0.0
    assert ranks["big"] > ranks["good"] > ranks["broken"]


def test_infinity_ranks_at_the_bottom_too() -> None:
    ranks = percentile_ranks({"a": 1.0, "overflow": math.inf})
    assert ranks["overflow"] == 0.0
