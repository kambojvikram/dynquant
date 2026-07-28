"""Paired comparison of two models scored on one problem set.

The tests that matter here are the ones about *misuse*, not the ones about
arithmetic. The arithmetic is a binomial tail and a variance formula, and both are
easy to check against hand-computed values. The dangerous failure is a comparison
that runs happily on data it should have rejected -- two vectors of different
lengths, or two vectors from different problem orders -- because the output is a
plausible p-value and nothing about it looks wrong.

The other property under test is that the paired interval is *narrower* than the
unpaired one on realistically correlated data. That is the entire reason this module
exists rather than a two-proportion z-test, so it is asserted rather than assumed.
"""

from __future__ import annotations

import math

import pytest

from dynquant.errors import DynQuantError
from dynquant.eval.compare import compare_paired, mcnemar_exact


def test_no_discordant_pairs_is_no_evidence() -> None:
    """Identical predictions cannot distinguish two models, at any accuracy."""
    hits = [True, False, True, True, False]
    result = compare_paired(hits, hits, label_a="a", label_b="b")

    assert result.discordant == 0
    assert result.delta_points == 0.0
    assert result.p_value == 1.0
    assert not result.separated()
    # Concordant problems contribute no variance: an exactly zero interval, which is
    # correct -- there is no sampling variability in "the two models agreed on every
    # problem", whatever the accuracy happens to be.
    assert result.standard_error_points == 0.0


def test_counts_partition_the_problem_set() -> None:
    hits_a = [True, True, True, False, False, False]
    hits_b = [True, False, False, True, True, False]
    result = compare_paired(hits_a, hits_b)

    assert (result.both_right, result.a_only, result.b_only, result.both_wrong) == (1, 2, 2, 1)
    assert result.total == 6
    assert result.accuracy_a == pytest.approx(0.5)
    assert result.accuracy_b == pytest.approx(0.5)
    assert result.delta_points == pytest.approx(0.0)


def test_mcnemar_exact_matches_the_binomial_by_hand() -> None:
    # 10 discordant pairs split 9/1. Two-sided p = 2 * (C(10,0) + C(10,1)) / 2^10.
    assert mcnemar_exact(9, 1) == pytest.approx(2 * (1 + 10) / 1024)
    # A 5/5 split doubles the whole lower half, which overshoots 1 and is clamped.
    assert mcnemar_exact(5, 5) == 1.0
    # Symmetric in its arguments: which model is called A cannot change the p-value.
    assert mcnemar_exact(3, 12) == mcnemar_exact(12, 3)


def test_mcnemar_exact_rejects_negative_counts() -> None:
    with pytest.raises(DynQuantError, match="cannot be negative"):
        mcnemar_exact(-1, 4)


def test_unequal_length_vectors_are_refused() -> None:
    """The failure mode this guard exists for produces a believable number.

    ``zip`` without ``strict`` would truncate to the shorter vector and compare
    problem *i* of one run against problem *i* of a different run for a while, then
    stop early. The result is a well-formed p-value computed from misaligned pairs.
    """
    with pytest.raises(DynQuantError, match="same problem set"):
        compare_paired([True, False, True], [True, False], label_a="4bit", label_b="3bit")


def test_a_lopsided_split_separates_even_at_a_small_gap() -> None:
    """The case the unpaired test cannot see.

    1319 problems, a 1.4-point gap, but the disagreements run 4:1 one way. Paired,
    that is decisive; unpaired, the same accuracies are nowhere near separated.
    """
    n = 1319
    a_only, b_only = 24, 6
    both_right = 700
    hits_a = [True] * both_right + [True] * a_only + [False] * b_only
    hits_b = [True] * both_right + [False] * a_only + [True] * b_only
    pad = n - len(hits_a)
    hits_a += [False] * pad
    hits_b += [False] * pad

    result = compare_paired(hits_a, hits_b, label_a="allocated", label_b="uniform")

    assert result.delta_points == pytest.approx((a_only - b_only) / n * 100)
    assert result.p_value < 0.05
    assert result.separated()

    # The same two accuracies, treated as independent proportions, are not separated.
    p_a, p_b = result.accuracy_a, result.accuracy_b
    unpaired_se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n) * 100
    assert unpaired_se > result.standard_error_points
    assert abs(result.delta_points) < 2 * unpaired_se


def test_a_balanced_split_does_not_separate_at_the_same_gap() -> None:
    """The complement: an identical accuracy gap that is genuinely noise.

    Here the discordant pairs are nearly even, so the gap comes from many
    disagreements cancelling rather than from a consistent advantage. Paired and
    unpaired agree that nothing is resolved -- and the point is that the paired test
    reaches that conclusion from the *split*, which is the thing that actually
    differs between this case and the previous one.
    """
    n = 1319
    a_only, b_only = 120, 102
    both_right = 600
    filler = n - both_right - a_only - b_only
    hits_a = [True] * both_right + [True] * a_only + [False] * b_only + [False] * filler
    hits_b = [True] * both_right + [False] * a_only + [True] * b_only + [False] * filler

    result = compare_paired(hits_a, hits_b)

    assert result.p_value > 0.05
    assert not result.separated()


def test_interval_brackets_the_delta() -> None:
    hits_a = [True] * 80 + [False] * 20
    hits_b = [True] * 60 + [False] * 40
    result = compare_paired(hits_a, hits_b)

    low, high = result.interval_points
    assert low < result.delta_points < high
    assert result.as_dict()["ci_low_points"] == low


def test_as_dict_is_json_ready() -> None:
    import json

    result = compare_paired([True, False], [False, False])
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["a_only"] == 1
    assert payload["separated"] is False
