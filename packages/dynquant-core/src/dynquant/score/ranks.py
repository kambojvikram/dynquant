"""Percentile ranking with tie averaging.

Ranking rather than normalising is the whole point. Activation RMS across a model
spans orders of magnitude and is dominated by a handful of outlier layers; a
min-max or z-score normalisation would compress every ordinary layer into a narrow
band and let two or three outliers decide the entire bit allocation. A percentile
rank keeps the *ordering* -- which is all the allocator needs -- and discards the
magnitudes, which were never comparable across roles anyway.

Ranks land in the *open* interval ``(0, 1)``, via the Hazen plotting position
``(rank - 0.5) / n`` rather than the more obvious ``(rank - 1) / (n - 1)``. The
difference is not cosmetic. Under the obvious formula the lowest-ranked member of
every group gets a rank of exactly 0, and since the importance score multiplies two
ranks together, a single 0 makes the score 0 -- and a module with score 0 is *free
to destroy*. The allocator, pricing damage as ``score * params * Δerror``, will take
it to the hard 2-bit minimum for no cost and gain nothing measurable in return. That
is what happened on Qwen3.5-2B: at a 4-bit target, exactly 20 modules landed at 2
bits, two per role -- the minimum of each role in each of the two signals -- with
relative reconstruction errors up to 0.61 on tensors that no evidence said were
unimportant. Being the least important attention ``v_proj`` in a model is not the
same as being worthless, and the arithmetic must not confuse the two.

The same formula fixes the mirror-image artifact at the top: a group with one member
gets ``0.5``, a neutral score, instead of ``1.0``. A role with a single instance --
``EMBEDDING`` on most architectures -- was otherwise handed the maximum score by
construction rather than by measurement, which is indistinguishable in the output
from having earned it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["percentile_ranks"]


def percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Map values to ranks in the open interval ``(0, 1)``, averaging within ties.

    Tie averaging matters more here than it looks. Whole groups of modules
    legitimately share a value -- every expert that was never routed to has
    activation RMS exactly 0, and an unexercised module has exactly 0 gradient
    variance. Breaking those ties by iteration order would hand one arbitrary
    member of the group a materially better rank than its identical siblings, and
    the allocator would then spend real bits on that arbitrary choice.

    Non-finite values (a NaN from a diverged step, an inf from an overflow) sort
    below everything real and receive rank 0 -- the one place a hard zero is
    correct, because it is the only case where the measurement did not happen.
    Losing precision on a module whose signal was never collected is the right
    outcome; a genuine measurement at the bottom of its group is not the same thing
    and keeps a small positive rank.

    Returns:
        One rank per key, strictly between 0 and 1 for every finite input. A single
        input gets ``0.5``: with nothing to compare against, neither "top of its
        group" nor "bottom" is a claim the data supports.
    """
    if not values:
        return {}

    items = [(key, float(value)) for key, value in values.items()]

    # Non-finite sorts first. Sorting on the key alone as a tiebreak makes the
    # result independent of the input mapping's iteration order.
    items.sort(key=lambda kv: (math.isfinite(kv[1]), kv[1] if math.isfinite(kv[1]) else 0.0, kv[0]))

    count = len(items)
    ranks: dict[str, float] = {}

    # The non-finite prefix is consumed first, one entry at a time. It cannot go
    # through the run-length scan below: that scan groups equal values, and NaN
    # compares unequal to itself, so the run would never advance past its first
    # element and the loop would never terminate.
    index = 0
    while index < count and not math.isfinite(items[index][1]):
        ranks[items[index][0]] = 0.0
        index += 1

    while index < count:
        end = index
        value = items[index][1]
        while end < count and items[end][1] == value:
            end += 1

        # Ranks 1..n mapped onto (0, 1); the tied block takes the mean of the
        # positions it occupies. See the module docstring for why the interval is
        # open -- a rank of exactly 0 zeroes the product score and makes the module
        # free for the allocator to destroy.
        average = (index + 1 + end) / 2.0
        percentile = (average - 0.5) / count
        for position in range(index, end):
            ranks[items[position][0]] = percentile
        index = end

    return ranks
