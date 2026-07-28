"""Compare two models scored on the same problems, with the pairing kept.

Every comparison this package exists to make -- quantized against source, allocated
against uniform, one target against another -- scores two models on one fixed
problem set. That is a paired design, and treating it as two independent samples
throws away most of the available power for no reason.

Why the pairing matters, concretely. Two models at 58.2% and 56.8% on 1319 GSM8K
problems: as independent proportions the difference is 1.4 points against a joint
2-sigma interval of about 3.8, so nothing is resolved. But the models are not
independent -- they are the *same weights* at different precisions, so they agree on
the great majority of problems, and only the disagreements carry information about
which is better. If 60 problems flip each way the difference is noise; if 80 flip one
way and 20 the other, the same 1.4-point gap is decisive. The unpaired test cannot
tell those two situations apart because it never looks at which problems flipped.

So the interval here is built from the discordant pairs only, and the significance
test is McNemar's, computed *exactly* against the binomial rather than through the
chi-square approximation. Exact because the interesting comparisons in quantization
are the close ones, close comparisons have few discordant pairs, and the chi-square
approximation is unreliable in precisely that regime -- which is where a wrong
answer would actually change a decision.

The one thing this cannot do is rescue a comparison that was never recorded as
pairs. :attr:`~dynquant.eval.gsm8k.Gsm8kResult.hits` exists for that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dynquant.errors import DynQuantError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PairedComparison", "compare_paired", "mcnemar_exact"]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Two models over one problem set, summarised by how they disagreed."""

    label_a: str
    label_b: str
    both_right: int
    a_only: int
    """Problems A got right and B got wrong. With :attr:`b_only`, the only two
    numbers that carry any information about which model is better."""

    b_only: int
    both_wrong: int
    p_value: float
    """Two-sided McNemar exact p-value: the probability of a split at least this
    lopsided if the two models were equally good."""

    @property
    def total(self) -> int:
        return self.both_right + self.a_only + self.b_only + self.both_wrong

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only

    @property
    def accuracy_a(self) -> float:
        return (self.both_right + self.a_only) / self.total if self.total else 0.0

    @property
    def accuracy_b(self) -> float:
        return (self.both_right + self.b_only) / self.total if self.total else 0.0

    @property
    def delta_points(self) -> float:
        """A minus B, in percentage points."""
        return (self.accuracy_a - self.accuracy_b) * 100.0

    @property
    def standard_error_points(self) -> float:
        """SE of the paired difference, in percentage points.

        Agresti's formula for correlated proportions. The ``(b - c)^2 / n`` term is
        what makes this narrower than the unpaired interval: concordant problems
        contribute nothing to the variance, because both models answered them the
        same way and neither is credited.
        """
        if not self.total:
            return float("nan")
        n = self.total
        inner = self.a_only + self.b_only - (self.a_only - self.b_only) ** 2 / n
        return math.sqrt(max(inner, 0.0)) / n * 100.0

    @property
    def interval_points(self) -> tuple[float, float]:
        """95% CI on the difference, in percentage points."""
        half = 1.96 * self.standard_error_points
        return (self.delta_points - half, self.delta_points + half)

    def separated(self, alpha: float = 0.05) -> bool:
        """Whether the difference is distinguishable from zero at ``alpha``.

        Read off the exact test rather than off the interval: for few discordant
        pairs the two can disagree, and the exact test is the one that is right.
        """
        return self.p_value < alpha

    def summary(self) -> str:
        low, high = self.interval_points
        verdict = "separated" if self.separated() else "not separated"
        return (
            f"{self.label_a} vs {self.label_b}: "
            f"{self.accuracy_a:.2%} vs {self.accuracy_b:.2%}  "
            f"delta {self.delta_points:+.2f} pts  "
            f"95% CI [{low:+.2f}, {high:+.2f}]  "
            f"discordant {self.a_only}/{self.b_only}  "
            f"p={self.p_value:.4g}  {verdict}"
        )

    def as_dict(self) -> dict[str, Any]:
        low, high = self.interval_points
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "accuracy_a": self.accuracy_a,
            "accuracy_b": self.accuracy_b,
            "delta_points": self.delta_points,
            "standard_error_points": self.standard_error_points,
            "ci_low_points": low,
            "ci_high_points": high,
            "both_right": self.both_right,
            "a_only": self.a_only,
            "b_only": self.b_only,
            "both_wrong": self.both_wrong,
            "discordant": self.discordant,
            "p_value": self.p_value,
            "separated": self.separated(),
        }


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided McNemar exact p-value from the two discordant counts.

    Under the null the discordant pairs are exchangeable, so each is an independent
    coin flip: ``a_only ~ Binomial(a_only + b_only, 0.5)``. The two-sided p-value
    doubles the smaller tail, clamped at 1 -- which is the conventional and slightly
    conservative choice for a discrete symmetric statistic.

    Returns 1.0 when there are no discordant pairs: identical predictions on every
    problem are no evidence of a difference, whatever the accuracies happen to be
    (they are necessarily equal).
    """
    if a_only < 0 or b_only < 0:
        raise DynQuantError(f"discordant counts cannot be negative: {a_only}, {b_only}")
    n = a_only + b_only
    if n == 0:
        return 1.0
    smaller = min(a_only, b_only)
    # Annotated because mypy types ``int ** int`` as Any (the exponent could be
    # negative), which would otherwise leak out of this function's return type.
    tail: float = sum(math.comb(n, k) for k in range(smaller + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def compare_paired(
    hits_a: Sequence[bool],
    hits_b: Sequence[bool],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> PairedComparison:
    """Build a :class:`PairedComparison` from two per-problem correctness vectors.

    The vectors must be the same length and in the same problem order -- the whole
    method rests on element *i* of each being the same question. Length equality is
    checked because a silent zip-truncation would produce a plausible number from
    misaligned pairs, which is the worst possible failure mode here: it looks like a
    result.
    """
    if len(hits_a) != len(hits_b):
        raise DynQuantError(
            f"paired comparison needs equal-length vectors, got "
            f"{len(hits_a)} ({label_a}) and {len(hits_b)} ({label_b}); "
            "these were not scored on the same problem set"
        )

    both_right = a_only = b_only = both_wrong = 0
    for hit_a, hit_b in zip(hits_a, hits_b, strict=True):
        if hit_a and hit_b:
            both_right += 1
        elif hit_a:
            a_only += 1
        elif hit_b:
            b_only += 1
        else:
            both_wrong += 1

    return PairedComparison(
        label_a=label_a,
        label_b=label_b,
        both_right=both_right,
        a_only=a_only,
        b_only=b_only,
        both_wrong=both_wrong,
        p_value=mcnemar_exact(a_only, b_only),
    )
