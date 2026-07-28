"""Paired comparison of the three 3.25-bit CaseHOLD arms.

Same 5314 problems, same order, same checkpoint, scored three times, so the
information about any difference lives entirely in the discordant pairs and McNemar's
exact test is the right analysis. The unpaired 2SE is printed beside it because it is
roughly twice as wide here, and a difference that clears one bar and not the other
should be visible as such rather than reported through whichever bar it passed.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from scipy.stats import binomtest

RUNS = Path("/workspace/runs/qwen35_2b_casehold")

ARMS = {
    "sensitivity": "stage5_3p25_sens",
    "rank-product": "stage5_3p25",
    "uniform 3b": "stage5_uniform3b",
}


def load(name: str):
    record = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    return record["hits"], record["accuracy"], record["average_bits"]


def mcnemar(left, right):
    # strict: two hit vectors of different lengths are two different problem sets,
    # and pairing them silently would report a quantization effect that is a
    # harness difference.
    only_left = sum(1 for a, b in zip(left, right, strict=True) if a and not b)
    only_right = sum(1 for a, b in zip(left, right, strict=True) if b and not a)
    n = only_left + only_right
    p = binomtest(only_left, n, 0.5).pvalue if n else 1.0
    delta = 100.0 * (only_left - only_right) / len(left)
    # Wald interval on the discordant proportion, rescaled to percentage points.
    if n:
        se = 100.0 * sqrt(n) / len(left)
        lo, hi = delta - 1.96 * se, delta + 1.96 * se
    else:
        lo = hi = 0.0
    return delta, only_left, only_right, lo, hi, p


def unpaired_2se(acc_a, acc_b, n):
    return 100.0 * 2 * sqrt(acc_a * (1 - acc_a) / n + acc_b * (1 - acc_b) / n)


def main() -> None:
    data = {}
    for label, name in ARMS.items():
        try:
            data[label] = load(name)
        except FileNotFoundError:
            print(f"missing arm: {name}.json")
            return

    print(f"{'arm':16s} {'stored bits':>12s} {'accuracy':>10s} {'correct':>9s}")
    for label, (hits, acc, bits) in data.items():
        print(f"{label:16s} {bits:12.4f} {100 * acc:9.2f}% {sum(hits):6d}/{len(hits)}")

    print(
        f"\n{'comparison':38s} {'delta':>8s} {'flips L/R':>12s} "
        f"{'paired 95% CI':>20s} {'p':>10s}  {'unpaired 2SE':>12s}"
    )
    pairs = [
        ("sensitivity", "uniform 3b"),
        ("rank-product", "uniform 3b"),
        ("sensitivity", "rank-product"),
    ]
    for left, right in pairs:
        lh, la, _ = data[left]
        rh, ra, _ = data[right]
        delta, ol, orr, lo, hi, p = mcnemar(lh, rh)
        u2se = unpaired_2se(la, ra, len(lh))
        verdict = "separated" if p < 0.05 else "not separated"
        if p < 0.05 and delta < 0:
            verdict = "separated, against"
        print(
            f"{left + ' vs ' + right:38s} {delta:+7.2f}  {ol:5d}/{orr:<5d}  "
            f"[{lo:+6.2f},{hi:+6.2f}] {p:10.2e}  {u2se:11.2f}   {verdict}"
        )


if __name__ == "__main__":
    main()
