"""Assemble the results table from the records the stages wrote.

Reads only what is on disk. Nothing here recomputes an accuracy or a size, so every
number in the table is a number some run produced, and a run that has not happened
shows up as a missing row rather than as a gap the reader has to notice.

The standard-error column is not decoration. At GSM8K's n=1319 one binomial SE is
about 1.3 points, so two arms seven points apart are separated and two arms one point
apart are not. A table that omits it invites both mistakes: reading noise as an effect,
and reading a real effect as noise.

The comparisons at the bottom are the ones the experiment was designed to answer.
Each is printed with the joint 2-SE interval beside it and an explicit verdict,
because "58.15 vs 65.13" and "65.13 vs 66.11" look like the same kind of statement
and are not.

The chance floor is printed as its own row, because on a 5-way task it is 20% and a
reader who assumes zero will read a collapsed arm as a mildly damaged one. Accuracy
minus that floor is the part of each number that is skill.

Both tests, deliberately
------------------------
Every comparison here is *paired*: the same problems, in the same order, scored twice.
So McNemar's exact test on the discordant pairs is the correct analysis, and it is the
verdict column. The unpaired joint-2SE column is printed next to it anyway, for one
reason: on this data the unpaired interval is roughly twice as wide, and a reader who
has seen the earlier version of these results -- where only the unpaired number existed
and the headline comparison came out "not separated" -- should be able to see exactly
where the change came from. Printing only the test that gives the more favourable
answer is how a reporting choice turns into a result.

Records written before ``hits`` was added carry no per-problem vector. Those rows say
so instead of falling back to the unpaired verdict, because a silent fallback would
make the two kinds of row indistinguishable in the output.
"""

from __future__ import annotations

import argparse
import math
import sys

from common import TASK, read_record

# (record name, row label). Controls are indented: they are not measurement points,
# they are the thing that says whether a measurement point means anything.
ROWS: tuple[tuple[str, str], ...] = (
    ("stage1_base", "1. base, no fine-tune (fp16)"),
    ("stage3_finetuned", "2. fine-tuned (fp16)"),
    ("stage5_4p25", "3. quantized ~4b, allocated"),
    ("stage5_uniform4b", "   control: uniform 4b"),
    ("stage5_3p25", "4. quantized ~3b, allocated"),
    ("stage5_uniform3b", "   control: uniform 3b"),
)

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("stage3_finetuned", "stage1_base", "did the fine-tune move the task?"),
    ("stage5_4p25", "stage3_finetuned", "cost of quantizing to ~4b"),
    ("stage5_4p25", "stage5_uniform4b", "did allocation beat uniform at ~4b?"),
    ("stage5_3p25", "stage3_finetuned", "cost of quantizing to ~3b"),
    ("stage5_3p25", "stage5_uniform3b", "did allocation beat uniform at ~3b?"),
)

FP16_BYTES_PER_PARAM = 2


def standard_error(accuracy: float, total: int) -> float:
    """One binomial SE, in percentage points."""
    return math.sqrt(accuracy * (1.0 - accuracy) / total) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=float,
        default=1.8821e9,
        help="parameter count for the fp16 size column",
    )
    args = parser.parse_args()
    fp16_gib = args.params * FP16_BYTES_PER_PARAM / 2**30

    from dynquant.eval.compare import compare_paired

    header = (
        f"{'measurement point':32s} {'bits':>6s} {'GiB':>6s} {'exact match':>12s} "
        f"{'±1SE':>5s} {'correct':>10s} {'unparsed':>8s}"
    )
    print(f"task: {TASK.key}   chance floor: {TASK.chance:.2%}\n")
    print(header)
    print("-" * len(header))

    found: dict[str, tuple[float, float]] = {}
    hits: dict[str, list[bool]] = {}
    for name, what in ROWS:
        data = read_record(name)
        if data is None or "accuracy" not in data:
            print(f"{what:32s} {'--':>6s} {'--':>6s} {'not run':>12s}")
            continue

        accuracy, total = data["accuracy"], data["total"]
        se = standard_error(accuracy, total)
        found[name] = (accuracy, se)
        if data.get("hits"):
            hits[name] = data["hits"]

        bits = data.get("average_bits")
        gib = data.get("quantized_gib")
        print(
            f"{what:32s} "
            f"{(f'{bits:.3f}' if bits else '16.000'):>6s} "
            f"{(f'{gib:.3f}' if gib else f'{fp16_gib:.3f}'):>6s} "
            f"{accuracy * 100:11.2f}% {se:5.2f} "
            f"{data['correct']:5d}/{total:<4d} {data['unparseable']:8d}"
        )

    if TASK.chance:
        # In the table, not just the header. An arm sitting on this line has collapsed
        # to guessing, and that is a different finding from "lost a few points" -- one
        # a reader should not have to compute.
        print(f"{'   (guessing)':32s} {'--':>6s} {'--':>6s} {TASK.chance * 100:11.2f}%")

    print()
    print(
        f"{'comparison':38s} {'delta':>7s} {'unpaired':>9s} "
        f"{'paired 95% CI':>18s} {'flips':>11s} {'p (McNemar)':>12s}  verdict"
    )
    print("-" * 108)
    for left, right, question in COMPARISONS:
        if left not in found or right not in found:
            print(f"{question:38s} (needs both arms)")
            continue
        (a_acc, a_se), (b_acc, b_se) = found[left], found[right]
        delta = (a_acc - b_acc) * 100.0
        joint = 2.0 * math.sqrt(a_se**2 + b_se**2)

        if left not in hits or right not in hits:
            unpaired = "separated" if abs(delta) > joint else "NOT separated"
            print(
                f"{question:38s} {delta:+6.2f}  {joint:8.2f} "
                f"{'no hits recorded':>18s} {'--':>11s} {'--':>12s}  {unpaired} (unpaired)"
            )
            continue

        paired = compare_paired(hits[left], hits[right], label_a=left, label_b=right)
        low, high = paired.interval_points
        verdict = "separated" if paired.separated() else "NOT separated"
        print(
            f"{question:38s} {delta:+6.2f}  {joint:8.2f} "
            f"{f'[{low:+.2f}, {high:+.2f}]':>18s} "
            f"{f'{paired.a_only}/{paired.b_only}':>11s} {paired.p_value:12.3g}  {verdict}"
        )

    print()
    print(
        "delta = left minus right, percentage points. unpaired = joint 2SE treating the\n"
        "two arms as independent proportions. paired CI and p = McNemar exact on the same\n"
        "problems; flips = problems only-left-got-right / only-right-got-right. The verdict\n"
        "follows the paired test, which is the correct one for this design."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
