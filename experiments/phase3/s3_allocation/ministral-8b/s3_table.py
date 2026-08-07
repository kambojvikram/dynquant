"""Turn the S3 records into the tables the report quotes.

Every number in ``docs/reports/phase3-s3-allocation.md`` comes from here, run against the
records committed beside this file, so the report can be re-derived without the GPU box.

Three things it does that a spreadsheet would not. It pairs: each cell stores per-item
``hits``, so every arm-vs-arm difference is an exact McNemar test rather than two independent
proportions -- the arms are the same weights at different allocations and agree on most items,
so only the disagreements carry information. It corrects: 28 comparisons are computed here and
a p-value read without accounting for that is not what it appears to be, so the Bonferroni
threshold is printed beside the raw ones. And it decomposes: ``dq - rtn`` splits into
``shuf - rtn`` (what the allocator is worth with the signal permuted) plus ``dq - shuf`` (what
the signal adds on top), which is the whole question S3 exists to answer.

    python experiments/phase3/s3_allocation/ministral-8b/s3_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dynquant.eval.compare import compare_paired

RECORDS = Path(__file__).parent / "records"
TASKS = ("gsm8k", "ifeval", "humaneval", "mbpp")
ANCHORS = {"3": "3.25 bits", "4": "4.25 bits"}
ARMS = ("rtn", "rank", "shuf", "dq")


def cell(arm: str, task: str) -> dict:
    return json.loads((RECORDS / f"{arm}.{task}.json").read_text(encoding="utf-8"))


def accuracy(arm: str, task: str) -> float:
    return cell(arm, task)["accuracy"] * 100.0


def mean_delta(a: str, b: str, anchor: str) -> float:
    """Arm ``a`` minus arm ``b`` at one anchor, averaged over the four tasks."""
    return sum(accuracy(f"{a}{anchor}", t) - accuracy(f"{b}{anchor}", t) for t in TASKS) / len(
        TASKS
    )


def main() -> None:
    base = {t: accuracy("bf16", t) for t in TASKS}

    print("=" * 96)
    print("Ministral-8B x Tulu-3, S3 allocation arms. Accuracy, and delta from the bf16 ceiling.")
    print("=" * 96)
    header = "arm     " + "".join(f"{t:>18}" for t in TASKS) + "     mean"
    print(header)
    print("bf16    " + "".join(f"{base[t]:>17.2f}%" for t in TASKS))
    for anchor in ANCHORS:
        for arm in ARMS:
            name = f"{arm}{anchor}"
            deltas = [accuracy(name, t) - base[t] for t in TASKS]
            row = "".join(
                f"{accuracy(name, t):>11.2f} ({d:+6.2f})"
                for t, d in zip(TASKS, deltas, strict=True)
            )
            print(f"{name:<8}{row}   {sum(deltas) / len(deltas):+7.2f}")

    # Every comparison the report makes, so the Bonferroni denominator is the real one and not
    # a count of the ones that happened to be interesting.
    pairs = [(a, b) for a in ("rank", "shuf", "dq") for b in ARMS if ARMS.index(b) < ARMS.index(a)]
    total = len(pairs) * len(TASKS) * len(ANCHORS)
    threshold = 0.05 / total

    for anchor, label in ANCHORS.items():
        print()
        print("-" * 96)
        print(f"anchor {label} -- exact McNemar, paired on stored per-item hits")
        print("-" * 96)
        for a, b in pairs:
            arm_a, arm_b = f"{a}{anchor}", f"{b}{anchor}"
            deltas = []
            for task in TASKS:
                c = compare_paired(
                    cell(arm_a, task)["hits"],
                    cell(arm_b, task)["hits"],
                    label_a=arm_a,
                    label_b=arm_b,
                )
                low, high = c.interval_points
                mark = " *" if c.p_value < threshold else ""
                deltas.append(c.delta_points)
                print(
                    f"{arm_a:>6} - {arm_b:<6} {task:<10} {c.delta_points:+6.2f}  "
                    f"[{low:+6.2f},{high:+6.2f}]  p={c.p_value:<10.4g} "
                    f"disc={c.discordant:>3} ({c.a_only}/{c.b_only}){mark}"
                )
            print(f"{arm_a:>6} - {arm_b:<6} {'MEAN':<10} {sum(deltas) / len(deltas):+6.2f}")
            print()

    print("-" * 96)
    print("Decomposition of the margin over uniform assignment, mean over the four tasks")
    print("-" * 96)
    for anchor, label in ANCHORS.items():
        allocator = mean_delta("shuf", "rtn", anchor)
        signal = mean_delta("dq", "shuf", anchor)
        whole = allocator + signal

        # A share of a total that is itself indistinguishable from zero is not a share of
        # anything -- at 4.25 bits dq - rtn is +0.74 with every p above 0.31, and dividing it
        # would print "the signal is 139% of the margin", which reads as a finding and is an
        # artefact of the allocator term happening to land slightly negative.
        resolved = any(
            compare_paired(
                cell(f"dq{anchor}", t)["hits"],
                cell(f"rtn{anchor}", t)["hits"],
                label_a="dq",
                label_b="rtn",
            ).p_value
            < 0.05
            for t in TASKS
        )
        share = (
            f"signal is {signal / whole * 100:.0f}% of it"
            if resolved and abs(whole) > 1.0
            else "not decomposable: dq - rtn is inside noise on every task"
        )
        print(
            f"{label}:  dq - rtn = {whole:+.2f}  =  allocator {allocator:+.2f}  +  "
            f"signal {signal:+.2f}   ({share})"
        )

    print()
    print(f"{total} comparisons; Bonferroni threshold p < {threshold:.5f}, marked * above.")


if __name__ == "__main__":
    main()
