"""Per-block wall clock from the panel's progress lines, aligned by item index.

The panel writes one ``[text2sql] N/12000`` line per 800 items and no timestamp. A
sampler on the box (``/workspace/rate.sh``) polls the log every 15 seconds and stamps
each new line, which turns the panel's own progress into an interval profile nobody had
to plan for. That profile is the only length evidence this campaign has: the eval records
keep ``accuracy``, ``errored``, ``unparseable`` and ``unfinished_reasoning``, and none of
those is a proxy for how many decode steps an arm took.

What makes the profile readable is that every arm evaluates the same 12,000 items in the
same order, so block *k* is the same 800 questions in every arm. Two arms can therefore be
divided block-for-block, and the question "is one arm uniformly slower, or slower in
places" has an answer.

The distinction matters because the panel's linearised arms cost 1.9-2.3x the banked ones
and two very different things could produce that:

  a fixed cost per forward   -- unpacking weights, or 22 grouped matmuls becoming 704
                                module calls. Constant work per decode step, so the
                                per-block ratio between two arms is flat.
  a variable number of steps -- decoding is greedy and batched at 32, and a ``generate``
                                call runs until every sequence in its batch has stopped,
                                so a batch costs its *longest* generation. One item in
                                thirty-two that fails to stop doubles a block. The ratio
                                then moves with the items, not with the arms.

A flat ratio is evidence for the first. A ratio that swings is evidence for the second,
and its minimum over the blocks is the largest the fixed component can be -- conditional
on the slower arm never taking *fewer* steps than the faster one on any block, which is
an assumption this script states and cannot check.

Two caveats that belong next to the numbers rather than in a footnote. The sampler polls
at 15 s, so every stamp is up to 15 s late; against blocks of 465-4636 s that is under 3%
and largely cancels between adjacent stamps. And the first stamp of an arm opens no
interval -- the gap before it contains a model load and a quantization, which is not eval
time and must never be divided by 800.

The same parser reads a second kind of log. ``rescore_eager.sh`` pipes the driver through
a stamper that timestamps each line as it arrives, which has no poll slop at all -- the
progress printer flushes, so the stamp is the line's own time. What it does have is a
one-second floor, and a run short enough to put two progress lines in one second yields a
zero-length block. Those are dropped from the ratios and counted, never divided by.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MARKER = "[text2sql]"


@dataclass(frozen=True)
class Stamp:
    """One progress line and the wall clock at which the sampler first saw it."""

    when: datetime
    done: int
    total: int


@dataclass(frozen=True)
class Block:
    """Items completed between two consecutive stamps of one arm."""

    end: int
    items: int
    seconds: float

    @property
    def per_item(self) -> float:
        return self.seconds / self.items


def parse(text: str) -> list[Stamp]:
    """Stamps, in file order, skipping anything that is not a progress line.

    Split rather than matched, because the sampler pastes the panel's line verbatim after
    its own timestamp and the panel's indentation has changed once already.
    """
    stamps: list[Stamp] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or MARKER not in parts:
            continue
        counter = parts[parts.index(MARKER) + 1]
        done, _, total = counter.partition("/")
        stamps.append(Stamp(datetime.strptime(parts[0], STAMP_FORMAT), int(done), int(total)))
    return stamps


def split_arms(stamps: list[Stamp]) -> list[list[Stamp]]:
    """Split where the counter stops increasing, which is the only arm boundary there is.

    Not on a time gap. The gap between arms holds a quantization -- 32 to 47 minutes for
    the baselines -- but block 9600 of ``awq_4b`` took 63 minutes on its own, so any
    threshold that catches the boundary also cuts an arm in half.
    """
    arms: list[list[Stamp]] = []
    for stamp in stamps:
        if not arms or stamp.done <= arms[-1][-1].done:
            arms.append([stamp])
        else:
            arms[-1].append(stamp)
    return arms


def blocks(arm: list[Stamp]) -> list[Block]:
    """Intervals within one arm. The opening stamp contributes none, by construction."""
    return [
        Block(right.done, right.done - left.done, (right.when - left.when).total_seconds())
        for left, right in itertools.pairwise(arm)
    ]


def align(left: list[Block], right: list[Block]) -> list[tuple[int, Block, Block]]:
    """Inner join on the item index the block ends at, never on position.

    The arms in a single log do not start at the same item -- the sampler was launched
    part-way through the first one -- so pairing the first block of each would divide
    block 4000 by block 1600 and call the result a per-forward cost.
    """
    by_end = {block.end: block for block in right}
    return [(one.end, one, by_end[one.end]) for one in left if one.end in by_end]


def compare(
    name_left: str, left: list[Block], name_right: str, right: list[Block]
) -> dict[str, Any]:
    """Block-for-block ratio of two arms, plus what its spread does and does not license.

    Both directions of the ceiling, because which arm is the slower one is a fact about
    the pair and not about the order the log happened to run them in. Each side's ceiling
    is the smallest that side ever costs relative to the other, and it bounds that side's
    *fixed* excess only while the side never takes fewer decode steps than the other. A
    ceiling under 1.0 is that condition failing on some block, in which case there is no
    positive fixed cost left to bound -- which is a finding, not a missing number.

    A block the clock could not separate from its neighbour is dropped rather than divided
    by. The panel's blocks run 18 minutes so it never arises there, but the re-score stamps
    lines as they arrive and a short ``--limit`` puts two of them in the same second.
    """
    joined = align(left, right)
    pairs = [row for row in joined if row[1].seconds > 0 and row[2].seconds > 0]
    dropped = len(joined) - len(pairs)
    if not pairs:
        return {
            "pair": f"{name_left} vs {name_right}",
            "refused": (
                "the two arms share no block boundary, so there is nothing to divide "
                "that would be the same 800 items on both sides"
            )
            if not joined
            else (
                f"all {dropped} shared block(s) took under one second on one side, which is "
                "the stamp resolution. A ratio between two numbers the clock could not "
                "separate is not a measurement"
            ),
        }
    rows = [
        {
            "block_end": end,
            name_left: round(one.per_item, 3),
            name_right: round(other.per_item, 3),
            "ratio": round(one.per_item / other.per_item, 3),
        }
        for end, one, other in pairs
    ]
    values = [one.per_item / other.per_item for _, one, other in pairs]
    low, high = min(values), max(values)
    ceilings = {name_left: round(low, 3), name_right: round(1.0 / high, 3)}
    return {
        "pair": f"{name_left} vs {name_right}",
        "blocks": rows,
        "blocks_dropped_below_clock_resolution": dropped,
        "ratio_min": round(low, 3),
        "ratio_max": round(high, 3),
        "ratio_spread": round(high / low, 3) if low else None,
        "fixed_cost_ceiling": ceilings,
        "ceiling_usable": {name: value >= 1.0 for name, value in ceilings.items()},
        "ceiling_holds_only_if": (
            "the named arm never takes fewer decode steps than the other on a block. "
            "A ceiling below 1.0 is that assumption failing in the open: that arm is "
            "faster somewhere, so none of its excess elsewhere can be a fixed per-forward "
            "cost"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="the sampler's output, one stamped line per block")
    parser.add_argument(
        "--arms",
        required=True,
        help="comma-separated arm names in the order the log covers them",
    )
    parser.add_argument("--out", default=None, help="write the report here as well as printing it")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    arms = split_arms(parse(args.log.read_text(encoding="utf-8")))
    if len(names) != len(arms):
        raise SystemExit(
            f"the log holds {len(arms)} arm(s) and {len(names)} name(s) were given. "
            "Naming them wrong is worse than not naming them: every ratio below would "
            "be labelled with the wrong pair of models."
        )

    profiles = {name: blocks(arm) for name, arm in zip(names, arms, strict=True)}
    for name, arm in zip(names, arms, strict=True):
        span = profiles[name]
        print(
            f"{name:10s} items {arm[0].done:>5d}-{arm[-1].done:<5d} "
            f"{len(span):2d} block(s)  {sum(b.seconds for b in span) / 3600:5.2f} h",
            flush=True,
        )

    payload: dict[str, Any] = {
        "arms": {
            name: [
                {
                    "block_end": b.end,
                    "items": b.items,
                    "seconds": b.seconds,
                    "s_per_item": round(b.per_item, 3),
                }
                for b in span
            ]
            for name, span in profiles.items()
        },
        "pairs": [
            compare(names[i], profiles[names[i]], names[j], profiles[names[j]])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
