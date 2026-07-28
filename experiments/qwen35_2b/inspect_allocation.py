"""Does the importance score actually drive the widths it was supposed to drive?

A width histogram cannot answer this. The supplement's allocator produced a complete,
plausible-looking bit map at its own headline 3-bit target while never reading the
scores at all -- inverting every one of them changed 0 of 282 modules -- and nothing
in the output said so.

So this script measures it directly: within each role, over every pair of modules
assigned different widths, how often does the wider one have the higher score?
Concordance near 0.5 means the score is not reaching the allocator and the widths are
being chosen by size alone. Comparing *within* role rather than globally is the point:
floors differ by role by design, so a global comparison is confounded by the very
structure the allocator is supposed to respect.

It also reports how many modules score exactly zero. That should be none: the score is
a product of two percentile ranks and the allocator prices damage as
``score * params * delta_error``, so a zero makes a module free to destroy. The ranker
maps onto the open interval to prevent it, and this is the check against the real graph
rather than a fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from common import RUN_DIR, load_model


def concordance(pairs: list[tuple[float, int]]) -> tuple[int, int]:
    """Agreeing and disagreeing (score, width) pairs. Ties in either are skipped:
    they carry no information about ordering and would dilute the ratio toward 0.5."""
    agree = disagree = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (score_i, width_i), (score_j, width_j) = pairs[i], pairs[j]
            if width_i == width_j or score_i == score_j:
                continue
            if (score_i > score_j) == (width_i > width_j):
                agree += 1
            else:
                disagree += 1
    return agree, disagree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--stats", default=str(RUN_DIR / "stats"))
    parser.add_argument("--bitmaps", default=str(RUN_DIR / "stage4_bitmaps.json"))
    parser.add_argument("--narrowest", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cpu",
        help="where to load the model. CPU by default: only shapes and names are read, "
        "and this script is often run while an evaluation holds the GPU.",
    )
    args = parser.parse_args()

    from dynquant.graph.classify import classify_model
    from dynquant.score.importance import score_modules
    from dynquant.signals.schema import load_stats

    graph = classify_model(load_model(args.model, device=args.device))
    report = score_modules(graph, load_stats(args.stats))
    scores = report.scores()

    values = sorted(scores.values())
    print(
        f"{len(scores)} modules scored: min {values[0]:.6f}  "
        f"median {values[len(values) // 2]:.6f}  max {values[-1]:.6f}"
    )
    zeroed = [name for name, score in scores.items() if score == 0.0]
    print(f"scoring exactly zero: {len(zeroed)}" + (f"  {zeroed[:5]}" if zeroed else "  (correct)"))
    print(f"unexercised: {len(report.unexercised)}   missing stats: {len(report.missing_stats)}")

    maps = json.loads(Path(args.bitmaps).read_text(encoding="utf-8"))["maps"]

    for target in sorted(maps, reverse=True):
        bits: dict[str, int] = maps[target]["bits"]
        print(f"\n=== target {target} ===")

        by_width: dict[int, list[str]] = defaultdict(list)
        by_role: dict[object, list[tuple[float, int]]] = defaultdict(list)
        for name, width in bits.items():
            by_width[width].append(name)
            by_role[graph[name].role].append((scores[name], width))

        for width in sorted(by_width):
            ranked = sorted(scores[name] for name in by_width[width])
            print(
                f"  {width}b  n={len(ranked):3d}  score min {ranked[0]:.4f}  "
                f"median {ranked[len(ranked) // 2]:.4f}  max {ranked[-1]:.4f}"
            )

        agree = disagree = 0
        for pairs in by_role.values():
            a, d = concordance(pairs)
            agree += a
            disagree += d
        total = agree + disagree
        if total:
            print(
                f"  within-role concordance of width with score: {agree}/{total} = "
                f"{agree / total:.3f}   (0.5 means the score is not reaching the allocator)"
            )
        else:
            print("  every module in every role got the same width: nothing to correlate")

        print("  narrowest modules, with the score that put them there:")
        for name, width in sorted(bits.items(), key=lambda kv: (kv[1], scores[kv[0]]))[
            : args.narrowest
        ]:
            info = graph[name]
            print(
                f"    {width}b  score {scores[name]:.4f}  {info.num_params / 1e6:6.1f}M  "
                f"{info.role.value:16s} {name}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
