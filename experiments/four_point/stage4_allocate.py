"""Score the collected signal map and emit bit maps at the requested targets.

This is where the method actually runs. Everything before it produced inputs: the
fine-tune produced the signals, the classifier produced the roles. Here the two are
combined into an importance score per module and a greedy ROI allocator turns that
into a width for every quantizable tensor under a size budget.

Two properties are worth checking in the output rather than assuming:

* The scores must *change* the map. The supplement's allocator returned the floor
  map whenever floors exceeded the budget -- which is every target below ~3.5 bits --
  so at its own headline 3-bit setting the importance signal had no effect at all.
  This script prints how many modules move between a run driven by the real scores
  and a run driven by uniform ones, which is the same check as the unit test but
  against the real 187-module graph instead of a fixture.
* The average must be *stored* bits. At group 128 asymmetric, a scale and an offset
  per 128 weights is 0.25 bits/weight, so a map whose payload averages 3.00 bits is
  a 3.25-bit checkpoint on disk. The number printed here is the one the filesystem
  will report.
"""

from __future__ import annotations

import argparse
import json
import sys

from common import RUN_DIR, load_model, set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--stats", default=str(RUN_DIR / "stats"))
    parser.add_argument("--targets", type=float, nargs="+", default=[4.0, 3.0])
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    set_seed()
    from dynquant.allocate.budget import Budget
    from dynquant.allocate.knapsack import allocate_bits
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.classify import classify_model
    from dynquant.score.importance import score_modules
    from dynquant.signals.schema import load_stats

    stats = load_stats(args.stats)
    print(f"loaded {len(stats.layers)} layer stats from {args.stats}", flush=True)

    model = load_model(args.model)
    graph = classify_model(model)
    print(
        f"classified {len(list(graph.quantizable()))} quantizable modules "
        f"({graph.total_params() / 1e9:.4f}B params, "
        f"{graph.unquantized_params() / 1e6:.1f}M staying at compute dtype)",
        flush=True,
    )

    report = score_modules(graph, stats)
    print("\n" + report.summary(), flush=True)

    scores = report.scores()
    # A uniform-score control. If the real scores produce the same map as this, the
    # allocator is running but the signal is not reaching it -- exactly the failure
    # mode that made the supplement's ablation table impossible to reproduce.
    uniform = dict.fromkeys(scores, 0.5)

    policy = AllocationPolicy(group_size=args.group_size)
    output: dict[str, object] = {"model": args.model, "stats": args.stats, "maps": {}}

    for target in args.targets:
        budget = Budget.from_target(graph, target_bits=target, group_size=args.group_size)
        real = allocate_bits(graph, scores, budget, policy)
        control = allocate_bits(graph, uniform, budget, policy)
        moved = sum(1 for name, bits in real.bits.items() if control.bits[name] != bits)

        print(f"\n=== target {target:.2f} bits ===", flush=True)
        print(real.summary(), flush=True)
        print(
            f"  signal effect: {moved}/{len(real.bits)} modules differ from a "
            f"uniform-score allocation at the same budget",
            flush=True,
        )

        params_at = {}
        for name, bits in real.bits.items():
            params_at[bits] = params_at.get(bits, 0) + graph[name].num_params
        print(
            "  params per width: "
            + ", ".join(f"{b}b={p / 1e6:.1f}M" for b, p in sorted(params_at.items())),
            flush=True,
        )

        output["maps"][f"{target:.2f}"] = {  # type: ignore[index]
            "target_bits": target,
            "average_bits": real.average_bits,
            "nbytes": real.nbytes,
            "group_size": args.group_size,
            "modules_differing_from_uniform": moved,
            "violations": [
                {
                    "name": v.name,
                    "role": v.role.value,
                    "floor_bits": v.floor_bits,
                    "assigned_bits": v.assigned_bits,
                    "num_params": v.num_params,
                }
                for v in real.violations
            ],
            "bits": real.bits,
        }

    path = RUN_DIR / "stage4_bitmaps.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n-> wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
