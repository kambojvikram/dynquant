"""Is a finished S2 signal map usable, and does the allocator actually spend it?

Run this on every S2 arm before its stats are handed to S3. A fine-tune that
completes and writes a well-formed file has not thereby produced a *usable* map:
the failure mode is a signal that is present, structurally valid, and degenerate --
identical across modules, or zero for a module large enough to dominate the
allocation. Both look like success in the trainer log.

Four checks, in the order they can invalidate each other:

1. **Structure.** Every quantizable module tracked, canonical names, ties recorded,
   and the Welford count equal to the optimizer-step count (the supplement's bug 10
   updated per micro-batch instead). ``forward_calls`` must be *uniform*: a module
   inside a gradient-checkpointed block fires its forward hook twice per micro-batch
   on identical data, which squares the EMA decay for that module alone, so a split
   in this number means the saliency of one part of the model is on a different
   footing from the rest.

2. **Spread.** A signal that does not vary cannot rank. Reported as a max/min ratio
   per signal plus the list of modules whose gradient signal never moved.

3. **Coverage against the graph.** The graph decides which modules exist; the stats
   file can be missing entries or carry stale ones. Modules that are missing or
   unexercised take a neutral rank, so this is the count of tensors allocated on no
   measurement at all.

4. **Whether it matters.** The one that pays for the other three, and it needs two
   measurements rather than one. Forcing a neutral module's score to 0.0 and to 1.0
   brackets every width any signal could have bought it -- but that bracket is wider
   than any real score, so a module can look "sensitive" to a value nothing would
   ever produce. So the bracket is paired with the *realistic* counterfactual: where
   the neutral module is a tie representative, substitute its tied partner's stats
   row, which describes the same tensor, and re-score. A gap costs nothing only if
   the realistic substitution lands on the same width.

On Phi-4-mini check 4 is the whole story, and the two measurements disagree in a
way that matters. ``model.embed_tokens`` is tied to ``lm_head``, and under the
``outer_exact`` estimator an embedding has no ``delta x^T`` to form -- its gradient
is a scatter-add -- so it carries no plasticity signal, and no channel moments
either, both deliberate (see ``signals/tracker.py``). It is also the only member of
its role group, so per-role ranking gives it 0.5 whatever it carries. Three separate
reasons for one neutral score on 16% of the model, and that module pays 68% of the
shortfall at a 3.25-bit target.

The bracket says the width spans 2 to 4 bits over the score range -- so the neutral
score is *not* structurally harmless. The realistic substitution says ``lm_head``'s
own signal ranks 0.93 and lands on 3 bits, which is where the neutral score already
put it. The gap costs this checkpoint nothing and would cost a differently-ranked
one a bit either way; that is a caveat to state, not a blocker to fix.

Usage::

    python experiments/phase3/s3_allocation/verify_signal_map.py \
        --stats experiments/phase3/s2_runs/phi4-mini.tulu3/stats/dynquant_stats.json \
        --model phi4-mini --out signal_map.phi4-mini.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _src in sorted((REPO / "packages").glob("*/src")):
    sys.path.insert(0, str(_src))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from floor_headroom import TARGETS, build  # noqa: E402

from dynquant.allocate.budget import Budget  # noqa: E402
from dynquant.allocate.knapsack import allocate_bits  # noqa: E402
from dynquant.graph.classify import classify_model  # noqa: E402
from dynquant.score.importance import score_modules  # noqa: E402
from dynquant.signals.schema import load_stats  # noqa: E402

#: Targets to probe for score sensitivity. Wider than S3's two so that a tensor
#: pinned by the budget at 3.25 can be seen coming unpinned as the budget loosens.
PROBE_TARGETS = (*TARGETS, 4.25, 4.5)


def _spread(values: list[float]) -> dict:
    """Min/median/max and the dynamic range, which is what ranking consumes."""
    lo, hi = min(values), max(values)
    return {
        "min": lo,
        "median": statistics.median(values),
        "max": hi,
        "ratio": (hi / lo) if lo > 0 else None,
        "zeros": sum(1 for v in values if v == 0.0),
    }


def _alias_score(graph, stats_path: Path, name: str, alias: str) -> float:
    """What ``name`` would score if it borrowed its tied partner's measurements.

    Ranked *globally*, not within role, and that is not a detail. A tie
    representative is the only member of its role group, so per-role ranking returns
    0.5 for it whatever the underlying numbers are -- which would hide the very
    thing this is trying to measure. Against the whole model is the only comparison
    a group of one admits.

    The role is carried over from the original row so the substitution changes the
    signal and nothing else. The result is injected into the shipped per-role score
    map by the caller: percentile ranks all live on (0, 1) and the knapsack only
    ever compares them across modules, so this reads as "give this tensor a score
    reflecting what was actually measured about it" -- but it is a hybrid, and a
    width that moves under it is a reason to look, not a finished number.
    """
    from dataclasses import replace

    from dynquant.score.importance import ScoreConfig

    patched = load_stats(Path(stats_path))
    patched.layers[name] = replace(patched.layers[alias], role=patched.layers[name].role)
    config = ScoreConfig(rank_within_role=False)
    return score_modules(graph, patched, config).modules[name].score


def verify(stats_path: Path, kind: str) -> dict:
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    prov = raw["provenance"]
    stats = load_stats(stats_path)

    model, cfg = build(kind)
    graph = classify_model(model, config=cfg)

    layers = stats.layers
    signals = {
        name: [getattr(layer, name) for layer in layers.values()]
        for name in ("activation_rms_ema", "grad_norm_var", "grad_norm_mean")
    }
    calls = sorted({layer.forward_calls for layer in layers.values()})
    steps = prov.get("num_optimizer_steps")
    counts = {layer.grad_norm_count for layer in layers.values()}

    report = score_modules(graph, stats)
    neutral = sorted(set(report.missing_stats) | set(report.unexercised))

    # Check 4. Forcing a score to 0.0 and to 1.0 brackets every allocation the
    # module could have received under *any* signal, so a width that is equal at
    # both ends is a width no measurement could have changed. Where the bracket is
    # open, `alias_score` below says whether a real measurement would have moved it.
    scores = report.scores()
    aliases = (prov.get("notes", {}).get("tied_parameters") or {}).get
    sensitivity: dict[str, dict] = {}
    for name in neutral:
        info = next(m for m in graph.quantizable() if m.name == name)
        alias = next((a for a in (aliases(name) or []) if a in layers), None)
        alias_score = _alias_score(graph, stats_path, name, alias) if alias else None
        per_target = {}
        for target in PROBE_TARGETS:
            budget = Budget.from_target(graph, target_bits=target)
            widths = []
            for forced in (0.0, 1.0):
                widths.append(allocate_bits(graph, {**scores, name: forced}, budget).bits[name])
            entry = {
                "bits_at_score_0": widths[0],
                "bits_at_score_1": widths[1],
                "sensitive": widths[0] != widths[1],
            }
            if alias_score is not None:
                entry["bits_at_alias_score"] = allocate_bits(
                    graph, {**scores, name: alias_score}, budget
                ).bits[name]
                entry["alias_changes_width"] = (
                    entry["bits_at_alias_score"] != allocate_bits(graph, scores, budget).bits[name]
                )
            per_target[f"{target}"] = entry
        sensitivity[name] = {
            "alias": alias,
            "alias_score": alias_score,
            "num_params": info.num_params,
            "param_fraction": round(info.num_params / graph.total_params(), 4),
            "floor_bits": info.floor_bits,
            "role_group_size": sum(1 for m in graph.quantizable() if m.role is info.role),
            "by_target": per_target,
        }

    # How much of the shortfall this module absorbs -- the reason its neutrality is
    # worth checking at all rather than noting and moving on.
    budget = Budget.from_target(graph, target_bits=3.25)
    result = allocate_bits(graph, scores, budget)
    deficit = graph.floor_cost_bits() - budget.total_bits
    for name, entry in sensitivity.items():
        info = next(m for m in graph.quantizable() if m.name == name)
        paid = info.num_params * (info.floor_bits - result.bits[name])
        entry["share_of_deficit_at_3.25"] = round(paid / deficit, 4) if deficit > 0 else 0.0

    return {
        "model": kind,
        "stats": str(stats_path).replace("\\", "/"),
        "structure": {
            "tracked_modules": len(layers),
            "quantizable_modules": len(list(graph.quantizable())),
            "leaves": dict(Counter(n.rsplit(".", 1)[-1] for n in layers)),
            "canonical_names": prov.get("canonical_names"),
            "grad_estimator": prov.get("grad_estimator"),
            "num_optimizer_steps": steps,
            "grad_norm_counts": sorted(counts),
            "welford_per_optimizer_step": counts <= {0, steps},
            "forward_calls_distinct": calls,
            "tied_parameters": prov.get("notes", {}).get("tied_parameters"),
            "channel_moment_modules": prov.get("notes", {})
            .get("channel_moments", {})
            .get("modules"),
        },
        "spread": {name: _spread(values) for name, values in signals.items()},
        "coverage": {
            "scored": len(report.modules) - len(neutral),
            "missing_stats": list(report.missing_stats),
            "unexercised": list(report.unexercised),
        },
        "neutral_module_sensitivity": sensitivity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("signal_map.json"))
    args = parser.parse_args(argv)

    result = verify(args.stats, args.model)
    s, cov = result["structure"], result["coverage"]
    print(
        f"{result['model']:<14} {s['tracked_modules']} modules tracked, "
        f"{cov['scored']} scored, {len(cov['unexercised'])} unexercised, "
        f"{len(cov['missing_stats'])} missing",
        flush=True,
    )
    print(
        f"    welford per optimizer step: {s['welford_per_optimizer_step']}  "
        f"forward_calls: {s['forward_calls_distinct']}",
        flush=True,
    )
    for name, spread in result["spread"].items():
        ratio = "inf" if spread["ratio"] is None else f"{spread['ratio']:.4g}"
        print(f"    {name:<20} med={spread['median']:.4g} ratio={ratio} zeros={spread['zeros']}")
    for name, entry in result["neutral_module_sensitivity"].items():
        moves = [t for t, a in entry["by_target"].items() if a["sensitive"]]
        print(
            f"    {name}: {100 * entry['param_fraction']:.1f}% of params, pays "
            f"{100 * entry['share_of_deficit_at_3.25']:.1f}% of the 3.25b deficit, "
            f"bracket open at {moves or 'no target'}",
            flush=True,
        )
        if entry["alias"]:
            flips = [t for t, a in entry["by_target"].items() if a.get("alias_changes_width")]
            print(
                f"        borrowing {entry['alias']} scores {entry['alias_score']:.4f} and "
                f"changes the width at {flips or 'no target'}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
