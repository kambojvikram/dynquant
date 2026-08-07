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

   That count is not the whole neutral set. A module in a role group of *one* also
   scores 0.5 under the shipped per-role ranking -- percentile rank of a single value
   against itself -- however complete its measurement is. It is not missing and not
   unexercised, so it appears nowhere in check 3, and on an untied checkpoint the LM
   head is exactly this: measured 1 492 times, ranked against nobody. Singletons are
   therefore enumerated separately and carried into check 4 alongside the neutral set.

4. **Whether it matters.** The one that pays for the other three, and it needs three
   measurements rather than one. Forcing a module's score to 0.0 and to 1.0 brackets
   every width any signal could have bought it -- but that bracket is wider than any
   real score, so a module can look "sensitive" to a value nothing would ever
   produce. So the bracket is paired with the *realistic* counterfactual: the score
   the module's own measurements produce when ranked globally, or, for a tie
   representative, its tied partner's stats row re-scored the same way. Both are
   ranked globally rather than within role, because a group of one admits no other
   comparison.

   The third measurement is the one Phi was read without. A counterfactual that lands
   the module on the same width has *not* shown the neutral score to be free: the
   knapsack is a single budget, so a score that changes where a large tensor sits in
   the ROI order changes how much budget reaches the tail even when the tensor itself
   does not move. So each counterfactual also reports how many *other* modules moved
   and what the byte total did. ``modules_moved`` is the honest headline;
   ``changes_width`` is a detail of it.

On Phi-4-mini ``model.embed_tokens`` is tied to ``lm_head``, and under the
``outer_exact`` estimator an embedding has no ``delta x^T`` to form -- its gradient
is a scatter-add -- so it carries no plasticity signal, and no channel moments
either, both deliberate (see ``signals/tracker.py``). It is also the only member of
its role group. Three separate reasons for one neutral score on 16% of the model, and
that module pays 68% of the shortfall at a 3.25-bit target. The bracket spans 2 to 4
bits, so the neutral score is not structurally harmless; the alias substitution ranks
0.93 and leaves that module on 3 bits at every target.

**That last fact was reported as "costs this checkpoint nothing", and the whole-map
measurement says otherwise**: at the same substitution the map moves 12 / 5 / 11 / 8
modules at 3.25 / 4.0 / 4.25 / 4.5, at byte totals equal to within 0.02%. Nothing was
free; a reallocation happened and one width was watched.

On Ministral-8B, untied, the same measurement is sharper. ``lm_head`` carries the
highest saliency in the model (global rank 0.998) and ranks 0.978 globally, against
the 0.5 its group of one hands it. At 3.25 b that is a whole bit on 6.7% of the
parameters -- 3 b shipped against 4 b measured -- paid for by 24 projections dropping
4 b to 3 b at an identical byte total. At 4.0 b the head's own width does not move and
23 other modules do. Which map is better is an eval question and an S3 arm; that the
shipped one is not the one the measurements imply is settled here.

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


def _global_scores(graph, stats) -> dict[str, float]:
    """Every module's score ranked against the whole model instead of its role.

    Ranking globally is not proposed as a better default -- per-role ranking exists
    because on an 18k-expert model a global ranking drowns attention and MLP. It is
    the only comparison a group of *one* admits, and a group of one is where the
    shipped ranking returns 0.5 regardless of what was measured.
    """
    from dynquant.score.importance import ScoreConfig

    return score_modules(graph, stats, ScoreConfig(rank_within_role=False)).scores()


def _alias_score(graph, stats_path: Path, name: str, alias: str) -> float:
    """What ``name`` would score if it borrowed its tied partner's measurements.

    Ranked globally for the reason in :func:`_global_scores`. The role is carried
    over from the original row so the substitution changes the signal and nothing
    else. The result is injected into the shipped per-role score map by the caller:
    percentile ranks all live on (0, 1) and the knapsack only ever compares them
    across modules, so this reads as "give this tensor a score reflecting what was
    actually measured about it" -- but it is a hybrid, and a map that moves under it
    is a reason to look, not a finished number.
    """
    from dataclasses import replace

    from dynquant.score.importance import ScoreConfig

    patched = load_stats(Path(stats_path))
    patched.layers[name] = replace(patched.layers[alias], role=patched.layers[name].role)
    config = ScoreConfig(rank_within_role=False)
    return score_modules(graph, patched, config).modules[name].score


def _counterfactual(graph, scores: dict[str, float], budget, name: str, forced: float) -> dict:
    """Re-allocate with one score replaced, and report what the *map* did.

    Not just ``name``'s width. The budget is shared, so moving a large tensor's
    position in the ROI order changes how much of it reaches everything ranked below,
    and a counterfactual that leaves the tensor itself on the same width can still
    have rewritten the tail. Phi's was read as costing nothing on exactly that basis
    while twelve other modules moved.
    """
    base = allocate_bits(graph, scores, budget)
    alt = allocate_bits(graph, {**scores, name: forced}, budget)
    moved = [n for n in base.bits if base.bits[n] != alt.bits[n]]
    params = {m.name: m.num_params for m in graph.quantizable()}
    return {
        "score": forced,
        "bits": alt.bits[name],
        "changes_width": alt.bits[name] != base.bits[name],
        "modules_moved": len(moved),
        "other_modules_moved": sum(1 for n in moved if n != name),
        "bytes_before": sum(params[n] * base.bits[n] for n in base.bits) // 8,
        "bytes_after": sum(params[n] * alt.bits[n] for n in alt.bits) // 8,
    }


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

    # A module alone in its role group scores 0.5 from the shipped ranker whatever it
    # measured, so it is uninformed in exactly the way the neutral set is -- and it is
    # in neither `missing_stats` nor `unexercised`, so check 3 never names it. Probe
    # it on the same terms.
    group_size = Counter(m.role for m in graph.quantizable())
    singletons = sorted(
        m.name for m in graph.quantizable() if group_size[m.role] == 1 and m.name not in neutral
    )

    # Check 4. Forcing a score to 0.0 and to 1.0 brackets every allocation the
    # module could have received under *any* signal, so a width that is equal at
    # both ends is a width no measurement could have changed. Where the bracket is
    # open, the counterfactuals below say what a real measurement would have done --
    # to this module and to the rest of the map.
    scores = report.scores()
    global_scores = _global_scores(graph, stats)
    aliases = (prov.get("notes", {}).get("tied_parameters") or {}).get
    sensitivity: dict[str, dict] = {}
    for name in sorted(set(neutral) | set(singletons)):
        info = next(m for m in graph.quantizable() if m.name == name)
        alias = next((a for a in (aliases(name) or []) if a in layers), None)
        alias_score = _alias_score(graph, stats_path, name, alias) if alias else None
        own_score = global_scores[name]
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
                # What this module's own measurements say, ranked globally. Equal to
                # the shipped 0.5 exactly when there was nothing to measure, which is
                # itself the answer for an unexercised module.
                "own": _counterfactual(graph, scores, budget, name, own_score),
            }
            if alias_score is not None:
                entry["alias"] = _counterfactual(graph, scores, budget, name, alias_score)
            per_target[f"{target}"] = entry
        # All the reasons, not the first one. Phi's tied embedding is uninformed three
        # times over -- no plasticity under `outer_exact`, no channel moments, and a
        # role group of one -- and reporting only the reason that happened to be
        # tested first makes a triply-determined 0.5 look incidental.
        reasons = []
        if name in report.missing_stats:
            reasons.append("no stats entry")
        if name in report.unexercised:
            reasons.append("unexercised or no gradient observations")
        if group_size[info.role] == 1:
            reasons.append("role group of one")

        sensitivity[name] = {
            "reasons": reasons,
            "alias": alias,
            "alias_score": alias_score,
            "own_score": own_score,
            "shipped_score": scores[name],
            "num_params": info.num_params,
            "param_fraction": round(info.num_params / graph.total_params(), 4),
            "floor_bits": info.floor_bits,
            "role_group_size": group_size[info.role],
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
            # Measured, and ranked against nobody. `scored` counts these as scored
            # because they are; the number that matters for check 4 is scored minus
            # this.
            "singleton_role_groups": singletons,
            "informed": len(report.modules) - len(neutral) - len(singletons),
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
        f"{cov['informed']} informed, {len(cov['unexercised'])} unexercised, "
        f"{len(cov['missing_stats'])} missing, "
        f"{len(cov['singleton_role_groups'])} ranked against nobody",
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
            f"    {name} ({'; '.join(entry['reasons'])}): "
            f"{100 * entry['param_fraction']:.1f}% of params, "
            f"pays {100 * entry['share_of_deficit_at_3.25']:.1f}% of the 3.25b deficit, "
            f"bracket open at {moves or 'no target'}",
            flush=True,
        )
        for key, label in (("own", "its own signal"), ("alias", f"borrowing {entry['alias']}")):
            arms = {t: a[key] for t, a in entry["by_target"].items() if key in a}
            if not arms:
                continue
            score = next(iter(arms.values()))["score"]
            flips = [t for t, a in arms.items() if a["changes_width"]]
            spill = {t: a["other_modules_moved"] for t, a in arms.items()}
            print(
                f"        {label} scores {score:.4f}, changes this width at "
                f"{flips or 'no target'}, and moves other modules {spill}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
