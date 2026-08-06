"""``dynquant inspect`` -- what the allocator decided, and whether to believe it.

A width histogram is not an inspection. The research supplement's allocator
produced a complete, plausible-looking bit map at its own headline 3-bit target
while never reading the importance scores at all -- inverting every score changed
0 of 282 modules -- and nothing in its output said so. So this command reports the
three things that would have caught that, none of which are visible in a list of
widths:

**Within-role concordance.** Over every pair of modules in the same role that got
different widths, how often is the wider one the higher-scoring one? Near 0.5
means the ordering is not reaching the allocator and size alone is choosing the
widths. Within role rather than globally, because floors differ by role by design,
so a global comparison is confounded by the very structure the allocator is
supposed to respect. Measured against whichever quantity actually drove the
widths -- the rank score, or measured ``dL`` when ``--moments`` was given -- and
the report names which, since a diagnostic that checks a signal the allocator did
not use is the same defect in a new place.

**Floor violations.** Soft floors are how scores get to matter at a 3-bit target
at all -- the alternative is returning the floor map and calling it an
allocation -- but a breached floor is a real risk that must be named, not a
footnote. Every one is listed with the role and the width it landed on.

**What could not be scored.** Modules with no stats entry, and modules present in
the stats but never exercised. A module the score never saw is a module allocated
blind, and silence about it looks identical to success.

Runs on CPU by default. Only names, shapes and the module tree are read, so the
weights never need to reach the GPU -- and holding a second copy of a model there
while an evaluation runs is a way to turn an analysis into an OOM in the run that
matters. ``--moments`` is the exception: cardinal sensitivity quantizes every
tensor at every candidate width, which is worth a GPU.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from . import _shared

if TYPE_CHECKING:
    import argparse

    from dynquant.allocate.knapsack import BitMap
    from dynquant.graph.classify import ModelGraph
    from dynquant.score.sensitivity import SensitivityTable

__all__ = ["concordance", "inspect_allocation", "ordering_values", "run"]


def concordance(pairs: list[tuple[float, int]]) -> tuple[int, int]:
    """Agreeing and disagreeing ``(score, width)`` pairs.

    Ties in either quantity are skipped: they carry no information about ordering
    and would dilute the ratio toward 0.5, which is the value that means "broken".
    """
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


def ordering_values(
    graph: ModelGraph,
    scores: dict[str, float],
    sensitivity: SensitivityTable | None,
) -> tuple[dict[str, float], str, list[str]]:
    """The per-module quantity that *ordered* the widths, whichever one that was.

    Concordance is only a check on the allocator if it is measured against what the
    allocator priced with. When ``--moments`` is given the widths come from measured
    ``dL``, and the rank score is then a different signal that disagrees with it by
    design -- the two orderings differ on roughly two-thirds of modules at 3.25 bits.
    Reporting the rank score's concordance in that case answers a question nobody
    asked, under a heading that says otherwise, which is the exact failure this
    command exists to catch.

    So with sensitivity the quantity is ``(dL(2) - dL(8)) / num_params``: the loss
    per parameter that going from the narrowest to the widest option recovers. That
    is not a new metric -- in the proxy path the same expression is
    ``score * (err(2) - err(8))``, a positive constant times ``score``, and
    concordance is scale-invariant, so the two definitions coincide exactly wherever
    sensitivity is absent. Dividing by ``num_params`` is what keeps the diagnostic
    pointed at its own failure mode: an undivided ``dL`` grows with tensor size, so
    concordance against it would read as healthy in precisely the case where size
    alone was choosing the widths.

    Returns ``(values, what they are, modules that had to fall back)``.
    """
    if sensitivity is None:
        return dict(scores), "score", []

    lo, hi = min(sensitivity.bit_options), max(sensitivity.bit_options)
    values: dict[str, float] = {}
    fallback: list[str] = []
    for name in bit_map_names(graph):
        gain = sensitivity.gain(name, lo, hi)
        if gain is None:
            fallback.append(name)
            continue
        values[name] = gain / max(graph[name].num_params, 1)
    return values, f"measured dL({lo}b)-dL({hi}b) per parameter", fallback


def bit_map_names(graph: ModelGraph) -> list[str]:
    return [info.name for info in graph.quantizable()]


def inspect_allocation(
    graph: ModelGraph,
    scores: dict[str, float],
    bit_map: BitMap,
    *,
    narrowest: int = 8,
    sensitivity: SensitivityTable | None = None,
) -> dict[str, Any]:
    """Everything worth knowing about one bit map, as data rather than as text."""
    ordering, quantity, fallback = ordering_values(graph, scores, sensitivity)

    by_width: dict[int, list[str]] = defaultdict(list)
    by_role: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for name, width in bit_map.bits.items():
        by_width[width].append(name)
        if name in ordering:
            # A module with no value in the driving quantity is left out rather
            # than entered as zero: zero is an ordering claim, and the reason it
            # is missing is that nothing was measured.
            by_role[graph[name].role.value].append((ordering[name], width))

    agree = disagree = 0
    per_role: dict[str, dict[str, int]] = {}
    for role, pairs in by_role.items():
        role_agree, role_disagree = concordance(pairs)
        agree += role_agree
        disagree += role_disagree
        if role_agree + role_disagree:
            per_role[role] = {"agree": role_agree, "total": role_agree + role_disagree}

    widths: dict[str, dict[str, float]] = {}
    for width in sorted(by_width):
        ranked = sorted(ordering.get(name, 0.0) for name in by_width[width])
        params = sum(graph[name].num_params for name in by_width[width])
        widths[str(width)] = {
            "modules": len(ranked),
            "params": params,
            "score_min": ranked[0],
            "score_median": ranked[len(ranked) // 2],
            "score_max": ranked[-1],
        }

    ordered = sorted(bit_map.bits.items(), key=lambda kv: (kv[1], ordering.get(kv[0], 0.0)))
    return {
        "target_label": bit_map.target_label,
        "average_bits": bit_map.average_bits,
        "nbytes": bit_map.nbytes,
        "quantity": quantity,
        "unordered": fallback,
        "widths": widths,
        "concordance": {
            "agree": agree,
            "total": agree + disagree,
            "value": (agree / (agree + disagree)) if agree + disagree else None,
            "per_role": per_role,
        },
        "violations": [
            {
                "name": v.name,
                "role": v.role.value,
                "floor_bits": v.floor_bits,
                "assigned_bits": v.assigned_bits,
                "num_params": v.num_params,
            }
            for v in bit_map.violations
        ],
        "narrowest": [
            {
                "name": name,
                "bits": width,
                "role": graph[name].role.value,
                "num_params": graph[name].num_params,
                "score": ordering.get(name, 0.0),
            }
            for name, width in ordered[:narrowest]
        ],
    }


def render(report: dict[str, Any]) -> str:
    # Named on every line that prints one, because a rank score and a loss delta
    # per parameter differ by nine orders of magnitude and by meaning, and a bare
    # "score" column that silently switches between them is the ambiguity this
    # whole command exists to remove.
    quantity = report.get("quantity", "score")
    lines = [
        f"=== {report['target_label']} ===",
        f"  achieved {report['average_bits']:.4f} avg bits, "
        f"{report['nbytes'] / 2**30:.3f} GiB stored",
    ]
    for width, stats in report["widths"].items():
        lines.append(
            f"  {width:>2}b  n={stats['modules']:3d}  {stats['params'] / 1e6:8.1f}M params  "
            f"{quantity} min {stats['score_min']:.4g}  median {stats['score_median']:.4g}  "
            f"max {stats['score_max']:.4g}"
        )

    con = report["concordance"]
    if con["value"] is None:
        lines.append("  every module in every role got the same width: nothing to correlate")
    else:
        verdict = "not reaching the allocator" if con["value"] < 0.55 else "ordered"
        lines.append(
            f"  within-role concordance of width with {quantity}: "
            f"{con['agree']}/{con['total']} = {con['value']:.3f}   ({verdict}; 0.5 is what "
            f"no ordering at all looks like)"
        )
    if report.get("unordered"):
        lines.append(
            f"  {len(report['unordered'])} module(s) left out of the concordance with no "
            f"measured value: {', '.join(report['unordered'][:3])}"
        )

    if report["violations"]:
        lines.append(f"  {len(report['violations'])} floor violation(s):")
        for v in report["violations"][:10]:
            lines.append(
                f"    {v['assigned_bits']}b (floor {v['floor_bits']}b)  {v['role']:16s} {v['name']}"
            )
        if len(report["violations"]) > 10:
            lines.append(f"    ... and {len(report['violations']) - 10} more")
    else:
        lines.append("  no floor violations")

    lines.append(f"  narrowest modules, with the {quantity} that put them there:")
    for entry in report["narrowest"]:
        lines.append(
            f"    {entry['bits']}b  {entry['score']:10.4g}  "
            f"{entry['num_params'] / 1e6:6.1f}M  {entry['role']:16s} {entry['name']}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    from dynquant._version import __version__
    from dynquant.errors import DynQuantError

    quiet = bool(args.json)
    model = _shared.load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    inputs = _shared.build_inputs(
        model,
        stats=args.stats,
        moments=args.moments,
        group_size=args.group_size,
        symmetric=args.symmetric,
        overrides=_shared.parse_overrides(args.role),
        soft_floors=not args.hard_floors,
        verbose=not quiet,
    )
    graph = inputs.graph
    scores = dict(inputs.scores)
    score_report = inputs.score_report

    payload: dict[str, Any] = {
        "dynquant_core": __version__,
        "model": args.model,
        "stats": args.stats,
        "allocator": inputs.allocator,
        "group_size": args.group_size,
        "model_type": graph.model_type,
        "modules": {
            "quantizable": len(graph.quantizable()),
            "unquantized": len(graph.unquantized()),
            "skipped": len(graph.skipped),
            "unclassified": list(graph.unclassified()),
        },
        "params": {
            "total": graph.total_params(),
            "unique": graph.unique_params(),
            "unquantized": graph.unquantized_params(),
        },
        # Empty rather than absent when no ``--stats`` was given: a reader that keys
        # into this dict gets three empty lists, which is the truth -- no module is
        # missing a signal when no signal was asked for -- instead of a KeyError.
        "scores": {
            "missing_stats": list(score_report.missing_stats) if score_report else [],
            "unexercised": list(score_report.unexercised) if score_report else [],
            "zero": [name for name, score in scores.items() if score == 0.0],
        },
        "targets": {},
    }

    if not quiet:
        print(graph.report(), flush=True)
        if graph.unclassified():
            print(
                f"\n{len(graph.unclassified())} module(s) fell through to a conservative "
                f"default and were not classified: "
                f"{', '.join(graph.unclassified()[:5])}",
                flush=True,
            )
        # A score of exactly zero makes a module free to destroy: the allocator
        # prices damage as score * params * delta_error. The ranker maps onto the
        # open interval to prevent it; this is the check against a real graph.
        if payload["scores"]["zero"]:
            print(
                f"\n{len(payload['scores']['zero'])} module(s) score exactly zero, which "
                f"makes them free for the allocator to destroy: "
                f"{', '.join(payload['scores']['zero'][:5])}",
                flush=True,
            )

    targets = list(args.target or [])
    if not targets and args.target_size is None and args.target_ratio is None and not args.uniform:
        if not quiet:
            print("\nno --target given: classification and scores only", flush=True)
        _emit(payload, quiet)
        return 0

    maps: dict[str, BitMap] = {}
    for target in targets:
        maps[f"{target:.2f}"] = _shared.allocate(inputs, target_bits=target)
    if args.target_size is not None:
        maps[args.target_size] = _shared.allocate(inputs, target_size=args.target_size)
    if args.target_ratio is not None:
        maps[f"ratio-{args.target_ratio:.3f}"] = _shared.allocate(
            inputs, target_ratio=args.target_ratio
        )
    for width in args.uniform or []:
        maps[f"uniform-{width}"] = _shared.uniform_map(
            graph, width, policy=inputs.policy, group_size=args.group_size
        )

    for key, bit_map in maps.items():
        report = inspect_allocation(
            graph,
            scores,
            bit_map,
            narrowest=args.narrowest,
            # The uniform arms are not allocated from anything, so there is no
            # driving quantity to check them against; they are reported against the
            # score so the control arm's column stays comparable across runs.
            sensitivity=None if key.startswith("uniform-") else inputs.sensitivity,
        )
        payload["targets"][key] = report
        if not quiet:
            print("\n" + render(report), flush=True)

    if args.save_map:
        written = _shared.write_bit_maps(
            args.save_map,
            maps,
            model=args.model,
            stats=args.stats,
            allocator=inputs.allocator,
            group_size=args.group_size,
        )
        payload["saved_map"] = str(written)
        if not quiet:
            print(f"\n-> wrote {written}", flush=True)

    _emit(payload, quiet)

    if args.fail_on_violation and any(t["violations"] for t in payload["targets"].values()):
        raise DynQuantError(
            "floor violations present and --fail-on-violation was given. The budget "
            "cannot be met without breaching a quality floor; raise the target or "
            "accept the listed breaches."
        )
    return 0


def _emit(payload: dict[str, Any], quiet: bool) -> None:
    if quiet:
        print(json.dumps(payload, indent=2, default=str))
