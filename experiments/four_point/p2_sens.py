"""Allocate the body from measured sensitivity, using the fine-tune's own moments.

The 3.25-bit map that scored 86.70% was built by ``allocate_bits(graph, scores,
budget, policy)`` -- four arguments, no ``sensitivity=``. So the widths came from the
rank-product proxy: a percentile rank standing in for importance, multiplied by a
*universal* ``4**-bits`` curve standing in for what a width change costs. The second
factor is identical for every module in the model, which means it can express "more
bits is better" and nothing else, while exploiting the difference between tensors is
the entire reason a bit allocator exists. ``knapsack.py``'s own docstring records the
two against measured loss disturbance on this exact model: rank product +0.231 mean
within-role Spearman, this estimator +0.521.

The estimator was available the whole time. What it needs -- ``E[x_c^2]`` and
``E[delta_r^2]`` per module -- is what the training hook already wrote to
``dynquant_moments.safetensors`` during the fine-tune, all 187 modules, both
accumulators complete.

That last point is the one worth being careful about. ``stage4_sensitivity.py``
collects moments in a post-hoc pass over the ``validation`` split, and its docstring
argues for that over ``train`` on the grounds that two epochs of SFT drive the
training loss to 0.0000, so a pass at the final weights reads ``E[delta^2]`` off a
regime the quantized model will never be in. That argument is about a *post-hoc* pass.
It does not apply to these moments, which were accumulated online across the whole
optimisation trajectory -- including the early steps where the loss was still large --
and so carry no held-out split, no calibration set, and no extra forward pass. Which
is the thing DynQuant claims to be able to do and GPTQ cannot. Using them here is not
a convenience; it is the claim under test.

The file is read, never written. ``save_moments`` would clobber the training-time
record with a calibration one and the row signal in ``p2_rowsignal.py`` reads the same
file, so the digest of what was actually loaded goes into the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from common import MODEL_ID, RUN_DIR, TASK, load_model, set_seed


def digest(path: Path) -> dict[str, object]:
    """Enough to prove later which moments an allocation was built from."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": h.hexdigest()[:16],
        "bytes": stat.st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
    }


def embed_name(graph) -> str:
    """The tied embedding/head module, matched by suffix like ``p2.py`` does."""
    hits = [
        info.name
        for info in graph.quantizable()
        if info.name.rsplit(".", 1)[-1] in ("embed_tokens", "lm_head")
    ]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one embedding/head module, found {hits}")
    return hits[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--stats", default=str(RUN_DIR / "stats"))
    parser.add_argument("--moments", default=str(RUN_DIR / "dynquant_moments.safetensors"))
    parser.add_argument("--targets", type=float, nargs="+", default=[3.25])
    parser.add_argument(
        "--target-total-bits",
        type=float,
        nargs="+",
        default=None,
        help=(
            "budget as an exact stored-bit count rather than an average. The claim "
            "under test is 'smaller than GPTQ's 5,932,207,432 bits', and an average "
            "rounded to two decimals moves the byte count by ~1 MB, which is the "
            "wrong side of a 3% margin."
        ),
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--clip",
        default="shipped",
        choices=("shipped", "deep"),
        help=(
            "clip grid the sensitivity is priced against. MUST match what p2.py "
            "quantizes with -- a map priced on one grid and spent on another is "
            "optimal for a checkpoint nobody builds."
        ),
    )
    parser.add_argument(
        "--weighted-clip",
        action="store_true",
        help=(
            "choose each group's clip ratio by E[x_c^2]-weighted error rather than "
            "plain MSE. Same requirement as --clip: p2.py must be given the matching "
            "--weighted-clip, because this changes what 'error' means and a map priced "
            "on one definition is not optimal for a checkpoint encoded on the other."
        ),
    )
    parser.add_argument(
        "--reserve-embed-bits",
        type=float,
        default=None,
        help=(
            "exclude the tied tensor from the allocator and reserve this many stored "
            "bits for it, so the body is allocated against what a row-partitioned tie "
            "actually leaves. Print the number with p2.py's stored_bits_rowwise."
        ),
    )
    parser.add_argument("--out", default=str(RUN_DIR / "p2_sens_bitmaps.json"))
    args = parser.parse_args()

    print(f"  MODEL_ID={MODEL_ID}  TASK={TASK.key}  RUN_DIR={RUN_DIR}", flush=True)
    if "qwen" not in RUN_DIR.name or TASK.key != "casehold":
        raise SystemExit(f"refusing to write into {RUN_DIR}; set DQ_MODEL and DQ_TASK")

    set_seed()
    from dynquant.allocate.budget import Budget
    from dynquant.allocate.knapsack import allocate_bits
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.classify import classify_model
    from dynquant.graph.roles import UNQUANTIZED_FLOOR
    from dynquant.quant.grid import CLIP_CANDIDATES, DEEP_CLIP_CANDIDATES
    from dynquant.score.importance import score_modules
    from dynquant.score.sensitivity import estimate_sensitivity, module_weights
    from dynquant.signals.moments import load_moments
    from dynquant.signals.schema import load_stats

    moments_path = Path(args.moments)
    provenance = digest(moments_path)
    print(f"  moments {provenance['sha256']} written {provenance['mtime']}", flush=True)

    moments = load_moments(moments_path)
    print(
        f"  {len(moments.complete_names())} modules with both accumulators "
        f"of {len(moments.names)} touched",
        flush=True,
    )

    model = load_model(args.model)
    model.config.use_cache = False
    graph = classify_model(model)
    emb = embed_name(graph)
    print(f"  {len(graph.quantizable())} quantizable modules, tie is {emb}", flush=True)

    candidates = DEEP_CLIP_CANDIDATES if args.clip == "deep" else CLIP_CANDIDATES
    objective = "E[x^2]-weighted" if args.weighted_clip else "unweighted MSE"
    print(
        f"  clip grid '{args.clip}': {len(candidates)} ratios down to {min(candidates)}, "
        f"chosen on {objective}",
        flush=True,
    )

    started = time.time()
    table = estimate_sensitivity(
        graph,
        moments,
        module_weights(model),
        group_size=args.group_size,
        candidates=candidates,
        weighted_clip=args.weighted_clip,
    )
    print(f"  priced in {time.time() - started:.0f}s\n  {table.summary()}", flush=True)
    if emb in table.unestimable:
        print(f"  NOTE: {emb} is unestimable and falls back to its score", flush=True)

    stats = load_stats(args.stats)
    scores = score_modules(graph, stats).scores()
    policy = AllocationPolicy(group_size=args.group_size)

    denominator = graph.total_params()
    fixed = float(graph.unquantized_params() * UNQUANTIZED_FLOOR)
    print(
        f"  {denominator:,} total params, {graph.unquantized_params():,} unquantized "
        f"= {fixed:,.0f} fixed bits",
        flush=True,
    )

    # (label, total stored bits) for every budget asked for, however it was asked.
    requests: list[tuple[str, float]] = [
        (f"{t:.2f}", float(denominator) * t) for t in (args.targets or [])
    ]
    requests += [(f"{b / 8 / 2**30:.4f}GiB", float(b)) for b in (args.target_total_bits or [])]

    # The allocator sees the whole model unless the tie is being handled separately,
    # in which case it sees the body and a budget already net of the tie's cost.
    if args.reserve_embed_bits is None:
        sub, reserved = graph, 0.0
    else:
        reserved = float(args.reserve_embed_bits)
        sub = replace(graph, modules={k: v for k, v in graph.modules.items() if k != emb})
        print(
            f"  tie excluded from allocation, {reserved / 8 / 2**20:.1f} MiB reserved "
            f"for it ({reserved / graph[emb].num_params:.4f} b/param)",
            flush=True,
        )

    output: dict[str, object] = {
        "model": args.model,
        "allocator": "sensitivity",
        # p2.py asserts against this: the grid the widths were priced with is the
        # grid the checkpoint has to be encoded with.
        "clip": args.clip,
        "clip_candidates": list(candidates),
        # And the objective the grid was searched on, for the same reason. Absent in
        # maps written before this existed, which read back as False -- correct, since
        # every one of them was priced on unweighted MSE.
        "weighted_clip": args.weighted_clip,
        "moments": provenance,
        "moments_source": "training-time hook, accumulated online during the fine-tune",
        "sensitivity": {
            "priced": len(table.values),
            "unestimable": list(table.unestimable),
        },
        "embed_module": emb,
        "reserved_embed_bits": reserved,
        "maps": {},
    }

    for key, total in requests:
        budget = Budget(
            total_bits=total,
            fixed_bits=fixed + reserved,
            denominator=denominator,
            label=f"{total / denominator:.4f} avg bits, {total / 8 / 2**30:.4f} GiB",
        )
        real = allocate_bits(sub, scores, budget, policy, sensitivity=table)
        ranked = allocate_bits(sub, scores, budget, policy)
        moved = sum(1 for name, bits in real.bits.items() if ranked.bits[name] != bits)

        bits = dict(real.bits)
        if emb not in bits:
            # p2.py locates the tie through this map even when it quantizes it
            # row-wise, so the entry has to exist. Nominal: what the reservation buys
            # if it were spent uniformly. Never used as a width in --embed rows mode.
            bits[emb] = 0

        print(f"\n=== target {key} ({total:,.0f} stored bits) ===", flush=True)
        print(real.summary(), flush=True)
        print(f"  {moved}/{len(real.bits)} modules differ from the rank-product map", flush=True)

        params_at: dict[int, int] = {}
        for name, width in real.bits.items():
            params_at[width] = params_at.get(width, 0) + graph[name].num_params
        print(
            "  params per width: "
            + ", ".join(f"{b}b={p / 1e6:.1f}M" for b, p in sorted(params_at.items())),
            flush=True,
        )
        ranked_at: dict[int, int] = {}
        for name, width in ranked.bits.items():
            ranked_at[width] = ranked_at.get(width, 0) + graph[name].num_params
        print(
            "  rank-product for comparison: "
            + ", ".join(f"{b}b={p / 1e6:.1f}M" for b, p in sorted(ranked_at.items())),
            flush=True,
        )

        output["maps"][key] = {  # type: ignore[index]
            "target_total_bits": total,
            "average_bits": real.average_bits,
            "nbytes": real.nbytes,
            "group_size": args.group_size,
            "modules_differing_from_rank_product": moved,
            "reserved_embed_bits": reserved,
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
            "bits": bits,
        }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n-> wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
