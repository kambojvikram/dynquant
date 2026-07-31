"""Does clipping on the objective that matters beat clipping on plain MSE?

The clip search minimises ``sum_c (W - Q)^2_rc`` per group. The quantity that
reaches the loss is ``sum_c E[x_c^2] (W - Q)^2_rc``. Those are the same objective
only if every input channel is equally consequential, and the entire premise of
the moments is that they are not.

The sweep found the gap empirically. Extending the clip grid to reach the 2-bit
optimum cut ``k_proj``'s relative error 16% while *raising* its Gauss-Newton
sensitivity 6.7%: a tighter clip buys resolution for the bulk of a group by
crushing its large-magnitude weights, and on some modules those are exactly the
weights the active channels multiply. Unweighted MSE cannot see that trade; the
weighted objective is the trade, written down.

This is as calibration-free as the rest of DynQuant -- ``E[x_c^2]`` is the same
training-time accumulator the allocator already spends, read from disk, no forward
pass and no held-out split. It costs one broadcast multiply per candidate.

Three arms per module, at the widths where the allocator actually has a choice:

    shipped   grid stops at 0.80, unweighted objective   (what ships today)
    deep      grid reaches 0.40, unweighted objective    (the sweep's change)
    deep+w    grid reaches 0.40, E[x_c^2]-weighted       (this file's question)

Reported as sensitivity, because that is what the allocator compares and what the
two objectives disagree about. Relative error is reported too and is *expected* to
get worse in the weighted arm -- it is no longer what is being minimised, and a
weighted arm that also won on unweighted MSE would mean the weighting did nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from common import RUN_DIR, TASK, load_model


def measure(
    weight: torch.Tensor,
    x2: torch.Tensor,
    d2: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    candidates: tuple[float, ...],
    weighted: bool,
) -> tuple[float, float, float]:
    """``(relative error, sensitivity, mean clip ratio)`` for one objective."""
    from dynquant.quant.grid import quantize_with_search

    with torch.no_grad():
        ref = weight.to(torch.float32)
        x2f = x2.to(weight.device, torch.float32)
        d2f = d2.to(weight.device, torch.float32)
        quantized, search = quantize_with_search(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=False,
            candidates=candidates,
            channel_weight=x2f if weighted else None,
        )
        squared = (ref - quantized.dequantize(dtype=torch.float32)) ** 2
        rms = float(torch.sqrt(torch.mean(ref**2)))
        rel = float(torch.sqrt(torch.mean(squared))) / rms if rms > 0 else 0.0
        sens = float((squared @ x2f) @ d2f)
        return rel, sens, float(search.ratios.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--moments", default=str(RUN_DIR / "dynquant_moments.safetensors"))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--per-role", type=int, default=2)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--out", default=str(RUN_DIR / "p2_wclip.json"))
    args = parser.parse_args()

    if "qwen" not in RUN_DIR.name or TASK.key != "casehold":
        raise SystemExit(f"refusing to write into {RUN_DIR}; set DQ_MODEL and DQ_TASK")

    from dynquant.graph.classify import classify_model
    from dynquant.quant.grid import CLIP_CANDIDATES, DEEP_CLIP_CANDIDATES
    from dynquant.score.sensitivity import module_weights
    from dynquant.signals.moments import load_moments

    moments = load_moments(Path(args.moments))
    model = load_model(args.model)
    graph = classify_model(model)
    weights = module_weights(model)

    by_role: dict[str, list[str]] = {}
    for info in graph.quantizable():
        if info.name in weights and weights[info.name].ndim == 2:
            by_role.setdefault(info.role.value, []).append(info.name)
    picked: list[str] = []
    for _role, names in sorted(by_role.items()):
        names.sort(key=lambda n: -graph[n].num_params)
        picked.extend(names[: args.per_role])

    arms = (
        ("shipped", CLIP_CANDIDATES, False),
        ("deep", DEEP_CLIP_CANDIDATES, False),
        ("deep+w", DEEP_CLIP_CANDIDATES, True),
    )
    records: list[dict[str, object]] = []
    # Aggregate the way the allocator would: sensitivity is an estimated loss
    # increase, so summing across modules is the meaningful comparison. A per-module
    # win rate would weight a 32k projection the same as a 508M embedding.
    totals: dict[tuple[int, str], float] = {}

    print(f"  {len(picked)} modules over {len(by_role)} roles, widths {args.bits}\n", flush=True)
    for name in picked:
        weight = weights[name]
        x2 = moments.input_sq.get(name)
        d2 = moments.output_grad_sq.get(name)
        if x2 is None or d2 is None:
            continue
        if x2.shape[0] != weight.shape[1] or d2.shape[0] != weight.shape[0]:
            continue
        print(f"{name}  {tuple(weight.shape)}  role={graph[name].role.value}", flush=True)
        print("    bits  arm       rel_err   sensitivity   clip   vs deep", flush=True)
        for bits in args.bits:
            baseline = None
            for arm, candidates, weighted in arms:
                rel, sens, clip = measure(
                    weight,
                    x2,
                    d2,
                    bits=bits,
                    group_size=args.group_size,
                    candidates=candidates,
                    weighted=weighted,
                )
                if arm == "deep":
                    baseline = sens
                delta = ""
                if arm == "deep+w" and baseline:
                    delta = f"{100 * (sens / baseline - 1):+7.1f}%"
                print(
                    f"    {bits:4d}  {arm:8s}  {rel:7.4f}  {sens:12.4e}  {clip:.3f}  {delta}",
                    flush=True,
                )
                key = (bits, arm)
                totals[key] = totals.get(key, 0.0) + sens
                records.append(
                    {
                        "module": name,
                        "role": graph[name].role.value,
                        "bits": bits,
                        "arm": arm,
                        "rel_err": rel,
                        "sensitivity": sens,
                        "mean_clip": clip,
                    }
                )
        print(flush=True)

    print("  total sensitivity over the sampled modules (lower is better):", flush=True)
    for bits in args.bits:
        deep = totals.get((bits, "deep"), 0.0)
        line = f"    {bits}b  "
        for arm, _c, _w in arms:
            total = totals.get((bits, arm), 0.0)
            rel = f" ({100 * (total / deep - 1):+.1f}% vs deep)" if deep else ""
            line += f"{arm}={total:.4e}{rel}   "
        print(line, flush=True)

    Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n-> wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
