"""Allocate width per output row of every body module, not per module.

The tie experiment established the principle and priced it. Splitting the 248,320
vocabulary rows across widths instead of forcing one width on all of them bought
+2.89 points for 7.4 MB -- 0.39 points per MB, against the body's marginal rate of
0.0109. Thirty-six times more efficient, from granularity alone.

Nothing in that argument was about the embedding. Every body arm so far has forced
one width on an entire ``[out, in]`` projection, and a module's rows are no more
interchangeable than a vocabulary's are: the Gauss-Newton sensitivity of row ``r``
is ``E[delta_r^2] * sum_c E[x_c^2] (W - Q)^2_rc``, and ``E[delta_r^2]`` is a
per-row quantity the hook already stores. A module-level width is that vector
averaged away before it is ever used.

Rows are the natural granularity for a *storage* format too, which is why this is
shippable rather than a paper trick. Groups run along the input dimension, so rows
are already quantized independently; a row block at its own width is exactly what
``RowPartition`` in the format spec describes, and what the fused-projection design
needs anyway. The metadata -- one scale and one offset per row per group -- does not
depend on the split at all. So the granularity upgrade costs bookkeeping and nothing
else, and the entire budget difference is payload.

Allocation is by Lagrangian relaxation rather than greedy ROI. For a multiplier
``lam`` every row independently takes ``argmin_b [ sens_r(b) + lam * cols * b ]``,
and bisecting ``lam`` walks the budget. That is exact on the convex hull of each
row's cost/sensitivity curve, it is one vectorised argmin over ~700k rows instead of
a heap of that many, and -- the reason it is worth the change here -- it applies
unmodified at either granularity. So the module-level counterfactual this script
reports is the *same solver on the same sensitivity table*, differing only in
whether the argmin is taken per row or per module. Any gap is granularity. It cannot
be the allocator, because it is the same allocator.

Output is a width per row as int8 safetensors, consumed by ``p2.py --row-body``.
This file evaluates nothing: it reports what the extra granularity is worth in
predicted loss before any GPU time is spent finding out what it is worth in accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from common import RUN_DIR, TASK, load_model

METADATA_BITS = 32
"""fp16 scale + fp16 offset per row per group. Independent of the row partition."""


def row_sensitivity(
    weight: torch.Tensor,
    x2: torch.Tensor,
    d2: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    candidates: tuple[float, ...],
    chunk: int = 4096,
) -> torch.Tensor:
    """Predicted loss increase from quantizing each output row to ``bits``.

    One value per row, in the same units as the module-level table -- summing this
    over rows reproduces :func:`estimate_sensitivity`'s number for the module, which
    is what makes the two granularities comparable at all.
    """
    from dynquant.quant.grid import quantize_with_search

    out = torch.empty(weight.shape[0], dtype=torch.float64)
    x2f = x2.to(weight.device, torch.float32)
    d2f = d2.to(weight.device, torch.float32)
    with torch.no_grad():
        for start in range(0, weight.shape[0], chunk):
            block = weight[start : start + chunk]
            quantized, _ = quantize_with_search(
                block,
                bits=bits,
                group_size=group_size,
                symmetric=False,
                candidates=candidates,
                row_offset=start,
            )
            squared = (block.to(torch.float32) - quantized.dequantize(dtype=torch.float32)) ** 2
            # (E * x2) over input channels is one number per row; d2 then weights it.
            # Never forms [out, in] twice and never sums across rows.
            out[start : start + chunk] = ((squared @ x2f) * d2f[start : start + chunk]).double()
            del squared, quantized
    return out


def solve(
    sens: torch.Tensor, cost: torch.Tensor, budget: float
) -> tuple[torch.Tensor, float, float]:
    """Lagrangian multiple-choice knapsack over independent items.

    ``sens`` is ``[items, widths]`` predicted loss, ``cost`` is ``[items, widths]``
    payload bits. Returns the chosen width index per item, the bits spent, and the
    total sensitivity. Bisects the multiplier until the spend fits, taking the
    cheapest solution that does -- ``lam`` large enough forces every item to its
    narrowest width, so a feasible point always exists provided the narrowest
    configuration fits at all.
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):  # grow until even the priciest item prefers its cheapest width
        choice = (sens + hi * cost).argmin(dim=1)
        if float(cost.gather(1, choice[:, None]).sum()) <= budget:
            break
        hi *= 4.0
    else:
        raise SystemExit("no multiplier makes the budget feasible")

    best = None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        choice = (sens + mid * cost).argmin(dim=1)
        spent = float(cost.gather(1, choice[:, None]).sum())
        if spent <= budget:
            best = (choice, spent, float(sens.gather(1, choice[:, None]).sum()))
            hi = mid  # feasible: try a smaller multiplier, which buys more bits
        else:
            lo = mid
    if best is None:
        raise SystemExit("bisection found no feasible point")
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--moments", default=str(RUN_DIR / "dynquant_moments.safetensors"))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4, 8])
    parser.add_argument("--clip", default="shipped", choices=("shipped", "deep"))
    parser.add_argument(
        "--target-total-bits",
        type=float,
        default=5_920_000_000.0,
        help="whole-checkpoint budget, tie included",
    )
    parser.add_argument(
        "--reserve-embed-bits",
        type=float,
        default=1_202_978_816.0,
        help="stored bits the tie's own row plan will consume, taken off the top",
    )
    parser.add_argument("--out", default=str(RUN_DIR / "p2_rowbody"))
    args = parser.parse_args()

    print(f"  TASK={TASK.key}  RUN_DIR={RUN_DIR}", flush=True)
    if "qwen" not in RUN_DIR.name or TASK.key != "casehold":
        raise SystemExit(f"refusing to write into {RUN_DIR}; set DQ_MODEL and DQ_TASK")

    from safetensors.torch import save_file

    from dynquant.graph.classify import classify_model
    from dynquant.graph.roles import UNQUANTIZED_FLOOR
    from dynquant.quant.grid import CLIP_CANDIDATES, DEEP_CLIP_CANDIDATES
    from dynquant.score.sensitivity import _moments_for, _tie_aliases, module_weights
    from dynquant.signals.moments import load_moments

    candidates = DEEP_CLIP_CANDIDATES if args.clip == "deep" else CLIP_CANDIDATES
    widths = sorted(args.widths)
    moments = load_moments(Path(args.moments))
    model = load_model(args.model)
    graph = classify_model(model)
    weights = module_weights(model)

    # Both ends of the tie appear here even though they are one tensor -- the bit map
    # carries a single entry because the allocator dedupes them, but the module tree
    # does not. Exclude both by suffix; the reserved-bits argument pays for the tensor
    # once, and charging it again under its other name would inflate the checkpoint.
    tie = {n for n in weights if n.rsplit(".", 1)[-1] in ("embed_tokens", "lm_head")}
    if not tie:
        raise SystemExit("found no embedding/head entry to exclude")
    print(f"  tie excluded and charged as reserved: {sorted(tie)}", flush=True)

    # Fixed cost: everything the row allocator does not get to choose. Norms and the
    # like at fp16, the tie's reserved plan, and -- the part that is easy to forget --
    # scales and offsets for every row-allocated module, which the partition cannot
    # change and which therefore must not be inside the thing being optimised.
    fixed = float(graph.unquantized_params()) * UNQUANTIZED_FLOOR + args.reserve_embed_bits

    print(f"  clip '{args.clip}', widths {widths}, group {args.group_size}", flush=True)
    print("  measuring per-row sensitivity", flush=True)

    aliases = _tie_aliases(graph)
    names: list[str] = []
    per_row: list[torch.Tensor] = []
    row_cols: list[torch.Tensor] = []
    skipped: dict[str, int] = {}
    for i, info in enumerate(graph.quantizable()):
        name = info.name
        if name in tie or name not in weights:
            continue
        weight = weights[name]
        pair = _moments_for(name, aliases, moments, weight)
        if weight.ndim != 2 or pair is None:
            # No usable moments: it cannot be priced per row, so it is not eligible.
            # Held at the widest width rather than guessed at, and charged as fixed.
            skipped[name] = info.num_params
            continue
        x2, d2 = pair
        rows, cols = weight.shape
        groups = max(1, math.ceil(cols / args.group_size))
        fixed += float(rows * groups * METADATA_BITS)
        table = torch.stack(
            [
                row_sensitivity(
                    weight,
                    x2,
                    d2,
                    bits=b,
                    group_size=args.group_size,
                    candidates=candidates,
                )
                for b in widths
            ],
            dim=1,
        )
        names.append(name)
        per_row.append(table)
        row_cols.append(torch.full((rows,), float(cols), dtype=torch.float64))
        if (i + 1) % 40 == 0:
            print(f"    {len(names)} modules priced", flush=True)

    if skipped:
        held = sum(skipped.values())
        fixed += float(held) * max(widths)
        print(
            f"  {len(skipped)} modules ({held:,} params) have no usable moments; "
            f"held at {max(widths)}b and charged as fixed",
            flush=True,
        )

    sens = torch.cat(per_row).to(torch.float64)
    cols_v = torch.cat(row_cols)
    widths_v = torch.tensor([float(b) for b in widths], dtype=torch.float64)
    cost = cols_v[:, None] * widths_v[None, :]
    budget = args.target_total_bits - fixed
    print(
        f"\n  {sens.shape[0]:,} rows over {len(names)} modules\n"
        f"  fixed {fixed:,.0f} bits, payload budget {budget:,.0f} bits "
        f"({budget / float(cols_v.sum()):.4f} bits/param over allocatable rows)",
        flush=True,
    )
    if budget <= float((cost[:, 0]).sum()):
        raise SystemExit("budget does not even cover every row at the narrowest width")

    choice, spent, total = solve(sens, cost, budget)

    # The counterfactual, deliberately the same solver and the same table: collapse
    # each module's rows to a single decision and let it choose one width for all of
    # them. Anything the row solution wins here is granularity, not search.
    offsets, cursor = [], 0
    for table in per_row:
        offsets.append((cursor, cursor + table.shape[0]))
        cursor += table.shape[0]
    m_sens = torch.stack([sens[a:b].sum(dim=0) for a, b in offsets])
    m_cost = torch.stack([cost[a:b].sum(dim=0) for a, b in offsets])
    m_choice, m_spent, m_total = solve(m_sens, m_cost, budget)

    print(
        f"\n  predicted loss increase at a matched {budget:,.0f}-bit payload budget:\n"
        f"    per module  {m_total:.6e}   ({m_spent:,.0f} bits spent)\n"
        f"    per row     {total:.6e}   ({spent:,.0f} bits spent)\n"
        f"    row / module = {total / m_total:.4f}"
        f"   -> {100 * (1 - total / m_total):+.1f}% predicted loss from granularity alone",
        flush=True,
    )

    print("\n  params per width:", flush=True)
    for j, b in enumerate(widths):
        rowp = float((cols_v * (choice == j).double()).sum())
        modp = sum(
            float(cols_v[a:e].sum())
            for (a, e), k in zip(offsets, m_choice.tolist(), strict=True)
            if k == j
        )
        print(f"    {b}b   per-row {rowp / 1e6:8.1f}M   per-module {modp / 1e6:8.1f}M", flush=True)

    # How many modules actually want to be split? If most end up uniform anyway the
    # granularity is not doing anything, whatever the aggregate says.
    split = sum(1 for a, e in offsets if len(set(choice[a:e].tolist())) > 1)
    print(f"\n  {split}/{len(names)} modules chose more than one width", flush=True)

    tensors = {
        name: torch.tensor([widths[k] for k in choice[a:e].tolist()], dtype=torch.int8)
        for name, (a, e) in zip(names, offsets, strict=True)
    }
    save_file(tensors, f"{args.out}_widths.safetensors")
    meta = {
        "clip": args.clip,
        "widths": widths,
        "group_size": args.group_size,
        "target_total_bits": args.target_total_bits,
        "reserve_embed_bits": args.reserve_embed_bits,
        "fixed_bits": fixed,
        "payload_budget_bits": budget,
        "payload_spent_bits": spent,
        "total_stored_bits": fixed + spent,
        "sensitivity_per_row": total,
        "sensitivity_per_module": m_total,
        "modules": names,
        "skipped": skipped,
        "held_width": max(widths),
    }
    Path(f"{args.out}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stored = fixed + spent
    print(
        f"\n-> {stored:,.0f} stored bits = {int(stored // 8):,} B "
        f"= {stored / 8 / 2**30:.4f} GiB, {stored / graph.total_params():.4f} avg bits\n"
        f"-> wrote {args.out}_widths.safetensors and {args.out}.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
