"""Phase 2: beat GPTQ at 3 bits by fixing DynQuant's allocation, not by adopting GPTQ's.

The target is one number. On Qwen3.5-2B / CaseHOLD, ``gptq_3b_head`` scores 88.03% at
3.1522 stored bits (0.6906 GiB) and ``dq_3p25`` scores 86.70% at 3.2494 (0.7118 GiB), so
DynQuant has to find +1.33 points *while getting 3% smaller*. It has to do it without a
calibration set and without error feedback, or it is not DynQuant any more.

The lever this file exists to test is one the allocator cannot currently pull. It assigns
**one width per module**, and on a tied model the largest module by far is the
embedding/LM-head pair -- 508,559,360 parameters, 27.0% of the checkpoint, all 248,320
vocabulary rows forced to share a single width. The signal to split them is already on
disk: the fine-tune's hook stored ``lm_head.output_grad_sq``, one gradient second moment
per vocabulary row.

That signal alone is not enough, and assuming it was would break the model. The tensor is
*tied*, so it plays two roles, and ``output_grad_sq`` only measures one of them:

* as the **LM head** it turns a hidden state into 248,320 logits, and on this task 10 rows
  carry 99.988% of the gradient mass -- CaseHOLD answers are single digits;
* as the **input embedding** it looks up legal prose, which touches a broad vocabulary
  that the head-side gradient says nothing about.

Crushing the tail on head evidence alone would quantize away the input representation of
every word in the corpus. So a row's importance is the *stronger* of its two claims -- a
max over percentile ranks rather than the rank product the module-level scorer uses. The
product is a soft AND, which is right when two signals are two views of one property; here
they are two independent jobs and a row needs precision if *either* job needs it, which is
a soft OR.

The embedding side is measured by how often the fine-tune actually looked each row up,
counted over the training sequences by way of ``Task.training_row`` so the count is over
the exact token stream the run saw. That is a training-time statistic, free to collect in
the hook with one ``bincount`` per batch and no GPU sync -- not a calibration pass. The
test split is never touched; a width chosen from test tokens would be leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from common import MODEL_ID, RUN_DIR, TASK, load_model, run_eval, set_seed

METADATA_BITS = 32
"""An fp16 scale plus an fp16 offset per group. Asymmetric, matching the allocator."""


def embed_name(bits: dict[str, int]) -> str:
    """The bit map's name for the tied embedding/head tensor.

    Matched by suffix rather than hardcoded: this model nests it at
    ``model.language_model.embed_tokens`` and the next one will not.
    """
    hits = [n for n in bits if n.rsplit(".", 1)[-1] in ("embed_tokens", "lm_head")]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one embedding/head entry, found {hits}")
    return hits[0]


def stored_bits_rowwise(
    rows: int, cols: int, plan: list[tuple[int, int]], group_size: int
) -> float:
    """Bits a row-partitioned tensor occupies, counted the way the allocator counts.

    Metadata is per row per group and so does not depend on how the rows are split --
    only the payload does. Which is worth noticing: at 2-bit payload on a 2048-wide row
    the scales and offsets are 0.25 of 2.25 bits, 11% of the tensor, and they are the
    same 15.9 MB whatever the partition does.
    """
    groups_per_row = max(1, math.ceil(cols / group_size))
    payload = sum(n * cols * width for width, n in plan)
    overhead = rows * groups_per_row * METADATA_BITS
    return float(payload + overhead)


def parse_plan(spec: str, rows: int) -> list[tuple[int, int]]:
    """``"8:2048,4:8192,2:*"`` -> ``[(8, 2048), (4, 8192), (2, 238080)]``.

    Order is by descending importance rank, not by row index: the first block gets the
    most important rows. Exactly one ``*`` block absorbs the remainder.
    """
    plan: list[tuple[int, int]] = []
    star = -1
    for i, part in enumerate(spec.split(",")):
        width, _, count = part.partition(":")
        if count == "*":
            star = i
            plan.append((int(width), 0))
        else:
            plan.append((int(width), int(count)))
    if star < 0:
        raise SystemExit(f"plan {spec!r} needs one '*' block to absorb the remaining rows")
    assigned = sum(n for _, n in plan)
    if assigned > rows:
        raise SystemExit(f"plan {spec!r} assigns {assigned} rows but the tensor has {rows}")
    plan[star] = (plan[star][0], rows - assigned)
    return plan


def quantize_rowwise(
    weight: torch.Tensor,
    order: torch.Tensor,
    plan: list[tuple[int, int]],
    *,
    group_size: int,
    candidates: tuple[float, ...] | None = None,
    channel_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    """Encode each importance block at its own width and write the result back.

    Rows are quantized independently already -- groups run along the input dimension --
    so a row block at a different width is exactly what a row-partitioned kernel would
    store, and this reconstruction is the values it would return. The rows are *not*
    permuted: ``order`` selects which rows belong to which block and each block is
    gathered, encoded and scattered back in place.

    ``channel_weight`` runs along the input dimension, which the row partition does not
    touch, so the same vector applies to every block unchanged -- gathering rows selects
    which outputs are in the block, never which inputs they read.
    """
    from dynquant.quant.device import quantize_tensor
    from dynquant.quant.grid import CLIP_CANDIDATES

    grid = CLIP_CANDIDATES if candidates is None else candidates
    stats: dict[str, float] = {}
    cursor = 0
    with torch.no_grad():
        for width, count in plan:
            if count == 0:
                continue
            idx = order[cursor : cursor + count]
            cursor += count
            block = weight[idx].detach()
            compute_dtype = block.dtype if block.dtype in (torch.float16, torch.bfloat16) else None
            quantized, _ = quantize_tensor(
                block,
                bits=width,
                group_size=group_size,
                symmetric=False,
                compute_dtype=compute_dtype,
                device=block.device,
                candidates=grid,
                channel_weight=channel_weight,
            )
            recon = quantized.dequantize(dtype=torch.float32)
            ref = block.to(dtype=torch.float32)
            rmse = float(torch.sqrt(torch.mean((ref - recon) ** 2)))
            rms = float(torch.sqrt(torch.mean(ref**2)))
            stats[f"rel_err_{width}b"] = rmse / rms if rms > 0 else 0.0
            weight[idx] = recon.to(weight.dtype)
            del recon, ref, block
    return stats


def channel_weights_for(graph, model, moments_path: str, names) -> dict[str, torch.Tensor]:
    """``E[x_c^2]`` per input channel for each name, ones where nothing was measured.

    Ones is not a fallback papering over a gap: it *is* the unweighted objective,
    written out. What it must not be is implicit. ``quantize_model`` refuses a partial
    channel-weight map precisely so the untouched modules get materialised here, in one
    place, where they can be counted and printed -- a module silently encoded on a
    different objective than it was priced with produces a checkpoint of exactly the
    right size that is simply worse, with nothing downstream able to tell.
    """
    from dynquant.score.sensitivity import _moments_for, _tie_aliases
    from dynquant.signals.moments import load_moments

    moments = load_moments(Path(moments_path))
    aliases = _tie_aliases(graph)
    out: dict[str, torch.Tensor] = {}
    unmeasured: list[str] = []
    for name in names:
        weight = model.get_submodule(name).weight
        pair = _moments_for(name, aliases, moments, weight)
        if pair is None:
            out[name] = torch.ones(weight.shape[-1], dtype=torch.float32, device=weight.device)
            unmeasured.append(name)
        else:
            out[name] = pair[0].to(weight.device, torch.float32)
    print(
        f"  weighted clip: E[x^2] for {len(out) - len(unmeasured)}/{len(out)} modules, "
        f"{len(unmeasured)} unmeasured and left on plain MSE"
        + (f": {unmeasured[:3]}" if unmeasured else ""),
        flush=True,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--bitmaps", default=str(RUN_DIR / "stage4_bitmaps.json"))
    parser.add_argument("--target", default="3.25", help="which map supplies the body widths")
    parser.add_argument(
        "--body",
        default="map",
        choices=("map", "fp16"),
        help="'fp16' leaves every non-embedding module unquantized, isolating the tie's damage",
    )
    parser.add_argument(
        "--embed",
        default="map",
        help="'map' | 'fp16' | 'uniform:N' | 'rows:8:2048,4:8192,2:*'",
    )
    parser.add_argument("--signal", default=str(RUN_DIR / "p2_rowsignal.json"))
    parser.add_argument(
        "--row-body",
        default=None,
        help=(
            "prefix written by p2_rowbody.py. Allocates the body per output row "
            "instead of per module; overrides the map's body widths entirely."
        ),
    )
    parser.add_argument(
        "--clip",
        default=None,
        choices=("shipped", "deep"),
        help=(
            "clip grid to encode with. Defaults to whatever the bit map was priced "
            "against, which is the only self-consistent choice; pass it explicitly "
            "only to run the mismatch deliberately."
        ),
    )
    parser.add_argument(
        "--weighted-clip",
        action="store_true",
        default=None,
        help=(
            "encode choosing each group's clip ratio by E[x^2]-weighted error. Defaults "
            "to whatever the bit map was priced with; the same self-consistency argument "
            "as --clip applies, and more sharply, because this changes the definition of "
            "error rather than the set of ratios searched over."
        ),
    )
    parser.add_argument("--moments", default=str(RUN_DIR / "dynquant_moments.safetensors"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # An env-derived RUN_DIR has previously sent one model's arms into another model's
    # directory with another model's tokenizer. Assert the resolved path, do not trust it.
    print(f"  MODEL_ID={MODEL_ID}  TASK={TASK.key}  RUN_DIR={RUN_DIR}", flush=True)
    if "qwen" not in RUN_DIR.name or TASK.key != "casehold":
        raise SystemExit(f"refusing to write into {RUN_DIR}; set DQ_MODEL and DQ_TASK")

    set_seed()
    from dynquant.allocate.budget import module_stored_bits
    from dynquant.graph.classify import classify_model
    from dynquant.graph.roles import UNQUANTIZED_FLOOR
    from dynquant.quant.grid import CLIP_CANDIDATES, DEEP_CLIP_CANDIDATES
    from dynquant.quant.quantizer import quantize_model

    doc = json.loads(Path(args.bitmaps).read_text(encoding="utf-8"))
    entry = doc["maps"][args.target]
    bits: dict[str, int] = dict(entry["bits"])
    group_size: int = entry["group_size"]
    emb = embed_name(bits)

    # Maps written before --clip existed carry no grid and were priced on the
    # shipped one. Inherit rather than default, so the encoder always matches the
    # allocator unless the mismatch was asked for by name.
    priced_with = doc.get("clip", "shipped")
    clip = args.clip or priced_with
    candidates = DEEP_CLIP_CANDIDATES if clip == "deep" else CLIP_CANDIDATES
    priced_weighted = bool(doc.get("weighted_clip", False))
    weighted = priced_weighted if args.weighted_clip is None else args.weighted_clip
    print(
        f"  bit map priced with '{priced_with}' clip grid, encoding with '{clip}' "
        f"({len(candidates)} ratios down to {min(candidates)})"
        + ("  *** MISMATCH, requested explicitly ***" if clip != priced_with else ""),
        flush=True,
    )
    print(
        f"  clip objective priced {'E[x^2]-weighted' if priced_weighted else 'unweighted'}, "
        f"encoding {'E[x^2]-weighted' if weighted else 'unweighted'}"
        + ("  *** MISMATCH, requested explicitly ***" if weighted != priced_weighted else ""),
        flush=True,
    )

    model = load_model(args.model)
    graph = classify_model(model)
    info = graph[emb]
    rows, cols = info.shape
    print(f"  tied tensor {emb} {tuple(info.shape)} = {info.num_params:,} params", flush=True)

    body = {k: v for k, v in bits.items() if k != emb}
    if args.body == "fp16":
        body = {}

    # Built over every name in the map, not per branch: the tie and the body are
    # encoded by different functions below and both need it, and a map assembled
    # twice is a map that can disagree with itself.
    cweights = channel_weights_for(graph, model, args.moments, bits) if weighted else None

    total_bits = float(graph.unquantized_params() * UNQUANTIZED_FLOOR)
    for name, width in body.items():
        total_bits += module_stored_bits(graph[name], width, group_size=group_size)
    if args.body == "fp16":
        for name in bits:
            if name != emb:
                total_bits += float(graph[name].num_params * UNQUANTIZED_FLOOR)

    started = time.time()
    extra: dict[str, object] = {}
    if args.row_body:
        # Row-partitioned body. The widths file replaces the map for every module it
        # names, so the map's contribution to total_bits has to be undone before the
        # row cost is added -- charging both would silently overstate the checkpoint.
        from safetensors.torch import load_file

        if not body:
            raise SystemExit("--row-body with --body fp16 would allocate nothing")
        row_widths = load_file(f"{args.row_body}_widths.safetensors")
        meta = json.loads(Path(f"{args.row_body}.json").read_text(encoding="utf-8"))
        held = int(meta["held_width"])
        # A module the allocator priced but the map does not carry would be dropped
        # silently -- left in fp16 and charged nothing, which reads as a free win.
        orphans = sorted(set(row_widths) - set(body))
        if orphans:
            raise SystemExit(
                f"{len(orphans)} row-allocated modules absent from the map: {orphans[:5]}"
            )
        print(
            f"  row-partitioned body: {len(row_widths)} modules from {args.row_body}, "
            f"{len(meta['skipped'])} held at {held}b, priced with '{meta['clip']}' clip",
            flush=True,
        )
        if meta["clip"] != clip:
            print(f"  *** row body priced with '{meta['clip']}', encoding '{clip}'", flush=True)

        for done, (name, width) in enumerate(body.items(), start=1):
            total_bits -= module_stored_bits(graph[name], width, group_size=group_size)
            w = model.get_submodule(name).weight
            if name in row_widths:
                widths_r = row_widths[name].to(torch.int64)
                if widths_r.numel() != w.shape[0]:
                    raise SystemExit(f"{name}: {widths_r.numel()} widths, {w.shape[0]} rows")
                # Descending width is descending importance, which is the order
                # quantize_rowwise walks; stable so ties keep their row index order.
                order = torch.argsort(-widths_r, stable=True).to(w.device)
                uniq, counts = torch.unique(widths_r, return_counts=True)
                mplan = [
                    (int(b), int(n))
                    for b, n in sorted(
                        zip(uniq.tolist(), counts.tolist(), strict=True), key=lambda p: -p[0]
                    )
                ]
                quantize_rowwise(
                    w,
                    order,
                    mplan,
                    group_size=group_size,
                    candidates=candidates,
                    channel_weight=None if cweights is None else cweights[name],
                )
                total_bits += stored_bits_rowwise(w.shape[0], w.shape[1], mplan, group_size)
            else:
                quantize_model(
                    model,
                    {name: held},
                    group_size=group_size,
                    in_place=True,
                    candidates=candidates,
                    channel_weights=cweights,
                )
                total_bits += module_stored_bits(graph[name], held, group_size=group_size)
            done += 1
            if done % 40 == 0 or done == len(body):
                print(f"  [body] {done}/{len(body)}", flush=True)
        extra["row_body"] = str(args.row_body)
        extra["row_body_split_modules"] = sum(
            1 for t in row_widths.values() if len(set(t.tolist())) > 1
        )
    elif body:

        def show(done: int, total: int) -> None:
            if done % 40 == 0 or done == total:
                print(f"  [body] {done}/{total}", flush=True)

        report = quantize_model(
            model,
            body,
            group_size=group_size,
            in_place=True,
            progress=show,
            candidates=candidates,
            channel_weights=cweights,
        )
        errors = sorted(r.relative_error for r in report.layers.values())
        extra["body_relative_error_median"] = errors[len(errors) // 2]

    # --- the tie ---
    weight = model.get_submodule(emb).weight
    if args.embed == "fp16":
        total_bits += float(info.num_params * UNQUANTIZED_FLOOR)
        plan_desc = "fp16"
    elif args.embed.startswith("uniform:"):
        width = int(args.embed.split(":", 1)[1])
        rep = quantize_model(
            model,
            {emb: width},
            group_size=group_size,
            in_place=True,
            candidates=candidates,
            channel_weights=cweights,
        )
        total_bits += module_stored_bits(info, width, group_size=group_size)
        plan_desc = f"uniform {width}b"
        extra["embed_relative_error"] = rep.layers[emb].relative_error
    elif args.embed.startswith("rows:"):
        plan = parse_plan(args.embed.split(":", 1)[1], rows)
        signal = json.loads(Path(args.signal).read_text(encoding="utf-8"))
        order = torch.tensor(signal["order"], dtype=torch.long, device=weight.device)
        if order.numel() != rows:
            raise SystemExit(f"signal has {order.numel()} rows, tensor has {rows}")
        extra.update(
            quantize_rowwise(
                weight,
                order,
                plan,
                group_size=group_size,
                candidates=candidates,
                channel_weight=None if cweights is None else cweights[emb],
            )
        )
        total_bits += stored_bits_rowwise(rows, cols, plan, group_size)
        plan_desc = " + ".join(f"{n:,} rows @ {w}b" for w, n in plan)
        extra["row_plan"] = [[w, n] for w, n in plan]
        extra["signal_recipe"] = signal.get("recipe")
    else:  # "map"
        width = bits[emb]
        rep = quantize_model(
            model,
            {emb: width},
            group_size=group_size,
            in_place=True,
            candidates=candidates,
            channel_weights=cweights,
        )
        total_bits += module_stored_bits(info, width, group_size=group_size)
        plan_desc = f"map {width}b"
        extra["embed_relative_error"] = rep.layers[emb].relative_error

    quantize_seconds = time.time() - started
    average_bits = total_bits / graph.total_params()
    nbytes = int(total_bits // 8)
    print(
        f"\n  body={args.body}  tie={plan_desc}\n"
        f"  -> {average_bits:.4f} stored bits, {nbytes:,} B = {nbytes / 2**30:.4f} GiB "
        f"(quantized in {quantize_seconds:.1f}s)",
        flush=True,
    )
    torch.cuda.empty_cache()

    run_eval(
        model,
        label=args.label or f"{args.name} ({average_bits:.4f}b)",
        name=args.name,
        limit=args.limit,
        extra={
            "phase": 2,
            "source_model": args.model,
            "body": args.body,
            "embed": args.embed,
            "tie_plan": plan_desc,
            "clip": clip,
            "clip_priced_with": priced_with,
            "average_bits": average_bits,
            "nbytes": nbytes,
            "quantized_gib": nbytes / 2**30,
            "quantize_seconds": round(quantize_seconds, 1),
            **extra,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
