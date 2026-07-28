"""Apply a stage-4 bit map to the fine-tuned model and score the result.

Measurement points 3 and 4. Quantization and evaluation happen in one process per
target, for two reasons. Reloading is avoidable work, and more importantly the
alternative -- ``save_pretrained`` then ``evaluate.py`` -- would write two 3.8 GB
fp16 directories to disk that are *not* quantized checkpoints and would sit there
looking like they were. The weights in memory are the exact values a packed
checkpoint reconstructs, so the accuracy measured here is the accuracy of the real
thing; the memory footprint is not, and nothing in this script claims otherwise.
Use ``--save`` if a checkpoint on disk is wanted anyway.

Evaluation goes through :func:`common.run_eval`, the same function stages 1 and 3
used, so the prompt, the exemplars, the decode settings and the scorer are shared
by construction rather than by having been copied carefully.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from common import RUN_DIR, load_model, load_tokenizer, record, run_eval, set_seed


def flatten_to_uniform(
    model: object, bits: dict[str, int], width: int, *, group_size: int
) -> tuple[dict[str, int], float, int]:
    """Every module at ``width``, except the ones a structural floor protects.

    The exemption is what makes this a control rather than a second experiment. A
    structural floor is a correctness constraint -- on this architecture the 36
    ``lin_attn.a``/``lin_attn.b`` projections carry the gate and decay coefficients of
    a recurrence, whose error compounds along the sequence instead of averaging out.
    Flattening those to 3 bits too would very likely break the model, and the
    comparison would then read as "score-driven allocation beats flat allocation" when
    what it actually showed was "a broken recurrence beats a working one". They are
    1.2M of 1.88B parameters, so holding them at 8 bits moves the stored average by
    about 0.003 bits and the two arms stay budget-matched to three decimals.

    Everything else -- including the whole embedding/LM-head tie, 27% of the model --
    is flattened. Those floors are quality preferences, and deciding where to spend
    precision among them is precisely the thing under test.

    Returns:
        The flattened map, its stored average bits, and its size in bytes -- computed
        through the same :func:`module_stored_bits` the allocator budgets with, so the
        two arms are compared on one accounting rather than on two that agree by
        assumption.
    """
    from dynquant.allocate.budget import module_stored_bits
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.classify import classify_model
    from dynquant.graph.roles import UNQUANTIZED_FLOOR

    policy = AllocationPolicy()
    graph = classify_model(model)

    flattened = dict.fromkeys(bits, width)
    total_bits = float(graph.unquantized_params() * UNQUANTIZED_FLOOR)
    for name in bits:
        info = graph[name]
        if policy.is_structural(info.role, info.tied_roles):
            flattened[name] = max(width, policy.floor_for(info.role, info.tied_roles))
        total_bits += module_stored_bits(info, flattened[name], group_size=group_size)

    return flattened, total_bits / graph.total_params(), int(total_bits / 8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--bitmaps", default=str(RUN_DIR / "stage4_bitmaps.json"))
    parser.add_argument(
        "--target",
        required=True,
        help="key into the bit-map file, e.g. 4.00 or 3.00",
    )
    parser.add_argument("--name", default=None, help="record name; defaults to stageN_<target>b")
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save", default=None, help="optional directory to write weights to")
    parser.add_argument(
        "--uniform",
        type=int,
        default=None,
        help=(
            "ignore the allocated widths and give every module this many bits, except "
            "those under a structural floor. The control the experiment is meaningless "
            "without: it separates damage caused by low precision from damage caused by "
            "where the allocator chose to spend it."
        ),
    )
    args = parser.parse_args()

    set_seed()
    from dynquant.quant.quantizer import quantize_model

    maps = json.loads(Path(args.bitmaps).read_text(encoding="utf-8"))["maps"]
    if args.target not in maps:
        print(
            f"no bit map for target {args.target!r}; available: {', '.join(sorted(maps))}",
            file=sys.stderr,
        )
        return 2
    entry = maps[args.target]
    bits: dict[str, int] = entry["bits"]
    group_size: int = entry["group_size"]
    average_bits: float = entry["average_bits"]
    nbytes: int = entry["nbytes"]

    suffix = (
        f"uniform{args.uniform}b" if args.uniform is not None else args.target.replace(".", "p")
    )
    name = args.name or f"stage5_{suffix}"

    model = load_model(args.model)
    print(f"loaded {args.model}", flush=True)
    if args.uniform is not None:
        bits, average_bits, nbytes = flatten_to_uniform(
            model, bits, args.uniform, group_size=group_size
        )
        protected = sum(1 for b in bits.values() if b != args.uniform)
        print(
            f"CONTROL: {len(bits) - protected} of {len(bits)} modules flattened to "
            f"{args.uniform} bits, {protected} held at their structural floor; "
            f"stored average {average_bits:.4f} bits vs {entry['average_bits']:.4f} allocated",
            flush=True,
        )
    else:
        print(
            f"applying the {args.target}-bit map: {len(bits)} modules, "
            f"stored average {average_bits:.4f} bits, "
            f"{nbytes / 2**30:.3f} GiB on disk, "
            f"{len(entry['violations'])} floors breached",
            flush=True,
        )
    label = args.label or (
        f"uniform {args.uniform}b ({average_bits:.2f}b stored)"
        if args.uniform is not None
        else f"allocated {average_bits:.2f}b stored"
    )
    widths: dict[int, int] = {}
    for width in bits.values():
        widths[width] = widths.get(width, 0) + 1
    print(f"  widths: {dict(sorted(widths.items()))}", flush=True)

    def show(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"  [quantize] {done}/{total} modules", flush=True)

    started = time.time()
    report = quantize_model(
        model,
        bits,
        group_size=group_size,
        in_place=True,
        progress=show,
    )
    quantize_seconds = time.time() - started
    print(f"\nquantized in {quantize_seconds:.1f}s", flush=True)
    print(report.summary(), flush=True)

    # The per-layer errors are the only diagnostic that separates "3 bits is simply
    # lossy" from "one tensor took catastrophic damage and dragged the model down",
    # and they are unrecoverable once the fp32 originals are gone.
    record(
        f"{name}_quant",
        {
            "target": args.target,
            "uniform": args.uniform,
            "average_bits": average_bits,
            "nbytes": nbytes,
            "group_size": group_size,
            "quantize_seconds": round(quantize_seconds, 1),
            "layers": {
                r.name: {
                    "bits": r.bits,
                    "num_params": r.num_params,
                    "rmse": r.rmse,
                    "relative_error": r.relative_error,
                    "clipped_fraction": r.clipped_fraction,
                    "clip_improvement": r.clip_improvement,
                }
                for r in report.layers.values()
            },
        },
    )

    torch.cuda.empty_cache()

    if args.save:
        model.save_pretrained(args.save)
        load_tokenizer().save_pretrained(args.save)
        print(f"-> wrote weights to {args.save} (fp16 storage of quantized values)", flush=True)

    run_eval(
        model,
        label=label,
        name=name,
        limit=args.limit,
        extra={
            "source_model": args.model,
            "target": args.target,
            "uniform": args.uniform,
            "average_bits": average_bits,
            "quantized_gib": nbytes / 2**30,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
