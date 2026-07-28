"""Per-layer reconstruction error from a stage-5 record, by width and by role.

Answers the question the accuracy number cannot: whether a drop is "every tensor lost
a little" or "one tensor was destroyed". Those look identical downstream and want
opposite fixes -- the first is the cost of the width, the second is a bug or a bad
allocation -- and the per-layer errors are the only place the difference is visible.

They are also unrecoverable once the model has been quantized in place, which is why
stage 5 writes them rather than printing them.

Read the numbers against theory, not against intuition. Group-128 asymmetric min/max
on roughly Gaussian weights gives a relative error near ``step/sqrt(12)`` with
``step ~ 5.8*sigma/(2**b - 1)``: about 0.10 at 4 bits, 0.21 at 3 bits, 0.50 at 2 bits,
0.006 at 8 bits. A 4-bit layer at 0.10 is working correctly. It is *also* a large
error, and no
amount of allocation makes plain round-to-nearest competitive with GPTQ's
error-compensated sequential encoding or AWQ's activation-aware scaling.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from common import read_record


def role_suffix(name: str) -> str:
    """A short grouping key. ``linear_attn.in_proj_a`` keeps two segments because a
    bare ``a`` says nothing; everything else is distinctive at one."""
    parts = name.split(".")
    if len(parts[-1]) <= 2:
        return ".".join(parts[-2:])
    return parts[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", help="stage-5 quant record name, e.g. stage5_4p25_quant")
    parser.add_argument("--worst", type=int, default=3, help="worst layers to name per width")
    args = parser.parse_args()

    data = read_record(args.record)
    if data is None:
        print(f"no record named {args.record!r}", file=sys.stderr)
        return 2

    layers = data["layers"]
    kind = "uniform" if data.get("uniform") else "allocated"
    print(
        f"{args.record}: {len(layers)} layers, {kind}, "
        f"stored average {data['average_bits']:.4f} bits, "
        f"{data['nbytes'] / 2**30:.3f} GiB, quantized in {data['quantize_seconds']:.0f}s"
    )

    by_width: dict[int, list[tuple[float, str]]] = defaultdict(list)
    by_role: dict[tuple[str, int], list[float]] = defaultdict(list)
    for name, entry in layers.items():
        by_width[entry["bits"]].append((entry["relative_error"], name))
        by_role[(role_suffix(name), entry["bits"])].append(entry["relative_error"])

    print("\nby width:")
    for bits in sorted(by_width):
        ranked = sorted(by_width[bits], reverse=True)
        errors = [error for error, _ in ranked]
        params = sum(layers[name]["num_params"] for _, name in ranked)
        print(
            f"  {bits}b  n={len(errors):3d}  {params / 1e6:7.1f}M params  "
            f"rel_err min {min(errors):.4f}  median {errors[len(errors) // 2]:.4f}  "
            f"max {max(errors):.4f}"
        )
        for error, name in ranked[: args.worst]:
            print(f"       worst  {error:.4f}  {layers[name]['num_params'] / 1e6:6.1f}M  {name}")

    print("\nby role and width:")
    for key in sorted(by_role):
        errors = by_role[key]
        print(
            f"  {key[0]:24s} @{key[1]}b  n={len(errors):3d}  "
            f"mean {sum(errors) / len(errors):.4f}  max {max(errors):.4f}"
        )

    gains = [entry["clip_improvement"] for entry in layers.values()]
    helped = sum(1 for gain in gains if gain > 0)
    print(
        f"\nMSE clip search improved SSE on {helped}/{len(gains)} layers, "
        f"mean {sum(gains) / len(gains) * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
