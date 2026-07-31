"""Two axes the allocator does not have, priced against the one it does.

``allocate_bits`` chooses a width from ``{2,3,4,8}`` and nothing else. Group size is
fixed at 128 for the whole model and the clip grid bottoms out at 0.80. Both are
inherited from the supplement, and neither was revisited when the estimator that
prices these choices stopped being a rank product.

They matter because *bits per parameter is not the same as width*. An fp16 scale and
offset per group is 32 bits of metadata, so the real cost is ``bits + 32/group_size``
and the choice set the allocator sees is much coarser than the one it could have::

    2b @ gs=128   2.25      3b @ gs=256   3.125     4b @ gs=256   4.125
    2b @ gs=64    2.50      3b @ gs=128   3.250     4b @ gs=128   4.250
    2b @ gs=32    3.00      3b @ gs=64    3.500     4b @ gs=64    4.500

``2b @ gs=32`` costs *less* than the ``3b @ gs=128`` the allocator treats as the only
option at that tier, and buys four times the scale resolution in exchange for one
fewer bit. Which of the two wins is an empirical question about the weights, and it
is the same question the sensitivity estimator already answers for width -- it
quantizes the actual tensor and measures the actual error. Nothing about extending it
to a second axis is new machinery.

The clip grid is the cheaper of the two to check. ``CLIP_CANDIDATES`` stops at 0.80,
which is a reasonable floor at 4 bits and cannot be one at 2: with four levels the
MSE-optimal shrink for anything resembling a Gaussian group is far tighter than 0.80,
so on every 2-bit group whose optimum is below the floor the search returns the floor
and reports it as a win. The fix is not a different algorithm, it is a grid that
reaches the answer.

Reported per candidate: relative reconstruction error, and the Gauss-Newton
sensitivity -- the estimate weighted by the channel moments, which is what the
allocator would actually compare. They disagree often enough to be worth printing
both: relative error treats every output row alike, and the whole point of the
moments is that the network does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from common import RUN_DIR, TASK, load_model

DEEP_CANDIDATES: tuple[float, ...] = (
    1.0,
    0.98,
    0.96,
    0.94,
    0.92,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
    0.45,
    0.40,
)
"""The shipped grid, continued down to where a 2-bit group's optimum actually lives.

Same spacing rule the original used -- fine near 1.0, coarse in the tail -- carried on
rather than replaced, so the first eight entries are bit-identical and any difference
is attributable to the extension and not to a re-tuning of what was already there.
"""


def cost_per_param(bits: int, group_size: int) -> float:
    """Stored bits per weight: payload plus an fp16 scale and offset per group."""
    return bits + 32.0 / group_size


def measure(
    weight: torch.Tensor,
    x2: torch.Tensor | None,
    d2: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    candidates: tuple[float, ...],
) -> tuple[float, float, float]:
    """``(relative error, sensitivity, mean clip ratio)`` for one candidate encoding."""
    from dynquant.quant.grid import quantize_with_search

    with torch.no_grad():
        quantized, search = quantize_with_search(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=False,
            candidates=candidates,
        )
        recon = quantized.dequantize(dtype=torch.float32)
        ref = weight.to(torch.float32)
        squared = (ref - recon) ** 2
        rms = float(torch.sqrt(torch.mean(ref**2)))
        rel = float(torch.sqrt(torch.mean(squared))) / rms if rms > 0 else 0.0
        if x2 is None or d2 is None:
            sens = float(squared.sum())
        else:
            per_output = squared @ x2.to(ref.device, torch.float32)
            sens = float(per_output @ d2.to(ref.device, torch.float32))
        return rel, sens, float(search.ratios.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--moments", default=str(RUN_DIR / "dynquant_moments.safetensors"))
    parser.add_argument("--bitmaps", default=str(RUN_DIR / "p2_sens_bitmaps.json"))
    parser.add_argument("--target", default="3.15")
    parser.add_argument("--per-role", type=int, default=2, help="modules sampled per role")
    parser.add_argument("--out", default=str(RUN_DIR / "p2_sweep.json"))
    args = parser.parse_args()

    if "qwen" not in RUN_DIR.name or TASK.key != "casehold":
        raise SystemExit(f"refusing to write into {RUN_DIR}; set DQ_MODEL and DQ_TASK")

    from dynquant.graph.classify import classify_model
    from dynquant.quant.grid import CLIP_CANDIDATES as CLIP_SHIPPED
    from dynquant.score.sensitivity import module_weights
    from dynquant.signals.moments import load_moments

    moments = load_moments(Path(args.moments))
    model = load_model(args.model)
    graph = classify_model(model)
    weights = module_weights(model)
    assigned: dict[str, int] = json.loads(Path(args.bitmaps).read_text(encoding="utf-8"))["maps"][
        args.target
    ]["bits"]

    # Sample by role so the answer is not driven by whichever role happens to be
    # largest, and keep the width the allocator actually chose -- the question is
    # what a *cheaper* encoding of this module would cost, not what an arbitrary one
    # would.
    by_role: dict[str, list[str]] = {}
    for info in graph.quantizable():
        if info.name in weights and weights[info.name].ndim == 2:
            by_role.setdefault(info.role.value, []).append(info.name)
    picked: list[str] = []
    for _role, names in sorted(by_role.items()):
        names.sort(key=lambda n: -graph[n].num_params)
        picked.extend(names[: args.per_role])

    grid = [
        (2, 32),
        (2, 64),
        (2, 128),
        (2, 256),
        (3, 64),
        (3, 128),
        (3, 256),
        (4, 128),
        (4, 256),
    ]

    print(
        f"  {len(picked)} modules over {len(by_role)} roles, {len(grid)} encodings each\n",
        flush=True,
    )
    records: list[dict[str, object]] = []

    for name in picked:
        weight = weights[name]
        x2 = moments.input_sq.get(name)
        d2 = moments.output_grad_sq.get(name)
        if (
            x2 is None
            or d2 is None
            or x2.shape[0] != weight.shape[1]
            or d2.shape[0] != weight.shape[0]
        ):
            x2 = d2 = None
        width = assigned.get(name, 0)
        print(
            f"{name}  {tuple(weight.shape)}  role={graph[name].role.value}  "
            f"allocator chose {width}b",
            flush=True,
        )
        print(
            "    bits gs    b/param   rel_err(shipped)  rel_err(deep)   sens(shipped)  sens(deep)  clip",
            flush=True,
        )
        for bits, gs in grid:
            cpp = cost_per_param(bits, gs)
            rel_s, sens_s, _ = measure(
                weight, x2, d2, bits=bits, group_size=gs, candidates=CLIP_SHIPPED
            )
            rel_d, sens_d, clip_d = measure(
                weight, x2, d2, bits=bits, group_size=gs, candidates=DEEP_CANDIDATES
            )
            mark = "  <- allocated" if bits == width and gs == 128 else ""
            print(
                f"    {bits:4d} {gs:3d}  {cpp:7.3f}   {rel_s:14.4f}   {rel_d:12.4f}   "
                f"{sens_s:12.4e}  {sens_d:10.4e}  {clip_d:.3f}{mark}",
                flush=True,
            )
            records.append(
                {
                    "module": name,
                    "role": graph[name].role.value,
                    "allocated_bits": width,
                    "bits": bits,
                    "group_size": gs,
                    "bits_per_param": cpp,
                    "rel_err_shipped": rel_s,
                    "rel_err_deep": rel_d,
                    "sensitivity_shipped": sens_s,
                    "sensitivity_deep": sens_d,
                    "mean_clip_deep": clip_d,
                }
            )
        print(flush=True)

    Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"-> wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
