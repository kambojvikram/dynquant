"""What fraction of the memory system the decode GEMV actually uses.

Decode is memory-bound, so the only question that matters about the GEMV is how
close it gets to the card's achievable read bandwidth. Everything else -- FLOPs,
occupancy, instruction counts -- is a means to that. This measures it directly, on
the shapes a real model contains, and prints the fraction rather than the raw time,
because a raw time is uninterpretable without knowing what the hardware could have
done.

Three things this is careful about, each of which produced a wrong number first:

*Wall clock measures Python, not CUDA.* An 8 us kernel behind a ~10 us dispatch
reads as 13 us no matter how fast it is, which makes every small shape look
identical and every optimization look worthless. Timing is therefore
``self_device_time_total`` over kernel-level profiler events. It has to be
kernel-level specifically: ``key_averages()`` also attributes device time to
op-level rows like ``aten::mm``, and summing both double-counts.

*The peak from the datasheet is the wrong denominator.* An A100 80GB PCIe is quoted
at 1935 GB/s and delivers about 1630. Reporting against the quoted figure understates
the gap; reporting against nothing at all is what the first pass of this measurement
did. So the achievable figure is measured here, on this card, at startup.

*bf16 can beat its own bandwidth.* A 25 MB weight fits in the A100's 40 MB L2, so a
benchmark loop reads it from DRAM once and from L2 forever, and the baseline posts
>100 % of achievable. That is real, it is not a bug, and it means the small-shape
speedup columns are pessimistic for the quantized path. The flag is printed next to
any row where it happens rather than left for the reader to notice.

Usage::

    python benchmarks/gemv_bandwidth.py                  # both paths, markdown table
    python benchmarks/gemv_bandwidth.py --json out.json  # also dump the raw numbers
    python benchmarks/gemv_bandwidth.py --only vectorized
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from dynquant.quant.grid import quantize_with_search

# (label, out_features, in_features) -- Qwen3.5-2B's distinct matmul shapes. The
# vocabulary projection is last because it is the only one big enough for
# "bandwidth" to be the right word: at 2048x2048 the whole weight is a few
# milliseconds of DRAM and the kernel is latency-bound whatever it does.
SHAPES = [
    ("linear_attn.out_proj", 2048, 2048),
    ("linear_attn.in_proj_qkv", 6144, 2048),
    ("mlp.gate_proj", 6144, 2048),
    ("mlp.down_proj", 2048, 6144),
    ("embed/lm_head", 248320, 2048),
]

BITS = [2, 3, 4, 8]
GROUP_SIZE = 128
DTYPE = torch.bfloat16


@dataclass
class Row:
    shape: str
    out_features: int
    in_features: int
    bits: int | None  # None == the bf16 baseline
    micros: float
    bytes_read: int
    gbytes_per_s: float
    max_rel_err: float


def _kernel_micros(fn, iters: int) -> float:
    """GPU time per call, from kernel-level events only.

    Warms up first: the first call of a templated kernel pays module load and
    JIT-of-PTX if the fat binary has no matching SASS, which is a one-off cost that
    would otherwise be divided into the average.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    total = sum(
        e.self_device_time_total for e in prof.key_averages() if e.device_type == DeviceType.CUDA
    )
    return total / iters


def achievable_bandwidth() -> tuple[float, float]:
    """Read and copy bandwidth this card actually delivers, in GB/s.

    A 1 GiB buffer: large enough that the 40 MB L2 cannot serve it, small enough to
    leave room for the weights under test.
    """
    n = 1 << 28  # 1 GiB of fp32
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    # Bound as defaults, not captured: `del src, dst` below is what lets
    # `empty_cache` actually return the gigabyte, and a closure still holding the
    # names would keep it alive as well as reading as unbound to a linter.
    read_us = _kernel_micros(lambda t=src: t.sum(), iters=20)
    copy_us = _kernel_micros(lambda d=dst, s=src: d.copy_(s), iters=20)

    read_gbs = (n * 4) / (read_us * 1e-6) / 1e9
    copy_gbs = (n * 4 * 2) / (copy_us * 1e-6) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return read_gbs, copy_gbs


def measure_shape(label: str, out_features: int, in_features: int) -> list[Row]:
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    x = torch.randn(1, in_features, dtype=DTYPE, device="cuda")

    rows: list[Row] = []

    dense_gpu = dense.to("cuda", DTYPE)
    # Big shapes are slow and stable; small ones need the repetitions to rise above
    # the profiler's own per-event cost.
    iters = 20 if out_features > 100_000 else 200
    micros = _kernel_micros(lambda w=dense_gpu: F.linear(x, w), iters)
    nbytes = out_features * in_features * 2
    rows.append(
        Row(
            label,
            out_features,
            in_features,
            None,
            micros,
            nbytes,
            nbytes / (micros * 1e-6) / 1e9,
            0.0,
        )
    )
    del dense_gpu
    torch.cuda.empty_cache()

    for bits in BITS:
        quantized, _ = quantize_with_search(
            dense, bits=bits, group_size=GROUP_SIZE, symmetric=False, compute_dtype=DTYPE
        )
        quantized = quantized.to("cuda")
        geom = quantized.geometry
        args = (
            x,
            quantized.packed,
            quantized.scales,
            quantized.offsets,
            bits,
            geom.effective_group,
            in_features,
        )

        # Against the *dequantized* weight, not the original. Comparing to the
        # original measures quantization error -- which at 2-bit is enormous and has
        # nothing to do with whether the kernel is right.
        reference = F.linear(x, quantized.dequantize(dtype=DTYPE))
        got = torch.ops.dynquant.gemv(*args)
        scale = reference.abs().max().clamp_min(1e-6)
        rel = ((got - reference).abs().max() / scale).item()

        micros = _kernel_micros(lambda a=args: torch.ops.dynquant.gemv(*a), iters)
        nbytes = (
            quantized.packed.numel() * 4
            + quantized.scales.numel() * 2
            + (quantized.offsets.numel() * 2 if quantized.offsets is not None else 0)
        )
        rows.append(
            Row(
                label,
                out_features,
                in_features,
                bits,
                micros,
                nbytes,
                nbytes / (micros * 1e-6) / 1e9,
                rel,
            )
        )
        del quantized
        torch.cuda.empty_cache()

    return rows


def run(path_label: str) -> dict:
    read_gbs, copy_gbs = achievable_bandwidth()
    rows: list[Row] = []
    for label, out_features, in_features in SHAPES:
        rows.extend(measure_shape(label, out_features, in_features))
    return {
        "path": path_label,
        "achievable_read_gbs": read_gbs,
        "achievable_copy_gbs": copy_gbs,
        "device": torch.cuda.get_device_name(0),
        "rows": [asdict(r) for r in rows],
    }


def render(result: dict, baseline: dict | None) -> str:
    read_gbs = result["achievable_read_gbs"]
    by_shape: dict[str, dict] = {}
    for row in result["rows"]:
        by_shape.setdefault(row["shape"], {})[row["bits"]] = row
    prev: dict[str, dict] = {}
    if baseline is not None:
        for row in baseline["rows"]:
            prev.setdefault(row["shape"], {})[row["bits"]] = row

    out = [
        f"{result['device']}  --  achievable {read_gbs:.0f} GB/s read, "
        f"{result['achievable_copy_gbs']:.0f} GB/s copy",
        "",
        "| shape | bf16 | " + " | ".join(f"{b} bit" for b in BITS) + " |",
        "|---|---|" + "---|" * len(BITS),
    ]
    for label, out_features, in_features in SHAPES:
        cells = by_shape[label]
        base = cells[None]
        base_pct = 100 * base["gbytes_per_s"] / read_gbs
        flag = " *" if base_pct > 100 else ""
        line = [
            f"`{label}` {out_features}x{in_features}",
            f"{base['micros']:.1f} us / {base_pct:.0f} %{flag}",
        ]
        for bits in BITS:
            cell = cells[bits]
            speed = base["micros"] / cell["micros"]
            pct = 100 * cell["gbytes_per_s"] / read_gbs
            text = f"{speed:.2f}x / {pct:.0f} %"
            if prev:
                gain = prev[label][bits]["micros"] / cell["micros"]
                text += f" ({gain:.2f}x)"
            line.append(text)
        out.append("| " + " | ".join(line) + " |")

    worst = max(r["max_rel_err"] for r in result["rows"])
    out += [
        "",
        "Speedup vs bf16 `F.linear`, and percent of achievable read bandwidth"
        + (", and in brackets the speedup over the general kernel." if prev else "."),
        f"Worst relative error against the dequantized oracle: {worst:.2e}.",
    ]
    if any(100 * by_shape[s][None]["gbytes_per_s"] / read_gbs > 100 for s, _, _ in SHAPES):
        out.append(
            "`*` bf16 exceeding 100 % is L2 residency across the benchmark loop, not a "
            "measurement error; it flatters the baseline on those rows."
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["vectorized", "general", "both"],
        default="both",
        help="'general' forces DYNQUANT_GEMV_SCALAR=1, the pre-optimization kernel.",
    )
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="comma-separated substrings; only matching shapes run. Most of this "
        "benchmark's wall time is the CPU-side clipping search on the 248320-row "
        "vocabulary weight, so `--shapes lm_head --only vectorized` is the loop to "
        "iterate a kernel change in.",
    )
    parser.add_argument("--_arm", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.shapes:
        wanted = [s.strip() for s in args.shapes.split(",") if s.strip()]
        global SHAPES
        SHAPES = [s for s in SHAPES if any(w in s[0] for w in wanted)]
        if not SHAPES:
            raise SystemExit(f"--shapes {args.shapes!r} matched nothing")

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a GPU")
    import dynquant_kernels

    if not dynquant_kernels.is_available():
        raise SystemExit("this benchmark needs the compiled kernels, not the torch fallback")

    # One arm per process: the path is chosen by an environment variable read once
    # into a static, so a single process cannot measure both.
    if args._arm is not None:
        print(json.dumps(run(args._arm)))
        return

    general = None
    if args.only in ("general", "both"):
        forward = ["--shapes", args.shapes] if args.shapes else []
        proc = subprocess.run(
            [sys.executable, __file__, "--_arm", "general", *forward],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "DYNQUANT_GEMV_SCALAR": "1"},
        )
        general = json.loads(proc.stdout)

    if args.only == "general":
        print(render(general, None))
        results = [general]
    else:
        vectorized = run("vectorized")
        print(render(vectorized, general))
        results = [r for r in (general, vectorized) if r is not None]

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
