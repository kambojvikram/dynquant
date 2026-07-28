"""``dynquant bench`` -- what fraction of the memory system decode actually uses.

Decode is memory-bound, so the only question that matters about the packed GEMV is
how close it gets to the card's achievable read bandwidth. Everything else --
FLOPs, occupancy, instruction counts -- is a means to that. This measures it
directly, on the shapes a real model contains, and prints the fraction rather than
the raw time, because a raw time is uninterpretable without knowing what the
hardware could have done.

Three things it is careful about, each of which produced a wrong number first:

**Wall clock measures Python, not CUDA.** An 8 us kernel behind a ~10 us dispatch
reads as 13 us no matter how fast it is, which makes every small shape look
identical and every optimization look worthless. Timing is therefore
``self_device_time_total`` over kernel-level profiler events. Kernel-level
specifically: ``key_averages()`` also attributes device time to op-level rows like
``aten::mm``, and summing both double-counts.

**The datasheet peak is the wrong denominator.** An A100 80GB PCIe is quoted at
1935 GB/s and delivers about 1630. Reporting against the quoted figure understates
the gap; reporting against nothing at all is what the first pass of this
measurement did. The achievable figure is measured here, on this card, at startup.

**A small weight can beat its own bandwidth.** A 25 MB weight fits in an A100's
40 MB L2, so a benchmark loop reads it from DRAM once and from L2 forever, and the
bf16 baseline posts over 100% of achievable. That is real, it is not a bug, and it
means the speedup column is pessimistic on those rows. Rows where it happens are
flagged rather than left for the reader to notice.

The shapes come from a real model (``--model``, read on the meta device so nothing
is downloaded twice or loaded into VRAM) or from ``--shape OUTxIN``. Nothing about
the measurement is model-specific; only which rectangles get measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dynquant.constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    import torch

__all__ = ["Row", "achievable_bandwidth", "kernel_micros", "measure_shape", "render", "run"]


@dataclass(frozen=True)
class Row:
    shape: str
    out_features: int
    in_features: int
    bits: int | None
    """``None`` is the dense baseline in the compute dtype."""
    rows: int
    micros: float
    bytes_read: int
    gbytes_per_s: float
    max_rel_err: float


def kernel_micros(fn: Callable[[], Any], iters: int) -> float:
    """GPU time per call, from kernel-level profiler events only.

    Warms up first: the first call of a templated kernel pays module load, and JIT
    of PTX if the fat binary has no SASS for this architecture -- a one-off cost
    that would otherwise be divided into the average.
    """
    import torch
    from torch.profiler import DeviceType, ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    total = sum(
        float(event.self_device_time_total)
        for event in prof.key_averages()
        if event.device_type == DeviceType.CUDA
    )
    return total / iters


def achievable_bandwidth() -> tuple[float, float]:
    """Read and copy bandwidth this card actually delivers, in GB/s.

    A 1 GiB buffer: large enough that no L2 can serve it, small enough to leave
    room for the weights under test.
    """
    import torch

    n = 1 << 28  # 1 GiB of fp32
    read_us, copy_us = _bandwidth_micros(n)
    # The buffers went out of scope with the frame above; hand the blocks back so a
    # 2 GiB benchmark scratchpad is not still resident when the weights arrive.
    torch.cuda.empty_cache()
    return (n * 4) / (read_us * 1e-6) / 1e9, (n * 4 * 2) / (copy_us * 1e-6) / 1e9


def _bandwidth_micros(n: int) -> tuple[float, float]:
    """Timed in its own frame so the two buffers are unreferenced on return."""
    import torch

    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    return (
        kernel_micros(lambda: src.sum(), iters=20),
        kernel_micros(lambda: dst.copy_(src), iters=20),
    )


def measure_shape(
    label: str,
    out_features: int,
    in_features: int,
    *,
    bits: Sequence[int] = BIT_OPTIONS,
    group_size: int = DEFAULT_GROUP_SIZE,
    rows: int = 1,
    dtype: torch.dtype | None = None,
    iters: int | None = None,
) -> list[Row]:
    """One rectangle, dense baseline plus every requested width."""
    import torch
    import torch.nn.functional as F

    from dynquant.quant.grid import quantize_with_search
    from dynquant.runtime.ops import quantized_matmul

    if dtype is None:
        dtype = torch.bfloat16

    torch.manual_seed(0)
    # On the GPU, because the clip search runs wherever the weight is. On CPU the
    # search over 8 candidates on a 248320x2048 embedding takes tens of minutes and
    # the command looks hung on its first row; on the device it is seconds. It is
    # also where every real quantization in this package happens, so the numbers
    # being measured come out of the same code path a user's model goes through.
    dense = torch.randn(out_features, in_features, dtype=torch.float32, device="cuda")
    x = torch.randn(rows, in_features, dtype=dtype, device="cuda")
    # Big shapes are slow and stable; small ones need repetitions to rise above the
    # profiler's own per-event cost.
    repeats = iters or (20 if out_features > 100_000 else 200)

    measured: list[Row] = []

    micros, nbytes = _dense_micros(x, dense, dtype, repeats)
    # Freed with the frame above: the dense copy of a large projection is the single
    # biggest allocation here, and holding it through the 2/3/4/8-bit arms would put
    # the peak at dense-plus-packed for no reason.
    torch.cuda.empty_cache()
    measured.append(
        Row(
            label,
            out_features,
            in_features,
            None,
            rows,
            micros,
            nbytes,
            nbytes / (micros * 1e-6) / 1e9,
            0.0,
        )
    )

    for width in bits:
        quantized, _ = quantize_with_search(
            dense, bits=width, group_size=group_size, symmetric=False, compute_dtype=dtype
        )
        quantized = quantized.to("cuda")

        # Against the *dequantized* weight, not the original. Comparing to the
        # original measures quantization error -- which at 2-bit is enormous and
        # has nothing to do with whether the kernel is right.
        reference = F.linear(x, quantized.dequantize(dtype=dtype))
        got = quantized_matmul(x, quantized)
        scale = reference.abs().max().clamp_min(1e-6)
        rel = float((got - reference).abs().max() / scale)

        # `partial`, not a lambda: the weight has to be bound now rather than read
        # from the loop variable when the timing loop runs.
        micros = kernel_micros(partial(quantized_matmul, x, quantized), repeats)
        # `nbytes` counts scales and offsets, so a 2-bit row is charged the 2.25
        # bits it really reads. Charging it 2 would credit the kernel with
        # bandwidth it never had to move.
        nbytes = quantized.nbytes
        measured.append(
            Row(
                label,
                out_features,
                in_features,
                width,
                rows,
                micros,
                nbytes,
                nbytes / (micros * 1e-6) / 1e9,
                rel,
            )
        )
        del quantized
        torch.cuda.empty_cache()

    return measured


def _dense_micros(
    x: torch.Tensor, dense: torch.Tensor, dtype: torch.dtype, repeats: int
) -> tuple[float, int]:
    """The baseline arm, in its own frame so the dense copy is released on return."""
    import torch.nn.functional as F

    dense_gpu = dense.to("cuda", dtype)
    micros = kernel_micros(lambda: F.linear(x, dense_gpu), repeats)
    return micros, dense_gpu.numel() * dense_gpu.element_size()


def shapes_from_model(
    path: str, *, limit: int = 8, trust_remote_code: bool = False
) -> list[tuple[str, int, int]]:
    """Distinct ``(label, out_features, in_features)`` rectangles a model contains.

    Built on the meta device: only shapes are needed, so nothing is downloaded
    beyond the config and nothing reaches VRAM. Deduplicated, because a 28-layer
    model has 28 identical ``down_proj`` shapes and measuring each would spend the
    run confirming the same number.
    """
    import torch
    from torch import nn

    from ._shared import _require_transformers

    transformers = _require_transformers()
    config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )

    seen: dict[tuple[int, int], str] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = (module.out_features, module.in_features)
        elif isinstance(module, nn.Embedding):
            key = (module.num_embeddings, module.embedding_dim)
        else:
            continue
        seen.setdefault(key, _short(name))

    ordered = sorted(seen.items(), key=lambda kv: -(kv[0][0] * kv[0][1]))
    return [(label, out, inp) for (out, inp), label in ordered[:limit]]


def _short(name: str) -> str:
    """``model.layers.11.mlp.down_proj`` -> ``mlp.down_proj``.

    The layer index is noise here: every layer has the same shape, and one of them
    is being measured on behalf of all of them.
    """
    parts = [p for p in name.split(".") if not p.isdigit()]
    return ".".join(parts[-2:]) if len(parts) >= 2 else name


def parse_shape(text: str) -> tuple[str, int, int]:
    """``4096x1024`` or ``label=4096x1024``."""
    from dynquant.errors import DynQuantError

    label, _, geometry = text.rpartition("=")
    try:
        out_text, _, in_text = geometry.partition("x")
        out_features, in_features = int(out_text), int(in_text)
    except ValueError as exc:
        raise DynQuantError(f"--shape expects OUTxIN, got {text!r}") from exc
    if out_features <= 0 or in_features <= 0:
        raise DynQuantError(f"--shape {text!r} has a non-positive dimension")
    return label or f"{out_features}x{in_features}", out_features, in_features


def render(result: dict[str, Any]) -> str:
    read_gbs = result["achievable_read_gbs"]
    bits = result["bits"]
    by_shape: dict[str, dict[Any, dict[str, Any]]] = {}
    for row in result["rows"]:
        by_shape.setdefault(row["shape"], {})[row["bits"]] = row

    out = [
        f"{result['device']}  --  achievable {read_gbs:.0f} GB/s read, "
        f"{result['achievable_copy_gbs']:.0f} GB/s copy, backend {result['backend']}, "
        f"{result['rows_per_call']} activation row(s)",
        "",
        f"| shape | {result['dtype']} | " + " | ".join(f"{b} bit" for b in bits) + " |",
        "|---|---|" + "---|" * len(bits),
    ]
    cached = False
    for label, out_features, in_features in result["shapes"]:
        cells = by_shape.get(label)
        if not cells:
            continue
        base = cells[None]
        base_pct = 100 * base["gbytes_per_s"] / read_gbs
        cached = cached or base_pct > 100
        line = [
            f"`{label}` {out_features}x{in_features}",
            f"{base['micros']:.1f} us / {base_pct:.0f} %" + (" *" if base_pct > 100 else ""),
        ]
        for width in bits:
            cell = cells.get(width)
            if cell is None:
                line.append("-")
                continue
            line.append(
                f"{base['micros'] / cell['micros']:.2f}x / "
                f"{100 * cell['gbytes_per_s'] / read_gbs:.0f} %"
            )
        out.append("| " + " | ".join(line) + " |")

    worst = max((row["max_rel_err"] for row in result["rows"]), default=0.0)
    out += [
        "",
        f"Speedup vs dense {result['dtype']} `F.linear`, and percent of achievable read bandwidth.",
        f"Worst relative error against the dequantized oracle: {worst:.2e}.",
    ]
    if cached:
        out.append(
            "`*` a baseline over 100 % is L2 residency across the benchmark loop, not a "
            "measurement error; it flatters the dense arm on those rows."
        )
    return "\n".join(out)


def run(args: argparse.Namespace) -> int:
    import torch

    from dynquant._version import __version__
    from dynquant.errors import DynQuantError
    from dynquant.runtime.ops import active_backend

    from ._shared import resolve_dtype

    if not torch.cuda.is_available():
        raise DynQuantError(
            "dynquant bench measures GPU memory bandwidth and needs a CUDA device. "
            "`dynquant doctor` reports what this machine has."
        )

    backend = active_backend().value
    if backend != "cuda" and not args.allow_fallback:
        raise DynQuantError(
            f"the active backend is {backend!r}, not the compiled kernels, so this "
            f"would benchmark the fallback -- which dequantizes the whole weight on "
            f"every call and is not what the packed runtime does on a working "
            f"install. Run `dynquant doctor`, or pass --allow-fallback to measure it "
            f"deliberately."
        )

    shapes: list[tuple[str, int, int]] = [parse_shape(text) for text in args.shape or []]
    if args.model:
        shapes = (
            shapes_from_model(
                args.model, limit=args.max_shapes, trust_remote_code=args.trust_remote_code
            )
            + shapes
        )
    if not shapes:
        raise DynQuantError("nothing to measure: give --model or at least one --shape")

    dtype = resolve_dtype(args.dtype)
    read_gbs, copy_gbs = achievable_bandwidth()
    measured: list[Row] = []
    for label, out_features, in_features in shapes:
        if not args.quiet:
            print(f"  {label} {out_features}x{in_features} ...", flush=True)
        measured.extend(
            measure_shape(
                label,
                out_features,
                in_features,
                bits=args.bits,
                group_size=args.group_size,
                rows=args.rows,
                dtype=dtype,
                iters=args.iters,
            )
        )

    result = {
        "dynquant_core": __version__,
        "device": torch.cuda.get_device_name(0),
        "backend": backend,
        "dtype": args.dtype,
        "group_size": args.group_size,
        "rows_per_call": args.rows,
        "bits": list(args.bits),
        "achievable_read_gbs": read_gbs,
        "achievable_copy_gbs": copy_gbs,
        "shapes": [list(s) for s in shapes],
        "rows": [asdict(row) for row in measured],
    }
    print("\n" + render(result), flush=True)

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n-> wrote {destination}", flush=True)
    if args.json:
        print(json.dumps(result, indent=2))
    return 0
