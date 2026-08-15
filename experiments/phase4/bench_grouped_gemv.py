"""Drive ``moe_grouped_gemv`` across a real sweep, and time it against the loop it replaces.

Two jobs in one file on purpose.

The sanitizer needs a *workload*. Running it over the parity suite looks like coverage and is
not: of the four grouped cases that reach cuda there, two reject their arguments before any
launch, one launches once, and the fourth -- the one that actually sweeps four widths over four
geometries -- does its work in a ``subprocess``, because the scalar-path flag it sets is read
once into a function-local static. ``--target-processes application-only`` does not follow a
child, so those sixteen launches are outside the tool. In-process grouped launches under the
sanitizer, before this file: one. That is the whole of what a clean report over the suite was
covering, and it is not what the phrase is taken to mean.

The timing needs the same geometries, because "one launch instead of E" is a claim about launch
count, not about arithmetic -- the win has to be measured where the launch count dominates,
which is decode, and it has to be shown *not* to appear where it does not, which is prefill.

The comparison is against :func:`dynquant.runtime.experts._grouped_linear_packed`'s own loop
rather than a hand-written one, so the ``seg.tolist()`` synchronization that function's docstring
names as the cost sits inside the measurement instead of being argued for beside it. A third
timing runs that same loop over an already-dense bank, which splits the win in two: fused vs
dense is the launch count and the sync, dense vs packed is ``bank[e]`` decoding one expert per
call with nothing cached. A single ratio against the packed loop credits the kernel for both,
and only one of them is the kernel's doing.

Usage on a GPU box, from the repo root::

    python experiments/phase4/bench_grouped_gemv.py --time --iters 50 --out results.json
    compute-sanitizer --tool memcheck --target-processes all \
        python experiments/phase4/bench_grouped_gemv.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch

from dynquant.quant.device import quantize_tensor
from dynquant.runtime import experts as experts_mod
from dynquant.runtime import ops
from dynquant.runtime.experts import _grouped_linear_packed
from dynquant.runtime.linear import DynQuantExpertBank

DEV = "cuda"
DTYPE = torch.float16

# (num_experts, out_features, in_features). Three shapes rather than one, so what moves with
# the expert count is separable from what moves with the row width: LFM2.5-8B-A1B's own
# 32-expert geometry, a wide-and-thin 64-expert one, and an 8-expert bank whose single expert
# is large enough that decoding it, not launching for it, is the cost.
GEOMETRIES = ((32, 1024, 2048), (64, 512, 2048), (8, 2048, 4096))
ROWSETS = (4, 8, 32, 128, 512, 2048)

# The parity suite's own criterion, restated here because the raw difference is not it: a
# measurement of |got - want| alone reports failures where the suite reports none.
ATOL = 2e-2
RTOL = 2e-2


def build(num_experts: int, out_features: int, in_features: int, bits: int) -> DynQuantExpertBank:
    """A packed expert bank of the given geometry, encoded on the card."""
    torch.manual_seed(num_experts * 1000 + out_features)
    dense = (torch.randn(num_experts, out_features, in_features) * 0.05).to(DTYPE)
    # `.to(DEV)` before the encode, not after. The bank is only scaffolding, and a CPU encode
    # of a 67M-element bank costs more wall clock than every kernel launch this file times.
    qt, _ = quantize_tensor(dense.to(DEV), bits=bits, device=torch.device(DEV))
    return DynQuantExpertBank(qt, out_dtype=DTYPE).to(DEV)


def even_bands(num_experts: int, rows: int) -> torch.Tensor:
    """Round-robin token to expert: every band non-empty and within one row of every other."""
    ids = torch.arange(rows, device=DEV) % num_experts
    counts = torch.bincount(ids, minlength=num_experts)[:num_experts]
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)


def skewed_bands(num_experts: int, rows: int) -> torch.Tensor:
    """Every token to expert 0: the load-imbalance case, and ``num_experts - 1`` empty bands."""
    counts = torch.zeros(num_experts, dtype=torch.long, device=DEV)
    counts[0] = rows
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)


def _set_probe(flag: Any) -> None:
    """Rebind both names: ``experts`` imported the module, ``ops`` owns the function."""
    ops.has_grouped_gemv = flag
    experts_mod.ops.has_grouped_gemv = flag


def loop_call(bank: Any, x: torch.Tensor, seg: torch.Tensor, out_features: int) -> torch.Tensor:
    """The dispatch's own loop, reached by telling the dispatch the op is not there."""
    saved = ops.has_grouped_gemv
    _set_probe(lambda: False)
    try:
        return _grouped_linear_packed(
            x, bank, seg, bias=None, is_transposed=False, out_features=out_features
        )
    finally:
        _set_probe(saved)


def _bank_getitem_via_reference(self: Any, index: Any) -> torch.Tensor:
    """``DynQuantExpertBank.__getitem__`` forced back onto the pure-torch reference.

    ``QuantTensor.dequantize`` is documented in its own body as the *reference*
    implementation: it materialises the codes in fp32, so a 67M-element bank moves about
    1.3 GB through five elementwise passes. ``runtime.ops.dequantize`` dispatches the same
    arithmetic to one CUDA kernel when the extension is loaded and falls through to that same
    reference when it is not.

    The bank used to call the reference directly, which is how this column was born: the first
    version of this sweep timed the fused kernel against a loop paying a dequantizer nobody
    would choose, and reported a speedup with that defect folded into it. The bank now indexes
    through ``ops`` (``runtime/linear.py``), so ``loop_ms`` is the repaired loop and this
    column is what the loop used to cost.

    It stays in the sweep as a standing measurement rather than a historical note: it is the
    price of the ``ops`` hop, so a revert of that fix shows up here as ``loop_ms`` climbing to
    meet ``loop_ref_ms`` instead of as a speedup quietly getting better.

    Patching the method rather than rewriting the loop keeps the loop under test the shipped
    one: the only thing that differs between ``loop_ms`` and ``loop_ref_ms`` is which
    dequantizer ``bank[e]`` reaches.
    """
    expert = int(index)
    band = self.out_features
    return self.weight_qt.rows(expert * band, (expert + 1) * band).dequantize(dtype=self.out_dtype)


def grouped_mm_reason() -> str:
    """One probe: is ``torch._grouped_mm`` usable on this card, and if not, why not?

    This is the reference the loop is not -- transformers' own default MoE path, dense and
    fused, one launch for the whole bank. Without it the only comparison available is against
    the code this kernel replaces, which can say the replacement is an improvement and cannot
    say whether it is fast. The probe is guarded rather than assumed because the op is a
    ``sm_90``-and-up CUTLASS path with its own dtype rules, and a *recorded* refusal is worth
    more than an omitted column: it is the reason the Python loop is the real alternative on
    whatever card returns it. Returns the empty string when the op ran.
    """
    if not hasattr(torch, "_grouped_mm"):
        return "torch has no _grouped_mm"
    probe_x = torch.randn(4, 64, device=DEV, dtype=DTYPE)
    probe_w = torch.randn(2, 64, 32, device=DEV, dtype=DTYPE)
    offs = torch.tensor([2, 4], device=DEV, dtype=torch.int32)
    try:
        torch._grouped_mm(probe_x, probe_w, offs)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - every refusal is a result, not an error
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return ""


def time_path(fn: Callable[[], torch.Tensor], iters: int) -> float:
    """Median milliseconds over ``iters``, after three warmups. CUDA events, not wall clock."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    times.sort()
    return times[len(times) // 2]


def sweep(iters: int, time_it: bool, gmm: bool) -> tuple[list[dict[str, Any]], int, float]:
    """Every (geometry, width, row count, band shape). Returns rows, launch count, worst ratio."""
    results: list[dict[str, Any]] = []
    launches = 0
    worst = 0.0
    for num_experts, out_features, in_features in GEOMETRIES:
        for bits in (3, 4, 8):
            bank = build(num_experts, out_features, in_features, bits)
            qt = bank.weight_qt
            dense = qt.dequantize(dtype=DTYPE)
            # Timed once per bank, not once per row count: it does not depend on the
            # tokens. It is here because it is the missing half of the prefill question.
            # `dense_ms` alone compares the fused kernel against a bank somebody already
            # dequantized for free, which nobody does; the real alternative at prefill is
            # to pay this once per forward and then run dense, so the comparison the
            # dispatch actually faces is `fused_ms` against `dequant_ms + dense_ms`.
            dequant_ms = (
                round(time_path(partial(qt.dequantize, dtype=DTYPE), iters), 4) if time_it else 0.0
            )
            for rows in ROWSETS:
                for band, make in (("even", even_bands), ("skewed", skewed_bands)):
                    seg = make(num_experts, rows)
                    torch.manual_seed(rows + bits)
                    x = torch.randn(rows, in_features, device=DEV, dtype=DTYPE) * 0.1
                    run_fused = partial(
                        ops.grouped_quantized_matmul, x, qt, seg, out_features=out_features
                    )
                    run_loop = partial(loop_call, bank, x, seg, out_features)
                    run_dense = partial(
                        _grouped_linear_packed,
                        x,
                        dense,
                        seg,
                        bias=None,
                        is_transposed=False,
                        out_features=out_features,
                    )
                    fused = run_fused()
                    launches += 1
                    loop = run_loop()
                    allowed = ATOL + RTOL * loop.float().abs()
                    spent = ((fused.float() - loop.float()).abs() / allowed).max().item()
                    worst = max(worst, spent)
                    row: dict[str, Any] = {
                        "experts": num_experts,
                        "out": out_features,
                        "in": in_features,
                        "bits": bits,
                        "rows": rows,
                        "band": band,
                        "spent": round(spent, 4),
                    }
                    # Timing on one band only. The kernel's cost model is not band-shaped --
                    # `skewed` is there to make the empty bands a correctness case, and timing
                    # it as well would double a 30-minute sweep for a second copy of the same
                    # curve.
                    if time_it and band == "even":
                        row["fused_ms"] = round(time_path(run_fused, iters), 4)
                        row["loop_ms"] = round(time_path(run_loop, iters), 4)
                        saved_getitem = DynQuantExpertBank.__getitem__
                        DynQuantExpertBank.__getitem__ = _bank_getitem_via_reference  # type: ignore[method-assign]
                        try:
                            patched = run_loop()
                            row["loop_ref_ms"] = round(time_path(run_loop, iters), 4)
                        finally:
                            DynQuantExpertBank.__getitem__ = saved_getitem  # type: ignore[method-assign]
                        # The swap has to be checked, not assumed: a patched `__getitem__`
                        # that quietly returned the wrong band would make the loop it is
                        # standing in for look fast for the wrong reason.
                        moved = (patched.float() - loop.float()).abs() / allowed
                        row["loop_ref_spent"] = round(moved.max().item(), 4)
                        row["dense_ms"] = round(time_path(run_dense, iters), 4)
                        row["dequant_ms"] = dequant_ms
                        row["vs_prefill"] = round(
                            (dequant_ms + row["dense_ms"]) / row["fused_ms"], 2
                        )
                        row["vs_loop_ref"] = round(row["loop_ref_ms"] / row["fused_ms"], 2)
                        if gmm:
                            run_gmm = partial(
                                torch._grouped_mm,
                                x,
                                dense.transpose(1, 2).contiguous(),
                                seg[1:].to(torch.int32),
                            )
                            row["grouped_mm_ms"] = round(time_path(run_gmm, iters), 4)
                        row["speedup"] = round(row["loop_ms"] / row["fused_ms"], 2)
                        row["vs_dense"] = round(row["dense_ms"] / row["fused_ms"], 2)
                    results.append(row)
    return results, launches, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", action="store_true", help="also time; skip under a sanitizer")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    reason = grouped_mm_reason() if args.time else "not probed without --time"
    print(f"grouped_mm={'available' if reason == '' else reason}")
    results, launches, worst = sweep(args.iters, args.time, reason == "")
    print(f"configurations={len(results)} launches={launches} worst_ratio={worst:.4f}")
    if worst > 1.0:
        raise SystemExit(f"grouped kernel disagrees with the loop at {worst:.4f} of tolerance")

    if args.time:
        print(
            "   E   out    in  b  rows  fused_ms   loop_ms   lpref_ms  dense_ms"
            "   deq_ms  vs_loop  vs_lpref vs_dense"
        )
        for r in results:
            if "speedup" not in r:
                continue
            print(
                f"{r['experts']:>4} {r['out']:>5} {r['in']:>5} {r['bits']:>2} "
                f"{r['rows']:>5} {r['fused_ms']:>9.4f} {r['loop_ms']:>9.4f} "
                f"{r['loop_ref_ms']:>10.4f} {r['dense_ms']:>9.4f} "
                f"{r['dequant_ms']:>8.4f} {r['speedup']:>8.2f} "
                f"{r['vs_loop_ref']:>9.2f} {r['vs_dense']:>8.2f}"
            )

    if args.out:
        # Self-describing, because a bare list of rows cannot say which card produced
        # it or why a column is missing, and the answer to the second question is a
        # result in its own right.
        record = {
            "device": torch.cuda.get_device_name(0),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "dtype": str(DTYPE),
            "iters": args.iters,
            "grouped_mm": reason or "available",
            "launches": launches,
            "worst_tolerance_spent": round(worst, 4),
            "rows": results,
        }
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
