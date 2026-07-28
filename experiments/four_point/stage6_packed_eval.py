"""The same bit map, with the weights actually packed in VRAM.

Stage 5 quantized in place. ``quantize_model(..., in_place=True)`` writes the
reconstructed values back over the bf16 parameters, so the accuracy it reports is
the accuracy of the real thing and the memory footprint is still the bf16 one --
its own docstring says exactly that. This script is the other half: same
checkpoint, same bit map, same evaluation, but the weights are held as packed
``int32`` words and the compiled kernels do the arithmetic. It is the first
measurement point in this experiment that can say anything about VRAM or speed.

What is measured, and why in this order
---------------------------------------
Packing happens with the model on the **CPU**, and the model arrives on the GPU
already packed. Quantizing on the GPU and measuring afterwards would report a peak
that includes the bf16 copy the packing consumed -- a number nobody serving the
model would ever see, since loading a quantized checkpoint never materialises one.
So ``memory_allocated()`` taken immediately after ``.to("cuda")``, with nothing
else on the device, is the resident weight footprint and nothing is being netted
out of it.

Decode is then timed, then accuracy is scored, with the peak counter reset between.
A batch-32 evaluation allocates activations and a KV cache that dwarf the weights,
and one peak spanning both would say nothing about either.

Three claims are on trial here
------------------------------
1. **Accuracy is unchanged from stage 5** -- and unchanged *exactly*. Both paths
   quantize through the same search over the same grid, so the values the kernels
   decode are bit-identical to the ones stage 5 wrote into the parameters. A
   difference of even one problem means the kernels disagree with the oracle
   somewhere the parity tests do not reach, and the script says so rather than
   leaving it to be noticed.
2. **Resident weight bytes match the manifest** the allocator wrote when it chose
   the map. The allocator's number is a prediction made from bit widths; this one
   is read off the device.
3. **Decode is faster than bf16.** At batch 1 a matmul reads far more weight than
   it does arithmetic, so the runtime is the time to stream the weights -- and the
   packed weights are a quarter of the size. Above ``gemv_max_rows`` the runtime
   falls back to dequantize-then-``F.linear``, which reads *more* memory than bf16
   does, so the batch sweep is where the crossover shows up rather than something
   to be quiet about.

The ``--dense`` arm runs every one of those measurements against the unquantized
fine-tuned model through the same code, so the comparison is not between a number
measured here and a number remembered from another script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from common import (
    DTYPE,
    MODEL_ID,
    RUN_DIR,
    TASK,
    load_task,
    load_tokenizer,
    record,
    run_eval,
    set_seed,
)

GIB = float(2**30)


def load_on_cpu(path: str) -> Any:
    """The fine-tuned checkpoint, on the CPU, with no accelerate dispatch.

    :func:`common.load_model` passes ``device_map``, which installs accelerate hooks
    and makes a later ``.to("cuda")`` either an error or a second dispatch depending
    on the transformers version. The weights loaded are the same weights -- same
    file, same dtype -- so the measurement points stay comparable; what differs is
    only that this one keeps the right to move the model itself, which is the whole
    point of packing before it reaches the device.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE)
    model.config.use_cache = True
    return model.eval()


def device_bytes() -> tuple[int, int, int]:
    """``(allocated, peak allocated, peak reserved)`` -- all three, deliberately.

    ``allocated`` is what the tensors need. ``reserved`` is what the caching
    allocator took from the driver and is what ``nvidia-smi`` shows; reporting only
    the first invites "but nvidia-smi says more" and reporting only the second
    charges the weights for the allocator's fragmentation.
    """
    torch.cuda.synchronize()
    return (
        torch.cuda.memory_allocated(),
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )


def resident_weight_bytes(model: Any) -> int:
    """Every parameter and buffer the model holds, each counted once.

    Deduplicated by storage, because a tied table reached through two modules is one
    allocation and counting it twice is precisely the error this runtime was changed
    to prevent. Walks the live tree rather than the bit map, so anything the packing
    missed is *included* rather than quietly excluded.
    """
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is None:
            continue
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.untyped_storage().nbytes()
    return total


def decode_prompt(shots: list[Any], example: Any) -> str:
    """A real evaluation prompt, not a synthetic one of the same length.

    Decode speed does not depend on the tokens, but prompt *length* sets the KV
    cache and therefore the attention cost per step, and this model interleaves
    linear-attention layers whose state cost differs from softmax attention's. Using
    the prompt the accuracy run actually sees keeps the speed number attached to the
    workload the accuracy number came from.
    """
    from importlib import import_module

    module = import_module(f"dynquant.eval.{TASK.key}")
    build = getattr(module, "build_prompt", None)
    if build is not None:
        return build(example, shots)
    return TASK.format_training_text(example)[0]


def time_decode(
    model: Any, tokenizer: Any, prompt: str, *, batch: int, new_tokens: int, repeats: int
) -> dict[str, Any]:
    """Tokens per second in the decode phase, with prefill subtracted rather than amortised.

    Two generations are timed, one of a single token and one of ``new_tokens``. The
    difference is ``new_tokens - 1`` decode steps with the prefill, the tokenization
    and the generate() overhead common to both and therefore cancelled. Reporting
    ``new_tokens / total`` instead would blend a compute-bound prefill into a
    memory-bound decode and quietly flatter whichever arm has the shorter prefill.

    The best of ``repeats`` is taken, not the mean: the contended runs are the ones
    with something else on the GPU, and the fastest observed run is the one closest
    to the machine's actual capability.
    """
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].repeat(batch, 1).to(model.device)
    mask = torch.ones_like(ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def generate(n: int) -> torch.Tensor:
        with torch.inference_mode():
            return model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=n,
                # Forces the full n steps. Without it an EOS ends the loop early and
                # the two timings would cover different amounts of work.
                min_new_tokens=n,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad,
            )

    generate(4)  # warm the kernels, the allocator and generate()'s own caches

    def best(n: int) -> float:
        fastest = float("inf")
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = generate(n)
            torch.cuda.synchronize()
            fastest = min(fastest, time.perf_counter() - started)
            assert out.shape[1] == ids.shape[1] + n, "generate stopped early"
        return fastest

    prefill = best(1)
    full = best(new_tokens)
    per_step = (full - prefill) / (new_tokens - 1)
    return {
        "batch": batch,
        "prompt_tokens": int(ids.shape[1]),
        "new_tokens": new_tokens,
        "prefill_seconds": round(prefill, 4),
        "decode_seconds_per_step": round(per_step, 6),
        "decode_tokens_per_second": round(batch / per_step, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument(
        "--bitmaps",
        default=str(RUN_DIR / "stage4_sensitivity.json"),
        help="the allocation to pack. Defaults to the sensitivity map, which is the "
        "arm stage 5 measured at 84.08%%.",
    )
    parser.add_argument("--target", default="3.25", help="key into the bit-map file")
    parser.add_argument("--name", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="skip packing: the bf16 baseline, measured through this same script so "
        "the VRAM and speed numbers are comparable by construction.",
    )
    parser.add_argument("--decode-batches", default="1,4,8,32")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("this script measures VRAM and decode speed; it needs a GPU", file=sys.stderr)
        return 2

    set_seed()

    from dynquant.runtime.backend import resolve_backend
    from dynquant.runtime.linear import pack_model, packed_bytes

    backend = resolve_backend()
    print(f"backend: {backend.value}", flush=True)
    if not args.dense and backend.value != "cuda":
        print(
            f"refusing to report kernel numbers from the {backend.value} backend; "
            f"build dynquant-kernels or set DYNQUANT_BACKEND=cuda to see why it is unavailable",
            file=sys.stderr,
        )
        return 2

    entry: dict[str, Any] = {}
    if not args.dense:
        maps = json.loads(Path(args.bitmaps).read_text(encoding="utf-8"))["maps"]
        if args.target not in maps:
            print(
                f"no bit map for {args.target!r}; have {', '.join(sorted(maps))}", file=sys.stderr
            )
            return 2
        entry = maps[args.target]

    name = args.name or (
        "stage6_dense_bf16" if args.dense else f"stage6_packed_{args.target.replace('.', 'p')}"
    )

    started = time.time()
    model = load_on_cpu(args.model)
    print(f"loaded {args.model} on cpu in {time.time() - started:.1f}s", flush=True)

    pack_seconds = 0.0
    report_summary = ""
    if not args.dense:
        bits: dict[str, int] = entry["bits"]
        group_size: int = entry["group_size"]

        def show(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                print(f"  [pack] {done}/{total} modules", flush=True)

        started = time.time()
        report = pack_model(model, bits, group_size=group_size, symmetric=False, progress=show)
        pack_seconds = time.time() - started
        report_summary = report.summary()
        print(f"\npacked in {pack_seconds:.1f}s\n{report_summary}", flush=True)
        if report.skipped:
            print(f"  WARNING: still dense: {report.skipped}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.to("cuda")
    allocated, _peak, reserved = device_bytes()
    weights = resident_weight_bytes(model)
    accounting = packed_bytes(model)

    manifest = int(entry.get("nbytes", 0))
    print(
        f"\nresident on device: {allocated / GIB:.4f} GiB allocated, "
        f"{reserved / GIB:.4f} GiB reserved by the allocator",
        flush=True,
    )
    print(f"  walked from the module tree: {weights / GIB:.4f} GiB", flush=True)
    print(
        f"  packed tensors {accounting['packed_bytes'] / GIB:.4f} GiB, "
        f"left dense {accounting['dense_bytes'] / GIB:.4f} GiB "
        f"({len(accounting['dense_modules'])} modules)",
        flush=True,
    )
    if manifest:
        print(
            f"  manifest predicted {manifest / GIB:.4f} GiB "
            f"({100.0 * allocated / manifest - 100.0:+.2f}% against what is on the device)",
            flush=True,
        )

    tokenizer = load_tokenizer()
    _train, test, shots = load_task()
    prompt = decode_prompt(shots, test[0])

    decode: list[dict[str, Any]] = []
    peak_decode: dict[str, float] = {}
    for batch in [int(b) for b in args.decode_batches.split(",") if b]:
        torch.cuda.reset_peak_memory_stats()
        measured = time_decode(
            model,
            tokenizer,
            prompt,
            batch=batch,
            new_tokens=args.decode_tokens,
            repeats=args.decode_repeats,
        )
        _alloc, peak, peak_reserved = device_bytes()
        measured["peak_allocated_bytes"] = peak
        measured["peak_reserved_bytes"] = peak_reserved
        peak_decode[str(batch)] = peak / GIB
        decode.append(measured)
        print(
            f"  decode b={batch}: {measured['decode_tokens_per_second']:.1f} tok/s "
            f"({measured['decode_seconds_per_step'] * 1e3:.2f} ms/step), "
            f"prefill {measured['prefill_seconds'] * 1e3:.1f} ms, "
            f"peak {peak / GIB:.3f} GiB",
            flush=True,
        )

    payload = {
        "task": TASK.key,
        "arm": "dense-bf16" if args.dense else f"packed-{args.target}",
        "source_model": args.model,
        "base_model": MODEL_ID,
        "backend": backend.value,
        "average_bits": entry.get("average_bits"),
        "manifest_bytes": manifest or None,
        "pack_seconds": round(pack_seconds, 1),
        "pack_summary": report_summary,
        "resident_allocated_bytes": allocated,
        "resident_reserved_bytes": reserved,
        "resident_walked_bytes": weights,
        "packed_bytes": accounting["packed_bytes"],
        "dense_bytes": accounting["dense_bytes"],
        "dense_modules": accounting["dense_modules"],
        "decode": decode,
        "peak_decode_gib": peak_decode,
    }
    record(f"{name}_runtime", payload)

    if args.skip_eval:
        return 0

    torch.cuda.reset_peak_memory_stats()
    label = args.label or (
        "bf16 fine-tuned" if args.dense else f"packed {entry['average_bits']:.2f}b, CUDA kernels"
    )
    result = run_eval(
        model,
        label=label,
        name=name,
        limit=args.limit,
        extra={
            "source_model": args.model,
            "arm": payload["arm"],
            "backend": backend.value,
            "average_bits": entry.get("average_bits"),
            "resident_allocated_bytes": allocated,
            "resident_gib": allocated / GIB,
            "manifest_gib": (manifest / GIB) if manifest else None,
            "decode": decode,
        },
    )
    _alloc, eval_peak, eval_reserved = device_bytes()
    print(
        f"\npeak during evaluation: {eval_peak / GIB:.3f} GiB allocated, "
        f"{eval_reserved / GIB:.3f} GiB reserved",
        flush=True,
    )

    # Rewritten rather than passed in, because the peak is only known once the
    # evaluation it measures has finished. The record run_eval wrote is the one the
    # results table reads, so the number has to land in that file and not beside it.
    path = RUN_DIR / f"{name}.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["peak_eval_allocated_bytes"] = eval_peak
    stored["peak_eval_reserved_bytes"] = eval_reserved
    path.write_text(json.dumps(stored, indent=2, default=str), encoding="utf-8")

    # The consistency check, stated loudly because a silent match is worth nothing.
    prior = RUN_DIR / f"stage5_{args.target.replace('.', 'p')}_sens.json"
    if not args.dense and prior.exists():
        was = json.loads(prior.read_text(encoding="utf-8"))
        agreement = "MATCH" if was["correct"] == result.correct else "DISAGREEMENT"
        print(
            f"\n{agreement}: stage 5 (simulated, bf16 weights) scored {was['correct']}/"
            f"{was['total']}; the packed kernels scored {result.correct}/{result.total}. "
            "These quantize through the same search, so the decoded values are "
            "bit-identical and anything but an exact match is a kernel bug.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
