"""Where one decode step's wall time actually goes, for the packed and bf16 arms alike.

Stage 6 reports that packed decode is *slower* than bf16 at batch 1 while the GEMV is
1.2-2.6x faster than `F.linear` on every shape this model contains. Both cannot be
explained by the kernel, so this script measures the step rather than arguing about it:
kernel launches, GPU-busy time, matmul time, and the wall time none of those account for.

Two things here are easy to get wrong and both were got wrong in an earlier pass of this
measurement:

* `torch.profiler` reports device time on *op-level* entries (`aten::mm`) as well as on
  the kernels those ops launched. Summing everything double-counts, and the first version
  of this measurement duly reported 58 % GPU-busy and 4551 launches instead of 29 % and
  2290. Only `DeviceType.CUDA` events are counted below, and the assertion at the end
  fails loudly if a future torch changes what that means.
* Profiling `generate()` measures prefill plus decode. Only the decode steps are wanted,
  so prefill runs outside the profiler and the profiled region is a hand-rolled loop over
  single-token forwards with a live KV cache -- which is what a decode step *is*.
* Wall time *inside* the profiled region is not the step's wall time. Tracing 2000 launches
  per step costs more than the step does: the same loop measures ~30 ms/step untraced and
  ~108 ms/step traced. So the loop is timed twice -- once clean for the denominator, once
  traced for the attribution -- and every fraction below divides device time by the clean
  wall. Dividing by the traced wall is what makes a launch-bound step look GPU-starved by a
  factor of three.

Usage:
    python stage7_profile_step.py            # packed, the 3.25-bit sensitivity map
    python stage7_profile_step.py --dense    # bf16 baseline through the same code
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from common import RUN_DIR, TASK, load_task, load_tokenizer, record, set_seed
from stage6_packed_eval import decode_prompt, load_on_cpu

STEPS = 8


def profile_decode(model: Any, tokenizer: Any, prompt: str, *, batch: int) -> dict[str, Any]:
    """Profile `STEPS` decode steps and attribute the wall time.

    The KV cache is built once by a real prefill outside the profiled region, then each
    profiled iteration feeds exactly one token per sequence -- so every iteration is a
    decode step with a cache of the same shape a generation would have.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].repeat(batch, 1).to(model.device)

    with torch.inference_mode():
        # Prefill, outside the profiler: this is the compute-bound phase and including it
        # would bury the launch-gap story the decode step is here to show.
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        step_ids = out.logits[:, -1:, :].argmax(dim=-1)

        def one_step() -> None:
            nonlocal cache, step_ids
            res = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
            cache = res.past_key_values
            step_ids = res.logits[:, -1:, :].argmax(dim=-1)

        for _ in range(3):  # warm the kernels and the allocator at this cache length
            one_step()

        # Clean pass: the denominator. Best of three, matching stage 6's method, so the
        # fraction is against the machine's capability rather than its worst moment.
        clean = float("inf")
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(STEPS):
                one_step()
            torch.cuda.synchronize()
            clean = min(clean, time.perf_counter() - started)

        torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(STEPS):
                one_step()
            torch.cuda.synchronize()
        traced = time.perf_counter() - started
        wall = clean

    device_us = 0.0
    launches = 0
    by_kernel: dict[str, float] = defaultdict(float)
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        device_us += event.self_device_time_total
        launches += event.count
        by_kernel[event.key] += event.self_device_time_total

    assert launches, "no CUDA kernel events -- has DeviceType.CUDA changed meaning?"

    def matching(*needles: str) -> float:
        return sum(us for k, us in by_kernel.items() if any(n in k.lower() for n in needles))

    matmul_us = matching(
        "gemm", "gemv", "cutlass", "s16816", "tensorop", "nn_128x", "nt_128x", "splitk"
    )
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1])[:12]
    return {
        "batch": batch,
        "steps": STEPS,
        "wall_ms_per_step": round(wall * 1e3 / STEPS, 3),
        "traced_wall_ms_per_step": round(traced * 1e3 / STEPS, 3),
        "gpu_busy_ms_per_step": round(device_us / 1e3 / STEPS, 3),
        "gpu_busy_fraction": round(device_us / 1e3 / (wall * 1e3), 4),
        "launches_per_step": round(launches / STEPS, 1),
        "matmul_ms_per_step": round(matmul_us / 1e3 / STEPS, 3),
        "matmul_fraction_of_wall": round(matmul_us / 1e3 / (wall * 1e3), 4),
        "top_kernels": [
            {"kernel": k[:90], "ms_per_step": round(us / 1e3 / STEPS, 3)} for k, us in top
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--bitmaps", default=str(RUN_DIR / "stage4_sensitivity.json"))
    parser.add_argument("--target", default="3.25")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dense", action="store_true", help="bf16 baseline through this same code")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("this script profiles GPU kernels; it needs a GPU", file=sys.stderr)
        return 2

    set_seed()
    from dynquant.runtime.backend import resolve_backend
    from dynquant.runtime.linear import pack_model

    backend = resolve_backend()
    print(f"backend: {backend.value}", flush=True)
    if not args.dense and backend.value != "cuda":
        print(
            f"refusing to profile the {backend.value} backend as if it were the kernels",
            file=sys.stderr,
        )
        return 2

    model = load_on_cpu(args.model)
    if not args.dense:
        entry = json.loads(Path(args.bitmaps).read_text(encoding="utf-8"))["maps"][args.target]
        started = time.time()
        pack_model(model, entry["bits"], group_size=entry["group_size"], symmetric=False)
        print(f"packed in {time.time() - started:.1f}s", flush=True)
    model.to("cuda")

    tokenizer = load_tokenizer()
    _train, shots, evalset = load_task()
    prompt = decode_prompt(shots[: TASK.n_shots], evalset[0])

    result = profile_decode(model, tokenizer, prompt, batch=args.batch)
    arm = "bf16" if args.dense else f"packed {args.target}b"
    print(
        f"\n{arm}, batch {result['batch']}, {result['steps']} decode steps\n"
        f"  wall            {result['wall_ms_per_step']:8.3f} ms/step  (untraced; "
        f"{result['traced_wall_ms_per_step']:.1f} under the profiler, which is why the "
        f"fractions below use the untraced figure)\n"
        f"  GPU busy        {result['gpu_busy_ms_per_step']:8.3f} ms/step "
        f"({result['gpu_busy_fraction'] * 100:.1f} % of wall; "
        f"the rest is launch gaps)\n"
        f"  kernel launches {result['launches_per_step']:8.1f} per step\n"
        f"  matmul kernels  {result['matmul_ms_per_step']:8.3f} ms/step "
        f"({result['matmul_fraction_of_wall'] * 100:.1f} % of wall)",
        flush=True,
    )
    print("\n  hottest kernels, ms/step:", flush=True)
    for row in result["top_kernels"]:
        print(f"    {row['ms_per_step']:7.3f}  {row['kernel']}", flush=True)

    record(args.name or f"stage7_profile_{'dense' if args.dense else 'packed'}", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
