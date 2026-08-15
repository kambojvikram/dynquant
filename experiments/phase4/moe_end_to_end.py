"""One real MoE, packed, decoding through the grouped kernel.

What this adds to `bench_grouped_gemv.py`
-----------------------------------------
That benchmark times the kernel against four denominators on synthetic banks. It
cannot say whether a model built out of those banks loads, routes, and decodes --
which is the last clause of P8's gate and the one thing the kernel report still
lists as unclaimed. This runs `LiquidAI/LFM2.5-8B-A1B`, whose expert banks are
91.5% of its parameters, all the way to generated text under three arms:

* `bf16`     -- the unpacked model, the reference for both text and rate.
* `eager`    -- packed, experts indexed one at a time through `bank[expert]`.
* `dynquant` -- packed, the whole bank handed to `moe_grouped_gemv` in one launch.

`eager` and `dynquant` differ in exactly one place, which is the point: same
weights, same bits, same group size, same prompts, one substitution at the dispatch.

Each arm runs in its own process, because peak VRAM is one of the numbers and a
process that has already held a bf16 copy cannot report the packed peak honestly.

Why the rate is a slope and not a division
------------------------------------------
`generate` time is prefill plus decode, and prefill is a GEMM that has nothing to
do with the kernel under test. Dividing tokens by total time therefore reports a
blend whose mixture changes with prompt length. Each arm is timed at two decode
budgets and the rate is the slope between them, which cancels prefill and load
exactly rather than approximately. The naive figure is recorded next to it so the
difference is visible instead of being taken on trust.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# Two prompts rather than one: a MoE routes on content, so a single prompt can
# leave whole experts cold and time a bank that never ran.
PROMPTS = (
    "Write a SQL query returning the three departments with the highest average salary.",
    "Explain in two sentences why memory bandwidth rather than FLOPs limits decoding.",
)
SHORT, LONG = 32, 96


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / 2**20


def _live_mib() -> float:
    """Currently-allocated bytes, which is what a served model actually costs.

    The peak is not that number and must not be quoted as it. Packing runs the clipping
    search on the GPU, so a dense copy of each weight is resident while its packed form
    is being built, and the peak over load-and-pack carries that workspace. A server
    loading an exported checkpoint never pays it. Recording both keeps the honest
    comparison available and the flattering one labelled.
    """
    return torch.cuda.memory_allocated() / 2**20


def _load(path: str, dtype: torch.dtype) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model.eval()
    return tok, model


def _pack(model: Any, bits: int, group_size: int) -> dict[str, Any]:
    """Pack in place on the CPU-resident model, so no dense weight reaches the GPU."""
    from dynquant.commands._shared import uniform_map
    from dynquant.graph.classify import classify_model
    from dynquant.graph.roles import ModuleRole
    from dynquant.runtime.linear import pack_model

    graph = classify_model(model)
    bit_map = uniform_map(graph, bits, group_size=group_size)
    # The router is dropped rather than packed, and the count is reported rather
    # than swallowed. `Lfm2MoeTopKRouter` owns a weight and calls `F.linear` on it
    # without being an `nn.Linear`, so the packed runtime has no forward to stand
    # in for -- `pack_model` refuses it by name, which is the right behaviour and
    # is how this arm failed the first time it ran. Leaving it dense is also what
    # the allocator's own floors want: a top-k decision is discrete, and a router
    # that rounds to a different argmax has not lost precision, it has lost the
    # token. What it costs in bytes is `router_params` in the record rather than an
    # adjective here, because a share that is asserted is a share nobody checked.
    routers = {n for n in bit_map.bits if graph[n].role is ModuleRole.MOE_ROUTER}
    packable = {n: b for n, b in bit_map.bits.items() if n not in routers}
    t0 = time.perf_counter()
    report = pack_model(model, packable, group_size=group_size)
    return {
        "modules_packed": len(packable),
        "routers_left_dense": len(routers),
        "router_params": sum(graph[n].num_params for n in routers),
        "pack_seconds": round(time.perf_counter() - t0, 1),
        "accounted_bytes": int(bit_map.nbytes),
        "accounted_bits": round(float(bit_map.average_bits), 4),
        # Measured from the tensors that exist, not predicted from the bit map --
        # the two disagree whenever a module is skipped or tied, and the predicted
        # one is the one that flatters.
        "fp16_bytes": int(report.fp16_bytes),
        "packed_bytes": int(report.packed_bytes),
        "modules_replaced": len(report.modules),
        "modules_tied": len(report.tied),
        "modules_skipped": len(report.skipped),
        "pack_moved_experts_from": report.experts_implementation,
        "pack_moved_experts_to": report.experts_dispatch,
    }


def _set_dispatch(model: Any, arm: str) -> dict[str, Any]:
    from dynquant.runtime.experts import use_dynquant_experts
    from dynquant.runtime.linear import use_eager_experts

    cfg = getattr(model, "config", None)
    before = getattr(cfg, "_experts_implementation", None)
    if arm == "dynquant":
        use_dynquant_experts(model)
    elif arm == "eager":
        use_eager_experts(model)
    after = getattr(cfg, "_experts_implementation", None)
    return {"experts_impl_before": before, "experts_impl_after": after}


def _render(tok: Any, prompt: str) -> str:
    """The instruct template, or the raw string if this checkpoint has none.

    Skipping it is not a cosmetic difference. The first run of this harness fed
    LFM2.5-8B-A1B a bare instruction and got fluent, grammatical, entirely
    contentless loops back -- "the problem is that the issue is that the problem
    is that" -- from the bf16 model. Timing was unaffected, but a coherence claim
    read off that output would have been a claim about the harness.
    """
    template = getattr(tok, "chat_template", None)
    if not template:
        return prompt
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    return str(rendered)


def _generate(
    model: Any, tok: Any, prompt: str, budget: int, device: str, *, force: bool = True
) -> tuple[float, str]:
    from transformers import GenerationConfig

    enc = tok(_render(tok, prompt), return_tensors="pt", add_special_tokens=False).to(device)
    # Built explicitly and passed per call: an unset field on transformers v5 is
    # refilled from the checkpoint's own generation config, which is worth points
    # and shows up as an arm that decoded a different number of tokens than another.
    cfg = GenerationConfig(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        max_new_tokens=budget,
        min_new_tokens=budget if force else None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, generation_config=cfg)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    produced = out[0, enc["input_ids"].shape[1] :]
    # The tripwire, asserted rather than eyeballed: two arms that decoded different
    # token counts produce a speedup that is a budget difference wearing a costume.
    if force and produced.numel() != budget:
        raise AssertionError(f"asked for {budget} tokens, got {produced.numel()}")
    return elapsed, tok.decode(produced, skip_special_tokens=True)


def _time_arm(model: Any, tok: Any, device: str, reps: int) -> dict[str, Any]:
    _generate(model, tok, PROMPTS[0], SHORT, device)  # warm the allocator and any autotune
    rows: list[dict[str, Any]] = []
    for prompt in PROMPTS:
        shorts = [_generate(model, tok, prompt, SHORT, device)[0] for _ in range(reps)]
        longs = [_generate(model, tok, prompt, LONG, device)[0] for _ in range(reps)]
        t_short, t_long = statistics.median(shorts), statistics.median(longs)
        text = _generate(model, tok, prompt, 160, device, force=False)[1]
        rows.append(
            {
                "prompt": prompt,
                "short_s": round(t_short, 4),
                "long_s": round(t_long, 4),
                # The slope: (LONG - SHORT) tokens bought (t_long - t_short) seconds,
                # and everything that is not decode cancelled.
                "decode_tok_s": round((LONG - SHORT) / (t_long - t_short), 2),
                "naive_tok_s": round(LONG / t_long, 2),
                "text": text,
            }
        )
    return {
        "prompt_style": "chat-template" if getattr(tok, "chat_template", None) else "raw",
        "rows": rows,
        "decode_tok_s": round(statistics.median([r["decode_tok_s"] for r in rows]), 2),
        "naive_tok_s": round(statistics.median([r["naive_tok_s"] for r in rows]), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="decode a real MoE through the grouped kernel")
    ap.add_argument("model")
    ap.add_argument("--arm", required=True, choices=("bf16", "eager", "dynquant"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="append the record to this JSON-lines file")
    args = ap.parse_args(argv)

    from dynquant.runtime import ops

    torch.cuda.reset_peak_memory_stats()
    record: dict[str, Any] = {
        "arm": args.arm,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        # Recorded, not assumed: an arm named `dynquant` on a build where the op is
        # missing silently becomes the loop and reports a speedup of 1.0x, which
        # reads as the kernel not helping rather than as the kernel not running.
        "has_grouped_gemv": bool(ops.has_grouped_gemv()),
    }

    tok, model = _load(args.model, torch.bfloat16)
    record["params"] = sum(p.numel() for p in model.parameters())
    if args.arm != "bf16":
        record.update(_pack(model, args.bits, args.group_size))
        record["bits"] = args.bits
    model.to(args.device)
    record.update(_set_dispatch(model, args.arm))
    record["peak_mib_loaded"] = round(_peak_mib(), 1)
    record["resident_mib_loaded"] = round(_live_mib(), 1)

    # Generation is measured against its own baseline, so the decode footprint is not
    # hidden underneath a load-time peak it never reaches.
    torch.cuda.reset_peak_memory_stats()
    record.update(_time_arm(model, tok, args.device, args.reps))
    record["peak_mib_generate"] = round(_peak_mib(), 1)
    record["peak_mib_total"] = max(record["peak_mib_loaded"], record["peak_mib_generate"])

    print(json.dumps(record, indent=2))
    if args.out:
        with Path(args.out).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
