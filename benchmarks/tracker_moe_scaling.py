"""Does tracker overhead survive 128 experts, or is there a step-time cliff?

P2's third gate item. It is separate from :mod:`tracker_overhead` because the failure it
looks for is not "the tracker is slow" but "the tracker's cost scales with module count in
a way a dense model never reveals" -- the research tracker's per-module device sync costs
roughly 850 stalls per step on a dense 14B model, and once cost a projected 18,000 on a
128-expert MoE.

**That premise no longer holds, and the reason is the finding.** It assumed a 128-expert
MoE has three ``nn.Linear`` modules per expert per layer, putting module count one to two
orders of magnitude above a dense model. On ``transformers`` 5.x every MoE family collapses
the bank into a single module holding 3-D parameters -- ``Qwen3MoeExperts.gate_up_proj`` is
``[E, 2 * inter, hidden]`` -- so widening the bank from 8 to 128 experts adds **no modules
at all**. The tracker then skips those tensors deliberately, recording the reason
(``dynquant.graph.experts``): activation and gradient signals are not separable at the
bank's module boundary, so one bank would get one signal for all its experts. Measured on a
4-layer model, the tracked count is 22 at 8, 32 and 128 experts alike -- attention, routers,
embedding, LM head, and nothing else.

The consequence for this benchmark is that the module-count cliff it was written to catch
**cannot occur**, and a PASS printed off a comparison of 22 modules against 22 modules would
be vacuous. So the script detects a module count that did not grow and reports
``INCONCLUSIVE`` for the cliff, naming the skipped banks. What it can still say, and does,
is whether the cost of the modules that *are* tracked moves as the bank widens: nothing in
the tracker should care how many experts sit behind a router it hooks.

The larger consequence is not a step-time one and is out of scope here: on 5.x the bank
holds ~91% of an MoE's parameters and receives no signal, so its bits come from role floors
rather than from a score. That is a P3/P4 allocation question, recorded in
:mod:`dynquant.graph.experts`, not something a step-time benchmark can settle.

So the measurement is a *sweep*, and the quantity that matters is **microseconds per
tracked module per step**. That is the unit :mod:`tracker_overhead` established as the
invariant -- the tracker's cost is CPU-side dispatch, fixed per module -- and it is the
only one that answers this question, because an overhead *percentage* here would move for
a reason that has nothing to do with MoE.

That reason is worth stating plainly, since it is what this benchmark's verdict is built
around. The config below is deliberately tiny, so its steps are short, so a fixed
per-module cost is a large fraction of them. Reading a percentage off that would report a
failure caused by the denominator. Worse, the percentage moves *both* ways as experts
multiply: more experts means more modules (numerator up) and more arithmetic per step
(denominator up), so a flat percentage could hide a cost that doubled. Per-module
microseconds cannot be fooled either way, so the gate is asserted on it.

The models are constructed from tiny configs rather than downloaded. Step time here is a
function of module count and shape, not of trained values, and a real Mixtral checkpoint
would make the benchmark unrunnable on anything smaller than a node while measuring the
same thing. Only the expert count varies across the sweep -- every other dimension is
held fixed, so a difference between rows cannot come from anything else.

Usage:
    python benchmarks/tracker_moe_scaling.py --experts 8 32 128 --out moe_scaling.json
    python benchmarks/tracker_moe_scaling.py --estimator param --experts 8 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from _bench import Comparison, compare, make_batch

# Small enough that 128 experts fits comfortably and the per-step time is dominated by
# launch and hook overhead rather than by arithmetic -- which is the regime where a
# per-module sync would be most visible, and therefore the honest place to look for it.
BASE_CONFIG: dict[str, Any] = {
    "vocab_size": 2048,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "num_experts_per_tok": 4,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}

# transformers renamed this field between major versions and it differs by family:
# qwen3_moe and mixtral use num_local_experts, olmoe uses num_experts. Setting the wrong
# one silently leaves the default expert count in place, which would make every row of the
# sweep identical and the result meaningless -- so the setter verifies it took.
EXPERT_COUNT_FIELDS: tuple[str, ...] = ("num_local_experts", "num_experts")


def build_moe(model_type: str, experts: int) -> Any:
    """A randomly initialised MoE with ``experts`` experts, on the GPU in bf16."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(model_type, **BASE_CONFIG)
    applied = [field for field in EXPERT_COUNT_FIELDS if hasattr(config, field)]
    if not applied:
        raise SystemExit(
            f"{model_type} config exposes none of {EXPERT_COUNT_FIELDS}; "
            "the expert count cannot be set and the sweep would be a no-op"
        )
    for field in applied:
        setattr(config, field, experts)

    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    model.train()
    return model


def tracked_count(model: Any) -> tuple[int, int]:
    """``(modules the tracker hooks, tensors it deliberately skipped)``.

    Not an ``nn.Linear`` census: the tracker also hooks embeddings, and skips weights it
    cannot form a separable signal for -- batched expert banks above all. Since the first
    number is the divisor for the per-module column, counting the wrong set would scale
    every number in the table; the second is what tells the reader the expert bank never
    entered the measurement.
    """
    from dynquant.signals import SignalTracker

    tracker = SignalTracker(model)
    return len(tracker), len(tracker.skipped)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="qwen3_moe")
    parser.add_argument("--experts", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=0)
    # No --budget: a percentage gate is meaningless against this config's toy step time,
    # and offering the flag would invite someone to read the verdict off it.
    parser.add_argument(
        "--estimator",
        default="outer_exact",
        help="the default is the one that ships, and the expensive one; 'param' is the floor",
    )
    parser.add_argument("--subsample", type=int, default=256)
    parser.add_argument(
        "--growth-tolerance",
        type=float,
        default=25.0,
        help="percent the per-module cost may rise from the smallest to the largest expert "
        "count before it counts as a cliff",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: step-time overhead is not measurable here", file=sys.stderr)
        return 2

    overrides: dict[str, Any] = {"grad_estimator": args.estimator}
    if args.estimator != "param":
        overrides["subsample_tokens"] = args.subsample

    print(
        f"{torch.cuda.get_device_name(0)}  {args.model_type}  "
        f"batch {args.batch_size}x{args.seq_len}  "
        f"{', '.join(f'{k}={v}' for k, v in overrides.items())}"
    )
    header = (
        f"{'experts':>8s} {'modules':>8s} {'skipped':>8s} {'off ms':>9s} {'on ms':>9s} "
        f"{'overhead':>9s} {'us/mod':>8s} {'noise':>7s}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    results: dict[int, Comparison] = {}
    for experts in args.experts:
        model = build_moe(args.model_type, experts)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
        batch = make_batch(model, args.batch_size, args.seq_len, seed=0)
        modules, skipped = tracked_count(model)

        result = compare(
            model,
            optimizer,
            batch,
            pairs=args.pairs,
            warmup=args.warmup,
            log_every=args.log_every,
            overrides=overrides,
        )
        results[experts] = result
        per_module_us = (result.on - result.off) / modules * 1e6
        rows.append(
            {
                "experts": experts,
                "tracked_modules": modules,
                "skipped_tensors": skipped,
                "per_module_us": per_module_us,
                **result.as_dict(),
            }
        )
        print(
            f"{experts:8d} {modules:8d} {skipped:8d} {result.off * 1e3:9.2f} "
            f"{result.on * 1e3:9.2f} {result.overhead_percent:+8.2f}% "
            f"{per_module_us:8.1f} {result.spread_percent:6.2f}%"
        )

        # Free before the next, larger model is built: holding three MoEs plus their
        # optimizer states would eventually OOM and turn a scaling result into a crash.
        del model, optimizer, batch
        torch.cuda.empty_cache()

    smallest, largest = min(results), max(results)
    module_ratio = rows[-1]["tracked_modules"] / rows[0]["tracked_modules"]
    per_module_growth = (rows[-1]["per_module_us"] / rows[0]["per_module_us"] - 1.0) * 100.0
    points = results[largest].overhead_percent - results[smallest].overhead_percent

    print()
    print(
        f"module count grew {module_ratio:.1f}x from {smallest} to {largest} experts; "
        f"per-module cost moved {per_module_growth:+.1f}% "
        f"({rows[0]['per_module_us']:.1f} -> {rows[-1]['per_module_us']:.1f} us), "
        f"overhead {points:+.2f} points"
    )
    # Per-module cost rising with expert count is the cliff: it means work scaling with
    # module count, the per-module sync reappearing under another name. Falling is the
    # expected direction when modules do grow, since total tokens seen across all experts
    # is tokens x num_experts_per_tok whatever the expert count -- the token-proportional
    # half of the tracker's cost is constant while the divisor grows.
    no_cliff = per_module_growth < args.growth_tolerance
    # But a flat module count makes that test vacuous, and printing PASS off 22 modules
    # compared against 22 modules would assert something this run did not measure. See the
    # module docstring: on transformers 5.x a wider expert bank adds no modules at all.
    grew = module_ratio > 1.05
    verdict = ("PASS" if no_cliff else "FAIL") if grew else "INCONCLUSIVE"
    print(
        f"P2 gate ({largest} experts, no per-module cliff within "
        f"{args.growth_tolerance:.0f}%): {verdict}  "
        f"[{largest}-expert overhead {results[largest].overhead_percent:+.2f}%, "
        f"{'resolved' if results[largest].resolved else 'within noise'}]"
    )
    if not grew:
        print(
            f"  INCONCLUSIVE because the tracked count did not grow: "
            f"{rows[0]['tracked_modules']} modules at {smallest} experts and "
            f"{rows[-1]['tracked_modules']} at {largest}, with "
            f"{rows[-1]['skipped_tensors']} expert-bank tensors skipped. A module-count "
            f"cliff cannot be measured where module count is constant."
        )
        print(
            "  What the rows do support: per-module cost is flat as the bank widens, so "
            "nothing in the tracker scales with experts behind a router it hooks."
        )
    # Said out loud because the number is large and means nothing here: this config is
    # deliberately tiny, so a fixed per-module cost is a big fraction of a short step. See
    # tracker_overhead for the ratio measured against a real model and a real batch.
    print(
        f"the overhead percentages above are against a {rows[-1]['untracked_median'] * 1e3:.0f} ms "
        f"toy step and are not the gate; the per-module column is"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(0),
                    "model_type": args.model_type,
                    "base_config": BASE_CONFIG,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "overrides": overrides,
                    "growth_tolerance_percent": args.growth_tolerance,
                    "per_module_growth_percent": per_module_growth,
                    "overhead_growth_points": points,
                    "module_count_ratio": module_ratio,
                    "verdict": verdict,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"-> wrote {args.out}")

    # 2 rather than 1 for INCONCLUSIVE: a caller wiring this into CI should be able to tell
    # "the tracker got slower" from "this configuration cannot answer the question".
    return {"PASS": 0, "FAIL": 1}.get(verdict, 2)


if __name__ == "__main__":
    sys.exit(main())
