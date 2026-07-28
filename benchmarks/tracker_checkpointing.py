"""Does gradient checkpointing make the tracker observe every module twice?

Every other benchmark in this directory calls ``gradient_checkpointing_disable()``, and
real fine-tunes of anything large do the opposite. That gap matters twice over:

*Correctness.* ``torch.utils.checkpoint`` replays a layer's forward during backward to
rebuild the activations it did not save. Module forward hooks are not suppressed on the
replay, so :meth:`SignalTracker._observe_forward` plausibly fires twice per module per
micro-batch -- once on the real forward, once on the recompute -- against *identical*
data. Two EMA updates with the same value do not move the fixed point, but they halve the
effective memory (beta_eff = beta^2), so the saliency signal would silently be a
shorter-horizon average than the config asks for. ``forward_calls`` lands in the stats
file, so the question is decidable by reading it rather than by reasoning about torch
internals: 2x the calls at the same step count is double observation. Reported as a
histogram rather than a ratio, because the affected set is not "everything in a block" --
see the note on the ``forward_call_histogram`` row.

*The step-time gate.* Checkpointing trades memory for a second forward pass. If the
tracker's forward work runs twice while the model's forward also runs twice, the overhead
ratio is roughly preserved; if the tracker's *backward* work is unaffected, the ratio
improves. Either way the number the P2 gate is judged on changes, and no run so far has
measured it.

This benchmark reports both from the same pair of runs. It does not decide what the right
semantics *are* -- if the count doubles, whether to deduplicate is a question about what
the saliency signal is defined to average over, and it is flagged, not fixed here.

Usage:
    python benchmarks/tracker_checkpointing.py --model Qwen/Qwen3-0.6B --batch-size 8 \
        --seq-len 2048 --out checkpointing.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from _bench import achieved_tflops, compare, make_batch

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def build(model_name: str, *, checkpointing: bool) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    if checkpointing:
        # use_reentrant=False is the non-deprecated implementation and what a current
        # Trainer defaults to; the reentrant one replays under a nested autograd call and
        # would be a different experiment.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    return model


def observe(model: Any, batch: dict[str, Any], steps: int, overrides: dict[str, Any]) -> Any:
    """Run ``steps`` optimizer steps under the tracker and return the stats file.

    ``optimizer`` is handed to ``track_signals`` because the Welford fold happens on the
    pre-optimizer-step hook; without it the gradient half never advances and only the
    saliency counts here would be meaningful.
    """
    from dynquant.signals.context import track_signals

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    with track_signals(model, out=None, optimizer=optimizer, log_every=0, **overrides) as tracker:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model(**batch, labels=batch["input_ids"]).loss.backward()
            optimizer.step()
        return tracker.snapshot()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--subsample", type=int, default=256)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument("--steps", type=int, default=3, help="steps for the count check")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--budget", type=float, default=3.0, help="gate, in percent")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: step-time attribution is not measurable here", file=sys.stderr)
        return 2

    overrides = {"grad_estimator": args.estimator, "subsample_tokens": args.subsample}
    print(
        f"{torch.cuda.get_device_name(0)}  {args.model}  batch {args.batch_size}x{args.seq_len}  "
        f"grad_estimator={args.estimator}, subsample_tokens={args.subsample}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for checkpointing in (False, True):
        label = "on" if checkpointing else "off"
        print(f"\n=== gradient checkpointing {label} ===", flush=True)
        model = build(args.model, checkpointing=checkpointing)

        batch = make_batch(model, args.batch_size, args.seq_len, seed=0)
        stats = observe(model, batch, args.steps, overrides)
        # max, not mean: a module outside the checkpointed decoder layers (embeddings, the
        # LM head) never replays, so an average would blend two populations and understate
        # the doubling. The max is the worst affected module.
        calls = [layer.forward_calls for layer in stats.layers.values()]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
        result = compare(
            model,
            optimizer,
            batch,
            pairs=args.pairs,
            warmup=args.warmup,
            log_every=0,
            overrides=overrides,
        )
        modules = len(calls)
        rows.append(
            {
                "checkpointing": checkpointing,
                "steps": args.steps,
                "modules": modules,
                "max_forward_calls": max(calls),
                "min_forward_calls": min(calls),
                # The full distribution, because min/max hides the shape. Non-reentrant
                # checkpointing stops recomputing once the saved tensors it needs exist, so
                # the last op of a block never replays: on Qwen3-0.6B that spared all 28
                # mlp.down_proj and left 168 of 198 modules doubled. A run that reported only
                # "3 to 6" would look like a clean two-population split, which it is not.
                "forward_call_histogram": {
                    str(n): calls.count(n) for n in sorted(set(calls), reverse=True)
                },
                "calls_per_step": max(calls) / args.steps,
                "per_module_us": (result.on - result.off) / modules * 1e6,
                "tflops": achieved_tflops(
                    result.off,
                    active_params=sum(p.numel() for p in model.parameters()),
                    tokens=args.batch_size * args.seq_len,
                ),
                **result.as_dict(),
            }
        )
        del model, optimizer, batch
        torch.cuda.empty_cache()

    print()
    header = (
        f"{'checkpointing':14s} {'calls/step':>11s} {'off ms':>9s} {'on ms':>9s} "
        f"{'overhead':>9s} {'us/mod':>8s} {'TFLOP/s':>8s} {'noise':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{('on' if row['checkpointing'] else 'off'):14s} {row['calls_per_step']:11.2f} "
            f"{row['untracked_median'] * 1e3:8.2f}ms {row['tracked_median'] * 1e3:8.2f}ms "
            f"{row['overhead_percent']:+8.2f}% {row['per_module_us']:8.1f} "
            f"{row['tflops']:8.1f} {row['untracked_spread_percent']:6.2f}%"
        )

    off, on = rows
    ratio = on["calls_per_step"] / max(off["calls_per_step"], 1e-9)
    print()
    if ratio > 1.05:
        print(
            f"DOUBLE-OBSERVED: {on['calls_per_step']:.2f} forward calls/step under "
            f"checkpointing vs {off['calls_per_step']:.2f} without ({ratio:.2f}x). The "
            f"recompute pass fires the hook again on identical data, so the saliency EMA "
            f"advances twice per micro-batch and its effective beta is beta^2."
        )
    else:
        print(
            f"no double observation: {on['calls_per_step']:.2f} vs "
            f"{off['calls_per_step']:.2f} forward calls/step ({ratio:.2f}x)"
        )
    # Printed either way. Partial doubling is the expected shape when it happens -- modules
    # outside a checkpointed block never replay, and under non-reentrant checkpointing neither
    # does the last op inside one -- so a max/min pair would read as a clean two-population
    # split that it is not. And when nothing doubled, two identical histograms are the
    # evidence for that, which a ratio of 1.00 states less directly.
    print(
        f"  forward_calls over {args.steps} steps, off {off['forward_call_histogram']} -> "
        f"on {on['forward_call_histogram']} (calls: modules)"
    )

    verdict = "PASS" if on["overhead_percent"] < args.budget else "FAIL"
    print(
        f"P2 gate under checkpointing ({args.budget:.0f}% of step time): {verdict}  "
        f"[{on['overhead_percent']:+.2f}%, {off['overhead_percent']:+.2f}% without]"
    )
    # Checkpointing inflates the denominator by design. If the ratio improved only because
    # the model got slower, that is a real number for a real training config -- but it is
    # not evidence the tracker got cheaper, and the us/mod column is what shows the
    # difference.
    print(
        f"the denominator moved {off['untracked_median'] * 1e3:.0f} -> "
        f"{on['untracked_median'] * 1e3:.0f} ms ({off['tflops']:.1f} -> {on['tflops']:.1f} "
        f"model TFLOP/s on the 6ND estimate, which does not bill the recompute); compare "
        f"us/mod ({off['per_module_us']:.1f} -> {on['per_module_us']:.1f}) to see whether the "
        f"tracker itself changed cost"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(0),
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "grad_estimator": args.estimator,
                    "subsample_tokens": args.subsample,
                    "budget_percent": args.budget,
                    "calls_ratio": ratio,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"-> wrote {args.out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
