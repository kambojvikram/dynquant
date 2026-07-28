"""Does the signal tracker cost less than 3% of step time on a real model?

This is P2's gate item, and it exists because the research code's claim of "zero additional
cost" does not survive its implementation: both hooks call ``float(...cpu().item())`` per
module per step, a full device sync each -- about 850 stalls per step on a dense 14B model.
Removing those was necessary and nowhere near sufficient, and the three years of wrong
guesses that followed are the reason this file carries numbers instead of adjectives.

**A percentage is the wrong unit for comparing runs**, because the ratio is a statement
about the *denominator* -- the user's model and batch size -- as much as about the tracker.
Every table here reports **microseconds per module per step** alongside it. That is what
makes rows comparable across models.

It is *not*, however, a constant, and the tempting shortcut is wrong. The cost has two
parts: one fixed per module, and one proportional to the tokens in the step (the saliency
read, below). So a bigger batch buys a larger denominator and a larger numerator at once,
and the gate does **not** reduce to a threshold on step time -- an earlier draft of this
docstring claimed it did. What actually happens is that the ratio falls toward the
token-proportional part's own share and flattens there:

=========  =========  ======  ========  ====================  =====================
model      batch      tokens  step      ``param``             ``outer_exact``
=========  =========  ======  ========  ====================  =====================
Qwen3-0.6B  2x2048      4096  232 ms    +3.58% (42.1 us)      +8.96% (105.3 us)
Qwen3-0.6B  4x2048      8192  422 ms    +2.63% (56.1 us)      +5.33% (114.1 us)
Qwen3-0.6B  8x2048     16384  795 ms    +2.31% (92.8 us)      +3.67% (148.3 us)
Qwen3-0.6B  16x2048*   32768  1920 ms   +1.52% (148.7 us)     **+2.30%** (222.9 us)
Qwen3-1.7B  4x2048      8192  752 ms    +1.96% (74.4 us)      +4.07% (155.1 us)
Qwen3-1.7B  6x2048     12288  1088 ms   +1.90% (104.6 us)     +3.13% (174.0 us)
=========  =========  ======  ========  ====================  =====================

\\* gradient checkpointing on, which is what makes that batch fit -- see below.

Octupling the tokens on Qwen3-0.6B moves ``outer_exact`` from 8.96% to 2.30% while its
per-module cost climbs 105 -> 223 us. The extrapolation from the first three rows put the 3%
crossing near 32k tokens/step and the fourth row lands there, so the model the gate names
does clear 3% on the default estimator -- but only at 32k tokens/step, which needs gradient
checkpointing to fit. ``param`` clears it from 8k tokens/step upward. The asymptote is ~1%.

Model size at *comparable step time* is the axis along which the per-module cost really is
flat, which is the sense in which the work is fixed per module:

=========  =======  =========  ========  =========  ==========  ================
model      modules  batch      step      TFLOP/s    ``param``   ``outer_exact``
=========  =======  =========  ========  =========  ==========  ================
Qwen3-0.6B  198     8x2048     795 ms    73.8       92.8 us     148.3 us
Qwen3-1.7B  198     4x2048     752 ms    112.5      74.4 us     155.1 us
Qwen3-4B    254     2x2048     864 ms    114.4      69.4 us     157.4 us
=========  =======  =========  ========  =========  ==========  ================

148, 155, 157 us across a 6.7x range of model size. Note what that table cannot do: every
row was given a different batch to hold step time near 800 ms, so its *percentages* fail
identically and say nothing. A sweep that holds the denominator constant cannot measure a
ratio whose free variable is the denominator -- which is why the batch series above exists.
Qwen3-0.6B is the pathological case for this gate either way: 198 modules against ~0.6
GFLOP/token, so a fixed per-module cost is maximally visible.

**Where the 151 us goes**, from :mod:`tracker_cost_ladder`, which disables one layer of the
tracker at a time:

=========================  =========  ==========
layer                      us/mod     share
=========================  =========  ==========
hook dispatch                    7.5   5%
saliency activation norm        75.6   50%
input Gram + bwd plumbing       18.7   12%
grad Gram + contraction         49.3   33%
=========================  =========  ==========

Half the cost is the saliency norm, and it is *not* dispatch and *not* removable. Summed
over every tracked module, reading each output activation once is ~16.3 GB per step on this
model -- the LM head's ``[8, 2048, 151936]`` is 5 GB of it by itself -- which is ~12 ms of
A100 bandwidth against 15 ms measured. The norm already runs at roughly 80% of achievable
bandwidth; the cost is the definition of the signal, not the implementation of it. Every
estimator mode pays it, which is why ``param`` cannot go below ~88 us/mod however much the
gradient path is optimised.

That is worth dwelling on, because three rounds of work went into the 7.5 us row. Dispatch
counts were cut from ~15.7 to 7.4 ATen calls per module by batching the saliency fold, then
the contraction, then sharing the input Gram between modules fed the same tensor -- and the
total moved 186 -> 148 us/mod, because the calls removed were cheap ones and the calls left
are the expensive ones. **A dispatch census tells you how many calls there are, never what
they cost**; :mod:`tracker_dispatch_count` counts and this file times, and neither substitutes
for the other. Two earlier diagnoses (device syncs, then the coherence cosine -- which is
disabled by default and so cannot have been in any of these measurements) were reached by
reasoning about counts and were both wrong.

The ``--subsample`` sweep separates the one term that *is* proportional to work from the
fixed ones. Only the Gram matrices scale with the token count, as ``T^2``:

=====  ========  =========
``T``  us/mod    overhead
=====  ========  =========
64        123.1  +3.05%
128       124.8  +3.07%
256       152.8  +3.75%
512       278.3  +6.81%
=====  ========  =========

Fitting ``fixed + c*T^2`` to the 256/512 pair puts ~42 us of Gram arithmetic at ``T=256``
and ~111 us in terms that do not move with ``T``. Below ``T=128`` the matmuls are
launch-bound and shrinking them further buys nothing.

TF32 is the third non-lever, and it is worth recording because the reasoning for it is
sound and the payoff still is not. ``gram`` upcasts to fp32, and fp32 matmuls run at 19.5
rather than 156 TFLOP/s on an A100 because torch disables TF32 by default -- while TF32's
concession is a 10-bit input mantissa, and these inputs are bf16 activations carrying 8, so
for bf16 training it cannot discard precision the data ever had. Measured directly, though,
the Gram shapes are occupancy-bound and not FLOP-bound: ``[256, 1024] @ [1024, 256]`` gains
1.11x, ``[256, 3072]`` gains 1.97x, and every shape floors near 18 us. Only the LM head's
grad Gram is wide enough to care -- ``[256, 151936]``, 1204 us -> 306 us, one module worth
1.2 ms of the step. Whole-tracker effect is ~13%, which does not buy a PASS anywhere and
would cost a process-global ``allow_tf32`` write issued from a backward hook that can run on
an autograd worker thread. There is no scoped API for it as of torch 2.13. Not shipped.

**Gradient checkpointing is the one real lever, and for two separate reasons.** It was
also a correctness bug: the recompute pass fires forward hooks again, so before the
``_in_backward`` guard a module inside a checkpointed block observed each micro-batch
*twice* -- ``forward_calls=8`` over 4 steps against ``lm_head``'s 4 -- squaring the EMA decay
for some modules and not others. Which ones depends on the checkpointing implementation: at
the transformers 5.x ``use_reentrant=False`` default, 168 of 198 modules doubled and every
``mlp.down_proj`` escaped, so ``up_proj`` and ``down_proj`` in one MLP ended up on different
horizons. Finding 11 in ``docs/legacy-audit.md`` has the detail; the
research code enables checkpointing unconditionally, so both shipped stats files carry it.
Fixing it took the checkpointed per-module cost from 221.1 back to parity with the
uncheckpointed path -- two paired runs read 152.1 / 157.0 and 154.4 / 154.2 us, so the
remaining difference is run-to-run variation and not a residue of the replay -- and left the
non-checkpointed path itself unmoved (150.6 -> 152.1, inside a 0.12% noise floor). The
post-fix ``forward_calls`` histogram is ``{3: 198}`` in both arms over 3 steps: identical,
which is what "the replay is not observed" looks like in the artifact.

What remains after the fix is a genuine amortization rather than a trick, and the two
effects should not be conflated. At the *same* 16k tokens/step, checkpointing improves
``outer_exact`` from +3.79% to +3.14% purely by lengthening the step -- 794 -> 988 ms, 73.8
-> 59.3 model TFLOP/s on the 6ND estimate, which is exactly the denominator inflation
``--min-tflops`` exists to flag, and the 6ND figure understates real utilisation because it
does not bill the extra forward. The part that is real is that freeing the activation memory
lets 32k tokens/step fit at all, and since only one term of the tracker's cost scales with
tokens, doubling them takes the ratio from +3.14% to +2.30% while per-module cost rises only
157 -> 223 us. That is the two-part cost model above making a prediction and being right.

**Two things that look like levers and are not.** A larger batch is capped, once activations
are out of the way, by the vocabulary-sized logits tensor, and checkpointing does nothing
about that because the logits are not inside a checkpointed block: at 16x2048,
``16*2048*151936`` in fp32 is 19.9 GB and an uncheckpointed run dies asking for 18.55 GiB;
at 24x2048 a *checkpointed* one dies asking for 27.82 GiB. So 16k tokens/step is the ceiling
on Qwen3-0.6B on an 80 GB A100 without checkpointing and 32k with it -- and 32k is where the
table above ends for that reason, not by choice.

And PEFT LoRA, which the gate's wording suggests, makes the step *slower* on a model this
small: ``r=16`` with ``target_modules="all-linear"`` measured 1026 ms untracked against 795 ms
for full fine-tuning at 8x2048 (38.1 vs 73.7 TFLOP/s), because two extra small matmuls per
module cost more than the base-weight gradients they skip. So the LoRA arm clears the gate
more easily, and that is the reason to distrust it rather than to quote it: at 16x2048 with
checkpointing it gives ``outer_exact`` **+1.68%** (229.5 us/mod) and ``param`` +1.15% (158.3
us/mod) against a 2697 ms step -- but 29.0 model TFLOP/s on the ``4ND`` estimate, less than
half the full-finetune arm's 61.0. The per-module cost is what shows nothing was really
gained: 229.5 us under LoRA against 222.9 us full-finetune at the same tokens. Same tracker,
same work, longer step. Treat +2.30% as the gate evidence and +1.68% as its corroboration.
Note also that ``--lora`` switches the throughput estimate to ``4ND``: a frozen base layer
computes an input gradient but no weight gradient, and leaving the factor at 6 would credit
the step with a third more work than it did.

The measurement design lives in :mod:`_bench`; read its docstring before trusting a number
from here. Companions: :mod:`tracker_cost_ladder` attributes the per-module cost to a layer
of the tracker, :mod:`tracker_dispatch_count` counts ATen calls, and
:mod:`tracker_moe_scaling` asks whether the per-module cost survives 128 experts.

``param`` is included in every sweep as the floor. It reads ``p.grad`` through one fused
``torch._foreach_norm`` and does no per-module linear algebra, so whatever survives in that
mode belongs to the hooks and the saliency read, and no change of estimator can fix it.

One figure reported here is not an overhead: **achieved model TFLOP/s**. A ratio is only as
trustworthy as its denominator, and every way of getting a slow step -- a kernel on a
reference fallback, gradient checkpointing left on, a shared GPU -- shrinks the percentage
without changing anything about the tracker, because the tracker's cost is fixed per module
per step. `Qwen3.5-2B` is precisely where this bites: it is a hybrid, and its
linear-attention layers fall back to a Python-level recurrence unless
`flash-linear-attention` is installed, which inflates the step by roughly an order of
magnitude. Below ``--min-tflops`` the script says so and withholds a PASS -- a FAIL under
those conditions still stands, since the true overhead can only be larger.

Usage:
    python benchmarks/tracker_overhead.py --out overhead.json
    python benchmarks/tracker_overhead.py \\
        --estimator param outer_exact --subsample 64 256 --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from _bench import achieved_tflops, compare, make_batch

DEFAULT_MODEL = "Qwen/Qwen3.5-2B-Base"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--pairs",
        type=int,
        default=15,
        help="interleaved (untracked, tracked) step pairs to time",
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--budget", type=float, default=3.0, help="gate, in percent")
    parser.add_argument(
        "--min-tflops",
        type=float,
        default=25.0,
        help="warn below this achieved TFLOP/s: a low figure means the step is slow for the "
        "hardware, so the overhead ratio has an inflated denominator and reads optimistic",
    )
    parser.add_argument("--out", default=None, help="write the raw timings here")
    parser.add_argument(
        "--estimator",
        nargs="+",
        default=["outer_exact"],
        help="estimator modes to sweep; 'param' is the hook-only floor",
    )
    parser.add_argument(
        "--subsample",
        nargs="+",
        type=int,
        default=[256],
        help="subsample_tokens values to sweep (ignored by the param estimator)",
    )
    parser.add_argument(
        "--lora",
        type=int,
        default=0,
        help="LoRA rank to wrap the model in; 0 full-finetunes. The P2 gate names a LoRA "
        "run, and it is not the easier test at fixed batch -- frozen base weights skip "
        "their weight gradient, so the step gets faster and the ratio worse. What it does "
        "change is which batch fits: AdamW state for every parameter is what caps the "
        "full-finetune arm, and the tracker's cost per module is the same either way.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="enable gradient checkpointing, as a real fine-tune of anything large does. "
        "Read the reported TFLOP/s alongside the percentage: this lengthens the step by "
        "re-running the forward, so it flatters the ratio for a reason that is not the "
        "tracker getting cheaper. What it legitimately buys is a batch that would not "
        "otherwise fit, and the fixed per-module cost then amortizes over more tokens.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: step-time overhead is not measurable here", file=sys.stderr)
        return 2

    from transformers import AutoModelForCausalLM

    print(f"loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    # Caching is for generation and is incompatible with a backward pass.
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()

    # Every parameter of a dense model runs on every token, so total is also active. An MoE
    # model would need the routed subset instead -- that is why this is computed here and
    # passed in, rather than inside achieved_tflops. Read before the adapter is attached, so
    # the LoRA arm is billed for the same base model as the full-finetune arm.
    active_params = sum(p.numel() for p in model.parameters())
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=args.lora, lora_alpha=2 * args.lora, target_modules="all-linear"),
        )
        model.train()

    # A near-zero learning rate keeps the optimizer's arithmetic and memory traffic
    # honest while stopping the weights from drifting into a regime with different
    # numerics over the run.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-9)
    batch = make_batch(model, args.batch_size, args.seq_len, seed=0)
    # The tracker's own count, not a Linear census: under LoRA every wrapped module gains
    # two more Linears, and counting those would divide the cost by three times too many
    # modules. This is also the number the per-module column has to be keyed to, since it
    # is what the hooks actually attach to.
    from dynquant.signals import SignalTracker

    tracked_modules = len(SignalTracker(model))

    print(
        f"{torch.cuda.get_device_name(0)}  batch {args.batch_size}x{args.seq_len}  "
        f"{tracked_modules} tracked modules  "
        f"{'lora r=' + str(args.lora) if args.lora else 'full finetune'}  "
        f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable  "
        f"grad-ckpt {'on' if args.gradient_checkpointing else 'off'}  "
        f"warmup {args.warmup}  pairs {args.pairs}",
        flush=True,
    )
    # The param estimator never reads subsample_tokens, so sweeping it there would
    # report the same configuration several times under different names.
    settings: list[dict[str, Any]] = []
    for estimator in args.estimator:
        if estimator == "param":
            settings.append({"grad_estimator": estimator})
            continue
        settings.extend(
            {"grad_estimator": estimator, "subsample_tokens": n} for n in args.subsample
        )

    runs: list[dict[str, Any]] = []
    for setting in settings:
        name = ", ".join(f"{k}={v}" for k, v in setting.items())
        print(f"\n=== {name} ===", flush=True)

        def report(index: int, off: float, on: float) -> None:
            print(
                f"  pair {index + 1}/{args.pairs}  off {off * 1e3:7.1f} ms   on {on * 1e3:7.1f} ms",
                flush=True,
            )

        result = compare(
            model,
            optimizer,
            batch,
            pairs=args.pairs,
            warmup=args.warmup,
            log_every=args.log_every,
            overrides=setting,
            on_pair=report,
        )
        print(f"  untracked median {result.off * 1e3:8.2f} ms", flush=True)
        print(f"  tracked   median {result.on * 1e3:8.2f} ms", flush=True)
        print(
            f"  overhead  {result.overhead_percent:+.2f}%  "
            f"(untracked spread {result.spread_percent:.2f}% 1sd, "
            f"{'resolved' if result.resolved else 'within noise'})",
            flush=True,
        )
        runs.append({"setting": setting, **result.as_dict()})

    print()
    header = (
        f"{'setting':44s} {'untracked':>10s} {'tracked':>10s} {'overhead':>9s} "
        f"{'us/mod':>8s} {'gate':>6s}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        label = ", ".join(f"{k}={v}" for k, v in run["setting"].items())
        passed = run["overhead_percent"] < args.budget
        # The ratio's denominator is the caller's batch size; this column is not.
        per_module_us = (
            (run["tracked_median"] - run["untracked_median"]) / tracked_modules * 1e6
            if tracked_modules
            else float("nan")
        )
        run["per_module_us"] = per_module_us
        print(
            f"{label:44s} {run['untracked_median'] * 1e3:9.2f}ms "
            f"{run['tracked_median'] * 1e3:9.2f}ms {run['overhead_percent']:+8.2f}% "
            f"{per_module_us:8.1f} {'PASS' if passed else 'FAIL':>6s}"
        )

    # The cross-setting consistency check the interleaving cannot provide on its own.
    # Every untracked arm did identical work, so a spread here is machine drift and it
    # bounds how much of any overhead difference above is real.
    baselines = [run["untracked_median"] for run in runs]
    drift = (max(baselines) / min(baselines) - 1.0) * 100.0
    print(f"\nuntracked baselines vary by {drift:.2f}% across settings (drift floor)")

    # Is the denominator representative? See _bench.achieved_tflops for why a percentage
    # alone cannot answer that. Measured against the fastest baseline, so the check is
    # conservative: it flags a slow step only when even the best arm was slow.
    tokens = args.batch_size * args.seq_len
    # 4ND under LoRA rather than 6ND: a frozen base layer computes an input gradient but no
    # weight gradient. Using 6 there would credit the step with a third more work than it
    # did and mask exactly the slow denominator this check exists to expose.
    factor = 4.0 if args.lora else 6.0
    tflops = achieved_tflops(
        min(baselines), active_params=active_params, tokens=tokens, flops_per_param_token=factor
    )
    suspect = tflops < args.min_tflops
    print(
        f"untracked step achieves {tflops:.1f} model TFLOP/s "
        f"({min(baselines) * 1e3:.1f} ms for {tokens} tokens, {factor:.0f}ND estimate)"
    )
    if suspect:
        print(
            f"  WARNING: below --min-tflops {args.min_tflops:.0f}. This step is slower than the\n"
            f"  hardware should manage, so every overhead above is a fraction of an inflated\n"
            f"  denominator and reads optimistic. Look for a kernel falling back to a reference\n"
            f"  implementation, gradient checkpointing left on, or a shared GPU.",
            file=sys.stderr,
        )

    worst = max(run["overhead_percent"] for run in runs)
    verdict = "PASS" if worst < args.budget else "FAIL"
    print(f"P2 gate (<{args.budget:.0f}% step time): {verdict} at the worst setting")
    if suspect and verdict == "PASS":
        # A FAIL on an inflated denominator is still a FAIL -- the real overhead can only be
        # larger. A PASS is the case that needs withholding.
        print("  -- but the denominator is suspect, so this PASS does not settle the gate.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "device": torch.cuda.get_device_name(0),
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "lora_rank": args.lora,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "tracked_modules": tracked_modules,
                    "active_params": active_params,
                    "log_every": args.log_every,
                    "budget_percent": args.budget,
                    "baseline_drift_percent": drift,
                    "achieved_tflops": tflops,
                    "min_tflops": args.min_tflops,
                    "denominator_suspect": suspect,
                    "verdict": verdict,
                    "runs": runs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"-> wrote {args.out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
