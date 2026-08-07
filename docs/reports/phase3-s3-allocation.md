# S3: what the allocator and the signal are each worth, at two budgets

**Run 2026-08-07** on Ministral-8B-Instruct-2410 fine-tuned on `allenai/tulu-3-sft-mixture`
(S2 arm 2, 11.23 h, final train loss 0.6335). Eight quantized arms across two budgets, four
benchmarks each, plus a bf16 ceiling: **36 eval cells**, all committed under
[`experiments/phase3/s3_allocation/ministral-8b/records/`](../../experiments/phase3/s3_allocation/ministral-8b/records/)
with per-item `hits`. Every number below is reproduced by
[`s3_table.py`](../../experiments/phase3/s3_allocation/ministral-8b/s3_table.py) from those
files, without the GPU box.

## The four arms

They differ only in how bits are assigned. Each anchor's four arms land on **exactly** the
same byte total — 3 257 925 632 B at 3.25, 4 260 364 288 B at 4.25, widest drift +0 B — so
nothing here is a size comparison in disguise.

| arm | allocator | priced from |
|---|---|---|
| `rtn` | uniform | nothing — every module at the target width |
| `rank` | rank-product | the published score: `Rank(plasticity) × Rank(saliency)` |
| `shuf` | sensitivity | the signal **permuted within role** — the control |
| `dq` | sensitivity | the measured signal |

`dq` − `rtn` is what the method is worth. `shuf` − `rtn` is what the *allocator* is worth with
the signal destroyed but its distribution preserved. `dq` − `shuf` is what the **signal** adds
on top, and it is the only quantity in this campaign that is about the fine-tune hook rather
than about knapsack mechanics.

## The result

| arm | GSM8K | IFEval | HumanEval | MBPP | mean Δ |
|---|---|---|---|---|---|
| bf16 | 78.92 % | 61.18 % | 75.00 % | 54.20 % | — |
| rtn3 | 64.14 (−14.78) | 48.43 (−12.75) | 45.73 (−29.27) | 42.80 (−11.40) | −17.05 |
| rank3 | 63.08 (−15.85) | 47.50 (−13.68) | **59.76** (−15.24) | 43.00 (−11.20) | −13.99 |
| shuf3 | 61.03 (−17.89) | 52.87 (−8.32) | 51.83 (−23.17) | 43.00 (−11.20) | −15.15 |
| **dq3** | **65.81** (−13.12) | **53.23** (−7.95) | 57.32 (−17.68) | 42.00 (−12.20) | **−12.74** |
| rtn4 | 73.77 (−5.16) | 56.93 (−4.25) | 64.02 (−10.98) | 50.00 (−4.20) | −6.15 |
| rank4 | 73.09 (−5.84) | **57.67** (−3.51) | 61.59 (−13.41) | 51.40 (−2.80) | −6.39 |
| shuf4 | **74.45** (−4.47) | 56.93 (−4.25) | 62.20 (−12.80) | 50.00 (−4.20) | −6.43 |
| **dq4** | 74.07 (−4.85) | 56.56 (−4.62) | **65.24** (−9.76) | **51.80** (−2.40) | **−5.41** |

`dq` has the best mean at both budgets. That is where the agreement between the two anchors
ends.

## At 3.25 bits the method works; at 4.25 nothing does

Mean over the four tasks, from the exact McNemar tables:

| | 3.25 bits | 4.25 bits |
|---|---|---|
| `dq` − `rtn` | **+4.31** | +0.74 |
| `dq` − `rank` | +1.26 | +0.98 |
| `dq` − `shuf` | **+2.41** | +1.03 |
| `shuf` − `rtn` | +1.91 | −0.29 |
| `rank` − `rtn` | +3.06 | −0.25 |

At 4.25 bits every one of the sixteen comparisons has p between 0.22 and 1.00. Not one arm is
distinguishable from any other, in either direction. The uniform baseline is already only 6.15
points off bf16 and there is nothing left for an allocator to buy — every role floor is
affordable at that budget, all three allocating arms report **zero floor violations**, and the
maps converge on the same widths for everything that matters. The discordant counts confirm it
is genuine inertness and not a power problem: HumanEval shows 20 flipped problems between
`rank4` and `rtn4` against 39 between `rank3` and `rtn3` — at the higher budget the two
allocators are producing nearly the same model.

This is [§11](README.md#11-phase-3--s2-arm-2-the-tensor-ranked-against-nobody)'s prediction
confirmed from the eval side. **The allocator earns its keep exactly where the role floors stop
being affordable**, and 4.25 bits is not that regime for this model.

## The signal is 56 % of the margin here — and 12 % on the last model

The 3.25-bit margin over uniform decomposes cleanly:

```
3.25 bits:  dq - rtn = +4.31  =  allocator +1.91  +  signal +2.41   (signal is 56% of it)
4.25 bits:  dq - rtn = +0.74  =  allocator -0.29  +  signal +1.03   (not decomposable)
```

The same decomposition at the same anchor and the same **module** granularity on
Qwen3.5-2B/CaseHOLD gave 22.62 allocator against 3.16 signal — the signal was **12 %**. Here it
is more than half, and the allocator term is the small one.

Both numbers are real; what they show is that the split is a property of the *campaign*, not of
the method. On Qwen the uniform baseline was catastrophically bad and almost any sane
reallocation recovered most of the gap, so the allocator dominated. Here uniform assignment is
merely mediocre, the reallocation head-room is 4.31 points rather than 25.78, and within that
smaller margin the question of *which* modules to protect is proportionally more of the answer.
The honest statement is the conditional one: **the signal's share of the margin grows as the
allocator's structural advantage shrinks**, and any single-campaign figure for it should be
quoted with its model and dataset attached. The 12 % figure is hereby scoped, not retracted.

The 4.25-bit row is left undecomposed on purpose. `dq` − `rtn` is +0.74 with every p above
0.31; the allocator term lands at −0.29, so a naive division prints *"the signal is 139 % of
the margin"*, which reads as a finding and is an artefact of dividing by noise. `s3_table.py`
refuses to compute the share unless at least one `dq` − `rtn` task comparison clears p < 0.05.

## Two of forty-eight comparisons survive correction

Forty-eight paired tests are computed here. At Bonferroni (p < 0.00104) exactly two survive:

| comparison | Δ | 95 % interval | p |
|---|---|---|---|
| `rank3` − `rtn3`, HumanEval | +14.02 | [+6.88, +21.17] | 0.00029 |
| `dq3` − `shuf3`, GSM8K | **+4.78** | [+2.51, +7.04] | 0.000048 |

The second is the campaign's central claim and it is the one that survives best: on GSM8K, at
identical bytes and with the same allocator, **priced from the measured signal beats priced
from a within-role permutation of it**. The control is doing real work — `shuf3` is not a
crippled arm, it beats uniform by +1.91 on average and beats every arm on IFEval among the
non-`dq` three — and `dq3` still separates from it decisively on the task with the most items.

Everything else is suggestive at best. `dq3` − `rtn3` on HumanEval is +11.59 at p = 0.0034,
which is striking and does *not* clear the corrected threshold. `dq3` − `rank3` on IFEval is
+5.73 at p = 0.011, likewise. They are reported because they are the record, not because they
are established.

## What does not work, stated plainly

**The rank-product score is a coin flip.** At 3.25 it wins HumanEval by +14.02 — the largest
single effect in the campaign — and *loses* GSM8K by 1.06 and IFEval by 0.92; at 4.25 its mean
is −0.25 with nothing significant. It has no consistent direction. That is the fourth
independent campaign in which the published score fails to be reliably better than doing
nothing, after it lost outright by 2.03 points on Qwen3.5-2B.

**MBPP separates nothing, anywhere.** All four 3.25-bit arms land in 42.00–43.00, and all four
4.25-bit arms in 50.00–51.80, with discordant counts split almost evenly every time (49–93
flips, never more than 58/42). It reads quantization damage — 11.40 points at 3.25 — but not
allocation. It should be kept as a damage sentinel and dropped from any headline that claims an
allocator difference.

**No arm wins every task at 3.25.** `dq3` takes GSM8K and IFEval, `rank3` takes HumanEval,
`rtn3` nominally takes MBPP. Reporting a single winner per budget hides that, so the mean is
quoted with the per-task table beside it and never instead of it.

## Where the damage concentrates

HumanEval is the task 3.25-bit quantization hurts most (−29.27 for uniform, against −11.40 on
MBPP) and also the task where reallocation recovers most (+14.02 for `rank3`, +11.59 for
`dq3`). Two code benchmarks, the same model, the same bits, and one is nearly three times as
sensitive as the other. Whatever HumanEval depends on is concentrated in modules that a uniform
map underserves and a score-driven map protects — which is the mechanism the method claims,
observed on the one task where it is unmistakable.

## Method notes

- **Paired throughout.** Each cell stores per-item `hits`, so every comparison is an exact
  binomial McNemar test on the discordant pairs, not two independent proportions. Exact rather
  than chi-square because the interesting comparisons are the close ones and the approximation
  is unreliable there — `rank3` − `rtn3` on HumanEval rides on 39 flips split 31/8, which is
  precisely that regime.
- **Matched bytes, not matched nominal width.** All four arms at an anchor are byte-identical,
  asserted in `tests/test_s3_allocation_arms.py` before any weight was written.
- **Arm directories hold dequantized bf16 weights** (~14.94 GiB each) so vLLM loads them as
  ordinary checkpoints; the quantized size is the manifest's. An arm's directory size is
  therefore the same at both anchors and says nothing about its width.
- **The maps were reused, and that was verified rather than assumed** — see
  [phase3-s3-reuse-guard.md](phase3-s3-reuse-guard.md). 7 of 7 reused, both stats variants
  unchanged, and the seven on the box sha256-identical to the seven committed.
- **The two anchors ran sequentially, gated on artifacts.** Each arm is ~15 GiB and four is 60
  GiB against 64 GiB free, so anchor 4 was chained behind anchor 3 — waiting not for the
  process to exit but for all 20 cells to exist and the disk to be reclaimed, because a crash
  and a clean finish look identical from outside.
