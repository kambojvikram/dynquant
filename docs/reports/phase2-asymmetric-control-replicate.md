# Phase 2's held claim, settled: the asymmetric GPTQ control

Phase 2 ([`dynquant-phase2-beating-gptq-3bit.tex`](dynquant-phase2-beating-gptq-3bit.tex)) reported
`p2_rb_agg` beating `gptq_3b_head` by +1.54 points at 4.5% fewer bytes, then held the claim: the
GPTQ arm had been fitted **symmetric** while every DynQuant arm on the panel was asymmetric, and
on Mistral-7B/text2sql that one knob alone was worth 69.4 points
([`byte-accounting-zero-point.md`](byte-accounting-zero-point.md)). The report said the claim was
"unattributed rather than wrong" until an asymmetric `gptq_3b_head` was run, and named the byte
figure it expected: 741,475,927 B.

The original checkpoint no longer exists — it lived on a box that has since been recycled — so
this is a **replicate**, not a re-measurement of the same weights: same recipe
(`/workspace/p2replicate.sh`, LoRA r=32, lr 1e-4, 2 epochs, Qwen3.5-2B-Base/CaseHOLD, n=5,314),
same driver, a fresh fine-tune, on torch 2.12.0+cu130 / transformers 5.10.1 (the original panel
used torch 2.13 / transformers 5.14). Absolute accuracy numbers will not match the destroyed
panel's to the decimal; the comparison that matters is arm-vs-arm inside this run, all four arms
scored on the same 5,314 items in the same order via stored per-item `hits`, same as every other
paired result in this project.

## The driver bug found getting here

`p2replicate.sh`'s freshness guard used bash's `-ot`/`-nt`, which compare mtimes at nanosecond
resolution. The training callback writes `dynquant_moments.safetensors` a fraction of a second
before the trainer finalizes the checkpoint's `config.json`, in the same save sequence — both
land in the same whole second, but the callback's file is nanoseconds older, so `[ "$MOM" -ot
"$FT/config.json" ]` read true and the driver exited immediately after an 8-hour fine-tune
completed, one line after printing `STEP finetune OK`. Fixed by comparing whole seconds
(`stat -c %Y`) instead of bash's sub-second test. Same defect class as
[[resume-guards-staple-stale-artifacts]]: the guard's stated reason for firing ("this predates the
checkpoint it describes") was never actually true — it was a same-run artifact a fraction of a
second younger than its sibling.

## The four-arm table

| arm | acc % | correct/total | bytes | avg bits |
|---|---:|---:|---:|---:|
| `p2_rb_agg` (DynQuant) | **89.57** | 4760/5314 | 708,087,552 | 3.0102 |
| `gptq_3b_head_asym` (GPTQ, asymmetric — the control) | 88.45 | 4700/5314 | 741,486,130 | 3.1522 |
| `gptq_3b_head` (GPTQ, symmetric) | 87.67 | 4659/5314 | 735,981,792 | 3.1288 |
| bf16 ceiling | 89.42 | 4752/5314 | 3,763,692,048 | 16 |

Every comparison below is an exact paired McNemar test (stdlib `math.comb`, not the χ² approximation)
on the stored `hits` arrays, matching [[paired-tests-on-stored-hits]].

| comparison | Δacc | discordant (b/c) | exact *p* | 95% CI | Δbytes |
|---|---:|---:|---:|---:|---:|
| `rb_agg` vs `gptq_3b_head_asym` | **+1.13** | 155/95 | **0.000178** | [+0.55, +1.71] | **−4.50%** |
| `rb_agg` vs `gptq_3b_head` (sym) | **+1.90** | 212/111 | **2.04e-08** | [+1.24, +2.56] | **−3.79%** |
| `rb_agg` vs bf16 ceiling | +0.15 | 81/73 | 0.573 (ns) | [−0.31, +0.61] | −81.19% |

## The held claim is settled

**The asymmetric control now exists, and DynQuant beats it.** +1.13 points, p = 0.000178, at 4.5%
fewer bytes — not matched bytes, *fewer* bytes. The original held claim worried that an honest
asymmetric GPTQ arm might close a 1.54-point gap using 1.71 points of headroom under the fp16
ceiling; the real asymmetric arm closed 0.41 of those 1.54 points (89.57 − 88.45 vs the old
89.57 − 88.03 read against the pre-fix byte figure) and DynQuant still wins comfortably outside
the 95% CI. This is no longer "unattributed" — it is a measured, paired, significant result against
the specific control the original report said was missing.

**DynQuant also beats the symmetric arm by a wider margin than the original panel reported**
(+1.90 here vs +1.54 there), because under the corrected byte accounting (below) the symmetric
arm is both cheaper and, on this replicate's checkpoint, less accurate than its destroyed-panel
counterpart.

**DynQuant statistically ties the bf16 ceiling** (+0.15, p = 0.57, CI crosses zero) while storing
5.32× fewer bytes (708,087,552 vs 3,763,692,048). This wasn't computed on the original panel and
is a new result: at this budget the quantization damage is not distinguishable from noise against
full precision.

## What this confirms about the original panel's own number

[`byte-accounting-zero-point.md`](byte-accounting-zero-point.md) fixed a bug where every symmetric
GPTQ arm across this project was charged `16 + bits` per group (scale plus a zero point) instead
of `16` (scale only, since a symmetric grid stores no zero point) — the asymmetric formula applied
regardless of which scheme actually ran. That report flagged the phase-2 headline as unable to
be corrected in place, because its checkpoint was gone, but predicted the direction: *"the
replicate's GPTQ arm will come in ~0.7% cheaper than the destroyed panel's did."*

That prediction is now checked directly:

| | bytes | Δ vs original panel's 741,475,927 B |
|---|---:|---:|
| this replicate's **symmetric** GPTQ | 735,981,792 | **−0.741%** |
| this replicate's **asymmetric** GPTQ | 741,486,130 | +0.0014% |

The predicted "~0.7% cheaper" lands on −0.741%, against the replicate's true symmetric arm. That
is not a coincidence worth hedging on. It also explains something the original report never
resolved: `gptq_3b_head` was fit and reported as **symmetric**, yet its byte figure
(741,475,927 B) sits within 10,203 bytes (0.0014%) of this replicate's **asymmetric** figure and
5,494,135 bytes (0.75%) away from this replicate's true symmetric figure. The original number was
a symmetric arm's weights, accounted through the pre-fix code path that priced it as if it stored
a zero point — i.e., it was already numerically an asymmetric-cost figure wearing a
symmetric-fit label. This also explains why the original report's own text asked for the asymmetric
control to be run "at 741,475,927 bytes" (§3, held-claim paragraph): under the bug, symmetric and
asymmetric were expected to cost the same, because the code never priced the difference between
them. They do not — asymmetric genuinely costs more (an extra per-group zero point), which is
exactly what this replicate's two controls now show (741,486,130 vs 735,981,792, a real 0.75% gap).

## What this does not change

* **No original-panel number is edited in place.** The destroyed checkpoint cannot be re-measured;
  the original panel's table stands as a record of what was measured then, under code that has
  since been fixed. This report supersedes its *held* status, not its printed digits.
* **The four-lever attribution is untouched.** All four levers were isolated at byte-identical
  budgets between DynQuant arms only; nothing about the GPTQ control changes those deltas.
* **Scope is still one panel.** Qwen3.5-2B-Base/CaseHOLD only — §10 of the phase-2 report
  ("What this does not overturn") stands unchanged.

## Fixed

* `p2replicate.sh`'s freshness guard now compares whole-second mtimes, not bash's sub-second
  `-ot`/`-nt`.
* The asymmetric `gptq_3b_head` control phase 2 was missing now exists and is checked into the
  arm table above; `docs/reports/README.md` row 4 and §4, the phase-2 `.tex`/PDF, and the
  `dynquant-beats-gptq-3bit` memory record are updated to cite it.

## Still open

* Re-run on the other two panels (Mistral-7B/Banking77, Qwen/Banking77) with their own asymmetric
  controls — §10 of the phase-2 report already predicts a smaller effect there and that has not
  been checked.
* The unrun control from phase 1 — GPTQ/AWQ handed DynQuant's own bit map rather than a uniform
  target — still has not been run on this panel.
