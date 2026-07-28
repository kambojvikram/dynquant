# Qwen3.5-2B-Base: measuring the allocator, then fixing it

The experiment the package was built to make possible: collect a signal map during a
real fine-tune, let the scorer and allocator choose per-module bit widths from it,
quantize, and measure the same benchmark at every stage.

Everything below is measured on this repository's own code. Nothing is taken from the
paper, and nothing is extrapolated.

**In short.** Both runs are complete, and they resolve in two stages.

*Stage one is a result against the method as implemented,* and it is the same on both
tasks. At a ~4.25-bit budget, where the score changes only 14–16 of 187 module widths,
score-driven allocation is indistinguishable from a same-size uniform control — GSM8K
+1.36 pts (p = 0.20), CaseHOLD −0.13 pts (p = 0.56). At a ~3.25-bit budget, where it
changes 70–79 of them, allocation **loses**: −7.88 pts on GSM8K (p = 1.1e−10) and
−2.03 pts on CaseHOLD (p = 5.8e−05). The more the rank-product score is allowed to drive
the widths, the more it costs.

*Stage two is what the diagnosis found underneath that.* The failure is not that a 3-bit
budget leaves nothing to allocate, and — contrary to what the first reading of these
tables concluded — it is not that 2-bit slots are unaffordable. It is that the published
score, `Rank(plasticity) × Rank(saliency)`, multiplies a factor that correlates with
measured damage by one that correlates **against** it (Spearman ρ = −0.21 over 187
modules), and then compares modules by ordinal rank when the decision the allocator has
to make is cardinal. Replacing it with a *measured* per-module sensitivity —
`Σ E[δ_r²] E[x_c²] (W − Q_b(W))²`, collected by the same tracker and priced per **move**
rather than per module — turns the same budget on the same checkpoint from −2.03 pts
against uniform into **+10.29 pts** (84.08 % vs 73.79 %, p = 1.3e−81). Quantization
damage at 3.25 bits falls from 16.73 points to **4.40**. See *The fix* below; the
stage-one sections are left standing as they were written, with the readings that stage
two overturned marked where they occur.

*Stage three ran the winning map through real CUDA kernels,* with the weights held packed
in VRAM rather than reconstructed into bf16. Accuracy is **exactly** unchanged — the same
4468 of 5314 problems — which retroactively licenses every simulated-quantization number
above. Resident VRAM is **0.7237 GiB against 3.5052 GiB at bf16 (4.84×)**, matching the
manifest the allocator wrote to 0.03 %; that is the claim the paper's Appendix F concedes
it cannot make. **Decode is slightly slower** (0.90× at batch 1), and the reason is measured
rather than guessed: a decode step on this model issues ~2000 kernel launches, leaves the
GPU idle ~70 % of the time, and spends 12 % of its wall clock in matmuls — so weight
compression has almost nothing to win here regardless of how good the kernel is. It does win
what there is to win: profiling both arms through one script puts in-model matmul time at
**1.42× faster** packed, with the entire drop in GPU-busy time accounted for by that one
change. It is simply outweighed, by the host-side cost of dispatching 187 packed modules
from Python — which is what CUDA Graphs (P8) exist to remove. Three
rounds of optimization then took the kernel *in isolation* from 0.64–1.83× bf16 to
**1.09–2.56×**, and from 26–41 % of achievable HBM bandwidth to **10–98 %** — the ≥70 %
target is met at 8 bits (98 %), three points short at 4 bits (67 %), and not met at 3 or 2
bits (52 %, 36 %). What remains is issue-bound rather than bandwidth-bound, is measured
against a computed instruction-issue floor, and needs the tensor-core path (P7) rather than
more of the same. See *Running it packed*.

**The two tasks are both here because they answer different questions.** GSM8K is the
paper's own task, and the fine-tune did not move it (−0.99 pts, p = 0.48) — not because
the fine-tune failed but because `Qwen3.5-2B-Base` was already at the supervised ceiling.
Its table therefore cannot read quantization damage against a fine-tuning gain, which is
the whole point of the design. CaseHOLD, chosen by measuring base-model headroom *first*,
gained **+53.35 points** from the identical recipe, and against that gain the quantized
arms become legible: ~4.25 bits keeps 96 % of the gain, ~3.25 bits keeps 69 %.

## Which task, and why the first one was the wrong one

GSM8K was chosen because it is the paper's own task. `Qwen3.5-2B-Base` scores 66.11 % on
it at 5-shot, and two epochs of SFT on GSM8K's own train split moved that −0.99 points.
The mistake was not the fine-tune; it was picking the task without checking whether a
fine-tune had anywhere to go.

The replacement was picked by measurement. [`screen_headroom.py`](screen_headroom.py)
applies a **two-sided** screen to candidate datasets: the base model must score far
*above* chance, or the task is simply beyond a 2 B model and the fine-tune has nothing to
build on; and far *below* what supervised training is known to reach, or there is nothing
left for it to teach. Both failures look identical afterwards — a flat line — which is
why the screen has to run first. It costs a few hundred examples of base-model eval.

| candidate | base few-shot | chance | supervised reference | to reference | verdict |
|---|---|---|---|---|---|
| CaseHOLD | 34.3 % | 20 % | ~69 % (BERT-base, Zheng et al. 2021) | ~35 pts | **pass** |
| sql-create-context | 54.0 % | 0 % | ~80 % | ~26 pts | pass |
| MedMCQA | 47.7 % | 25 % | ~45–50 % | ~0 pts | reject |
| GSM8K *(control)* | 66.1 % | 0 % | 65.1 %, **measured here** | ~0 pts | reject |

GSM8K stays in the candidate list as the control: a screen that does not reject the task
already known to have failed is not a screen. Its reference is the only measured entry in
the column, and measuring it is exactly what the screen exists to avoid having to do
again.

CaseHOLD was taken. One further requirement is a judgement rather than a threshold, so
the script does not mechanise it: the answer format must be conveyable by the few-shot
prefix. CaseHOLD's answer is a single digit, so the exemplars supply the format and any
gain is task skill rather than the model learning to answer in the right shape — which is
the cheap way to manufacture a gain and would not be a real one.

Screen numbers are a few hundred examples on a subsample and **do not appear in any
results table below**. Only figures produced by the shared `common.run_eval` on a full
test split do. The screen's job is to choose the task, and it is checked afterwards by
the real measurement, not trusted in place of it.

## What was run

Everything except the task is shared between the two runs, deliberately: the comparison
the second run rests on is "same recipe, different task headroom", so a step count or a
learning rate tuned per task would introduce a second difference and make the two tables
answer different questions.

| | |
|---|---|
| Model | `Qwen/Qwen3.5-2B-Base` — `Qwen3_5ForCausalLM`, 28 layers, hybrid linear/full attention |
| Parameters | 1.8821 B total; 1.8817 B across 187 quantizable modules, 0.4 M staying at compute dtype |
| Fine-tune | 2 epochs SFT on the task's `train` split, lr 1e-5 cosine, warmup 3 %, effective batch 32, bf16, signals collected by `DynQuantCallback` |
| Quantizer | group-128 asymmetric min/max, MSE-optimal clip search over α ∈ {1.00 … 0.80} |
| Budgets | 4.25 and 3.25 stored bits, each against a same-size uniform control |
| Hardware | 1 × A100 80 GB PCIe, torch 2.13.0+cu130, transformers 5.14.1 |

What differs is in [`tasks.py`](tasks.py), and only these rows:

| | GSM8K | CaseHOLD |
|---|---|---|
| test split | full, n = 1319 | full, n = 5314 |
| train examples | 7 468 | 42 507 |
| chance floor | 0 % (open-ended) | **20 %** (5-way choice) |
| few-shot | 5 | 2 (the prompt is ~400 tokens, not ~90) |
| decode | 320 new tokens, stop `"\n\nQuestion:"` | 8 new tokens, stop `"\n\n"` |
| scored by | exact match on the final number | exact match on the holding index |
| overlong training rows | dropped | **left**-truncated |
| loss masked to | the worked solution | the single answer digit |

The last two rows are not housekeeping. CaseHOLD's holdings and its `Answer:` cue sit at
the *end* of a long prompt, so right truncation removes the question and scores the model
on something it never saw — which reads as quantization damage rather than as a harness
bug. And a CaseHOLD completion is one digit after ~400 tokens of case-law prose: train on
the concatenation and the answer carries under 1 % of the gradient, so the run teaches
appellate prose while the training loss falls convincingly the whole way down.

The few-shot exemplars are drawn from `train` by a fixed seed and **held out of the
fine-tuning set**. They are used at every measurement point, including the fine-tuned
model — dropping the prefix after tuning would confound "got better at the task" with
"learned this output format".

Every point goes through one function, `common.run_eval`. Two call sites that each
build their own prompt would agree today and drift later, and the drift would be
indistinguishable from an accuracy difference the quantizer caused. This was very
nearly not true: `stage1_eval_base.py` was a hand-copied duplicate of `run_eval`'s
body, and it had already drifted. It is now a caller.

### What the control is, and why it holds the structural floors

The uniform arm flattens every module to the target width **except** those under a
structural floor — on this architecture the 36 `linear_attn.in_proj_a` / `in_proj_b`
projections, which carry the gate and decay coefficients of a recurrence. Their error
compounds along the sequence instead of averaging out. Flattening them too would very
likely break the model, and the comparison would then read as "score-driven allocation
beats flat allocation" when what it actually showed was "a working recurrence beats a
broken one". They are 1.2 M of 1.88 B parameters, so holding them at 8 bits moves the
stored average by 0.003 bits and the arms stay budget-matched.

Everything else is flattened, including the whole tied embedding / LM-head tensor —
27 % of the model. Those floors are quality preferences, and choosing among them is
exactly what is under test.

### Stored bits, not payload bits

`4.249` is what the filesystem would report, metadata included. At group 128 with an
fp16 scale and offset per group, metadata costs 0.25 bits per weight, so the *payload*
is 4.0 bits. This is the same convention as a "4-bit g128" GPTQ checkpoint. Targeting
`4.00` stored — which an earlier run did — gives a 3.75-bit payload and is not
comparable to anything the field calls 4-bit.

### What these numbers are not

Accuracy here is real: the quantizer writes the dequantized values back in place, and
those are bit-for-bit the values a packed checkpoint reconstructs (pinned by
`tests/test_quantizer.py`). **In the tables that follow, memory and speed are not
measured and are not claimed.** The GiB column is the size the packed checkpoint would
occupy on disk, computed from the format's own accounting. The model as evaluated sat in
memory at bf16.

Those two claims are made, and measured, in [Running it packed](#running-it-packed-the-kernels-and-what-they-buy)
further down, once the kernels exist. That section also settles whether simulated
quantization was telling the truth about accuracy: it was, exactly.

## Task 1 — GSM8K: the fine-tune had nowhere to go

Accuracy is exact match on n = 1319. `±1SE` is one binomial standard error in
percentage points — about 1.3 here, which is the scale any two numbers below have to
be compared against.

| measurement point | stored bits | GiB | exact match | ±1SE | correct |
|---|---|---|---|---|---|
| 1. base, no fine-tune (fp16) | 16.000 | 3.506 | **66.11 %** | 1.30 | 872 / 1319 |
| 2. fine-tuned (fp16) | 16.000 | 3.506 | **65.13 %** | 1.31 | 859 / 1319 |
| 3. quantized ~4 bit, DynQuant allocation | 4.249 | 0.931 | **58.15 %** | 1.36 | 767 / 1319 |
| &nbsp;&nbsp;&nbsp;control: uniform 4 bit, same budget | 4.255 | 0.932 | 56.79 % | 1.36 | 749 / 1319 |
| 4. quantized ~3 bit, DynQuant allocation | 3.249 | 0.712 | **13.57 %** | 0.94 | 179 / 1319 |
| &nbsp;&nbsp;&nbsp;control: uniform 3 bit, same budget | 3.256 | 0.713 | **21.46 %** | 1.13 | 283 / 1319 |

Unparseable completions: 2 at the base model, 0 everywhere else.

### The comparisons

Every comparison here is *paired* — the same 1319 problems, in the same order, scored
twice — so McNemar's exact test on the discordant pairs is the correct analysis and it
is what the verdict follows. "Flips" is how many problems only the left arm got right
against how many only the right arm did; those are the only problems carrying
information about the difference.

| question | Δ | flips (L/R) | paired 95 % CI | p | verdict | unpaired 2SE |
|---|---|---|---|---|---|---|
| did the fine-tune move GSM8K? | −0.99 pts | 140 / 153 | [−3.53, +1.56] | 0.48 | **not separated** | 3.70 |
| cost of quantizing to ~4 bit | −6.97 pts | 99 / 191 | [−9.48, −4.47] | 7.1e−08 | **separated** | 3.78 |
| did allocation beat uniform at ~4 bit? | +1.36 pts | 97 / 79 | [−0.61, +3.33] | 0.20 | **not separated** | 3.85 |
| cost of quantizing to ~3 bit | −51.55 pts | 22 / 702 | [−54.43, −48.68] | 1.2e−176 | **separated** | 3.23 |
| did allocation beat uniform at ~3 bit? | −7.88 pts | 79 / 183 | [−10.25, −5.52] | 1.1e−10 | **separated, against** | 2.94 |

The unpaired column is kept deliberately. On this data it is roughly twice as wide as
the paired interval, and an earlier version of these results reported only the unpaired
number — so the column makes visible exactly which conclusions changed when the correct
test replaced it, rather than quietly upgrading them. Two did: the ~3-bit allocation
loss and the ~3-bit quantization cost were already separated and became overwhelming;
**the headline ~4-bit comparison did not change verdict.** Halving the interval was not
enough to resolve +1.36 points.

Recording this needed a re-run: the first pass stored only summary counts, and a paired
test cannot be reconstructed from those. All six arms were re-scored, and every count
reproduced exactly — 872, 859, 767, 749, 179, 283 — which is also the determinism check
on the harness that greedy decoding on fixed prompts should pass.

### Reading this honestly

**The fine-tune did not improve GSM8K, and that is a real result.** 66.11 % → 65.13 %
is a 0.99-point *decrease*, well inside one standard error. Qwen3.5-2B-Base is already
strong at 5-shot GSM8K, and two epochs of SFT on GSM8K's own train split at lr 1e-5
did not add to it. The one thing that visibly improved is output format: unparseable
completions went from 2 to 0.

This does not weaken the experiment, because the fine-tune's other job was to produce
the signal map, and it did. But the four-point table must not be presented as
"fine-tuning helps, then quantization costs a little". It shows fine-tuning flat and
quantization expensive.

**The ~4-bit drop is the cost of round-to-nearest, not a defect.** Per-layer
reconstruction errors track quantization theory closely: group-128 asymmetric min/max
on roughly Gaussian weights predicts a relative error near `step/sqrt(12)` with
`step ≈ 5.8σ/(2**b − 1)`, i.e. ≈ 0.10 at 4 bits and ≈ 0.21 at 3. Measured medians were
0.099 at 4 bits, 0.189 at 3 bits, and 0.0075 at 8 bits. No layer took anomalous
damage; the worst 4-bit layer was 0.145 and the worst 3-bit layer 0.195. The MSE clip
search improved SSE on 146 of 187 layers, 9.4 % on average.

Seven points is nonetheless a large drop, and the reason is that this is **plain RTN**.
GPTQ compensates each column's error against the remaining ones through a Hessian;
AWQ rescales channels by activation magnitude first. DynQuant's contribution as
implemented here is *where the bits go*, not *how each tensor is encoded*, and on a 2 B
model evaluated by chain-of-thought exact match — where one wrong token derails a whole
solution — the encoder is the dominant term. Pairing the allocator with a GPTQ-style
encoder is the obvious next step and is not part of these numbers.

**Allocation beat uniform by 1.36 points at ~4 bit, which this experiment cannot call
a win.** 97 problems flipped to allocation and 79 the other way: p = 0.20, and the
paired interval [−0.61, +3.33] contains zero. The sign is in the right direction at
both a matched budget (4.249 vs 4.255 stored bits — the arms differ by 0.006 bits per
weight) and a matched encoder, so nothing is confounded; the effect is simply smaller
than n = 1319 resolves. The paired test was added specifically to give this comparison
its best honest chance, and it still does not separate. Claiming a win would require
either a larger test set or a setting where the widths diverge more than 16 of 187
modules.

### The ~3-bit result: score-driven allocation lost, decisively

At the ~3-bit budget the allocator did not merely fail to help — it did **7.88 points
worse than flat 3-bit** at the same size (13.57 % vs 21.46 %, 79 / 183 flips,
p = 1.1e−10). This is the strongest single result in the table and it is against the
method as implemented.

The mechanism is fully traceable and is not a bug:

- To lift 30 modules (210.8 M params) to 4 bits, the allocator pushed 22 modules
  (223.3 M params, 12 % of the model) down to **2 bits**, at relative errors of
  0.41–0.45. The uniform control held everything at 3 bits, relative error 0.194.
- The allocator prices a move as `score × num_params × Δerror` against a cost of
  `num_params × Δbits`. `num_params` cancels, so the decision is `score × Δerror /
  Δbits`.
- Its error model is *correct*: `_error_scale(bits) = 4**−bits`, i.e. relative error
  ∝ `2**−b`, which the measured per-layer errors match at every width (0.0075 at 8,
  0.099 at 4, 0.189 at 3, ~0.43 at 2).
- What fails is the implicit **damage** model. Scores span 0.0109 to 0.9588 — an 88×
  range — while the per-bit error ratio is only 4×. So the score term dominates and the
  allocator will always trade a low-score module's precision for a high-score module's.
  That is correct if end-to-end task damage is linear in injected weight error. It is
  not: at relative error 0.43 a projection is qualitatively broken, not "twice as bad
  as 0.19", and concentrated damage on 12 % of parameters is worse than uniform damage
  on 100 % of them.

Sample generations from the ~3-bit allocated arm confirm it: the 5-shot format is
preserved (0 unparseable) but the arithmetic is wrong, and at least one completion
degenerates into a repeated-digit loop. The model is mostly destroyed, and the flat
control is merely badly damaged.

Two implications, neither of which this experiment can dodge:

1. **The allocator needs a floor on achievable error, not only on bits.** A move that
   takes a module past the width where its encoder still works should be priced as
   unavailable, not as cheap. That is a change to `allocate/knapsack.py`, not to the
   score.
2. **2-bit RTN is not a usable operating point,** so a bit map that spends 2-bit slots
   is spending currency it does not have. This is the same argument the ~4-bit drop
   makes for a GPTQ/AWQ-style encoder, in a sharper form: a better encoder does not
   just recover points, it changes which allocations are legal.

The ~4-bit and ~3-bit results are therefore consistent, not contradictory. Where every
width the allocator can reach is one the encoder handles, allocation is neutral-to-
slightly-positive. Where it can reach a width the encoder cannot handle, allocation is
actively harmful — and nothing in the current pricing stops it going there.

> **Superseded — point 2 is wrong, and *The fix* below is the measurement that refutes
> it.** The sensitivity allocation puts **twice** as many parameters at 2 bits as this
> map does (440 M vs 220 M, 23.4 % vs 11.7 % of the model) at the identical budget, and
> scores 12.33 points *higher*. So 2-bit RTN is not an unaffordable width; it is a width
> that has to be spent on the right modules, and this score picked the wrong ones. Point
> 1 survives in weakened form — pricing by measured `sens(b) − sens(b′)` already makes a
> ruinous move look expensive without a hand-set error floor, because the measurement
> that says a module is fragile is the same measurement that says its next step down is
> costly. The reasoning that produced point 2 is left above because it was the honest
> reading of what was then in evidence, and the shape of the mistake — inferring a
> property of the *width* from the behaviour of one allocation over it — is the useful
> part.

### Does the score actually drive the widths?

The failure mode this experiment most needed to rule out is an allocator that produces
a plausible bit map while ignoring the signal — which is what the supplement's
allocator does at its own headline 3-bit target, where inverting every score changes 0
of 282 modules. `inspect_allocation.py` measures it directly, as within-role pairwise
concordance of assigned width with importance score, on the GSM8K signal map:

| target | modules scoring exactly 0 | unexercised | within-role concordance |
|---|---|---|---|
| 4.25 | 0 | 0 | 255 / 255 = **1.000** |
| 3.25 | 0 | 0 | 682 / 682 = **1.000** |

Read this as a passed sanity check, not as a strong result. Within a role on a dense
model most modules have identical parameter counts, so the greedy ROI ratio
`score / (params × Δbits)` reduces to the score ordering and perfect concordance is
close to tautological. What it does establish is that the score reaches the allocator
at all, at both targets, including the 3-bit target where the legacy code path silently
stopped reading it. The comparison is within role because floors differ by role by
design; globally it would be confounded by the structure the allocator is meant to
respect.

Score range was 0.0109 to 0.9588, median 0.1576, with **no module at exactly zero** and
no unexercised or unmeasured module. The narrowest modules at both targets are the
layer-0 and layer-20/21 MLP projections.

## Task 2 — CaseHOLD: a fine-tuning gain to read the damage against

CaseHOLD is a 5-way multiple choice over US case law: given a citing paragraph, pick which
of five holding statements it cites. It passed the headroom screen on both sides and was run
through the identical six-arm pipeline.

Accuracy is exact match on the holding index over the full test split, n = 5314. `±1SE` is
one binomial standard error in percentage points, and it is **not** constant down the
column — binomial variance shrinks toward the extremes, so the same n gives 0.65 points at
35 % and 0.44 at 88 %.

| measurement point | stored bits | GiB | exact match | ±1SE | correct | unparseable |
|---|---|---|---|---|---|---|
| 1. base, no fine-tune (fp16) | 16.000 | 3.506 | **35.13 %** | 0.65 | 1867 / 5314 | 7 |
| 2. fine-tuned (fp16) | 16.000 | 3.506 | **88.48 %** | 0.44 | 4702 / 5314 | 0 |
| 3. quantized ~4 bit, DynQuant allocation | 4.249 | 0.931 | **86.49 %** | 0.47 | 4596 / 5314 | 0 |
| &nbsp;&nbsp;&nbsp;control: uniform 4 bit, same budget | 4.255 | 0.932 | 86.62 % | 0.47 | 4603 / 5314 | 0 |
| 4. quantized ~3 bit, DynQuant allocation | 3.249 | 0.712 | **71.75 %** | 0.62 | 3813 / 5314 | 0 |
| &nbsp;&nbsp;&nbsp;control: uniform 3 bit, same budget | 3.256 | 0.713 | **73.79 %** | 0.60 | 3921 / 5314 | 0 |
| 5. quantized ~3 bit, **measured sensitivity** | 3.249 | 0.712 | **84.08 %** | 0.50 | 4468 / 5314 | 0 |
| &nbsp;&nbsp;&nbsp;*(guessing)* | — | — | *20.00 %* | — | — | — |

Row 5 is the fix, and it is described in full two sections below. It is listed here
rather than kept apart because it is the same checkpoint, the same quantizer, the same
`run_eval`, and — exactly, not approximately — the same budget as row 4: both maps spend
5 637 144 576 payload bits over the same 1.8813 B parameters, so the two differ only in
*which* modules got them.

The base arm's 35.13 % agrees with the screen's 34.3 % subsample estimate inside its
sampling error — the check the screen is supposed to survive rather than substitute for.

**Read the 20 % floor into every number above.** On an open-ended task a destroyed model
scores near zero, so damage is obvious; here it scores near 20, and a collapsed arm sits
much closer to a merely damaged one. The `unparseable` column is what tells them apart, and
it settles the question for this run: every post-tuning arm emitted a parseable digit on all
5314 problems, including the ~3-bit arm with 83 breached floors and 23 modules at 2 bits. So
none of the damage below is broken format compliance. It is all degraded reasoning.

### The comparisons

Same analysis as Task 1 — paired McNemar exact on the same 5314 problems in the same order,
with the unpaired joint 2SE kept alongside for comparison.

| question | Δ | flips (L/R) | paired 95 % CI | p | verdict | unpaired 2SE |
|---|---|---|---|---|---|---|
| did the fine-tune move CaseHOLD? | **+53.35 pts** | 2976 / 141 | [+51.87, +54.83] | &lt; 1e−300 | **separated** | 1.58 |
| cost of quantizing to ~4 bit | −1.99 pts | 120 / 226 | [−2.68, −1.31] | 1.3e−08 | **separated** | 1.28 |
| did allocation beat uniform at ~4 bit? | −0.13 pts | 50 / 57 | [−0.51, +0.25] | 0.56 | **not separated** | 1.32 |
| cost of quantizing to ~3 bit | −16.73 pts | 175 / 1064 | [−17.95, −15.51] | 1.1e−155 | **separated** | 1.51 |
| did allocation beat uniform at ~3 bit? | −2.03 pts | 301 / 409 | [−3.01, −1.05] | 5.8e−05 | **separated, against** | 1.73 |
| did **sensitivity** beat uniform at ~3 bit? | **+10.29 pts** | 712 / 165 | [+9.20, +11.39] | 1.3e−81 | **separated, for** | 1.57 |
| sensitivity vs the rank-product allocation | **+12.33 pts** | 857 / 202 | [+11.13, +13.53] | 1.7e−96 | **separated, for** | 1.59 |
| cost of quantizing to ~3 bit *(sensitivity)* | −4.40 pts | 136 / 370 | [−5.23, −3.57] | 4.8e−26 | **separated** | 1.33 |

### The fine-tune worked, so the damage column means something

+53.35 points, 2976 problems gained against 141 lost. This is precisely what GSM8K's table
was missing, and it converts the quantized arms from bare accuracies into a retention
figure — damage measured against the gain it is eating, which is what the design was for:

| budget | accuracy | of the +53.35-point gain, retained |
|---|---|---|
| fp16, fine-tuned | 88.48 % | 100 % |
| ~4.25 bits | 86.49 % | **96 %** |
| ~3.25 bits, rank-product score | 71.75 % | **69 %** |
| ~3.25 bits, measured sensitivity | 84.08 % | **92 %** |

The fine-tuned model also clears the ~69 % fine-tuned-BERT-base figure the screen used as
its supervised reference, by 19 points. The screen was directionally right and
quantitatively conservative: it predicted ~35 points of room from that reference, and a
fully fine-tuned 2 B decoder found 53. No overfitting collapse either — a 500-step probe
scored 75.67 % and the full 2658-step run 88.48 %.

### The same quantizer costs 3.5× less here, and that is about the metric

Identical weights-to-bits pipeline, identical 4.2486 stored bits, identical 0.931 GiB — and
the ~4-bit cost is **1.99 points** where GSM8K's was 6.97. At ~3 bits the gap is far
starker: 71.75 % here against **13.57 %** there, from the same 3.2494-bit map breaching the
same 83 floors.

The difference is decode length, and the GSM8K section above predicted it before this run.
GSM8K scores exact match on the final number of a ~320-token chain, so a model damaged
enough to slip occasionally almost never completes a correct chain and the damage compounds
multiplicatively over the decode. CaseHOLD scores one digit. Most of what GSM8K's ~3-bit arm
measured was decode-length amplification, not weight damage — which also means a
quantization result reported on chain-of-thought exact match is not transferable to
single-token tasks, in either direction.

Stated as a caveat rather than left to be discovered: the two tasks also sit at different
accuracies (65 % vs 88 %) and differ in kind, so this is not a clean isolation of decode
length. The direction and rough magnitude are what the compounding argument predicted, and
the argument was written down first.

### The ~3-bit result replicates, and it is the finding

| budget | modules the score moves off uniform (GSM8K · CaseHOLD) | GSM8K | CaseHOLD |
|---|---|---|---|
| ~4.25 bits | 16 · 14 of 187 | +1.36 (p = 0.20) | −0.13 (p = 0.56) |
| ~3.25 bits | 70 · 79 of 187 | **−7.88 (p = 1.1e−10)** | **−2.03 (p = 5.8e−05)** |

Read down the columns. Where the score barely changes the allocation, allocation ties
uniform. Where the score actually drives the widths, allocation is worse than flattening
everything to 3 bits under the same floors — on both tasks, at two accuracy levels, at two
decode lengths, and with the control holding marginally *more* storage in every case. The
uniform arms are 0.0067 bits per weight larger at both budgets, because the same 36 floor
modules are held in both, so no comparison here is flattered by size.

The 4-bit tie is not a resolution failure this time. n = 5314 gives a paired interval of
[−0.51, +0.25], a quarter of GSM8K's width, and only 107 of 5314 problems flip at all —
which is what 177-of-187 identical widths looks like from the output side. The two 4-bit
maps produce very nearly the same model. Note also that GSM8K's +1.36 was the only
pro-method number anywhere in either run, and it was already not separated; at four times
the sample size the same comparison lands at −0.13. The honest joint reading is that at a
~4.25-bit budget the score has **no measurable effect on accuracy in either direction**.

The mechanism at 3 bits is the one Task 1 traced. To fund 4-bit slots the allocator drives
**23 modules to 2 bits**, where round-to-nearest gives relative errors around 0.43 and the
tensor is qualitatively broken rather than proportionally worse; the uniform control never
goes below 3. So the result reduces to *the 2-bit modules cost more accuracy than the 4-bit
modules they pay for* — now on two tasks, which is the difference between an anecdote and a
reason to change [`allocate/knapsack.py`](../../packages/dynquant-core/src/dynquant/allocate/knapsack.py).

> **Half superseded.** The italicised sentence is true of *this* map and false as a
> general claim, which is the distinction *The fix* below turns on: the same 2-bit slots,
> sold to different modules, buy +10.29 points instead of −2.03. The change it motivated
> in `allocate/knapsack.py` was the right file for the wrong reason — what needed
> replacing was the price tag, not the width limit.

What this does **not** indict is the paper's Phi-4 and Qwen3-14B numbers, which came from a
hand-written map because their allocator's `remaining` went negative and the greedy loop
never ran at all (bug 4 in the audit). What it does establish is that making the floors soft
so the score *can* bite at a 3-bit target — the fix for that bug — produces a map measurably
worse than not scoring, until sub-encoder widths are priced as unavailable rather than cheap.

### The prediction, and how it resolved

Recorded here before the CaseHOLD arms ran, verbatim: *"the expectation is therefore that
the ~3-bit allocated arm loses to uniform-3b again, and the ~4-bit arms stay close."* Both
held — the ~3-bit arm lost by 2.03 points at p = 5.8e−05, and the ~4-bit arms landed 0.13
points apart.

The prediction also carried a caveat, and the caveat was warranted. The width histograms it
quoted — `{2: 26, 3: 90, 4: 35, 8: 36}` at the 3.25 target — came from the **smoke run's**
60-step signal map rather than the real 2657-step one, so it predicted only that the *shape*
would survive, not the counts. The real map came back `{2: 23, 3: 99, 4: 29, 8: 36}` at 3.25
and `{3: 8, 4: 138, 8: 41}` at 4.25. Shape survived; the counts moved.

What did **not** replicate is the severity. GSM8K's ~3-bit allocated arm fell to 13.57 %,
below its own uniform control's 21.46 % and effectively destroyed; CaseHOLD's holds 71.75 %,
still 36.6 points above base fp16. The ordering replicated, the catastrophe did not — see
the decode-length section above.

### An unexpected observation: the two tasks produce nearly the same bit map

Not predicted, not the reason for the second run, and recorded because it bears directly on
the method's premise. Comparing the two tasks' allocations module by module at the same
target:

| target | identical widths | modules at 2 bits |
|---|---|---|
| 4.25 | **177 / 187** | none on either side |
| 3.25 | 149 / 187 | 23 (CaseHOLD) vs 22 (GSM8K), overlapping on 15 |

The deviations are mostly shared rather than task-specific. At 4.25, CaseHOLD's allocation
differs from uniform on 14 modules and GSM8K's on 16, while the two allocations differ from
*each other* on only 10 — so roughly two-thirds of the score's departure from flat is common
to both tasks. At 3.25 the corresponding figures are 79, 70 and 38: about three-quarters
shared.

The method's premise is that signals harvested during fine-tuning reveal *task-specific*
importance. Most of what these two signal maps encode appears to be model-intrinsic instead.
This is an observation from **two tasks, one model, one recipe** — not a conclusion — but it
is cheap to test properly, and it makes a sharp prediction: a signal map should transfer
across tasks with little loss. If it does, that is a useful engineering property and a
problem for the premise at the same time.

Chasing that observation is what produced the next section. The two-thirds-shared
allocation is what a score dominated by module *shape* looks like from the output side,
and asking which factor was doing that is what turned up the sign error.

## The fix: measure the damage instead of ranking the module

Everything above measures one score. This section measures the score itself, finds it
carrying a factor with the wrong sign, replaces it, and re-runs the losing arm.

### Step 1 — measure what a bit is actually worth, per module

Nothing above establishes that any score *could* have worked, so the first thing built was
the answer key. [`marginal_3to4.json`](../../stats/qwen35_2b_casehold/marginal_3to4.json)
holds it: quantize the whole model to 3 bits, then for each of the 187 modules **on its
own** restore it to 4 and measure the change in task loss on a fixed batch. 187 forward
passes, one controlled variable each, no allocator involved. `fp16` loss 0.1049, uniform-3b
0.1514, so a perfect policy has 0.0466 of loss to recover and every module's `gain` is a
share of it.

Against that key, Spearman ρ over all 187 modules:

| candidate | ρ vs measured gain | p |
|---|---|---|
| plasticity (`Var_t ‖∇W‖²`) alone | **+0.3131** | 1.3e−05 |
| fisher (`Σ E[δ²] E[x²] (W−Q(W))²`) | **+0.2951** | 4.3e−05 |
| `num_params` | +0.1492 | 0.042 |
| weight error `‖W − Q(W)‖²` alone | +0.0645 | 0.38 |
| gram (`E[x²]` term alone) | +0.0099 | 0.89 |
| **the shipped score** | **+0.0040** | 0.96 |
| saliency (`EMA ‖X‖`) alone | **−0.2059** | 0.0047 |

Read the first and last rows together. The published score is
`Rank(plasticity) × Rank(saliency)`, a product of the **best** single predictor in the
table with the one predictor that is significantly **anti**-correlated with damage. The
product lands at ρ = 0.0040 — statistically indistinguishable from ranking the modules at
random. That is the whole finding in one line: multiplying by saliency destroys the signal
plasticity carries.

The same picture holds within role, which is where the allocator actually compares things
(floors differ by role, so cross-role comparisons are confounded by design):

| candidate | mean ρ over 12 roles | roles where ρ > 0 |
|---|---|---|
| fisher | **+0.281** | **11 / 12** |
| plasticity | +0.266 | 10 / 12 |
| the shipped score | −0.001 | 5 / 12 |
| gram | −0.151 | 4 / 12 |
| saliency | −0.214 | **2 / 12** |

Saliency is not noise. It is informative and it points the wrong way — activation
magnitude flags the modules whose precision matters *least* here, which is why an ordinal
product with it is worse than either factor alone rather than merely diluted.

### Step 2 — three changes

1. **Drop the saliency factor.** It survives only under the `paper-3.15` preset, whose job
   is to reproduce published numbers, not to be right.
2. **Replace the rank product with a cardinal quantity.** Ranks answer "is A more
   important than B"; the allocator has to answer "is A's next step down cheaper than B's",
   which needs magnitudes. The replacement is a diagonal Gauss-Newton / empirical-Fisher
   estimate of the loss increase from quantizing module *i* to *b* bits,

   ```
   sens_i(b) = Σ_rc  E[δ_r²] · E[x_c²] · (W − Q_b(W))²_rc
   ```

   where `δ = ∇_Y L` is the gradient arriving at the module's output and `x` its input —
   both already available to a backward hook, and both accumulated as running per-channel
   means by [`SignalTracker`](../../packages/dynquant-core/src/dynquant/signals/tracker.py)
   under `collect_channel_moments`. This is new instrumentation, not a re-read of the
   existing stats file: the old tracker stored an activation-RMS EMA and a scalar gradient
   variance, neither of which is a per-channel second moment. It costs one extra reduction
   per hook, gated behind `channel_moment_every` (default 16).
3. **Price the move, not the module.** The old cost function multiplied a per-module score
   by `Δerror`; the new one asks the estimator directly what *this specific step* costs,
   `sens_i(b) − sens_i(b′)`, and hands that to the same greedy knapsack. A module can then
   be important overall and still be the cheapest thing to cut at its current width, which
   the module-level formulation cannot express.

### Step 3 — the policy shootout

Before spending an eval, every candidate was run in-batch: build a bit map at a fixed
budget, quantize, measure loss on the same fixed batch. Percentages are the share of the
uniform-3-bit damage recovered; 100 % is fp16 and the `oracle` row is a greedy fill using
the measured `gain` column itself, so it is the ceiling any predictor could reach.

| policy | 3.125 bits | 3.250 bits |
|---|---|---|
| **oracle** *(has seen the answers)* | *84.87 %* | *104.15 %* |
| `fisher_move` — `Σ E[δ²]E[x²](W₃−W₄)²` | 77.21 % | **105.67 %** |
| **`fisher_diff` — `sens(3) − sens(4)`, what ships** | **85.54 %** | 95.76 % |
| plasticity alone | 86.81 % | 81.58 % |
| **the shipped score** | **28.49 %** | 85.17 % |
| weight error alone | **−29.46 %** | 48.51 % |
| `num_params` (biggest tensors first) | 52.01 % | 26.40 % |
| role-aware control (no signal at all) | 65.58 % | 95.08 % |
| random × 5 | −6.45 … 45.39 % | −5.86 … 59.15 % |

Three things to take from this table. The shipped score at 3.125 bits (28.49 %) sits
*inside* the band five random allocations span and below "upgrade the biggest tensors
first" — which is the in-batch version of the end-to-end loss the tables above measured.
Weight error alone at −29.46 % is worse than not reallocating, and it is the control that
matters most: it is the sensitivity formula with the two measured expectations removed, so
it is what the method degenerates to without training signal. And the role-aware control
recovers 65.58 % / 95.08 % with no signal whatsoever, which is the bar any signal-driven
policy has to clear to have earned the hooks.

`fisher_diff` ships because it is the one candidate above 85 % at both budgets. Above the
oracle at 3.25 bits is not a paradox — the oracle is greedy over *independently measured*
single-module gains, and those do not add.

### Step 4 — the end-to-end re-run

Same checkpoint, same quantizer, same `run_eval`, same 5314 problems, and the same
5 637 144 576 payload bits as the rank-product arm — the two maps are budget-identical to
the bit, not to four decimals.

| arm at 3.25 stored bits | accuracy | correct | GiB |
|---|---|---|---|
| **measured sensitivity** | **84.08 %** | 4468 / 5314 | 0.7118 |
| uniform 3 bit control | 73.79 % | 3921 / 5314 | 0.7133 |
| rank-product score (as shipped) | 71.75 % | 3813 / 5314 | 0.7118 |

| comparison | Δ | flips (L/R) | paired 95 % CI | p |
|---|---|---|---|---|
| sensitivity vs uniform 3b | **+10.29 pts** | 712 / 165 | [+9.20, +11.39] | 1.3e−81 |
| sensitivity vs rank-product | **+12.33 pts** | 857 / 202 | [+11.13, +13.53] | 1.7e−96 |
| rank-product vs uniform 3b *(from above)* | −2.03 pts | 301 / 409 | [−3.02, −1.05] | 5.8e−05 |

The sign flips and the magnitude is five times the loss it replaces. Against fine-tuned
fp16 the sensitivity arm gives up **4.40 points** where the shipped allocation gave up
16.73, and it keeps 92 % of the fine-tuning gain at 3.25 bits against the shipped map's
69 %. The winning arm is also fractionally *smaller* on disk than the uniform control it
beats, because the control has to hold the 36 recurrence modules at 8 bits too.

### What the reallocation looks like, and why point 2 above was wrong

Average assigned bits per role, weighted by parameters — the whole difference between two
maps that cost the same:

| role | params | sensitivity | rank-product | Δ |
|---|---|---|---|---|
| `self_attn.k_proj` | 6.3 M | **4.00** | 3.17 | **+0.83** |
| `self_attn.v_proj` | 6.3 M | **4.00** | 3.50 | **+0.50** |
| `self_attn.q_proj` *(fused Q + output gate)* | 50.3 M | 3.33 | 3.00 | +0.33 |
| `linear_attn.out_proj` | 75.5 M | 3.33 | 3.11 | +0.22 |
| `linear_attn.in_proj_qkv` | 226.5 M | 3.33 | 3.17 | +0.17 |
| `self_attn.o_proj` | 25.2 M | 3.33 | 3.17 | +0.17 |
| `linear_attn.in_proj_z` | 75.5 M | 3.17 | 3.11 | +0.06 |
| `embed_tokens` *(tied, 27 % of the model)* | 508.6 M | 3.00 | 3.00 | 0.00 |
| `linear_attn.in_proj_a` / `_b` *(recurrence)* | 1.2 M | 8.00 | 8.00 | 0.00 |
| `mlp.gate_proj` | 302.0 M | 3.12 | 3.12 | 0.00 |
| `mlp.down_proj` | 302.0 M | **2.67** | 2.79 | **−0.12** |
| `mlp.up_proj` | 302.0 M | **2.67** | 2.83 | **−0.17** |

604 M parameters of MLP up/down give up precision so that 88 M parameters of attention
can have it, and `k_proj` and `v_proj` go to a full 4 bits everywhere. Step 1's answer key
points the same way — its within-role correlations are highest in exactly these roles
(`ATTN_O` +0.714, `ATTN_V` +0.657, `ATTN_K` +0.429) — but there are only 6 modules in each
of those roles on this architecture, so treat that as consistent rather than as
confirmation. `ATTN_Q_GATE` is the one place the two disagree: −0.086 in the sweep against
+0.33 assigned bits in the map.

This is also what refutes the "2-bit RTN is not a usable operating point" reading recorded
in Task 1. Parameters by assigned width, both maps at the same budget:

| width | sensitivity | rank-product |
|---|---|---|
| 2 bit | **440.4 M (23.4 %)** | 220.2 M (11.7 %) |
| 3 bit | 1011.9 M (53.8 %) | 1452.3 M (77.2 %) |
| 4 bit | **427.8 M (22.7 %)** | 207.6 M (11.0 %) |
| 8 bit | 1.2 M (0.1 %) | 1.2 M (0.1 %) |

The winning map is the *more* aggressive one. It doubles the parameters sent to 2 bits and
doubles the parameters lifted to 4, and beats the timid map by 12.33 points. So the
earlier inference — that spending 2-bit slots is spending currency the encoder cannot cash
— was reading a property of one bad ordering as a property of the width. 2-bit RTN is
affordable; it just has to be sold to modules that can absorb it, and identifying those is
what the measurement is for. A better encoder would still help, and that argument is
untouched; it is simply not what was wrong here.

### Where the calibration data came from

The moments are collected from CaseHOLD's **`validation`** split, which the fine-tune
never saw and the evaluation never scores. Three notes, because a sensitivity estimate is
only as trustworthy as the batch it was measured on:

- **Not `test`.** Building the allocation from statistics of the split it is then graded
  on would be leakage, and the +10.29 would be uninterpretable. (The step-1 marginal sweep
  *does* use `test` for both, correctly — it asks whether a proxy predicts a loss, a
  question about predictability, not about generalisation.)
- **Not `train`, though that would not have been leakage either.** Two epochs of SFT put
  this model at a training loss of 0.0000, so `E[δ_r²]` comes out around 1e−9 and is being
  read off a regime the quantized model is never in. It is not degenerate — 0 all-zero
  modules, within-module p99/p50 of 41.3× — merely uninformative about the thing being
  priced. `--calib-split train` is kept so that can be shown rather than asserted.
- **The splits were checked, not assumed.** In this parquet mirror `validation` and `test`
  both have exactly 5314 rows, which looks alarming enough to verify: they have different
  first-row hashes, different answers, and set intersection on `citing_prompt` gives
  `validation ∩ test` = **1** and `validation ∩ train` = **1** of 5314 — a duplicate in
  the source dataset, not a split leak.

512 rows, 204 693 tokens.

### What this does not establish

- **Collected post-hoc, not streamed.** The moments were taken in one pass at the final
  weights rather than accumulated across the fine-tune, which is what `DynQuantCallback`
  does and what the package ships. `stage4_sensitivity.py` runs the real `SignalTracker`
  on the real hook path, so the code is the same; the expectations were taken at a
  different time. Whether streamed moments give the same map is not measured here.
- **One model, one task, one budget.** The 3.25-bit CaseHOLD arm is where the shipped
  score lost, so it is where the fix had to be demonstrated. GSM8K's 3.25-bit arm has not
  been re-run with the estimator, and the 4.25-bit budget — where nothing separates — has
  not either.
- **The estimate is first-order in the module.** It assumes one module's error at a time,
  so it over-counts in aggregate (measured ~1.4–2× against the joint effect) and predicts
  magnitude rather than sign. It is a ranking device that happens to be cardinal, not a
  loss prediction.
- **The premise question is still open.** Step 1 shows the *signal* carries real
  information; it does not show that information is task-specific rather than
  model-intrinsic, which is what the previous section flagged and what a same-weights,
  two-task collection would settle.

## Running it packed: the kernels, and what they buy

Everything above was measured with the weights reconstructed into bf16. This section is
the other half: same checkpoint, same bit map, same 5314 problems, but the weights are
held as packed `int32` words and compiled CUDA kernels do the arithmetic. One A100 80GB
PCIe, `stage6_packed_eval.py`.

Packing runs on the **CPU**, so the model reaches the GPU already packed and
`memory_allocated()` taken immediately after `.to("cuda")` is the resident weight
footprint with nothing netted out of it. Quantizing on the device and measuring afterwards
would report a peak that includes the bf16 copy the packing consumed — a number nobody
loading a quantized checkpoint would ever see. Decode is timed next, then accuracy scored,
with the peak counter reset between: a batch-32 evaluation allocates a KV cache that
dwarfs the weights, and one peak spanning both would say nothing about either.

### Accuracy: exactly unchanged

| arm | accuracy | correct |
|---|---|---|
| base, no fine-tune | 35.13 % | 1867 / 5314 |
| bf16 fine-tuned | 88.48 % | 4702 / 5314 |
| 3.25 bit sensitivity, simulated (stage 5) | 84.0798 % | 4468 / 5314 |
| **3.25 bit sensitivity, packed + CUDA kernels** | **84.0798 %** | **4468 / 5314** |

Not "within noise" — the same 4468 problems. Both paths quantize through the same search
over the same grid, so the values the kernels decode are bit-identical to the ones stage 5
wrote into the parameters, and anything but an exact match is a kernel bug rather than a
rounding difference. The script asserts it and prints `MATCH` or `DISAGREEMENT` rather than
leaving it to be noticed.

This is what licenses every accuracy number earlier in this document. They were all
measured with simulated quantization, on the assumption that a packed model would compute
the same thing. That assumption is now a measurement.

### Memory: the manifest was right to 0.03 %

| | packed 3.25 bit | bf16 | ratio |
|---|---|---|---|
| manifest prediction | 0.7118 GiB | — | — |
| model tensors, walked from the module tree | **0.7120 GiB** | 3.5052 GiB | 4.92× |
| process allocated after `.to("cuda")` | 0.7237 GiB | 3.5052 GiB | 4.84× |
| allocator reserved (what `nvidia-smi` shows) | 0.7480 GiB | 3.5645 GiB | 4.77× |
| peak during decode, batch 1 | 0.882 GiB | 3.662 GiB | 4.15× |
| peak during decode, batch 32 | 5.382 GiB | 8.164 GiB | 1.52× |

187 of 187 modules packed, 0 left dense. The allocator predicted 764 317 696 bytes when it
chose the map; the live model holds 764 530 560 — **212 864 bytes over, 0.03 %**, which is
the norms and biases the map never claimed to cover. This is the claim the paper's own
Appendix F concedes it cannot make, and it is the first number in this project that
required the kernels to exist.

Three definitions are reported because each invites a different objection. `allocated` is
what the tensors need; `reserved` is what the caching allocator took from the driver.
The 12.6 MB between the walked total and `allocated` is not a leak — enumerating every live
CUDA tensor after the move finds **zero** outside the module tree — it is the caching
allocator's block granularity, and `reserved` is the honest upper bound. The saving is
4.8× on whichever of the three you prefer.

The batch-32 row is the one worth reading twice. Weights stop dominating once the KV cache
is real, so a 4.9× smaller model is only a 1.5× smaller process. Quantization buys headroom
in proportion to how much of your memory is weights, and at serving batch sizes that
fraction is not 100 %.

### Speed: decode is slightly *slower*, and the reason is not the kernel

| batch | packed tok/s | bf16 tok/s | ratio | packed ms/step | bf16 ms/step |
|---|---|---|---|---|---|
| 1 | 29.9 | 33.1 | **0.90×** | 33.39 | 30.19 |
| 4 | 116.3 | 126.2 | 0.92× | 34.40 | 31.71 |
| 8 | 235.6 | 250.9 | 0.94× | 33.95 | 31.89 |
| 32 | 610.2 | 1012.6 | 0.60× | 52.44 | 31.60 |

Prefill is subtracted rather than amortised — two generations are timed, of 1 and of 64
tokens, and the difference is 63 decode steps with tokenization, prefill and `generate()`
overhead cancelled. Best of three within a run, and each cell above is the best of **three
bf16 runs and two packed runs**, all in one session on one card.

That last detail is not bookkeeping, and an earlier version of this table got it wrong. It
reported **0.98×** at batch 1, from a packed run and a bf16 run measured in *different*
sessions. Repeating the measurement shows why that was not a number: across three bf16 runs
batch-1 decode came out at 33.1, 29.5 and 33.1 tok/s — a 12 % spread between best-of-three
runs, wide enough that the slow bf16 run (29.5) is *slower than either packed run* (29.9,
28.7). A step that is two thousand launches and 69 % idle is paced by the host, and host
jitter moves it further than the entire effect being measured. So the arms have to be
sampled repeatedly and in one session, and the honest batch-1 figure is 0.90×, not 0.98×:
packed decode is about **10 % slower**, not at parity.

The batch-32 collapse is expected and was predicted in the script's docstring: above
`gemv_max_rows() == 8` the runtime falls back to dequantize-then-`F.linear`, which reads
*more* memory than bf16 does. The sweep is reported rather than stopping at the flattering
batch-1 point.

The batch-1 result is the interesting one: **4.9× less weight traffic costs 10 % of the
wall clock.** A 2 B model streaming 3.5 GiB at the 1665 GB/s this card actually achieves
should decode in ~2.3 ms; bf16 measures 30.2. So weight traffic was never what the step was
spending, and
profiling a decode step says exactly where it goes — both arms, through the same script,
[stage7_profile_step.py](stage7_profile_step.py):

| per decode step, batch 1 | bf16 | packed 3.25 bit |
|---|---|---|
| GPU busy (kernel time) | 8.801 ms | **7.763 ms** |
| matmul kernels, all of them | 3.519 ms | **2.486 ms** — 1.42× faster |
| matmul as a share of GPU-busy | 40.0 % | 32.0 % |
| kernel launches | 2013 | 1980 |
| wall time | 28.4 ms | 35.7 ms |
| GPU busy as a share of wall | 31 % | 22 % |

Read the device-side rows and the wall row with different confidence. Kernel time and launch
counts are device measurements and are stable; the wall figures are single samples of a
quantity shown above to swing 12 % run to run, so the *magnitude* of the deficit comes from
the multi-run table above (0.90×) and not from dividing 35.7 by 28.4.

Everything else is counted from kernel-level profiler events only. `torch.profiler` also
attributes device time to op-level entries such as `aten::mm`, and summing both
double-counts — an earlier pass of this measurement did exactly that and reported 58 % and
4551 launches. The script asserts on the event type rather than trusting it.

**The kernel does precisely what it was built to do, and it is still not enough.** In-model
matmul time falls 3.519 → 2.486 ms, a **1.42×** speedup — which independently corroborates
the isolated benchmark below, since a 1.2–1.5× blend over this model's shapes with one much
faster vocabulary projection is exactly what 1.42× looks like. And the whole of the 1.038 ms
drop in GPU-busy is accounted for by the 1.033 ms drop in matmul: nothing else moved, which
is what a correct weight-only change should look like.

But 1.04 ms against a ~30 ms step is **3.5 %**, and the packed path gives back more than
that on the host. It issues *fewer* kernels than bf16 (1980 vs 2013) and yet takes longer,
so the extra cost is per-call host work — `DynQuantLinear.forward` is Python, and a
`torch.ops.dynquant.gemv` dispatch costs more than `F.linear`'s. Pricing the gap two ways —
against the paired stage-6 walls (+3.2 ms) and against the profiled walls (+7.3 ms), both
after crediting back the 1.04 ms of GPU time saved — puts it at roughly **20–45 µs per
packed module per step** across 187 modules. That is an order of magnitude, not a
measurement, but it is the right order: in a step that is ~2000 launches and ~70 % idle,
host cost per module is the currency, and the kernel is not spending it.

The rest of those launches are why the ceiling is so low to begin with. This build has no
`flash-linear-attention` fast path installed, so the linear-attention recurrence runs a
torch fallback, and the hottest non-matmul kernels in both arms are elementwise copies and
reductions from it — none of which touch a weight matrix. Even a matmul made *free* would
return only the 12.4 % of wall that bf16 spends on it.

So the honest statement is not "quantization does not speed up decode", nor "the kernel is
slow". It is: on this model, on this stack, the GPU is idle roughly 70 % of the decode step
waiting to be given work; the kernel wins 1.42× on the fraction that is real work, and that
win is smaller than the host-side cost of dispatching it. CUDA Graphs (P8) remove the
per-call host work and the launch gaps together, which is the one change that would let this
speedup reach the wall clock; installing the FLA fast path removes most of the 2000
launches. Neither is done, and neither is a kernel change.

### The kernel on its own, which is where the real gap was

Model-level numbers cannot separate "the kernel is slow" from "the model gives it nothing
to do". Timing the GEMV directly on this architecture's real shapes, at M = 1, GPU time
only, against measured achievable bandwidth (**1665 GB/s** read, 1675 GB/s copy, measured
on this card at startup):

| shape | bf16 | 2 bit | 3 bit | 4 bit | 8 bit |
|---|---|---|---|---|---|
| `linear_attn.out_proj` 2048×2048 | 8.6 µs / 58 % | 1.23× / 10 % (1.52×) | 1.23× / 15 % (1.55×) | 1.18× / 18 % (1.78×) | 1.09× / 33 % (1.94×) |
| `linear_attn.in_proj_qkv` 6144×2048 | 13.3 µs / 114 % \* | 1.26× / 20 % (1.32×) | 1.24× / 29 % (1.53×) | 1.21× / 37 % (1.47×) | 1.16× / 68 % (1.63×) |
| `mlp.gate_proj` 6144×2048 | 13.3 µs / 114 % \* | 1.26× / 20 % (1.32×) | 1.24× / 29 % (1.53×) | 1.21× / 37 % (1.47×) | 1.16× / 68 % (1.63×) |
| `mlp.down_proj` 2048×6144 | 22.7 µs / 67 % | 1.51× / 14 % (1.81×) | 1.50× / 20 % (1.85×) | 1.40× / 25 % (2.09×) | 1.29× / 44 % (2.36×) |
| `embed/lm_head` 248320×2048 | 606.2 µs / 101 % \* | **2.56×** / 36 % (1.36×) | **2.53×** / 52 % (1.96×) | **2.51×** / 67 % (1.61×) | **1.89×** / 98 % (1.44×) |

Speedup vs bf16 `F.linear`, percent of achievable read bandwidth, and in brackets the
speedup over the general kernel — which is the *unmodified* kernel this section originally
measured, still compiled, still under test, and still reachable with
`DYNQUANT_GEMV_SCALAR=1`. So the bracketed column is a like-for-like before/after on the
same card in the same process. Worst relative error against the dequantized oracle across
the whole sweep: 6.96e−3.

Against what this document previously reported — 0.64–1.83× and 26–41 % of achievable — the
kernel is now **1.09–2.56× and 10–98 %**, and every shape at every width beats bf16 rather
than four of five losing to it.

The table was then re-measured from a clean rebuild of the shipping source, because the
binary these numbers came from predated some late edits to `gemv.cu` — comments and one
host-side simplification of the grid expression, but "only comments" is a claim to verify and
not to assert. The rebuild is codegen-identical: 1384 SASS instructions in the 2-bit M = 1
kernel with the same `FFMA` 264 / `I2F.U16` 256 / `LOP3` 244 / `SHF` 240 histogram, the same
64 registers (72 at 3-bit) and `STACK:0`. It re-measures `embed/lm_head` at 2.55× / 2.54× /
2.51× / 1.90× and 36 / 52 / 67 / 99 % — the same numbers inside run-to-run noise — and passes
all 417 parity tests, which is what actually checks the grid change, since an undersized grid
leaves output rows untouched rather than faulting.

#### What was actually wrong

The original kernel read the activation **one scalar at a time**, and that single fact
accounted for nearly all of the gap. Lane L owned values `[kVals·L, kVals·L + kVals)`, so at
any fixed position within a block the 32 lanes of a warp read 32 two-byte activations spaced
`kVals·2` bytes apart. That one instruction touches sixteen 32-byte sectors and uses two
bytes of each. The activation is only 4 KB and lives in L1 — but L1 is charged per sector,
not per useful byte, so the warp paid 8× the transactions it needed, **once per value rather
than once per word**. The number of values in a row is `K` regardless of bit width, so that
cost did not shrink when the weights did. That is the signature the first measurement showed
and that I misread: 2/3/4-bit all landing within 40 % of each other in absolute time while
reading 1.4–2× different amounts of weight.

Making a lane's values *consecutive* turns its activations into a contiguous run readable
with one 128-bit load, and across the warp those loads are contiguous too, so the
sixteen-sector access becomes four. That is the whole of round 1, and it is where the
26–41 % → 35–99 % came from. The cost is that chunk sizes have to be made to line up: a
chunk must be a whole number of values *and* a whole number of vector loads, which is why
3-bit reads `uint2` (6 words, 64 values) where the other widths read `uint4`. The invariant
`kValues · BITS == kWords · 32` holds at every width, which is what keeps chunks
value-aligned and lets `decode_value`'s straddle path stay in bounds without a guard.

#### Retraction: the 3-bit claim in the previous version of this section was wrong

That version read:

> The 3-bit path is the worst of the three widths despite reading less than 4-bit (441 vs
> 666 GB/s), which points at a specific cause: 32 values per 3 words leaves group loads
> unaligned to 128 bits, so the `uint4` vectorized loads the design calls for are not
> happening there.

The diagnosis was wrong and the evidence was over-read. The `uint4` loads were not happening
**at any width** — the kernel had no vectorized load path at all, for weights or
activations — so "not happening there" identified 3-bit as the exception to something that
was in fact universal. Once loads were vectorized the timings became monotone in width
(243 / 247 / 261 / 318 µs at 2/3/4/8-bit), which retires the alignment story directly: an
alignment defect specific to 3-bit would have survived the change. What remains at 3-bit is
narrower and cheaper than claimed — a 64-bit load instead of 128, two straddling values per
32 handled by an extra shift-or, and 10.67 values per word against 8 — and it now shows up
as 3-bit sitting *between* 2-bit and 4-bit where it belongs, not below both.

#### Two further rounds, and what they measured

**Round 2 — arithmetic per weight.** The SASS showed 257 `I2F.F32.U32` against 512 `FFMA` in
the 2-bit inner loop. On sm_80 the programming guide rates conversions *from* 8- and 16-bit
integers at 64 results per SM per clock and **all other conversions at 16** — so the
`(float)(uint32_t)` on every decoded weight was a quarter-rate instruction costing more issue
slots than both FFMAs that consumed it. Narrowing through `unsigned short` first is lossless
for a ≤8-bit value and emits `I2F.F32.U16` at full rate. Separately, scale and offset were
hoisted out of the per-value loop using

```
sum_i (q_i·scale + offset)·x_i  ==  scale·sum_i q_i x_i  +  offset·sum_i x_i
```

which is legal because a chunk lies inside one group (the host checks this before selecting
the kernel), costs one FFMA per value instead of two, shares one activation sum across the
rows a warp owns, and is *more* accurate — the reconstructed weight is never materialised, so
there is one rounding per value where there were two. SASS confirms it: `FFMA` 512 → 264,
total inner-loop instructions 1568 → 1384.

Wall-clock gain: **2.5 % at 2-bit, 7 % at 4-bit.** I had predicted ~2×. That miss is the
most useful measurement in this section, because a −11.7 % instruction count buying −2.5 %
time means the kernel is not FP-pipe bound — it is **issue- and memory-latency bound**, and
the arithmetic was never the ceiling.

**Round 3 — rows per warp.** A warp reads all of `x` to produce `kRowsPerWarp` outputs, so
the activation is re-read `num_rows / kRowsPerWarp` times per launch: on this vocabulary
projection at 2-bit that is 254 MB of activation traffic against 127 MB of weight, and the
ratio halves with each doubling of bit width — suggestively the order the shortfall falls in.
Taking more rows per warp cuts that traffic and costs registers. Both neighbours lose
(`embed/lm_head`, M = 1, percent of achievable at 2/3/4/8-bit):

| rows/warp | registers | 2 bit | 3 bit | 4 bit | 8 bit |
|---|---|---|---|---|---|
| 2 | 48–56 | 34 % | 50 % | 65 % | 98 % |
| **4** | **64–72** | **36 %** | **52 %** | **67 %** | **98–99 %** |
| 8 | 96–128, spills at M ≥ 4 | 36 % | 49 % | 64 % | 97 % |

So neither activation re-read nor occupancy is the binding constraint: the maximum is flat
and in the middle, which is what an issue-bound kernel looks like. The hypothesis that
motivated the sweep was wrong in both directions, and 4 — the value already in the file —
is optimal. It is now recorded in `gemv.cu` as a measured constant rather than an argued one.

#### The gate, and where the remaining gap actually is

`embed/lm_head` is the only shape in this model where "bandwidth" is the right word at all;
everything else is an 8–23 µs kernel where both paths are latency-bound. Against the ≥70 %
target:

| width | % of achievable | gate |
|---|---|---|
| 8 bit | 98 % | **met** |
| 4 bit | 67 % | 3 points short |
| 3 bit | 52 % | not met |
| 2 bit | 36 % | not met |

**The gate is not met at 2, 3 and 4 bits.** It is, however, now *located* rather than
guessed at, and the instruction accounting says where. Per (value, row) the inner loop costs
about 5.4 instructions — 1384 SASS instructions over the 4 rows × 64 values a warp iteration
covers, of which `FFMA` 264, `I2F.U16` 256, `LOP3` 244, `SHF` 240, `FADD` 84, `PRMT` 74, so
per (value, row): `FFMA` 1.03, `I2F.U16` 1.00, `LOP3` 0.95, `SHF` 0.94, `FADD` 0.33,
`PRMT` 0.29, and ~0.86 of loop overhead. At 108 SMs × 4 schedulers × 1 warp-instruction per
clock at 1.41 GHz that is 19.5e12 thread-instruction slots per second, so 248320 × 2048
value-row pairs at 5.4 instructions each has a **pure issue-rate floor of ~141 µs** against
the 237 µs measured — about 60 % issue efficiency. The rest is latency that cannot be hidden
where it is: at K = 2048 and 64 values per chunk there are exactly 32 chunks per row, one per
lane, so **each warp executes exactly one iteration of the main loop** and there is no
intra-warp memory-level parallelism at all. Latency is hidden only across warps.

Getting a further large multiple therefore is not more of this. It needs the weights to stop
becoming floats one at a time: `LOP3`/`PRMT` straight into `half2` pairs and accumulation
through `mma.sync` tensor cores, which is the AWQ/Marlin route and is **P7**, a separate
phase with its own gate.

One shortcut that does not work here, recorded so it is not attempted again: the
magic-number trick (`__int_as_float(0x4B000000 | q)`, which makes an integer into a float
with no conversion instruction) requires folding a −2²³ bias into the group offset. In fp32
that makes `offset' = −(2²³ + z)·scale`, a magnitude around 8.4e6·scale storable to only 24
bits of mantissa — leaving 0.5·scale of absolute error on a result whose magnitude at 2-bit
is at most 3·scale, i.e. ~17 % error. Folding the correction into a per-chunk activation-sum
term fails the same way, since `2²³·xsum` swamps `Σ q·x`. Marlin and AWQ get away with it
because they work in fp16, where the analogous constant is 1024 and every step stays exact.
An explicit per-value `FADD` would be exact but costs the same issue slot as the
`I2F.U16` it replaces, so it buys nothing.

The bf16 column exceeding 100 % on three shapes is L2 residency across the benchmark loop —
a 25 MB weight fits the A100's 40 MB L2 and gets read once. That flatters bf16 here relative
to a real decode loop, which touches every weight once per token, so those speedup cells are
pessimistic. The in-model figure is the unflattered one: the 187 matmuls stream 3.504 GiB in
3.58 ms, which is ~1051 GB/s, 63 % of achievable. That is the number the packed kernel has to
beat, and on the vocabulary projection at 4 and 8 bits it now does.

### What this section establishes, and what it does not

Established, on hardware: the kernels are numerically exact against the simulated path
(4468 = 4468 on 5314 problems, and ≤6.96e−3 relative to the dequantized oracle in
isolation, which is bf16 accumulation noise); resident VRAM matches the manifest to 0.03 %
and is 4.8× below bf16; **417 kernel-parity tests** pass on this card across every geometry
× width × M × dtype, inside a full suite of **1169 passed, 34 skipped, 0 failed**; and all
four `compute-sanitizer` tools — **memcheck, initcheck, racecheck, synccheck** — report 0
errors and 0 hazards over that parity suite.

The four-tool sweep matters more than it did before this round. The rewritten kernel reads
weights through 128-bit `uint4` loads and activations through vector loads at a computed
byte offset, both of which are exactly the constructs that turn an off-by-one in the chunk
geometry into an out-of-bounds read that produces *plausible* numbers rather than a fault.
memcheck is the check that the `kValues · BITS == kWords · 32` invariant actually holds at
every width rather than merely being asserted in a comment.

Not established: any decode speedup, and the ≥70 %-of-achievable-bandwidth gate at three of
the four widths. In isolation the kernel is **1.09–2.56×** bf16 on every shape at every
width, and in the model its matmul time is **1.42×** faster — but model decode is **0.90×**
at batch 1 and 0.60× at batch 32, because the win is 3.5 % of a step whose remaining 96.5 %
this phase does not touch. On bandwidth the gate is **met at 8 bits (98 %)** and missed at
4, 3 and 2 bits (67 %, 52 %, 36 %). Both shortfalls are now attributed rather than guessed —
instruction-issue saturation in the kernel, host-side dispatch cost in the model — and both
fixes are named phases (P7, P8) rather than further tuning of this one.

Also not measured here: prefill (it goes through dequantize-then-GEMM, and at batch 32 the
packed and bf16 prefills are 1605 ms and 1599 ms — indistinguishable, as they should be);
any model other than this one; any tensor-core path (P7); MoE (P8). The full 5314-problem
evaluation took 355.1 s packed against 344.6 s bf16, **3 % slower end to end**, because the
evaluation is prefill-dominated and prefill is the same work either way.

Packing 187 modules on the CPU took 447 s, which is a quantizer cost rather than a runtime
one and is the thing P5's CUDA path is for.

## A defect this experiment found

The first ~4-bit run scored 20.83 % on a 24-prompt smoke. The bit map looked entirely
reasonable — plausible width histogram, target hit to four decimals, floor violations
listed without comment — but 20 of 187 modules sat at the **2-bit minimum at a 4-bit
target**, with relative errors up to 0.61.

The cause was in `percentile_ranks`, not in the encoder or the allocator. Ranks were
mapped onto the closed interval `[0, 1]` via `(rank − 1) / (n − 1)`, so the
lowest-ranked member of every group scored exactly 0. The importance score is a
*product* of two ranks, and the allocator prices damage as `score × params × Δerror` —
so one zero zeroed the score, and a module with score zero is free to destroy. The
breakdown was exactly two modules per role across ten roles: the minimum of each role
in each of the two signals.

The fix is the Hazen plotting position `(rank − 0.5) / n`, which maps onto the open
interval. After it, no module is below 3 bits at a 4-bit target. The same change
removes the mirror-image artifact at the top, where a single-member role — `EMBEDDING`,
the largest tensor in the model — was handed rank 1.0 by construction rather than by
measurement.

Two things about this are worth recording. First, the encoder was suspected and
cleared: an independently written brute-force search agreed with `grid.py` exactly on
every per-group clip ratio and SSE, and the per-bit errors matched theory. Second,
**the scorer had no test file at all** before this, despite being the part of the
method every downstream decision reads. A defect there does not raise; it produces a
complete, plausible bit map driven by the wrong ordering.

## A second defect: the harness could not run the right test

The first pass stored 12 sample predictions per arm and the summary counts, nothing
else. That is enough to report an accuracy and enough to compare two arms as
independent proportions — and it is not enough for the test this design calls for.

Scoring two models on the same fixed problem set is a *paired* experiment. Most of the
statistical power lives in the pairing: of 1319 problems, only the ~180 that flipped
carry any information about the difference, and the 1140 that both arms got right or
both got wrong contribute no variance at all. Treating the two accuracies as
independent throws that away, and on this data it roughly doubles the interval.

The failure was not that the wrong number was printed. It was that the *right number
was unrecoverable* after the GPU-hours were spent. Fixing it meant adding
`Gsm8kResult.hits` — one boolean per problem, mandatory and unsampled — and re-running
all six arms. `eval/compare.py` now implements McNemar's exact test and Agresti's
paired standard error, with tests that concentrate on refusing misuse (mismatched
vector lengths, negative counts) rather than on the arithmetic, because a comparison
run on the wrong data returns a plausible p-value and nothing about it looks wrong.

McNemar is used *exact* rather than as a chi-square. The interesting comparisons here
are the close ones, close comparisons have few discordant pairs, and few discordant
pairs is precisely where the chi-square approximation stops being trustworthy.

## A third defect: the fallback priced two populations off different rungs

Found by a test written for the sensitivity work, before it could reach a run.

Not every module is guaranteed a measured `sens_i(b)` — a module the tracker never saw a
backward pass through has no moments — so `_apply_fallback_scale` in
[`allocate/knapsack.py`](../../packages/dynquant-core/src/dynquant/allocate/knapsack.py)
prices the unmeasured ones with the old `score × params` formula, calibrated so the two
populations are comparable: take the median next-step value of each and scale one onto the
other. Mixing an unscaled `score × params` into a knapsack full of measured sensitivities
would otherwise be arbitrary units against arbitrary units.

The calibration took each median over that population's step **from its own floor**. MLP
up/down floor at 3 bits, so their next step is 3→4 and spans `4⁻³ − 4⁻⁴` = 0.011719;
attention floors at 4, so its next step is 4→8 and spans 0.003891 — three times narrower.
Both formulas already carry that span factor, so calibrating on the raw step values
applies it twice. Measured median 191.2, proxied median 576, `scale = 0.332`: whichever
population happened to sit at the lower floor was scaled down by two-thirds and became the
first thing the downgrade pass cut, on no evidence at all. On this architecture that is
MLP up/down — 604 M parameters, a third of the model.

The fix is to normalise both medians by their own step's span before taking the ratio. On
the fixture the scale goes 0.332 → exactly 1.000.

Two notes. The bug is invisible when every module is measured (`unestimable` is empty in
the run above, so **the +10.29 result does not depend on this fix**) and invisible again
when both populations share a floor — it needs an architecture where they differ, which is
most of them. And the test that caught it asserts an *ordering*, not a constant: a
median-ratio calibration is entitled to absorb any constant, so a test pinning one would
have failed for the wrong reason on the next legitimate change.

## A fourth defect: the tied table untied itself on `.to("cuda")`

Found while preparing the packed run, by nothing more than asking what `model.to("cuda")`
does to a tensor that two modules share — and it would have silently falsified the headline
memory number.

`pack_model` follows tied weights, so a tied embedding and LM head get one packed table
between them. The obvious implementation registers the *same* tensor as a buffer on both
modules, and on the CPU it is genuinely one tensor: the pointers match, and the test that
asserted they match passed.

`nn.Module._apply` — which is all `.to(device)` is — calls its conversion function once per
registered buffer and memoizes nothing. Two buffers that happen to be the same tensor
therefore arrive on the GPU as **two** tensors. Confirmed directly before fixing it: 8192
bytes moved for one 4096-byte tensor.

On this model the tied table is 508.6 M of 1881.3 M parameters — 27 % — or 0.192 GiB once
packed. The model would have reached the GPU 27 % larger than the manifest said, and the
`resident on device` line above would have read 0.9161 GiB against a 0.7118 GiB prediction.
The packed runtime's entire reason for existing, falsified at the exact moment the weights
reach the device it is claimed about.

The fix is structural rather than a memo: the tied module registers no table at all and
reads the owner's through a `holder` property, with the owner held via
`object.__setattr__` so it does not appear a second time in `named_modules()` or in every
`state_dict` key prefix. There is one tensor because there is one buffer. As a side effect
a tied module's `state_dict` now carries only its bias, which is the convention
`transformers` already uses for tied weights and is what a checkpoint should contain — one
table, written once.

Three notes. The regression test asserts through `.to()` rather than through pointer
equality on the CPU, because CPU pointer equality is what passed while the bug was live.
A dtype-only `.to(torch.float64)` exercises the same `_apply` path a real `.cuda()` does,
so the test needs no GPU. And the packed run is the confirmation: `packed 187 modules:
3.504 GiB -> 0.711 GiB (4.93×); 1 tied module(s) share 0.192 GiB of that, counted once`.

## Reproducing

```bash
cd experiments/qwen35_2b
export PYTHONPATH=/path/to/packages/dynquant-core/src

python screen_headroom.py   # stage 0: which task has room? ~20 min, no fine-tune

export DQ_TASK=casehold     # or gsm8k; selects the task in tasks.py
./run_all.sh                # all six arms, resumable, then the table
```

Stage 0 is separate from `run_all.sh` and comes before it because it chooses the argument
you pass. Running it is optional for reproducing either table and was not optional for
producing them: skipping it is what cost the GSM8K run.

`run_all.sh` skips any stage whose output already exists and prints every skip, so an
interrupted run resumes without silently reusing a stale checkpoint. The stages are also
individually runnable, which is how the run was actually driven — it is long and
GPU-bound, and a failure in quantization should not cost the fine-tune:

```bash
python stage1_eval_base.py                     # measurement 1
python stage2_finetune.py                      # writes finetuned/ and stats
python evaluate.py --model "$DQ_RUN_DIR/finetuned" --name stage3_finetuned --label "fine-tuned"
python stage4_allocate.py --targets 4.25 3.25  # writes stage4_bitmaps.json
python stage5_quantize.py --target 4.25
python stage5_quantize.py --target 4.25 --uniform 4
python stage5_quantize.py --target 3.25
python stage5_quantize.py --target 3.25 --uniform 3

python results_table.py                        # the table above
python inspect_errors.py stage5_4p25_quant     # per-layer error, by width and role
python inspect_allocation.py                   # does the score drive the widths?
```

The sensitivity arm is a separate stage 4, because it needs a calibration pass the
rank-product path does not. It writes a stage-5-compatible bit map, so stage 5 is unchanged
apart from being pointed at it:

```bash
python stage4_sensitivity.py --targets 3.25 --calib-split validation --calib-rows 512
python stage5_quantize.py --target 3.25 --bitmaps "$DQ_RUN_DIR/stage4_sensitivity.json" \
                          --name stage5_3p25_sens --label "3.25 bit, measured sensitivity"
python compare_3p25_arms.py                    # the paired tests in *The fix*
```

Stage 6 is the packed run, and it needs a GPU with the compiled kernels — it refuses to
report kernel numbers from the torch backend rather than quietly producing plausible ones:

```bash
python stage6_packed_eval.py --target 3.25          # packed weights, CUDA kernels
python stage6_packed_eval.py --dense --skip-eval    # the bf16 baseline, same code path
```

The bf16 arm runs through the same script on purpose. Comparing a number measured here
against one remembered from another script is how a 3 % harness difference becomes a
reported speedup.

`stage4_sensitivity.py` collects through the shipped `SignalTracker`
(`collect_channel_moments=True`) rather than through hooks written for the experiment. An
estimate that only works when the moments come from a bespoke script is not a feature of
the package, and the difference would not show up in any number it prints.

To check the whole chain in ~15 minutes before committing a real run — worth doing,
because stages 4 and 5 consume the stats file stage 2 writes and a schema mismatch
between them otherwise surfaces hours in:

```bash
DQ_RUN_DIR=/tmp/smoke LIMIT=32 MAX_STEPS=60 ./run_all.sh
```
