# Mistral-7B-Instruct-v0.3 on Banking77: does the allocator hold on a 7B?

[`RESULTS.md`](RESULTS.md) established the sensitivity allocator on Qwen3.5-2B-Base across
two tasks. Everything it concluded was measured on one model, at one scale, under full
fine-tuning. This run changes four things at once and asks whether the finding survives:

| | `RESULTS.md` | here |
|---|---|---|
| model | Qwen3.5-2B-Base, 1.88 B | Mistral-7B-Instruct-v0.3, 7.25 B |
| architecture | hybrid linear/full attention, **tied** embedding | dense GQA, **untied** embedding |
| task | GSM8K, CaseHOLD | Banking77 |
| regime | full fine-tune, lr 1e-5 | **LoRA r=32**, lr 1e-4 |

Changing four things at once is normally how you learn nothing. It is the right design
here because the question is not "which of these mattered" but "does the method transfer
at all" — and a method that only works on one 2 B model with tied embeddings and a
full fine-tune is not a method. If the qualitative result replicates across all four
changes simultaneously, that is stronger evidence than four one-at-a-time runs, each of
which would still share three conditions with the original.

## Why Banking77

Headroom is a property of the model/dataset **pair**, not of the dataset, so the screen
in [`../screen/screen_datasets.py`](../screen/screen_datasets.py) was re-run against base
Mistral rather than inherited. The rule it enforces was learned from GSM8K, where a full
six-arm run was spent before anyone checked that the base model already scored 66% and
there was therefore no fine-tuning gain for quantization damage to be read against.

| candidate | base few-shot | chance | supervised reference | headroom |
|---|---|---|---|---|
| **Banking77** | **36.3%** (full 3 080) | 1.3% | ~93% (BERT-base) | **~57 pts** |
| MNLI | 71.0% (300) | 33.3% | ~90% (BERT-large) | ~19 pts |
| PubMedQA | 57.5% (200) | 33.3% | ~73% | ~15 pts |
| LogiQA | 41.3% (300) | 25.0% | ~40% (RoBERTa-large) | none |

LogiQA is the informative rejection: the base model is already *above* the supervised
reference, so a fine-tune there could only move the number down.

The 1.3% chance floor is the reason this task is the most sensitive of the four to
quantization damage. CaseHOLD bottoms out at 20% and GSM8K at 0%; here a model whose
weights have been destroyed cannot land on the right intent by luck, so the distance
between "still works" and "broken" is nearly the full range of the scale. There is no
cushion for a small regression to hide under, which is the direction an evaluation should
err in.

## What was run

| | |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.3` — `MistralForCausalLM`, 32 layers, hidden 4096, intermediate 14336, vocab 32768, GQA 32/8 heads |
| Parameters | 7.2478 B across 226 quantizable modules; embedding and LM head **untied**, 134 M each |
| Fine-tune | 2 epochs SFT on `train`, LoRA r=32 α=64 dropout 0.05 on `all-linear`, lr 1e-4 cosine, effective batch 32, bf16, 626 steps in 60.4 min, signals collected by `DynQuantCallback` over 226 modules with the `outer_exact` estimator |
| Training rows | 9 999 = 10 003 minus the 4 few-shot exemplars, which are drawn from `train` by a fixed seed and held out of the fine-tuning set. Zero dropped for length: a row is the 77-line taxonomy plus one sentence plus one index, so nothing approaches the 1024-token cap, and the task is configured so that a nonzero drop count would be a config error rather than a data property |
| Quantizer | group-128 asymmetric min/max, MSE-optimal clip search over α ∈ {1.00 … 0.80}; 15.1 s for the whole 7 B |
| Allocator | `combine="plasticity"` — plasticity rank within role, the package default — plus role floors and a greedy ROI knapsack. The signal reaches the map: 24 / 226 modules differ from a uniform-score allocation at 4.25 bits, 89 / 226 at 3.25 |
| Budgets | 4.25 and 3.25 stored bits, each against a same-size uniform control |
| Eval | full 3 080-row `test` split, 4-shot, 6 new tokens, exact match on the intent index |
| Hardware | 1 × A100 80 GB PCIe, torch 2.13.0+cu130, transformers 5.14.1, peft 0.19.1 |

### Why LoRA, and why it costs nothing here

A 7 B full fine-tune needs ~14.5 GB of bf16 parameters, ~14.5 GB of gradients and ~58 GB
of fp32 AdamW moments — ~87 GB before a single activation, against an 80 GB card. LoRA is
not a preference, it is the only way this run fits.

It costs nothing in signal fidelity, and that is worth stating precisely rather than
hoping. The default `outer_exact` estimator does not read `.grad` off the base weight —
it stashes `x` in a forward hook, takes `δ = ∇_Y L` from a full backward hook, and
reconstructs `∇W = δxᵀ`. Since `y = Wx + s·BAx`, the tensor `∂L/∂y` is *identical*
whether or not an adapter is attached. LoRA changes what the model learns; it does not
change how faithfully the signal is measured. 83.9 M parameters trained, 1.14% of the
model, final train loss 0.141 — low for a task where only ~4 tokens per sequence are
unmasked, which says the model learned the query→intent mapping rather than the output
format.

The learning rate is resolved from the regime rather than shared with the Qwen runs.
1e-5 on a rank-32 adapter trains to a flat result that is indistinguishable from the task
having no headroom — which is exactly the conclusion this experiment exists to draw, and
the one it must not draw by accident.

The adapter is merged before the checkpoint is written. Saving adapter weights would
leave the fine-tuned arm scoring the *base* model under the label "fine-tuned": an arm
that looks measured, sits in the table, and is not.

### Stored bits, not payload bits

`4.25` is what the filesystem reports, metadata included. At group 128 with an fp16 scale
and offset per group, metadata costs 0.25 bits per weight, so the *payload* is 4.0 bits —
the same convention as a "4-bit g128" GPTQ checkpoint.

### What these numbers are not

Accuracy here is real: the quantizer writes dequantized values back in place, and those
are bit-for-bit the values a packed checkpoint reconstructs (pinned by
`tests/test_quantizer.py`). **Memory and speed are not measured in this run and are not
claimed.** The GiB column is the size the packed checkpoint would occupy on disk, computed
from the format's own accounting; the model as evaluated sat in memory at bf16. Those two
claims are measured for the packed-kernel path in
[`RESULTS.md`](RESULTS.md#running-it-packed-the-kernels-and-what-they-buy), which also
establishes that simulated quantization tells the truth about accuracy — exactly.

## What the allocator chose

Both maps spread across three widths with clear role separation, so the allocator is
doing work rather than returning a floor map.

| target | width histogram | mean |
|---|---|---|
| 4.25 | 3b × 10, 4b × 212, 8b × 4 | 4.027 |
| 3.25 | 2b × 35, 3b × 137, 4b × 54 | 3.084 |

Per role, sorted by what the signal was willing to pay for:

| role | n | 4.25 map | 3.25 map |
|---|---|---|---|
| `lm_head` | 1 | **8.000** | 3.000 |
| `v_proj` | 32 | **4.250** | 3.188 |
| `k_proj` | 32 | **4.125** | 3.188 |
| `q_proj` | 32 | 4.000 | 3.188 |
| `gate_proj` | 32 | 4.000 | 3.188 |
| `o_proj` | 32 | 4.000 | 3.156 |
| `embed_tokens` | 1 | 4.000 | 3.000 |
| `up_proj` | 32 | 3.844 | **2.844** |
| `down_proj` | 32 | 3.844 | **2.844** |

This replicates the CaseHOLD reallocation on a different model, a different task, a
different architecture family and a different fine-tuning regime: **bits come out of MLP
up/down and go into attention k and v.** Two independent runs agreeing on the direction is
evidence the signal is measuring something about transformer structure rather than fitting
one checkpoint.

The untied embedding is what lets `lm_head` reach 8 bits at the 4.25 target. On
Qwen3.5-2B the embedding and LM head are one tensor carrying 27% of the model, so pricing
the head up drags the embedding with it and the budget forbids it. Here they are two
134 M tensors the allocator prices independently — a structural difference that changes
what the same policy can express.

## Results

Full 3 080-row test split, every arm, no subsampling.

| measurement point | stored bits | GiB | exact match | ±1SE | correct | unparsed |
|---|---|---|---|---|---|---|
| 1. base, no fine-tune (fp16) | 16.000 | 13.500 | **36.27%** | 0.87 | 1117 / 3080 | 15 |
| 2. fine-tuned (fp16) | 16.000 | 13.500 | **94.51%** | 0.41 | 2911 / 3080 | 0 |
| 3. quantized ~4 bit, DynQuant allocation | 4.250 | 3.586 | **94.32%** | 0.42 | 2905 / 3080 | 0 |
| &nbsp;&nbsp;&nbsp;control: uniform 4 bit, same budget | 4.250 | 3.586 | 94.22% | 0.42 | 2902 / 3080 | 0 |
| 4. quantized ~3 bit, DynQuant allocation | 3.250 | 2.742 | **93.34%** | 0.45 | 2875 / 3080 | 0 |
| &nbsp;&nbsp;&nbsp;control: uniform 3 bit, same budget | 3.250 | 2.742 | 91.98% | 0.49 | 2833 / 3080 | 0 |
| &nbsp;&nbsp;&nbsp;(guessing) | — | — | 1.30% | | | |

| comparison | delta | paired 95% CI | flips | p (McNemar) | verdict |
|---|---|---|---|---|---|
| did the fine-tune move the task? | **+58.25** | [+56.45, +60.04] | 1816 / 22 | ~0 | **separated** |
| cost of quantizing to ~4 bit | −0.19 | [−0.53, +0.14] | 11 / 17 | 0.345 | not separated |
| did allocation beat uniform at ~4 bit? | +0.10 | [−0.04, +0.24] | 4 / 1 | 0.375 | not separated |
| cost of quantizing to ~3 bit | −1.17 | [−1.69, −0.65] | 16 / 52 | 1.4e−05 | **separated** |
| did allocation beat uniform at ~3 bit? | **+1.36** | [+0.73, +2.00] | 71 / 29 | 3.2e−05 | **separated** |

`flips` is problems only-the-left-arm got right / only-the-right-arm got right. Every
comparison is paired — the same 3 080 problems in the same order, scored twice — so
McNemar's exact test on the discordant pairs is the verdict column.

The fine-tune moved the task 58.25 points, to above the ~93% BERT-base supervised
reference, and it did so almost without trading anything away: 1 816 problems fixed
against 22 broken. It also took the unparseable count from 15 to 0, which is worth
separating from the accuracy gain — part of what two epochs bought was the model
learning to emit a bare intent index instead of a sentence. This is the headroom the
screen predicted, and it is what makes the rest of the table readable: quantization
damage is now being measured against 58 points of real, task-specific capability rather
than against a model that never learned anything to lose.

### At 4.25 bits there is nothing to recover

Quantizing to 4.25 stored bits costs **−0.19 points, p = 0.345 — not separated**. A 7 B
model at a 4-bit payload is, on this task, indistinguishable from bf16 while being
**3.77× smaller**. That is the practically interesting result in this half of the table.

The allocator's own comparison is **+0.10, p = 0.375 — also not separated**, and the flip
counts say why more clearly than the p-value does: **4 problems changed one way and 1 the
other, out of 3 080**. Five discordant pairs is not a small effect, it is no effect. Nobody
should read a direction into the sign.

This is a null result for the allocator and it should be reported as one. It is also the
expected result: an allocator redistributes damage, and here there is no damage to
redistribute. Both arms sit within noise of fp16, so the ceiling on what any allocation
could recover is roughly two tenths of a point. The same pattern held on Qwen3.5-2B under
a *different* scorer — GSM8K +1.36 (p = 0.20), CaseHOLD −0.13 (p = 0.56), neither
separated — so across two models, three tasks and two scoring rules, 4.25 bits is
consistently *not* where this method earns anything. That is a robust negative: it does
not depend on which score drives the allocation, which is what you would expect if the
cause is that there is no damage to redistribute.

The honest summary of the 4-bit regime is that uniform quantization is already good
enough, and the reason to run DynQuant there is not accuracy.

### At 3.25 bits the allocator recovers half the damage

At 3.25 stored bits, damage finally exists, and the arms separate.

Uniform 3-bit surrenders **2.53 points** of the fine-tuning gain (94.51 → 91.98).
Allocated 3-bit surrenders **1.17** (94.51 → 93.34). The difference is **+1.36 points,
p = 3.2e−05, separated**, with **71 problems fixed against 29 broken** — not a uniform
softening of the loss but a genuine trade, and the trade nets +42 problems. Allocation
therefore recovers **53.8%** of what uniform quantization destroys, at exactly the same
2.742 GiB on disk and **4.92×** smaller than bf16.

Three things about that number deserve to be said plainly, and the first is a caution
against reading it next to the wrong number.

**This is the plasticity scorer, not the sensitivity one.** These arms were allocated with
`combine="plasticity"`, the package default. That default exists because the paper's
`Rank(saliency) × Rank(plasticity)` product *lost* to uniform on CaseHOLD by 2.03 points,
and it is not the same thing as the Gauss-Newton `fisher_diff` allocator that won there by
+10.29. **That allocator was not run here.** So +1.36 should not be compared to +10.29;
they are different methods, and the CaseHOLD run's own comparison of them is the reason
to expect the gap to be real. What this run *does* establish is something the CaseHOLD
work could not: until now the plasticity default had only ever been justified by an
in-batch loss proxy, where it recovered 81.58% of uniform-3-bit damage at this budget on
one model and one task. This is its **first end-to-end confirmation, on a second model and
a second task** — 53.8% of the damage, measured in accuracy on 3 080 held-out problems
rather than in loss on a fixed batch. A default that had been chosen on a proxy is now a
default that has been measured.

The two figures are computed the same way — share of uniform-3-bit damage recovered — but
the earlier one is a loss proxy and this one is accuracy, so 81.58% and 53.8% are not two
attempts at the same measurement and the gap between them should not be read as a decline.

**The obvious next arm is the one that won last time.** `fisher_diff` reached 95.76%
against plasticity's 81.58% on the same in-batch screen, so there is likely room above
+1.36 on this model. Running it here is the single highest-value follow-up, and until it
is run the honest claim for Mistral/Banking77 is the +1.36 that was measured.

**Allocation reduces the cost of 3-bit quantization; it does not eliminate it.** The
allocated arm is still separated from fp16 (−1.17, p = 1.4e−05). Anyone choosing 3.25 bits
is spending about a point of accuracy for 4.92× compression, and the allocator's
contribution is that the point is one and not two and a half.

## What replicated

Across all four changed conditions at once:

1. **The allocator helps where damage exists and not otherwise** — null at 4.25 bits on
   both models, separated at 3.25 bits on both.
2. **The reallocation direction** — bits out of MLP `up_proj`/`down_proj`, into attention
   `k_proj`/`v_proj`. Two runs sharing no model, task, architecture family or training
   regime agreeing on which roles are worth paying for is evidence the signal measures
   transformer structure rather than one checkpoint.
3. **The current default scorer is on the right side of zero end-to-end**, which had never
   been shown. On CaseHOLD it was screened in-batch; the arm that was measured end-to-end
   there was the rank product, and it lost.

What did *not* transfer, and could not: the magnitude. A 7 B model at 3 bits loses 2.53
points to uniform quantization where the 2 B lost 14.69. The method's value scales with
the damage, which means it matters most exactly where compression is most aggressive —
and correspondingly, that a bigger model buys robustness that reduces how much any
allocator can add.
