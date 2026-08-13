# Phase 4, S4: Mistral-7B on text-to-SQL — fine-tune, allocate, publish

**Measured 2026-08-12/13**, on `mistralai/Mistral-7B-Instruct-v0.3` fine-tuned for two epochs
on a three-source text-to-SQL mixture and quantized at two byte budgets against GPTQ and AWQ.
Scripts: [`scripts/run_s2_finetune.py`](../../scripts/run_s2_finetune.py),
[`experiments/phase4/arms_lfm2.py`](../../experiments/phase4/arms_lfm2.py),
[`experiments/phase4/panel_table.py`](../../experiments/phase4/panel_table.py),
[`experiments/phase4/publish_panel.py`](../../experiments/phase4/publish_panel.py),
[`experiments/phase4/push_to_hub.py`](../../experiments/phase4/push_to_hub.py).
Commit `5959fe0`. Companion: [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md), which
built and debugged the benchmark this campaign is scored on.

This is the first campaign in the series run end to end on a **dense** model. Every phase-4
result before it was measured on `lfm2_moe`, where 91.5% of the parameters live in batched
expert banks and the dominant findings — the proxy pricing, the packed-bank publish path, the
`grouped_mm` dispatch confound — are all properties of that bank. None of them apply here.
`experts` reads `{found: eager, ran: eager}` on every arm because there is nothing to
dispatch, so this panel's margins are a difference in quantization and nothing else.

**The result, up front.** At the 4-bit anchor nothing separates — all six comparisons return an
adjusted p of 1.000, and the quantized arms are indistinguishable from the unquantized ceiling.
At the 3-bit anchor DynQuant scores **75.22%** against AWQ's **74.16%** at the same byte budget,
and that **+1.06 does not clear significance** (McNemar p = 0.149, adjusted 0.595). GPTQ at the
same anchor collapses to **6.68%**, but it is symmetric where the other two are asymmetric, so
that 68-point gap is not attributable to allocation and **must not be cited as a win** (§8.3).
What DynQuant 3-bit does establish standing alone: **4.73× compression retaining 96.2% of the
fine-tuned model's accuracy**, with 35% of the model allocated below its role floors.

## 1. The run

| | |
|---|---|
| model | `mistralai/Mistral-7B-Instruct-v0.3`, 7 248 023 552 params, untied |
| training mixture | `gretelai/synthetic_text_to_sql` + `Salesforce/wikisql` + `b-mc2/sql-create-context` |
| evaluation sources | gretel + wikisql + **spider** |
| regime | LoRA rank 32, `outer_exact` estimator, effective batch 32, lr 1e-4, `max_len` 2048 |
| epochs / steps | 2.0 / **2472** optimizer steps |
| final train loss | **0.0540260114363458** |
| wall clock | 13 738.5 s = **3.816 h** (3:48:58), 5.755 samples/s |
| tracked modules | **226**, equal to the stats file's module count |
| box | one NVIDIA RTX PRO 6000 Blackwell Workstation Edition |

226 tracked modules is `32 × 7 + 2` — every `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj` across 32 layers, plus `embed_tokens` and `lm_head`.
Mistral-7B-v0.3 is untied, so those last two are separate tensors and both are scored; the
tied-model arithmetic that has bitten three earlier campaigns does not arise.

## 2. Why this campaign has two boxes, and what the first one was doing

The fine-tune was launched twice. The first attempt reached step 337 of 2472 at **52 s/it**,
projecting 34 hours. I attributed that to a CPU-bound allocation dry run sharing the host and
killed the dry run; the rate did not move. The actual cause was the host's GPU:

| | first box | second box |
|---|---|---|
| SM clock | **510 MHz**, flat | 2325–2790 MHz |
| temperature | **31 °C**, flat under 100% reported utilization | 52 → 70 °C |
| power | ~600 W claimed | 527–550 W |
| step time, same commit and data | **52 s/it** | **5.64–6.73 s/it** |

A card pinned at 510 MHz while reporting full utilization and full board power, with the die
never warming, is not doing the work the counters describe. `nvmlClocksThrottleReasonSwPowerCap`
(`0x4`) was set on both boxes and is *normal* on a 600 W part — the diagnostic is the clock it
settles at and whether the temperature rises, not the throttle flag. The second box, same
commit, same command, same data, ran 9.2× faster and finished in 3.8 hours.

Two things are worth carrying forward from this. **My first diagnosis was wrong**, and it was
wrong in a way that cost a running process: the dry run I killed was healthy and had to be
re-run. And the first attempt's 337 steps were unrecoverable because `save_strategy` was
`"no"`. The relaunch added `--save-steps 100`, which matters more than a checkpoint normally
does here — `DynQuantCallback.on_save` writes the signal map as of that step, and the signal
map is the one artifact the panel cannot run without and cannot recompute without retraining
from zero.

Nothing in this report is measured across the two machines. Every number below comes from the
second box.

## 3. The mixture, and the 3 342 items the decontamination filter took out

| | seen | kept | dropped |
|---|---|---|---|
| conversations | 40 000 | **39 531** | 469, **all** "too long" (1.17%) |
| tokens | | 15 858 075 | |
| supervised tokens | | 1 426 125 | **8.99%** of the total |

The supervised share is low and is *supposed* to be. A text-to-SQL example is a large schema
and question in the prompt and a short query in the answer; only the query is scored, so nine
percent is the shape of the task, not a masking failure. `unmaskable_rate` is **0.0**.

Requested composition was even thirds — `gretel` 13 334, `wikisql` 13 333, `create-context`
13 333 — and the decontamination filter then removed:

| source | removed |
|---|---|
| gretel | 4 |
| wikisql | 16 |
| **`b-mc2/sql-create-context`** | **3 342** |

That 200-to-1 asymmetry is this campaign inheriting a defect the mixture report found and
fixed. §11 of [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) established that
**189 of 200 WikiSQL evaluation items are questions in `b-mc2/sql-create-context`** — the two
corpora share ancestry — and that the guard which reported the mixture clean could not have
fired. It fires now, and what it catches is concentrated exactly where that finding predicted:
one training source is 25% overlap with an evaluation source, the other two are noise.
`sources_overlapping_an_eval_task` is **`[]`** after the filter runs.

Masking resolved to **`template`** from `mask_mode_requested: auto`. The probe was a dead tie —
31 for `template` against 31 for `assemble` — which is the tokenizer telling you that Mistral's
chat template and a hand-assembled turn agree on where the assistant's tokens begin. Ministral
resolved the same probe to `assemble` and Phi to `template`; it is settled per model, never per
campaign (see [`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md)).

## 4. Spider is genuinely held out

Training is gretel + wikisql + **create-context**. Evaluation is gretel + wikisql + **spider**.
So one of the three scored sources is a corpus no arm has seen a single example from, in any
form, and it is not a paraphrase of one either — Spider's schemas are real multi-table
databases with a median schema of 5 849 characters against a few hundred for the other two.

`--limit 2454` is **818 × 3**. Spider's dev split admits exactly 818 items after the execution
filter (1 034 read, 168 returning nothing, 48 degenerate), and `--limit` is divided evenly
across sources rather than taken as a prefix, so 2 454 is the largest total that uses all of
Spider *and* weights the three sources equally. A larger number would keep adding gretel and
wikisql items to a Spider contribution that cannot grow, and the headline figure would drift
toward whichever source survives its filter most often.

`--sources gretel wikisql spider` is spelled out in the launcher rather than left to the
default, because the panel driver defaults to the two sources its banked LFM2 arms were scored
on — a default that tracks the registry would have silently rescoped this run the moment Spider
was registered. `--batch-size 16` rather than the LFM2 panel's 32: sources are interleaved, a
batch pads to whichever Spider item it caught, and an OOM six arms in costs more than the
throughput does.

## 5. The ceiling

The bf16 merge, scored first, over the same 2 454 items every other arm sees in the same order.

| | |
|---|---|
| accuracy | **78.16%** (1 918 / 2 454) |
| exact string match | 1 144 — so **774 of the 1 918** are correct by execution equivalence, not by text |
| unparseable | **0** |
| errored | 50 (SQL that did not execute) |
| `unfinished_reasoning` | **0** — nothing was cut off by the decode budget |
| decode | greedy, `max_new_tokens` 1024, `max_prompt_tokens` 3072, batch 16 |
| wall clock | 590.6 s |

By source:

| source | ceiling | in-mixture? |
|---|---|---|
| wikisql | **93.89%** (768/818) | yes |
| gretel | **77.02%** (630/818) | yes |
| **spider** | **63.57%** (520/818) | **no** |

That is the headroom screen's requirement met on all three — no source is near ceiling, so
quantization damage has room to show — and it is also the generalization result in its own
right: the two trained-on sources land 13 and 30 points above the held-out one. `unparseable:
0` and `unfinished_reasoning: 0` together mean the ceiling's 21.84% of misses are wrong
answers, not harness artifacts. A query cut mid-clause would have scored as a syntax error
rather than as a wrong answer, which is why the driver runs the ceiling first and checks it for
censoring before spending six more arms on the same decode budget.

## 6. "At matched bytes" — the two anchors, reconciled to the byte

DynQuant is pinned to byte counts, never to nominal widths. Its format spends **32 bits per
group of 128** (an fp16 scale and an fp16 zero point) where `compressed-tensors` spends
`16 + bits`, so anchoring the comparison on nominal 4-bit would hand DynQuant 2.3% more bytes
than the methods it is being compared against, *inside the arm whose accuracy is the claim*.

The anchors come from
[`baselines_lfm2.accounted_bytes`](../../experiments/phase4/baselines_lfm2.py), computed
against a **meta-device copy of the architecture** — free, GPU-less, and immune to the observer
state llm-compressor leaves on the model it hands back (walking one of those reported 14.34
bits per weight for a 3-bit checkpoint). They reproduce exactly from the format rule:

```
quantized_bits = P·b + (P/128)·(16 + b)          P = 7 113 539 584 quantized params
dense_bits     = D·16                            D = 134 483 968 (embed_tokens + norms)

b=4:  4.15625 bits/param  →  3 964 674 048 B    ✓ matches the launched anchor exactly
b=3:  3.1484375           →  3 068 534 784 B    ✓
```

`IGNORE` is empty — nothing is left in fp16 by choice. The 7 113 539 584 quantized params are
`32 × 7` layer projections plus `lm_head` = **225 Linears**; `model.embed_tokens` is an
`nn.Embedding`, not a Linear, so `compressed-tensors` holds it dense at 268 435 456 B and the
norms at 532 480 B. DynQuant quantizes it, which is why it reaches **226** modules against the
baselines' 225 — it spends the same total bytes over a larger quantized pool.

Whole-model averages, which is what "matched bytes" actually equalizes:

| anchor | baseline bytes | bits/param | DynQuant bytes | bits/param | drift |
|---|---|---|---|---|---|
| 4-bit | 3 964 674 048 | 4.376166 | **3 964 149 760** | 4.375588 | −524 288 B (**−0.013%**) |
| 3-bit | 3 068 534 784 | 3.387017 | **3 067 617 280** | 3.386004 | −917 504 B (**−0.030%**) |

Both DynQuant arms land *under* their anchor. The tolerance is 0.1% and the realised drift is
an order of magnitude inside it, in the direction that cannot flatter DynQuant.

## 7. The two allocations

Allocator `sensitivity`, quantity **"measured dL(2b) − dL(8b) per parameter"**, group size 128.
Structural census, identical at both budgets:

| | |
|---|---|
| quantizable modules | **226** |
| unquantized | **0** |
| skipped | 65 (norms), 266 240 params |
| **`unclassified`** | **`[]`** |
| **`missing_stats`** | **`[]`** — all 226 stats keys matched module names |
| `unexercised` | `['model.embed_tokens']` |
| priced by proxy | **1 module, 134 217 728 params, 1.85%** |

The 1.85% proxy share is the number to compare against the LFM2 campaign's **91.5%**. There,
a batched expert bank had no boundary at which the Gauss–Newton form exists, so the rank-product
proxy chose the widths for almost the whole model and the concordance guard could only read the
remaining 8.46%. Here the proxy prices one tensor — the embedding, which is `unexercised`
because an embedding's Gram axes are transposed relative to how its sensitivity would be read —
and **the measured price chooses the widths for 98.15% of the parameters**. This is the first
phase-4 model on which the campaign's headline mechanism is doing essentially all of the work.

### 7.1 The 4-bit map: no floor is breached

| bits | modules | params |
|---|---|---|
| 3 | 14 | 822 083 584 |
| 4 | 153 | 5 993 660 416 |
| 8 | 59 | 432 013 312 |

**`violations: []`.** At this budget the floors stopped binding entirely: every role got at
least what its floor demanded and the allocator still had bytes left to promote 59 modules to
8 bits. Concordance — does the measured score order agree with the proxy's, where both exist —
is **560/561 = 0.9982**, and the single disagreement is one `attn.k` (`mlp.down` 199/199,
`mlp.up` 175/175, `attn.v` 31/31).

### 7.2 The 3-bit map: 57 breaches, reported by name

| bits | modules | params |
|---|---|---|
| 2 | 7 | 243 269 632 |
| 3 | 111 | 5 796 528 128 |
| 4 | 107 | 1 203 765 248 |
| 8 | 1 | 4 194 304 |

Average **3.385880 bits**, **3 067 617 280 B** against the 3 068 534 784 B anchor — 917 504 B
under, −0.030%, inside the 0.1% tolerance. Those figures reconcile to the byte from the map
alone: 7 247 757 312 quantized params at their assigned widths, plus 32 bits of scale and zero
per group of 128, plus the 266 240 norm params held dense at fp16 (532 480 B, the same constant
in both maps and in the packer's own note).

**57 floor violations**, covering 2 533 359 616 params — **35.0% of the model**:

| role | floor → assigned | count |
|---|---|---|
| `mlp.gate` | 4 → 3 | 29 |
| `attn.q` | 4 → 3 | 10 |
| `attn.o` | 4 → 3 | 9 |
| `attn.o` | 4 → 2 | 3 |
| `mlp.gate` | 4 → 2 | 1 |
| `attn.q` | 4 → 2 | 1 |
| `mlp.up` | 3 → 2 | 1 |
| `mlp.down` | 3 → 2 | 1 |
| **`lm_head`** | **8 → 4** | 1 |
| `model.embed_tokens` | 4 → 3 | 1 |

This is the P4 soft-floor fix doing exactly what it was built for, and the contrast between the
two budgets is the cleanest demonstration of it in the record. The supplement's allocator
returns the floor map whenever `budget − base − floors` goes negative, which at a 3-bit target
means the scores never run and the shipped map is hand-written. Here the budget does go
negative against the floors, the allocator downgrades by lowest ROI anyway, and **it names
every breach** instead of silently handing back a map the score never touched.

`lm_head` at 4 bits against an 8-bit floor is the single most aggressive decision in either
map. The seven 2-bit modules are the second, and they are not scattered — three of them are
`layers.0`'s `q_proj`, `up_proj` and `gate_proj`, and the other four are `o_proj` in layers
**27, 28 and 29** plus `layers.29`'s `down_proj`. The measured sensitivity puts the first
block's inputs and the last blocks' attention outputs at the bottom of the model, which is a
shape the score produced rather than one anybody encoded — no floor, role rule or preset
mentions layer index. The single 8-bit survivor is `model.layers.2.self_attn.v_proj`, a
4 194 304-param tensor the score priced highly enough to keep at full width while 35% of the
model went below its floor.

Concordance at this budget — does the measured score order agree with the proxy's, where both
exist — is **690/690 = 1.0000**, perfect across every role the guard can read.

### 7.3 What the fine-tune moved, and a map I first read off the wrong checkpoint

The allocation was run twice: once as a feasibility check before the panel, and once inside the
panel itself. The two disagree, and the reason is not the allocator — it is that the check ran
against `mistralai/Mistral-7B-Instruct-v0.3` off the Hub while the panel ran against
`runs/s4/mistral7b-v03.text2sql/merged`. Same stats file, same allocator, same group size, same
target, same `dynquant-core` 0.4.0; different weights. **An earlier draft of §7.2 was written
from the feasibility map and described a model this panel never quantized.** The numbers above
are the panel's own.

The disagreement is also a measurement worth keeping, because it prices what two epochs of
fine-tuning do to the allocation:

| budget | widths that moved | of | histogram | average bits | bytes |
|---|---|---|---|---|---|
| 4-bit | **6** | 226 | *identical* — 3b×14 / 4b×153 / 8b×59 | identical to 10 dp | identical |
| 3-bit | **4** | 226 | 2b×8→7, 3b×109→111, 4b×108→107, 8b×1 | identical to 10 dp | identical |

At the 4-bit budget the histogram is preserved exactly — the six moves are three swaps
(`layers.25.k_proj` 8→4 against `layers.31.k_proj` 4→8; two `mlp.up_proj` 4→3 against
`layers.30.up_proj` and `layers.31.down_proj` 3→4). So fine-tuning reordered *which* tensors
deserve the width without changing *how many* do. That is the same result the phase-3 campaign
found across two tasks, one level down: the map is mostly a property of the architecture, and
the training run perturbs it at the margin — here 2.7% of modules at 4 bits and 1.8% at 3.

It is also the reason a feasibility run on the base model is a fine check that the stats keys
match and the budget is reachable, and **not** a preview of the map. Nothing downstream read
the feasibility map; the packed checkpoints and every number in §8 come from the panel's.

## 8. The panel

Seven arms, every one scored over the same 2 454 items in the same order, so every pairwise
comparison is a McNemar test on stored per-item hits rather than two independent proportions.
Everything below comes out of `panel_table.py --json-out`; none of it is typed.

| arm | bytes | bits/param | accuracy | correct | exact | errored | eval s |
|---|---|---|---|---|---|---|---|
| bf16 ceiling | 14 496 047 104 | 16 | **78.16%** | 1918 | 1144 | 50 | 591 |
| GPTQ 4-bit | 3 964 674 048 | 4.3762 | **78.28%** | 1921 | 1137 | 58 | 1 839 |
| AWQ 4-bit | 3 964 674 048 | 4.3762 | **77.91%** | 1912 | 1126 | 51 | 2 029 |
| DynQuant 4-bit | 3 964 149 760 | 4.3754 | **78.08%** | 1916 | 1135 | 50 | 706 |
| GPTQ 3-bit | 3 068 534 784 | 3.3869 | **6.68%** | 164 | 69 | **1796** | 23 290 |
| AWQ 3-bit | 3 068 534 784 | 3.3869 | **74.16%** | 1820 | 1088 | 57 | 2 302 |
| DynQuant 3-bit | 3 067 617 280 | 3.3859 | **75.22%** | 1846 | 1069 | 62 | 655 |

Unparseable is 0 and unfinished-reasoning is 0 for every arm, including the collapsed one — the
decode budget was never the binding constraint, so nothing here is a censoring artifact.

### 8.1 At 4 bits, nothing separates

| comparison | Δ points | 95% CI | discordant | p | p adjusted |
|---|---|---|---|---|---|
| DynQuant vs GPTQ | −0.204 | [−1.053, +0.645] | 113 (54/59) | 0.707 | 1.000 |
| DynQuant vs AWQ | +0.163 | [−0.726, +1.052] | 124 (64/60) | 0.788 | 1.000 |
| GPTQ vs AWQ | +0.367 | [−0.459, +1.193] | 107 (58/49) | 0.439 | 1.000 |
| DynQuant vs bf16 | −0.081 | [−0.831, +0.668] | 88 (43/45) | 0.915 | 1.000 |
| GPTQ vs bf16 | +0.122 | [−0.541, +0.786] | 69 (36/33) | 0.810 | 1.000 |
| AWQ vs bf16 | −0.244 | [−1.051, +0.562] | 102 (48/54) | 0.621 | 1.000 |

Every adjusted p-value is 1.000 and no interval excludes zero. At 3.66× compression on a 7B
model, all three methods and the unquantized ceiling are one accuracy. **This campaign has
nothing to say about allocation at 4 bits**, and the honest reading of the 4-bit row is that
the budget is too loose for any allocator to matter.

### 8.2 At 3 bits, one collapse and one difference that does not clear significance

| comparison | Δ points | 95% CI | discordant | p | p adjusted | separated |
|---|---|---|---|---|---|---|
| DynQuant vs GPTQ | **+68.541** | [+66.659, +70.423] | 1708 (1695/13) | ~0 | ~0 | **yes** |
| GPTQ vs AWQ | −67.482 | [−69.393, −65.571] | 1690 (17/1673) | ~0 | ~0 | **yes** |
| **DynQuant vs AWQ** | **+1.059** | **[−0.323, +2.442]** | 300 (163/137) | **0.149** | **0.595** | **no** |
| DynQuant vs bf16 | −2.934 | [−4.161, −1.707] | 238 (83/155) | 3.6e−06 | 1.4e−05 | yes |
| AWQ vs bf16 | −3.993 | [−5.286, −2.700] | 266 (84/182) | 1.8e−09 | 9.0e−09 | yes |
| GPTQ vs bf16 | −71.475 | [−73.301, −69.650] | 1776 (11/1765) | ~0 | ~0 | yes |

**The +1.06 over AWQ does not separate.** 163 items DynQuant answers and AWQ misses, 137 the
other way, and a paired test on 2 454 items cannot call a 26-item margin. That is the result,
and it is a weaker one than this project's earlier 3-bit campaigns produced.

**The +68.5 over GPTQ is real but is not a statement about allocation.** See §8.3.

The one place DynQuant looks better with significance attached is the gap to the ceiling:
−2.93 against AWQ's −3.99, both separated from bf16 at p < 1e−05. It is tempting to read those
two numbers as an ordering. **They do not establish one** — they are two separate tests against
a third arm, and the direct paired test between the two arms is the one that answers the
question. It says 0.595. Reading a pair of ceiling gaps as a ranking is precisely the error the
paired test exists to prevent.

What DynQuant 3-bit does establish on its own terms: **4.73× compression — 14 496 047 104 B to
3 067 617 280 B — retaining 96.2% of the fine-tuned model's accuracy** (75.22 of 78.16), with
`lm_head` at half its floor and 35% of the model below role floors.

### 8.3 The GPTQ 3-bit collapse, and why it must not be cited as a win

GPTQ at the 3-bit anchor scores 6.68% and **1796 of 2454 generations fail to execute** — 73% of
the benchmark. By source it is wikisql 5/818 (0.61%), spider 49/818 (5.99%), gretel 110/818
(13.45%). The eval took 23 290 s against the same arm's 1 839 s at 4 bits, because a model
emitting non-executing SQL runs its batches toward the decode cap.

The first question is whether the arm is real, and the instrumentation says it is:

- `weights_moved: 0`, `max_weight_delta: 0.0` — the documented expected shape for GPTQ, whose
  scales are fitted and whose weights are already on the grid when `materialize_quantization`
  re-reads them. The fixed-point check passed on all 8 probes or it would have raised.
- `probe_unique_values_per_row: [169, 181, 172, 168, 159, 176, 175, 163]`. A 4096-wide row at
  group 128 has 32 groups; at 3 bits that is at most 256 distinct values, at 4 bits at most 512.
  The same probe on `gptq_4b` returned 298–350. **These weights are genuinely 3-bit.**
- `accounted_bytes: 3 068 534 784` — byte-identical to `awq_3b`.

So the collapse is not a broken arm. But it is also **not evidence that DynQuant's allocation
beats GPTQ's**, for a reason visible in the recipe rather than the results. `_llmc.py` builds
`GPTQModifier(config_groups=groups, ignore=ignore, dampening_frac=0.01)` — each method at its
own published default, which for GPTQ means **symmetric** quantization and **no activation
ordering**, and for AWQ means **asymmetric**. DynQuant is asymmetric too: min/max with a stored
zero point.

At 4 bits that difference is worth nothing — GPTQ symmetric is the best arm in the panel. At 3
bits a symmetric grid has 8 levels pinned around zero against an asymmetric grid's 8 levels
fitted to the actual range, and that is a large handicap on weight distributions that are not
centered. The 67.5-point gap between two arms that differ in *both* allocation and grid
symmetry cannot be attributed to either one.

**The comparison that is apples-to-apples at 3 bits is DynQuant vs AWQ** — both asymmetric,
both group-128, both at the same byte anchor, differing only in how bits are distributed across
modules. That comparison is +1.06 points, p_adj = 0.595.

Two controls would settle what the GPTQ number means, and neither has been run:

1. **GPTQ 3-bit asymmetric, with activation ordering.** If it recovers to AWQ's neighbourhood,
   the collapse was the grid and the default, not the method.
2. **GPTQ and AWQ handed DynQuant's own bit map**, already on this project's backlog. That
   isolates allocation from everything else by holding the quantizer fixed.

Until at least the first is run, this report's position is that **GPTQ 3-bit at
llm-compressor's defaults collapses on this model**, which is a useful thing to know about a
default, and not that DynQuant beats GPTQ by 68 points.

### 8.4 By source

Correct out of 818 per source, every arm on the same items:

| arm | gretel | spider | wikisql |
|---|---|---|---|
| bf16 | 630 (77.02%) | 520 (63.57%) | 768 (93.89%) |
| GPTQ 4-bit | 636 (77.75%) | 517 (63.20%) | 768 (93.89%) |
| AWQ 4-bit | 630 (77.02%) | 516 (63.08%) | 766 (93.64%) |
| DynQuant 4-bit | 630 (77.02%) | 516 (63.08%) | **770 (94.13%)** |
| GPTQ 3-bit | 110 (13.45%) | 49 (5.99%) | 5 (0.61%) |
| AWQ 3-bit | 618 (75.55%) | 446 (54.52%) | 756 (92.42%) |
| DynQuant 3-bit | 607 (74.21%) | **480 (58.68%)** | 759 (92.79%) |

The 3-bit margin over AWQ is not spread evenly. It is **+34 items on Spider** (480 vs 446, the
held-out source with the largest schemas), **+3 on wikisql**, and **−11 on gretel** — signs in
both directions, which is the same heterogeneity the phase-3 panel found at 4 bits.

That pattern is worth a caveat rather than a claim: `panel_table.py` returned an empty
`head_to_head_by_source`, so **no per-source paired test was computed** and the counts above are
descriptive only. Whether the Spider margin survives a McNemar test on Spider's 818 items, and
whether Cochran's Q calls the three sources heterogeneous, are open. The unadjusted whole-panel
test is the only inferential statement this campaign supports, and it says 0.149.

## 9. Publication

Three artifacts go to the Hub: the bf16 fine-tune, and the two DynQuant arms at each anchor.

| arm | repo |
|---|---|
| `bf16` | `VikramPal/mistral-7b-instruct-v0.3-text2sql-bf16` |
| `dq_4b` | `VikramPal/mistral-7b-instruct-v0.3-text2sql-DynQuant-4bit` |
| `dq_3b` | `VikramPal/mistral-7b-instruct-v0.3-text2sql-DynQuant-3bit` |

Pushed 2026-08-13, public, all three complete. The Hub reports 14 499 764 397 B for the
ceiling, 3 968 081 867 B for `dq_4b` and 3 071 549 838 B for `dq_3b` — each the exported
directory plus its generated card and `.gitattributes`, and each quantized repo carrying the
`dynquant_manifest.json` its loader reads. The 3-bit directory is 0.13% over the 3 067 617 280 B
its arm was scored at, which is the container's own overhead and not a different allocation.

```bash
export HF_TOKEN=...                     # env only; see below
python experiments/phase4/publish_panel.py     --arms   runs/s4/panel-mistral/arms.json     --out    runs/s4/published-mistral --only dq_4b,dq_3b
python experiments/phase4/push_to_hub.py     --table     runs/s4/panel-mistral/table.json     --finetune  runs/s4/mistral7b-v03.text2sql/s2_finetune.json     --published runs/s4/published-mistral     --repo-prefix VikramPal/mistral-7b-instruct-v0.3-text2sql     --only bf16,dq_4b,dq_3b
```

`--only bf16,...` rather than `--include-ceiling`: the two flags are not additive. `--only`
replaces the label list outright, so `--include-ceiling` beside it is read and discarded, and a
push that named the two DynQuant arms and asked for the ceiling as an extra would have shipped
the two quantized repos and silently no fine-tune. The ceiling resolves to the trainer's own
`output` path — it is the merged fine-tune, which lives where the trainer put it, while every
other arm lives where the publish pass put it.

**Only the DynQuant arms are exported.** This is a cost decision with a structural reason
behind it. `publish_panel.py` produces a baseline's weights by *re-running its recipe* — GPTQ
and AWQ score in process and keep no checkpoint — which is another two to three hours of GPU
(≈32 min for GPTQ, ≈47 min for AWQ, at each of two anchors). A DynQuant arm has no second pass:
its map **is** the artifact its score came from, so the export is minutes and is exactly the
thing being published. The baselines stay in the table, where they are the comparison.

That asymmetry is worth stating plainly rather than hiding, because it is a real property of
the method and not a convenience: a signal-driven allocation is reproducible from a map file,
and an error-feedback recipe is reproducible only by re-running it.

### 9.1 What the push refuses to do

[`push_to_hub.py`](../../experiments/phase4/push_to_hub.py) runs every one of these before it
uploads a byte:

- every publishable arm has a directory, a `config.json`, and weights;
- every directory agrees with the merge on `architectures` and `vocab_size` — the guard
  against pushing one model's weights under another model's name, which is the failure a
  campaign that ran four Mistral arms into a Qwen directory has already produced once;
- every quantized directory carries a DynQuant `quantization_config`, and the **ceiling carries
  none**;
- a repo that already holds files is refused without `--force`.

The gap it cannot close is stated in the file rather than papered over: a swap *between two
quantized arms* passes every check above, because both are Mistral and both carry a config. So
the plan prints **how many distinct widths each map carries** and leaves a human to read it —
4 bits uniform and an allocation are not the same object, and the count is the one line that
separates them.

**The token comes from `HF_TOKEN` and from nowhere else.** Not a flag — an argument lands in
shell history and in `ps`, where every other process on a shared box can read it — and not a
file either. The script refuses any argv element beginning with `hf_`.

The published `generation_config.json` carries `bos_token_id`, `eos_token_id` and nothing
else — no `temperature`, no `top_p`, no `do_sample`. That is the state it has to be in.
transformers v5 refills fields a caller left unset from the checkpoint's own config, so a
checkpoint that ships sampling defaults silently overrides a downstream greedy request; on
GSM8K that was worth 19 points. Here there is nothing to refill from.

Each card is **generated** from `panel_table.py --json-out` and the fine-tune's own record
rather than read off disk, so the repo id baked into the usage snippet always matches where the
directory is actually pushed, and no accuracy figure on a card is typed by hand. Uploads are
restricted to `*.safetensors`, `*.json`, `*.model`, `*.txt`, `*.md`, `*.jinja`.

## 10. What this campaign settles, and what it does not

**Settles.** It is the first phase-4 result on a dense model, and it removes every caveat the
MoE campaign carried: no expert banks, so no `grouped_mm`-against-`eager` dispatch confound
(`experts: {found: eager, ran: eager}` on all seven arms), no batched-bank publish path, and
**1.85% proxy pricing instead of 91.5%** — the measured Gauss–Newton sensitivity chooses the
widths for 98.15% of the parameters. Both anchors reconcile to the byte from the baselines'
format rule. Both DynQuant arms land inside 0.03% of their anchor, under it. Spider is a
genuinely unseen third of the evaluation.

**Does not settle.** Four controls this panel does not contain:

1. **A GPTQ 3-bit arm that is not handicapped by its own default.** The panel's GPTQ collapsed
   to 6.68% at the 3-bit anchor, and §8.3 shows the arm is genuinely 3-bit — but it is also
   *symmetric* and without activation ordering, against an asymmetric AWQ and an asymmetric
   DynQuant. The 68-point gap spans two differences at once and attributes to neither. A GPTQ
   3-bit asymmetric arm with act-order is the control, and it has not been run. **Nothing in
   this campaign should be cited as "DynQuant beats GPTQ at 3 bits."**
2. **The null.** No `--score-null` arm ran, so this campaign cannot say what share of any
   DynQuant margin is the *signal* rather than the shape of the map. The two campaigns that
   have measured it disagree by construction — 12% on Qwen3.5-2B against a constant-score
   control, 56% on Ministral-8B against a within-role shuffle — and those are different
   questions, not a range.
3. **The bit-map handoff.** GPTQ and AWQ have never been handed DynQuant's own bit map. Until
   they are, "allocation beats uniform" and "DynQuant's encoder beats theirs" are not separated.
4. **RTN.** No round-to-nearest arm at either anchor, so the panel has no floor to quote the
   margins against — and on Qwen3.5-2B at matched bytes, RTN *tied* DynQuant at 4 bits.

**A cost this campaign measured by accident.** The measured-moments allocation runs on CPU —
`dynquant inspect` takes no device flag — and each anchor costs about **two hours** at ~24
cores while the GPU sits at 180 MHz and 0%. `dq_3b`'s dry-run allocation took 1 h 58 m 42 s
(03:34:44 → 05:33:26) and the panel's `dq_4b` pass ran the same length. That is roughly four
GPU-hours of a rented box spent idle, per panel, and it is the single largest avoidable line in
this campaign's cost. It is not a correctness problem: the map is a deterministic function of
(weights, stats, moments, group size, target), so CPU and GPU would have to agree bit-for-bit
before the allocation could be moved without risking two arms allocated by different
arithmetic. That check is the prerequisite, and it has not been run.

The 3-bit map's 57 floor breaches are also a live question rather than a settled one. 35.0% of
the model sits below the width its role's floor asks for, `lm_head` most aggressively at 4
against 8. The soft-floor mechanism is working — it names them instead of silently returning
the floor map — but whether the allocation it produced is *better* than the hard-floor map it
replaced is a claim only the results table can make.
