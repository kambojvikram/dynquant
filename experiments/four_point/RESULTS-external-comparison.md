# DynQuant against GPTQ, AWQ, RTN and bnb-NF4

Two models, two tasks, one harness, one process. Each model is fine-tuned on the task its own
base-model headroom screen selected for it, then quantized six ways from that one fine-tuned
checkpoint, and every arm is scored on the same held-out split in the same order:

| model | task | test rows | 2-shot / 4-shot | chance | base | fine-tuned |
|---|---|---:|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | Banking77, 77-way intent | 3,080 | 4-shot | 1.30% | 33.7% | 94.38% |
| Qwen3.5-2B-Base | CaseHOLD, 5-way holding selection | 5,314 | 2-shot | 20.0% | 32.33% | 89.74% |

The Qwen pairing gives up the sharper chance floor — 20% against Banking77's 1.3%, because CaseHOLD
is 5-way and Banking77 is 77-way — and buys the wider fine-tuning gain: **+57.4 points** against
Mistral's +60.7 but from a base that is genuinely near chance rather than accidentally competent.
Both tasks land the fine-tuned model around 90–94%, which is the range that matters: enough
headroom that a quantization regression has somewhere to show, without the ceiling effect that
made GSM8K useless for this model.

**Why not one dataset for both models.** Because a shared dataset answers a different
question. Pairing a model with a task it is already at ceiling on measures nothing: quantization
damage has to be read against a fine-tuning gain, and if there is no gain there is no scale. That
is not hypothetical here — GSM8K was the first choice for Qwen3.5-2B and it produced six flat
arms before the cause turned out to be that the base model already sat at the supervised ceiling.
So each model gets screened before it gets trained, and it gets the task where the screen showed
room. Holding the *dataset* constant across two models would trade the thing being measured for a
symmetry that only looks tidier. What is held constant instead is everything that touches the
comparison: harness, split order, fine-tuning regime (LoRA r=32, lr 1e-4), calibration protocol,
accounting convention, and the six methods.

The internal experiment — does the signal-driven allocator beat a uniform one — is
[RESULTS-mistral7b-banking77.md](RESULTS-mistral7b-banking77.md). This document asks the
other question: does it beat what people actually ship.

A third run exists, Qwen3.5-2B on Banking77 (58.0% base → 93.41% fine-tuned), from an earlier
pass that read "same dataset" as the design. It has real headroom and is kept as a secondary
panel at the end, because it happens to isolate one variable the main panels cannot: the same
model and the same allocator on a *different* task, which is the only place task-specificity of
the bit map is visible at all.

## Read this before the tables

**The methods do not see the same information, and that is the honest headline.**
DynQuant reads gradient-variance and activation-magnitude statistics collected *during*
the fine-tune, by a hook attached to the training loop. GPTQ and AWQ see forward
activations on a 256-row calibration set and nothing else. RTN and NF4 see only the
weights. So DynQuant is being given something the baselines are not, and any advantage it
shows is partly an advantage of *access*, not of algorithm. That is a real property of the
method — the signal is free if you were fine-tuning anyway, which is the deployment path
this targets — but it is not a like-for-like algorithmic comparison and should never be
presented as one. Where DynQuant loses, that caveat makes the loss worse, not better.

The asymmetry is not entirely one-directional, and naming the other half keeps the framing
honest. GPTQ spends its 256 calibration rows on an inverse-Hessian **error compensation**
sweep — it rounds one column at a time and pushes the residual into the columns not yet
quantized. DynQuant does not do this at all: it allocates a width per module and then
rounds to nearest within it. So DynQuant has more information about *which* weights matter
and less machinery for *reducing the cost of rounding them*, and the two effects are
separable. Panel 2's 3-bit tier measures both, and it is the one place the missing
compensation shows up as a real deficit.

**Sizes are measured, not nominal.** No arm here costs what its name says. A "4-bit g128"
checkpoint keeps `lm_head` in fp16 — the convention every published GPTQ and AWQ
checkpoint follows, and llm-compressor's own examples — and pays an fp16 scale plus a
4-bit zero point per group of 128 on everything else. On Mistral-7B that is **4.5953 bits
per weight, not 4**. DynQuant's 4.25 quantizes the embedding and the head too, and the
4.25 already includes its metadata. Ranking these arms by nominal width would credit the
baselines for bytes they are still spending, so every table below ranks on the bits they
measurably cost, computed under one accounting convention for all six methods.

**The baselines were not hand-tuned to lose.** They are llm-compressor (the vLLM project's
own library, v0.12) at its documented settings: `w{4,3}a16` asymmetric, group 128, 256
calibration rows at 1024 tokens drawn from that model's own fine-tuning train split — Banking77
for Mistral, CaseHOLD for Qwen — sequential pipeline, GPTQ dampening 0.01. The calibration set
matching the evaluation task is the setting that favours the baselines, so it is the one used.
bnb-NF4 is bitsandbytes' double-quantized NF4 at load time.
Each arm's quantization was verified to have actually happened — see *Provenance* below,
because the first time it was not.

## Provenance: why each arm is believed

llm-compressor separates calibration from compression. `oneshot()` fits scales and
zero-points, attaches them, and sets `quantization_status=FROZEN` — and for
`QuantizationModifier` it **never rounds the weight tensor**. Rounding happens later,
inside `save_pretrained(save_compressed=True)`. GPTQ is the exception: its algorithm writes
corrected weights back as it sweeps.

The first RTN arm therefore scored *exactly* the bf16 arm's 2907/3080 with a bit-identical
hit vector, because the in-process path never saved and the evaluation ran on the
unquantized checkpoint. Four-bit quantization of a 7B model cannot leave 3080 predictions
untouched, so hit-vector equality with bf16 is now a standing check — and it is the one
failure a reader cannot spot from an accuracy column, because the number looks excellent
rather than broken.

Three guards run on every arm:

- **Fixed-point invariant.** Re-quantizing an already-quantized weight must return it
  bit-identically. This catches "never rounded" and "the write did not land" together.
  Writes go through `compressed_tensors.utils.update_offload_parameter` — assigning
  `module.weight.data` directly does not stick, even with no accelerate hook attached.
- **`weights_moved`.** Expected 0 for GPTQ, whose weights are already on the grid
  (max delta 0.0), and the full Linear count for RTN and AWQ. Mistral: 224 (32 layers × 7
  Linears). Qwen3.5-2B: 186 (6 full-attention layers × 4 + 18 linear-attention × 5 + 24
  MLP × 3). AWQ's max delta is large — 1.32 at 4-bit — because smoothing rescales the
  weights before rounding, which is the method working, not a fault.
- **Distinct values per row.** A cheap width check on eight probed modules. Mistral 4-bit
  g128 collapses a row from ~1589 distinct values to 292–337, 3-bit to 150–180. On
  Qwen3.5-2B, measured against the unquantized checkpoint read straight out of safetensors:

  | row width | unquantized bf16 | 4-bit g128 arm | ceiling at 4-bit g128 |
  |---|---:|---:|---:|
  | 2048 | 1010–1059 | 174–183 | 256 |
  | 6144 | 1515–1552 | 418–423 | 768 |

  The ceiling is **16 levels × (width / 128) groups**, not 16, because every group of 128
  carries its own scale and the dequantized row is a union of per-group grids. Reading the
  probe as "a 4-bit row must show ≤16 values" would condemn every correctly quantized arm in
  this document, and a 6144-wide row legitimately showing 423 distinct values is the reason
  the check records the width alongside the count. Note also that unquantized bf16 tops out
  near 1000–1550 rather than at the row length: bf16 has 8 mantissa bits, so the *format* is
  the binding constraint on a full-precision row, not the data.

Every arm below passed all three. `baselines_table.py` refuses to print an arm that fails
the first, and flags an arm with no materialization proof.

**These three guards protect the baseline arms and none of them protects the DynQuant arms**, and
that asymmetry produced a real defect. All three ask "was this weight tensor actually quantized" — a
question about the *output*. A DynQuant arm has an input the baselines do not: a signal map, which is
a separate file produced by a separate earlier stage, and nothing above notices if it was built from
a different checkpoint than the one being quantized. That is what happened on Mistral (see Panel 1).
The missing check is one line and belongs next to the other three: **assert the signal map is newer
than the stats directory it claims to summarize, and the stats newer than the fine-tune that wrote
them.** Existence tests cannot substitute for freshness tests when any upstream stage is re-runnable.

## Panel 1 — Mistral-7B-Instruct-v0.3 on Banking77 (untied embedding)

| arm | bits | GiB | accuracy | correct |
|---|---:|---:|---:|---:|
| fine-tuned bf16 | 16.000 | 13.501 | 94.38% | 2907/3080 |
| **DynQuant 4.25b** | **4.250** | **3.586** | 94.19% | 2901 |
| GPTQ 4b g128 | 4.595 | 3.877 | 94.25% | 2903 |
| AWQ 4b g128 | 4.595 | 3.877 | **94.42%** | 2908 |
| bnb NF4 | 4.567 | 3.854 | 94.25% | 2903 |
| RTN 4b g128 | 4.595 | 3.877 | 94.25% | 2903 |
| **DynQuant 3.25b** | **3.250** | **2.742** | 93.57% | 2882 |
| GPTQ 3b g128 | 3.625 | 3.059 | 94.38% | 2907 |
| AWQ 3b g128 | 3.625 | 3.059 | 94.03% | 2896 |
| RTN 3b g128 | 3.625 | 3.059 | 93.34% | 2875 |

McNemar on the discordant pairs, DynQuant on the left so a positive delta means DynQuant
ahead:

| comparison | delta | 95% CI | flips | p | verdict |
|---|---:|---:|---:|---:|---|
| DQ 4.25 vs GPTQ 4b | −0.06 | | | 0.85 | not separated |
| DQ 4.25 vs AWQ 4b | −0.23 | | | 0.21 | not separated |
| DQ 4.25 vs NF4 | −0.06 | | | 0.85 | not separated |
| DQ 4.25 vs RTN 4b | −0.06 | | | 0.83 | not separated |
| GPTQ 4b vs RTN 4b | +0.00 | | 24/24 | 1.0 | not separated |
| **DQ 3.25 vs GPTQ 3b** | **−0.81** | **[−1.36, −0.26]** | **25/50** | **0.0052** | **separated** |
| DQ 3.25 vs AWQ 3b | −0.45 | | | 0.11 | not separated |
| DQ 3.25 vs RTN 3b | +0.23 | | | 0.48 | not separated |
| GPTQ 3b vs RTN 3b | +1.04 | | | 0.0001 | separated |
| AWQ 3b vs RTN 3b | +0.68 | | | 0.0075 | separated |
| bf16 vs DQ 4.25 | +0.19 | | | 0.26 | not separated |
| bf16 vs DQ 3.25 | +0.81 | | | 0.0026 | separated |
| bf16 vs GPTQ 3b | +0.00 | | | 1.0 | separated? no — **lossless** |

Three things to take from this panel, in order of how much they matter.

**At 4 bits nothing separates.** Five methods, a 0.23-point spread, every pairwise p above
0.2. On a 77-way classification task a 7B model at ~4.5 bits is simply not under
pressure — the task has headroom the quantization does not consume. DynQuant is the
smallest of the five (3.586 GiB against 3.877) and gives up 0.06 points to GPTQ, which is
two problems out of 3080. The right statement is "indistinguishable at 6% fewer bytes",
not a win.

**At 3 bits GPTQ beats DynQuant, and the row is doing two jobs.** −0.81 points, CI
excluding zero, p=0.0052. That is a real loss, and it is the only separated DynQuant-vs-
baseline row in the table. But GPTQ 3b costs 3.6249 bits against DynQuant's 3.250 — a
0.375-bit budget gap, 11.6% more bytes — so the row is a method difference *plus* a budget
difference and cannot say which. The iso-size arms exist to split those apart; see below.

**GPTQ 3-bit is lossless here in aggregate.** 2907/3080, exactly the bf16 count, p=1.0 —
though not the same 2907 problems, so this is aggregate parity and not an unchanged model.
It is a genuinely strong baseline result on this task and worth stating plainly.

### Provenance defect in the two DynQuant rows

The two bolded rows above were quantized from a **different fine-tune than the one that produced
their signal map**, and the number as published should be read with that in mind.

The pipeline re-ran the Mistral fine-tune — a configuration-identical replicate (LoRA, lr 1e-4,
626 steps, 2 epochs; train_loss 0.1405 → 0.1373, 3623s → 3450s) — which overwrote `finetuned/`
and `stats/`. It then skipped stage3, stage4 and all four stage5 steps, because each of their
outputs was already on disk. So the refit weights were quantized through a map built from the
first fine-tune's signals. `md5sum` pins it: `stage8_dq_4p25_quant.json` is byte-identical to
`stage5_refit_4p25_quant.json` and differs from `stage5_4p25_quant.json`. The mismatched arms are
retained as `stage8_dq_*_ft1map*` and the map that produced them as `stage4_bitmaps.ft1.json`.

**Scope is narrow.** Only these two rows are affected. bf16 and all seven baseline rows were
measured against the refit weights with no signal map in the loop, so every non-DynQuant number in
Panel 1 is clean, as are all of Panels 2 and 3 (a different model and a different run directory).
A map rebuilt from the refit stats differs from the one used on **24 of 226 widths** at 3.25 bits,
balanced 12 up / 12 down, concentrated in attention projections (`q_proj` 9, `k_proj` 6, `v_proj`
4, `o_proj` 3, `gate_proj` 2).

**The mismatch looks close to harmless, and that appearance cannot be trusted yet.** A fully
self-consistent pair from the first fine-tune also survives on disk, so both can be read against
their own fp16 reference:

| weights | signal map | fp16 ref | 4.25b | 3.25b |
|---|---|---:|---:|---:|
| fine-tune #1 | fine-tune #1 | 94.51% | 94.32% (−0.19) | 93.34% (−1.17) |
| refit (published) | fine-tune #1 | 94.38% | 94.19% (−0.19) | 93.57% (−0.81) |

Identical degradation at 4.25 bits and a *smaller* one at 3.25. But the two rows differ in weights
as well as in map, so this compares two things at once and establishes nothing about either. The
missing cell — refit weights, refit map — is the one being measured now.

**How it surfaced.** `run_isosize.sh` compares a freshly built map against the one on disk before
using it, and refused to proceed. Its diagnostic said "the allocator is not deterministic," which
was wrong: scoring and allocating twice in one process from the same files gives 0 differing
scores and 0 differing widths. The guard was right to stop and wrong about why. What it had
actually detected was that the *inputs* had moved underneath a published result — the map is dated
01:23 and the stats it should have been built from were written at 03:08, an hour and three
quarters later. A resume guard that skips on the existence of an output cannot notice that the
output is older than its input, which is the general form of this bug.

**One thing the repair established before it produced a single accuracy number: at 4.25 bits the
defect has no effect on the bit map at all.** The map rebuilt from the refit stats differs from the
ft1 map on **0 of 226 widths** at that target — only the 3.25-bit map moves, on 24. So the 4.25 row's
provenance problem is real in bookkeeping and empty in consequence: `stage8_dq_4p25_ft1map` used a map
byte-identical to the corrected one. The 3.25 row is the only one where the repair can change
anything.

### Two failed repairs, and the guards they bought

Worth recording because both failures were silent in the way that matters — each reported plausible
output while doing the wrong thing.

**The premise guard fired as a false positive.** The first repair script refused to run, on: *"target
4.25: corrected map differs from ft1 map on 0 of 226 widths — nothing to repair, aborting."* It had
inferred from an identical map that the map had never been rebuilt. It had been: four targets instead
of two, mtime 2026-07-30 11:52, well after the 2026-07-29 03:08 stats. The map was identical because
4.25 is *reproducible across fine-tune replicates*, which is a finding, not a fault. A content diff
cannot separate "never rebuilt" from "rebuilt identically" — only mtimes can. The guard is now a
provenance test (`map` newer than `stats` newer than `weights`) that *reports* per-target diffs
instead of aborting on them. This is the second time in this project that a guard's stated reason for
firing turned out to be a hypothesis rather than a diagnosis.

**The relaunched repair wrote every record into the wrong model's directory.** The hand-written
driver — not one of the committed runners, all five of which set all three variables — pinned
`RUN` in a shell variable and passed `--model "$RUN/finetuned"` and `--bitmaps "$RUN/..."`
explicitly, and exported `DQ_TASK` — but not `DQ_MODEL`. `common.RUN_DIR` is
`runs/{model_slug(DQ_MODEL)}_{DQ_TASK}` with `DQ_MODEL` defaulting to the Qwen model, so all four
arms wrote into the **Qwen Banking77** directory, and the same defaulted `MODEL_ID` handed a **Qwen
tokenizer to Mistral weights** — out-of-vocab ids, `device-side assert triggered`, every arm dead
during eval after a full 7B quantize had been paid for. Pinning the input pinned nothing about the
output; the output path lived in env-derived module state the script never touched.

Damage was limited by accident of ordering: `record()` runs after eval, so all published Qwen
accuracy numbers survive with their original timestamps. Three `_quant.json` companions were
overwritten and are now in `/workspace/quarantine/` with a note — kept out of the Mistral directory
on purpose, since a `_quant.json` with no matching accuracy record is exactly the
partially-completed-arm artifact that caused the original defect. The companions were regenerated
under `*_redo` names rather than the originals, which costs nothing and buys a determinism check on
two published numbers.

Both runners now open with two explicit fatal guards, in this order: **where output goes**, then
**what the input is**. The first imports the same `common` module the arms import — a
reimplementation of the path rule would drift, an import cannot — and is validated by a negative
control (same script, `DQ_MODEL` unset) that must exit nonzero.

A guard in a runner only protects the runners that carry it, though, and the driver that caused this
was written by hand in an afternoon. So both checks now also live inside `stage5_quantize.py` itself,
which every arm goes through: it refuses to start if `--model` resolves outside `RUN_DIR`, or if the
bit map is older than either the weights or the stats it claims to describe. `--allow-stale`
downgrades the second to a warning, which is what the deliberately-mismatched `*_ft1map` arms need.
The check runs before the model loads, so when it fires it costs nothing.

## The finding that came out of running a second model

Qwen3.5-2B-Base ties `embed_tokens` and `lm_head` into **one 508.6M tensor — 27% of the
1.882B-parameter text decoder**. Mistral unties them, where the head is 5.5% of the
checkpoint and the fp16-head convention costs the baselines 0.35 bits: a footnote.

On a tied model the same convention stops being a footnote. `ignore=["lm_head"]` pins that
shared quarter of the model at fp16, so measured on the fine-tuned checkpoint:

| convention | "4-bit g128" | "3-bit g128" |
|---|---|---|
| `ignore=["lm_head"]`, as shipped | **7.3605 bits / 1.6125 GiB** — 58.8% of stored bits are fp16 | **6.6253 bits / 1.4514 GiB** — 65.3% fp16 |
| `ignore=[]`, tie quantized | 4.1597 bits / 0.9113 GiB | 3.1522 bits / 0.6906 GiB |
| DynQuant, measured | 4.2486 bits / 0.9307 GiB | 3.2494 bits / 0.7118 GiB |

The first two rows are properties of the architecture meeting the recipe, so they do not depend
on which task the checkpoint was tuned on. The third does in principle — the allocator reads
task-specific statistics — and the way it depends on the task is instructive. The **bit width** is
identical to ten significant figures on both of this model's runs (4.2485829466 on CaseHOLD,
4.2485829466 on Banking77) because the budget is a constraint the allocator hits, not a target it
approaches. The **byte count** differs, but only in the fourth decimal — 0.9307 GiB here against
0.9313 on Banking77 — because which specific modules got the wide widths moved, and modules have
different parameter counts. So the task changes the *distribution* of bits under a budget the
budget itself pins. That is the same conclusion the module-by-module diff reaches further down,
arrived at from the accounting side.

The published recipes, run at their documented settings on a tied-embedding model, reach
**7.36 bits when asked for 4** — 2.2× compression where the name implies 4×. That is not a
DynQuant result; it is a property of the convention meeting an architecture, and it is
worth knowing independently of anything in this repo.

It also means the default panel cannot be read as a method comparison. Comparing DynQuant
at 4.2486 bits against a baseline at 7.3605 would be reporting that a model keeping a
quarter of itself in full precision does better than one that does not — true, and
uninformative. So both conventions are run and reported as separate panels:

- **Panel 2 (default)** is what a reader gets today from the documented recipe. The size
  gap in it is the finding above.
- **Panel 3 (`--include-head`)** re-runs the identical recipes with the tie quantized. Those
  land at 4.1597 and 3.1522 bits — **2.1% and 3.0% below DynQuant's own measured width**.
  This is the panel that isolates the allocator, and the residual budget error runs
  *against* DynQuant, so it cannot manufacture a win.

Neither panel is the "real" one. Deleting the first would hide where the shipped recipes
quietly stop compressing; deleting the second would let a budget difference masquerade as
a method difference.

<!-- PANEL-2-3-PENDING: the ten standard arms and the six tied-head arms are still
     running. Filled from baselines_table.py output, not by hand. -->

## Panel 2 — Qwen3.5-2B-Base on CaseHOLD, shipped convention

All ten arms landed. Rows are grouped by tier and ordered by accuracy within tier; every number is
read back from the arm's own JSON record rather than typed, and the published table is regenerated by
`baselines_table.py`.

| arm | bits | GiB | ×smaller | accuracy | correct | unparseable |
|---|---:|---:|---:|---:|---:|---:|
| fine-tuned bf16 | 16.0000 | 3.5050 | 1.00× | 89.74% | 4769/5314 | 0 |
| GPTQ 4b g128 | 7.3605 | 1.6125 | 2.17× | 89.74% | 4769/5314 | 0 |
| **DynQuant 4.25b** | **4.2486** | **0.9307** | **3.77×** | 89.25% | 4743/5314 | 0 |
| AWQ 4b g128 | 7.3605 | 1.6125 | 2.17× | 89.18% | 4739/5314 | 0 |
| bnb NF4 (block 64, dq) | 7.3391 | 1.6078 | 2.18× | 89.01% | 4730/5314 | 0 |
| RTN 4b g128 | 7.3605 | 1.6125 | 2.17× | 88.93% | 4726/5314 | 0 |
| GPTQ 3b g128 | 6.6253 | 1.4514 | 2.42× | 88.97% | 4728/5314 | 0 |
| **DynQuant 3.25b** | **3.2494** | **0.7118** | **4.92×** | 86.70% | 4607/5314 | 0 |
| AWQ 3b g128 | 6.6253 | 1.4514 | 2.42× | 83.36% | 4430/5314 | 0 |
| RTN 3b g128 | 6.6253 | 1.4514 | 2.42× | **65.53%** | 3482/5314 | 0 |

**Every arm produced zero unparseable generations.** That matters more here than on Banking77:
CaseHOLD answers are single digits 0–4, so a model whose output formatting had degraded would show
up as unparseable rows rather than as wrong ones, and an accuracy column alone would not distinguish
"chose the wrong holding" from "stopped emitting a holding". Across ten arms including one that lost
24 points, the failure mode is always wrong answers, never broken formatting.

**The 4-bit tier does not separate the methods.** Five arms span **0.81 points** — 89.74% down to
88.93% — and no pairwise DynQuant comparison in it is significant: against AWQ +0.08 (p = 0.85),
against NF4 +0.24 (p = 0.45), against RTN +0.32 (p = 0.34), against GPTQ −0.49 (p = 0.10). DynQuant
places second of five on accuracy and the ranking below itself is noise. What the tier *does*
separate is size, and by a wide margin: **0.9307 GiB against 1.61 for all four baselines**, 42%
fewer bytes for a difference in accuracy no test here can distinguish from zero. At this budget the
result is a size claim, not an accuracy claim, and it should be stated that way.

One detail worth keeping because it is a trap. GPTQ 4-bit scores **exactly** 4769/5314, the same as
bf16 — and it is not the same model. The two disagree on **116 examples**, 58 each way. Identical
accuracy from a coin-flip-balanced set of disagreements is what a lossy transform looks like when it
is small relative to the task's noise floor; reading it as "GPTQ 4-bit is lossless here" would be
wrong. The same caution applies to DynQuant's own 4.25-bit row. Against the bf16 checkpoint it was
quantized from it gives up **0.49 points**, not separated — CI [−0.07, +1.05], 128 examples bf16 got
right that DynQuant got wrong against 102 the other way, p = 0.099 — but that net is built from
**230 disagreements** on 5,314 rows. The two models differ on 4.3% of the test set and happen to
break close to even. Reporting only the net would overstate how similar they are.

The 3.25-bit row is a different story. It loses **3.05
points** (86.70% against 89.74%) and that gap *is* separated — paired CI [+2.33, +3.77], flips
271/109, p = 4.9e-17 — at 4.92× compression. On Mistral/Banking77 the same target cost 0.81 points.
So the 3-bit regime is roughly four times more expensive on this model-and-task pair than on the
other, and the honest reading is that 4.25 bits is free here while 3.25 is not.

The obvious structural candidate was the tied embedding. DynQuant's budget covers the **tied
embedding/LM-head — 27% of this model** — so its 3.25-bit arm quantizes the tensor that produces
the output distribution, while every baseline row in this panel leaves that same tensor at fp16 by
convention. Mistral's head is untied and 5.5% of the checkpoint, so the equivalent pressure there
is small.

**The RTN 3-bit row refutes that explanation, and does so in DynQuant's favour.** RTN keeps the tie
in fp16 — 65% of its stored bits are full precision — and still collapses to **65.53%**, 24.2 points
below bf16 and barely three times chance on a 5-way task. So it is not the tie that makes 3 bits
hard on this model. (Panel 3 later quantifies the tie's own contribution: quantizing it costs RTN a
further 4.61 points at 3 bits. Real, and small beside the 24.21 the body weights cost on their own.) DynQuant's allocation of an average 3.2494 bits across everything, tie included,
retains 86.70%: **+21.17 points at 0.712 GiB against 1.451**, better accuracy from half the bytes.

**GPTQ at 3 bits then narrows what that proves.** It reaches **88.97%** — 0.77 below bf16, separated
but small (CI [+0.18, +1.36], flips 149/108, p = 0.012) — at the same 6.6253 bits and the same fp16
tie that RTN had. Same width, same convention, **+23.45 points** over RTN (CI [+22.00, +24.89],
1389/143, p ≈ 1e-256). So the body weights of this model *can* go to 3 bits with almost no loss.
RTN's collapse was a failure of naive rounding, not evidence that the budget is infeasible, and the
+21.17 over RTN should be read as *allocation beats uniform naive rounding*, not as DynQuant beating
a strong baseline. AWQ, below, is that comparison.

With AWQ landed the tier is complete, and it separates the four methods cleanly — the only tier in
this document that does. Each baseline sits at 6.6253 bits and 1.4514 GiB; DynQuant at 3.2494 and
0.7118, **2.04× smaller than all three**:

| 3-bit arm | what it does beyond rounding | GiB | accuracy | vs bf16 | vs RTN 3b |
|---|---|---:|---:|---:|---:|
| GPTQ 3b g128 | inverse-Hessian error feedback | 1.4514 | 88.97% | −0.77 | +23.45 |
| **DynQuant 3.25b** | **per-module width from training signal** | **0.7118** | **86.70%** | −3.05 | +21.17 |
| AWQ 3b g128 | activation-aware channel smoothing | 1.4514 | 83.36% | −6.38 | +17.84 |
| RTN 3b g128 | nothing | 1.4514 | 65.53% | −24.21 | — |

Every adjacent gap is statistically separated: GPTQ over DynQuant **+2.28** (CI [+1.55, +3.00],
flips 253/132, p = 7.0e-10), DynQuant over AWQ **+3.33** (CI [+2.46, +4.21], 370/193, p = 7.5e-14),
AWQ over RTN **+17.84** (1143/195, p ≈ 2e-163).

So DynQuant places **second of four, ahead of AWQ by more than it trails GPTQ, at half the storage
of both.** That is the strongest result in this document, and it is the first one that is a win
against a method people actually ship rather than against a straw baseline. The AWQ comparison is
the load-bearing one: AWQ also uses calibration data, is also a published production method, and is
beaten here by 3.33 points while spending twice the bytes.

The mechanism column is why the ordering is worth reporting rather than just the numbers. These are
three different things to do with a bit budget, and no arm in this document does two of them.
**DynQuant's quantizer has no error compensation and no smoothing** — it is MSE-clipped
round-to-nearest inside a width the allocator chose. GPTQ rounds one column at a time and pushes the
residual into the columns not yet quantized. AWQ rescales channels by activation magnitude before
rounding and never revisits them. Ranked by what each buys at 3 bits on this model: error feedback
(+23.45) ≳ allocation (+21.17) > smoothing (+17.84).

That ranking is the honest frame for the GPTQ gap, and it is not a stacking term. The 2.28 points is
compensation-alone against allocation-alone, with allocation doing its work at half the storage —
**not** "compensation is worth +2.3 on top of allocation," which is a claim no arm here supports.
The combination is unmeasured. Running a GPTQ-style sweep inside the per-module widths DynQuant
already picks is the obvious next experiment; nothing here says the two would add, and nothing says
they would conflict.

One caveat keeps the GPTQ comparison from being a clean loss: the arms are not the same size, and
that difference favours DynQuant. Panel 3 is the like-for-like test, and the question it answers is
now sharp. If GPTQ with the tie quantized drops toward 86%, the 2.28 points were the tie. If it
holds near 89% at DynQuant's byte count, the gap is the missing error compensation and it is a real
deficit to fix — in which case the AWQ win still stands, since AWQ is in the same panel under the
same convention.

The bits column is where this panel stops being a method comparison. Every baseline row sits at
**7.36 or 6.63 bits** — the tied-embedding effect from the section above — against DynQuant's
4.25 and 3.25. Panel 3 is the one that compares methods.

## Panel 3 — Qwen3.5-2B-Base on CaseHOLD, tie quantized (`--include-head`)

This is the panel that compares methods rather than conventions. `--include-head` empties the
`ignore` list, so the tied embedding/LM-head — 27.02% of this model, confirmed by the gate at run
time (`tied embedding check: yes 0.2702`) — is quantized alongside everything else, exactly as
DynQuant's budget already does. The arms land at **4.1597 and 3.1522 bits**, which is 2.1% and 3.0%
*below* DynQuant's 4.2486 and 3.2494, so any residual budget error in this panel runs against
DynQuant, not for it.

All six arms landed.

| arm | bits | GiB | accuracy | correct | modules | fp16 share |
|---|---:|---:|---:|---:|---:|---:|
| GPTQ 4b g128 +head | 4.1597 | 0.9113 | **89.76%** | 4770/5314 | 187 | 0.11% |
| **DynQuant 4.25b** | **4.2486** | **0.9307** | **89.25%** | 4743/5314 | — | — |
| AWQ 4b g128 +head | 4.1597 | 0.9113 | 88.93% | 4726/5314 | 187 | 0.11% |
| RTN 4b g128 +head | 4.1597 | 0.9113 | 88.80% | 4719/5314 | 187 | 0.11% |
| GPTQ 3b g128 +head | 3.1522 | 0.6906 | 88.03% | 4678/5314 | 187 | 0.15% |
| **DynQuant 3.25b** | **3.2494** | **0.7118** | **86.70%** | 4607/5314 | — | — |
| AWQ 3b g128 +head | 3.1522 | 0.6906 | 83.31% | 4427/5314 | 187 | 0.15% |
| RTN 3b g128 +head | 3.1522 | 0.6906 | **60.91%** | 3237/5314 | 187 | 0.15% |

**Two results from the first arm, and one of them is negative for DynQuant.**

The first is about the convention rather than the method. Quantizing the tie costs RTN almost
nothing at 4 bits: 88.93% with it at fp16, 88.80% with it at 4 bits — **0.13 points, not separated**
(CI [−0.18, +0.44], flips 39/32, p = 0.48) — while cutting the checkpoint from 1.6125 GiB to
**0.9113**, a 43% saving. So the fp16-`lm_head` convention that every published GPTQ and AWQ
checkpoint follows is, on this model at this width, buying 43% of the bytes for a difference no test
here can distinguish from zero. That is a finding about the baselines' default settings, and it is
the honest reason Panel 2's baseline rows measure 7.36 bits: not because the methods need the head
in fp16, but because nobody changed the flag.

The second is the like-for-like comparison, and it does not favour the allocator. At matched size —
DynQuant 0.9307 GiB against RTN's 0.9113, DynQuant *2.1% larger* — DynQuant is **+0.45 points ahead
and not separated** (CI [−0.16, +1.06], flips 150/126, p = 0.17). Plain round-to-nearest with the tie
included, given the same byte budget, is statistically indistinguishable from the signal-driven
allocation. At 4.25 bits the allocator earns nothing measurable over uniform rounding once the
baseline is allowed the same storage.

That is consistent with what the internal experiment already found and should be read alongside it:
at a 4.25-bit target the knapsack is barely under pressure, 177 of 187 module widths are identical
across two different tasks, and the map is close to a fixed architectural prior. A method whose
allocation is nearly uniform should not be expected to beat uniform. The place the allocator has to
earn its keep is the tight budget, where Panel 2 showed it worth +21.17 over naive uniform rounding —
and the 3-bit rows of this panel are the test that matters, because they put that comparison at
matched bytes for the first time.

**GPTQ's +head arm is the strongest absolute result in this document, and it is a baseline.** It
scores **89.76%** — 4770 of 5314, *one example above* the bf16 checkpoint — at **0.9113 GiB, 3.85×
compression**. The difference from bf16 is −0.02 points and unmeasurable (flips 65/66, p = 1.00), so
the honest statement is that GPTQ 4-bit with the tie quantized reproduces full-precision accuracy on
this task at just over a quarter of the bytes. It also confirms the convention finding a second time
and more sharply than RTN did: against its own fp16-head twin the difference is −0.02 points with 30
and 31 flips (p = 1.00), for the same 43% saving. Two methods, same conclusion — `ignore=["lm_head"]`
is costing 43% of the checkpoint and buying nothing measurable.

**Three methods now agree the convention is free.** Each `+head` arm against its own fp16-head twin:
RTN +0.13 (p = 0.48), GPTQ −0.02 (p = 1.00), AWQ +0.24 (p = 0.13). Not one is separated, the largest
is a quarter of a point, and all three save the same 43% of the checkpoint. Three independent
quantization methods, three independent calibrations, same answer: on a tied model at 4 bits,
`ignore=["lm_head"]` buys nothing and costs 43% of the bytes. That is the cleanest finding in this
document, and it is a finding about how these tools are configured in practice rather than about any
method's algorithm.

With the 4-bit tier complete, the matched-byte ordering is four-deep, and its shape is worth reading
carefully because it is not a simple loss:

| matched-byte 4-bit arm | GiB | accuracy | vs DynQuant | separated? |
|---|---:|---:|---:|---|
| GPTQ 4b +head | 0.9113 | 89.76% | +0.51 (p = 0.092) | no |
| **DynQuant 4.25b** | **0.9307** | **89.25%** | — | — |
| AWQ 4b +head | 0.9113 | 88.93% | −0.32 (p = 0.30) | no |
| RTN 4b +head | 0.9113 | 88.80% | −0.45 (p = 0.17) | no |

**DynQuant is not separated from any of the three**, and the tier collapses further than that: AWQ is
not separated from RTN either (−0.13, p = 0.70), so its smoothing buys nothing here. The *only*
separated comparison in the whole tier is GPTQ over RTN, +0.96 (CI [+0.38, +1.54], flips 151/100,
p = 0.0016).

So at 4 bits and matched storage, three of the four arms — DynQuant, AWQ, RTN — are mutually
indistinguishable across a 0.45-point spread, and error compensation is the single mechanism whose
benefit clears the noise floor. DynQuant's point estimate places it second, above both other
calibration-using methods, but "DynQuant beats AWQ" and "GPTQ beats DynQuant" are both over-readings
of 5,314 examples. The defensible statement is narrower and more useful: **at 4 bits this task cannot
tell these methods apart, so the tier is decided on cost, and DynQuant is the most expensive of the
four** — 0.9307 GiB against 0.9113, spending 2.1% more bytes for an accuracy difference nothing here
can measure.

That reading also disposes of the tie hypothesis for the 4-bit tier specifically. The 2.1% budget
error runs against DynQuant, the tie is quantized in every arm, and the arms still do not separate.
Whatever distinguishes these methods on this model is not visible at 4 bits under any convention.

### Why the 4-bit tier is flat: the allocator barely allocates there

The null result above has a mechanical explanation that is visible in the allocator's own log and
does not require an accuracy column to find. `stage4_allocate.py` runs the knapsack twice at every
target — once with the real scores, once with every score replaced by the same constant
(`dict.fromkeys(scores, 0.5)`), holding the graph, the floors, the group size and the budget fixed —
and counts the modules whose width differs. That control is an exact in-silico ablation of the signal.

| target | modules the signal moves | share of the 187 |
|---:|---:|---:|
| 7.36 bits | 88 | 47% |
| 6.63 bits | 90 | 48% |
| **4.25 bits** | **12** | **6%** |
| **3.25 bits** | **87** | **47%** |

**At 4.25 bits the signal changes 12 module widths out of 187.** The reason is that 4.25 is almost
exactly the budget at which this model's floor map is affordable: the map is `{3b: 8, 4b: 140, 8b: 39}`
and only 5 floors are breached, so nearly every module receives the width its role would have been
given anyway, and the ROI knapsack has almost nothing left to trade. The DynQuant 4.25-bit arm is
therefore **~94% a fixed architectural prior with the tie forced down to 4 bits**, and the fact that
it cannot be told apart from RTN at matched bytes is close to a prediction rather than a
disappointment: at that budget there is very little signal-driven allocation in it to detect. At
3.25 bits the same allocator moves 87 modules, and that is the tier where it wins by 25 points.

This also corrects a conclusion recorded earlier in this project. The observation that the CaseHOLD
and Banking77 maps agree on **177 of 187 widths** at 4.25 bits was read as "the bit map is mostly
architecture, not task." That denominator is the problem — 170 of those 187 modules are ones the
signal moved in *neither* task, so they are pinned by floors and budget and would agree across any
pair of tasks whatsoever. Conditioning on the modules the allocator actually moved:

| target | signal-driven slots | unconditional agreement | **disagreement given the signal moved it** |
|---:|---:|---:|---:|
| 4.25 bits | 17 / 187 (9%) | 177/187 = 95% | **10 / 17 = 59%** |
| 3.25 bits | 103 / 187 (55%) | 161/187 = 86% | **26 / 103 = 25%** |

So the signal is *not* mostly measuring architecture. Where it has freedom at 4.25 bits it disagrees
across tasks more often than it agrees, and the 95% figure was measuring how constrained the knapsack
was rather than how task-specific the signal is. (The containment itself — every cross-task difference
falling inside the signal-driven set — is arithmetic, not evidence: a module the signal moved in
neither task equals the control in both maps and therefore equals itself. Only the ratio is a
finding.) The union is also a lower bound on the allocator's freedom, since a module it could have
moved but left at the control width is not counted.

One honest gap this opens. DynQuant's +25.78 over RTN at 3 bits is the combined effect of **two**
things — everything the allocator does without consulting the signal, and the signal on top of that —
and nothing in the panels above separates them, because RTN is uniform-width and the control map had
never been quantized and evaluated. The arm that settles it is cheap and specific: **quantize with
the uniform-score control map and evaluate it.** That arm is measured under "Ablating the signal"
below, and the split is **22.62 for the allocator and 3.16 for the signal**, both separated. Read
that section before quoting the 25-point figure as a result about training signals.

One thing to get right about what such an arm can and cannot isolate. It is tempting to call the
control a "role floors only" arm, and at 3.25 bits that is wrong: the floors are unaffordable at this
budget for *both* maps, and the control breaches 64 of them while the real map breaches 79. Nor is
the control degenerate — with every score equal, the ROI metric `score / (params × Δbits)` reduces to
pure size, so the control is a real strategy: *spend bits where they are cheapest per parameter*. What
the arm isolates is therefore the signal against a competent size-aware allocator, not against a
straw man. It is the sharper question of the two.

### The 3-bit tier, matched bytes

**This is where the allocator's contribution becomes unambiguous.** RTN 3-bit with the tie quantized
lands at **60.91%** — 0.6906 GiB, 3.1522 bits, 3.1% *fewer* bytes than DynQuant's 0.7118 — against
DynQuant's 86.70%. That is **+25.78 points** for DynQuant (CI [+24.26, +27.30], flips 1531/161,
p ≈ 3e-280), at matched storage, with the residual budget error running the wrong way for DynQuant.

Note the word *allocator*, deliberately. "Ablating the signal" below decomposes this 25.78 into
+22.62 for the allocator run with the training signal switched off and +3.16 for the signal itself.
Both separate; the larger part is the architectural prior, not the fine-tune.

The matched-byte number is *larger* than Panel 2's half-byte comparison (+21.17), and the reason is
the second finding here: **at 3 bits the fp16-head convention stops being free.** RTN loses 4.61
points when the tie is quantized — 65.53% to 60.91%, separated (CI [+3.60, +5.63], flips 501/256,
p = 3.7e-19) — where at 4 bits the same change cost 0.13 points and was unmeasurable. So the
convention is a genuine accuracy purchase at 3 bits and an unexamined default at 4. Any blanket claim
that `ignore=["lm_head"]` is free needs the width attached to it.

Put together, the two facts sharpen what DynQuant is doing. Uniform 3-bit rounding of this
checkpoint's body costs 24.21 points; also rounding the tie costs another 4.61. DynQuant spends an
average of 3.2494 bits across the same 187 modules *including* the tie and gives up 3.05. It is not
protecting the tie by exempting it — it cannot, the tie is inside its budget — so it must be
distributing width in a way that survives both pressures at once. That is the allocator doing the
thing it was designed to do, measured against the only fair baseline for it, and the margin is 25
points rather than a fraction of one.

**And GPTQ still wins at matched bytes.** Its 3-bit +head arm scores **88.03%** at 0.6906 GiB — 1.34
points above DynQuant, separated (CI [−2.07, −0.60], flips 162/233, p = 4.2e-04), while storing 3.1%
*less*. This is the clearest head-to-head loss in the document: matched budget, budget error running
in DynQuant's favour, and DynQuant behind by a margin the test can resolve.

Matching the bytes did help, and the mechanism is legible. GPTQ paid a **0.94-point** penalty for
taking the tie to 3 bits (88.97% → 88.03%, CI [+0.47, +1.41], p = 1.1e-04) where RTN paid 4.61 — so
the Hessian sweep absorbed about **80%** of the tie penalty that destroyed naive rounding. DynQuant
had the tie inside its budget from the start and paid nothing extra for it. Net effect: the deficit
narrowed from Panel 2's 2.28 points (where GPTQ was spending 2.04× the bytes) to 1.34 at matched
storage — a 41% reduction, and not a reversal.

With AWQ landed the matched-byte 3-bit tier is complete, and **every adjacent gap is separated** —
the only tier in this document where the full ordering is statistically resolved:

| matched-byte 3-bit arm | GiB | accuracy | vs DynQuant | separated? |
|---|---:|---:|---:|---|
| GPTQ 3b +head | 0.6906 | 88.03% | +1.34 (p = 4.2e-04) | **yes** |
| **DynQuant 3.25b** | **0.7118** | **86.70%** | — | — |
| AWQ 3b +head | 0.6906 | 83.31% | −3.39 (p = 1.5e-13) | **yes** |
| RTN 3b +head | 0.6906 | 60.91% | −25.78 (p ≈ 3e-280) | **yes** |

DynQuant places **second of four with both neighbours resolved**: it beats AWQ by 3.39 points and
trails GPTQ by 1.34, while being the largest arm in the tier at 0.7118 GiB against 0.6906. Adjacent
baseline gaps for reference: AWQ over RTN +22.39 (p ≈ 2e-216), GPTQ over AWQ +4.72 (p = 1.3e-26),
GPTQ over RTN +27.12 (p ≈ 5e-306).

The conclusion is symmetrical rather than flattering. **Allocation is worth ~26 points over naive
rounding at this budget and ~3.4 over activation smoothing, and a Hessian error-compensation sweep at
uniform width is worth ~1.3 points more than allocation.** All three mechanisms are recovering most of
the same 26-point cliff by different routes; compensation currently does it slightly better and
slightly cheaper, and DynQuant's real win here is over AWQ, a shipped production method, by a margin
that is not close to the noise floor.

One refinement to the convention finding, now that all three methods have both variants at both
widths. The tie penalty is **method-dependent and tracks how much machinery the method has for
absorbing it**: at 3 bits RTN pays 4.61 points (separated), GPTQ 0.94 (separated), AWQ 0.06 (not
separated, p = 0.90). At 4 bits all three pay nothing measurable. So "quantizing the tie is free" is
true at 4 bits for everyone, and at 3 bits only for methods that do something beyond rounding. The
AWQ reading carries a confound worth stating: at 83% it has already lost 6.4 points to the body
weights, so it may simply have less left to lose on the tie rather than be handling it well.

That is the finding to carry forward, and it points at one experiment rather than a defence. Neither
arm has both mechanisms. DynQuant chooses widths and then rounds naively inside them; GPTQ rounds
carefully at a width it never chooses. The 4-bit tier showed allocation contributes nothing measurable
when the budget is loose; this tier shows it contributes ~26 points when the budget is tight but still
trails careful rounding by 1.3. Running the sweep *inside* the allocated widths is the obvious test of
whether the two compose, and until it is run this document's honest summary is that DynQuant's
allocator is a large win over the naive baseline and a small, statistically real loss to the strongest
one.

## Equal-byte arms

Both panels still compare arms at slightly different budgets, and on Mistral the 3-bit row
has a 0.375-bit gap doing real work in a separated result. `run_isosize.sh` closes it from
the DynQuant side: it reads the baselines' *recorded* `accounted_bits` — not a hardcoded
constant, because the measured width depends on vocabulary share and whether the embedding
is tied — and re-runs the allocator at exactly that budget.

Because `stage4_allocate.py` overwrites its output file, the script re-emits the two
already-measured targets in the same pass and hard-fails if their bit maps changed. If the
allocator is not deterministic then the existing table is not reproducible, which is a
finding worth stopping for rather than quietly replacing the maps under a published result.

**The guard passed on Qwen and fired on Mistral, and both outcomes were informative.** On Qwen the
log records `allocator reproduced targets 3.25, 4.25 bit-identically` before any new arm was
quantized, so every number in Panels 2 and 3 still describes the maps it was measured on. On Mistral
it refused to proceed, which is how the provenance defect in Panel 1 was found — see "Provenance
defect in the two DynQuant rows" above. Its stated reason, allocator nondeterminism, was wrong; what
it had actually caught was that the fine-tune had been re-run underneath a published map. The guard
earned its keep by comparing rather than by explaining.

### Qwen/CaseHOLD at 7.3605 bits — the shipped-convention budget

This is the arm that answers "what if DynQuant simply had the bytes the baselines actually used?"
The baselines' `ignore=["lm_head"]` recipe measures 7.3605 bits / 1.6125 GiB (Panel 2), so the
allocator was re-run at that target and landed at **7.3602 bits / 1.612 GiB** — a 0.004% budget
difference, the tightest matched comparison in this document.

| arm at ≈1.61 GiB | bits | accuracy | vs DynQuant iso | separated? |
|---|---:|---:|---:|---|
| fine-tuned bf16 (3.5050 GiB) | 16.0000 | 89.74% | +0.11 (p = 0.33) | no |
| GPTQ 4b g128 | 7.3605 | 89.74% | +0.11 (p = 0.64) | no |
| **DynQuant iso-7.36b** | **7.3602** | **89.63%** | — | — |
| AWQ 4b g128 | 7.3605 | 89.18% | −0.45 (p = 0.070) | no |
| bnb NF4 (block 64, dq) | 7.3391 | 89.01% | −0.62 (p = 0.016) | **yes** |
| RTN 4b g128 | 7.3605 | 88.93% | −0.70 (p = 0.019) | **yes** |

**This is the first 4-bit-class tier in the document where DynQuant separates from anything.** It
beats RTN by +0.70 (CI [+0.13, +1.26], flips 137/100) and NF4 by +0.62 (CI [+0.13, +1.11], 105/72),
ties GPTQ (60/54 — the two disagree on 114 rows and land 6 apart), and edges AWQ by a margin that
just misses (p = 0.070). Against the bf16 checkpoint it was quantized from it differs on **26 of
5,314 predictions** and gives up 0.11 points, which is as close to lossless as this harness can
resolve.

The contrast with the 4.25-bit tier is the point, and it is not a budget effect alone. At 7.36 bits
the allocator moves **88 of 187** module widths off the uniform-score control; at 4.25 bits it moves
**12**. Same model, same signal, same code — the tier where the knapsack has room to act is the tier
where the method separates from naive rounding, and the tier where it is pinned by its own floors is
the tier where it cannot be told apart. DynQuant iso-7.36 is also +0.38 over DynQuant 4.25b
(p = 0.20, not separated), so most of that 0.70 over RTN is coming from *how* the bits are spent
rather than from the extra bits themselves.

The honest caveat: 7.36 bits is only a meaningful target because the baselines' own convention
inflates them there. Nobody would choose a 7.36-bit checkpoint on purpose — Panel 3 shows the same
methods reaching 4.16 bits with the tie quantized at no measurable accuracy cost. So read this tier
as *"given the bytes these tools actually ship, the allocator spends them better than RTN, NF4 and
AWQ, and as well as GPTQ"*, not as a recommendation to run at 7 bits.

### Qwen/CaseHOLD at 6.6253 bits — the one tier DynQuant wins outright

The companion arm targets the baselines' *3-bit* convention budget, 6.6253 bits / 1.4514 GiB, and
landed at **6.6247 bits / 1.451 GiB**. It is the strongest result in this document.

| arm at ≈1.45 GiB | bits | accuracy | vs DynQuant iso | separated? |
|---|---:|---:|---:|---|
| **DynQuant iso-6.63b** | **6.6247** | **89.71%** | — | — |
| GPTQ 3b g128 | 6.6253 | 88.97% | **−0.73 (p = 0.017)** | **yes** |
| AWQ 3b g128 | 6.6253 | 83.36% | −6.34 (p = 2.1e-44) | **yes** |
| RTN 3b g128 | 6.6253 | 65.53% | −24.18 (p ≈ 7e-264) | **yes** |

**This is the only place in the document where DynQuant beats GPTQ with statistical separation** —
+0.73 points, CI [+0.14, +1.32], flips 147/108. It also beats AWQ by 6.34 and RTN by 24.18, so all
three neighbours are separated, and it does it while being **statistically indistinguishable from the
bf16 checkpoint**: 89.71% against 89.74%, 70 disagreements out of 5,314, p = 0.905. At 2.42×
compression this arm is, as far as this harness can measure, free.

Why it wins here when it ties at 1.61 GiB and loses at 0.69 is the same mechanism as everywhere else,
and it is about *allocation*, not about rounding. At 1.451 GiB the baselines spend their bytes the
only way their recipe allows — the tied embedding at fp16 and every other module at a uniform 3 bits.
That is a bad split, and GPTQ's error-compensation sweep has to spend itself repairing 3-bit body
weights. DynQuant given the identical byte count chooses `{3b: 2, 4b: 77, 8b: 108}` and puts nothing
at fp16, so no module is ever pushed to a width that needs repairing. The allocator moves **90 of 187**
widths off the uniform-score control at this budget — **the most active of the four tiers** (90 here,
88 at 7.36, 87 at 3.25, 12 at 4.25) — and that is where the 0.73 comes from.

### The full picture across four matched-byte budgets

Collecting every iso-byte DynQuant-vs-GPTQ comparison in the document, the result is
budget-dependent and the ordering reverses twice:

| budget | GPTQ | DynQuant | Δ (DynQuant − GPTQ) | separated? | signal moves |
|---|---:|---:|---:|---|---:|
| ≈1.61 GiB | 89.74% | 89.63% | −0.11 (p = 0.64) | no | 88/187 |
| **≈1.45 GiB** | 88.97% | **89.71%** | **+0.73 (p = 0.017)** | **yes, DynQuant** | 90/187 |
| ≈0.92 GiB | 89.76% | 89.25% | −0.51 (p = 0.092) | no | 12/187 |
| ≈0.69 GiB | 88.03% | 86.70% | −1.34 (p = 4.2e-04) | **yes, GPTQ** | 87/187 |

So "GPTQ is the stronger method" — the reading this document carried while only Panels 2 and 3
existed — was an artifact of which budgets had been measured. The defensible statement is narrower
and more interesting: **DynQuant wins where allocation has room and the budget is loose enough that
nothing needs repairing; GPTQ wins at the tightest budget, where every module is forced low enough
that error compensation matters more than placement.** The crossover on this model sits between 0.92
and 1.45 GiB.

That also explains why the two mechanisms look complementary rather than redundant. Allocation avoids
putting weights at widths that hurt; compensation reduces the damage once they are there. The first
runs out of room before the second does. Neither arm here has both, so the combination remains
unmeasured — and the 0.73 and the 1.34 are the two numbers that say it is worth measuring.

Note the two iso arms are indistinguishable from each other (−0.08, p = 0.72) despite an 11% byte
difference, and both are indistinguishable from bf16. Above ~1.45 GiB this model is saturated and
extra bytes buy nothing, which is why the interesting comparisons all live below it.

## Ablating the signal

Every DynQuant arm above measures the allocator and the training signal together. This section takes
the signal out and leaves everything else fixed.

**The answer, up front: the signal is worth +3.16 points at 3.25 bits (p = 1.9e-15) and +0.19 at
4.25 bits (p = 0.15). The allocator without it is worth +22.62 over RTN at 3.25. So the method's
25-point margin is roughly 88% architectural prior and 12% training signal, and the signal only
earns its keep at budgets tight enough to breach the role floors.**

`stage4_allocate.py` has always computed a **uniform-score control** — the same knapsack with every
module's score replaced by one constant, holding the graph, the role floors, the group size and the
budget unchanged — but only to *count* how many modules the signal moves, never to quantize with. So
"the allocator is worth 25 points over RTN" had never been split into what the allocator does without
the signal and what the signal adds. These arms are that split, and they are the only comparison in
the document where exactly one input changes.

The control is byte-exact against the arm it is compared with, not approximately: net `params × bits`
between the real and control maps is **+0** at both budgets, because the budget binds exactly. Both
report 4.2486 bits / 0.9307 GiB at 4.25 and 3.2494 / 0.7118 at 3.25. Recomputing size directly from
(module, width, params) independently of the allocator gives 0.9299 and 0.7110 GiB — a 0.1% metadata
difference, identical for both maps, so the accounting is measured rather than echoed from the target.

Two things about what this control is, because both are easy to get wrong:

- **It is not a "role floors only" arm.** At 3.25 bits the floors are unaffordable for either map: the
  control breaches 64 of them and the real map breaches 79. At 4.25 they nearly bind — 1 breach and 5.
- **It is not a straw man.** With all scores equal the ROI metric `score / (params × Δbits)` reduces to
  pure size, so the control is a coherent strategy — *spend bits where they are cheapest per
  parameter* — and it makes a strong, specific choice: it puts the embedding at **2 bits** at 3.25.
  The signal's single most consequential decision at that budget is to rescue the embedding to 3 and
  pay for it by spreading the loss across more modules (79 breached floors against 64).

### At 4.25 bits the signal is worth nothing measurable

| arm | GiB | what it has | accuracy | correct |
|---|---:|---|---:|---:|
| fine-tuned bf16 | 3.5050 | — | 89.74% | 4769/5314 |
| **DynQuant 4.25b** | **0.9307** | floors + size-ROI + **signal** | **89.25%** | 4743 |
| **uniform-score control 4.25b** | **0.9307** | floors + size-ROI | **89.07%** | 4733 |
| RTN 4b +head | 0.9113 | uniform rounding | 88.80% | 4719 |

The first two rows are byte-exact against each other, which is the comparison this section exists
for. RTN +head is 2.1% smaller than both, so the step from it to the control is flattered by a small
byte advantage — which matters not at all here, since that step does not separate anyway.

| comparison | delta | 95% CI | flips | p | separated? |
|---|---:|---:|---:|---:|---|
| DQ 4.25 vs control 4.25 — **the signal** | **+0.19** | [−0.05, +0.42] | **25/15** | 0.154 | no |
| control 4.25 vs RTN 4b +head — **the allocator** | +0.26 | [−0.34, +0.87] | 142/128 | 0.429 | no |
| control 4.25 vs RTN 4b (7.36 b) | +0.13 | [−0.48, +0.74] | 141/134 | 0.718 | no |
| control 4.25 vs AWQ 4b | −0.11 | [−0.71, +0.48] | 127/133 | 0.757 | no |
| control 4.25 vs GPTQ 4b | −0.68 | [−1.25, −0.10] | 103/139 | 0.024 | yes |
| control 4.25 vs bf16 | −0.68 | [−1.24, −0.12] | 97/133 | 0.021 | yes |

**This is a clean confirmation of a mechanism that had only been inferred.** The 4-bit tier was
already known to be flat against every baseline, and the explanation offered was that the signal
barely allocates there — 12 of 187 modules moved, about 6.4% of parameters. That was an inference from
the map. It is now a measurement: removing the signal entirely costs **0.19 points and cannot be
resolved**, on only 40 discordant predictions out of 5314. The pairing is that tight because the two
checkpoints genuinely are nearly the same object.

It also serves as the validity check on the construction, and this is why 4.25 was run as well as
3.25. Where the signal moves almost nothing, an iso-byte control must land close to the real arm. If
it had not, the control would be measuring something other than the signal and the 3.25 result could
not be read either. It landed 0.19 points away, so the construction is sound.

The decomposition at this budget is therefore complete and entirely null: uniform rounding 88.80,
plus the allocator 89.07 (+0.26, p = 0.43), plus the signal 89.25 (+0.19, p = 0.15), against bf16
89.74. **Not one adjacent gap separates.** What separates is only the distance to bf16 and to GPTQ,
and both the control and the real arm are behind those by the same 0.68 and 0.49–0.68. At 4.25 bits on
this model, nothing DynQuant does is distinguishable from rounding to the nearest grid point — which
is the honest reading of a tier where the floor map is almost exactly affordable.

### At 3.25 bits the signal is worth 3.16 points

The same ablation at the tight budget, where the signal moves **87 of 187** modules instead of 12,
gives the opposite answer and gives it decisively.

| arm | GiB | what it has | accuracy | correct |
|---|---:|---|---:|---:|
| fine-tuned bf16 | 3.5050 | — | 89.74% | 4769/5314 |
| GPTQ 3b +head | 0.6906 | error feedback, uniform width | 88.03% | 4678 |
| **DynQuant 3.25b** | **0.7118** | floors + size-ROI + **signal** | **86.70%** | 4607 |
| **uniform-score control 3.25b** | **0.7118** | floors + size-ROI | **83.53%** | 4439 |
| AWQ 3b +head | 0.6906 | channel smoothing, uniform width | 83.31% | 4427 |
| RTN 3b +head | 0.6906 | uniform rounding | 60.91% | 3237 |

| comparison | delta | 95% CI | flips | p | separated? |
|---|---:|---:|---:|---:|---|
| DQ 3.25 vs control 3.25 — **the signal** | **+3.16** | [+2.38, +3.95] | **310/142** | 1.9e-15 | **yes** |
| control 3.25 vs RTN 3b +head — **the allocator** | **+22.62** | [+21.16, +24.08] | 1383/181 | 2.8e-229 | **yes** |
| control 3.25 vs AWQ 3b +head | +0.23 | [−0.70, +1.15] | 318/306 | 0.660 | no |
| control 3.25 vs GPTQ 3b +head | −4.50 | [−5.36, −3.63] | 155/394 | 5.3e-25 | **yes** |
| control 3.25 vs bf16 | −6.21 | [−7.10, −5.32] | 127/457 | 1.2e-44 | **yes** |

**So DynQuant's ≈25-point margin over RTN at 3 bits splits 22.62 / 3.16 — about 88% architecture,
12% signal — and both halves separate.** The bulk of the method is the non-uniform prior: role
floors, an honest byte budget, and a knapsack that degrades by ROI instead of rounding everything
to the same width. That is not a small finding, but it is a finding about the *allocator*, and it
needs no fine-tune, no hook and no training signal. The signal is the smaller half. It is also not
decoration: 3.16 points on 452 discordant predictions, p = 1.9e-15, is as separated as anything in
this document.

The placement of the control tells the story more sharply than the margin does. **Without the
signal, DynQuant at 3.25 bits is statistically AWQ** — 83.53% against 83.31%, p = 0.66, and AWQ is
3% *smaller*. Everything that makes the 3-bit tier look like a distinct method rather than a
reshuffle of an existing one is in that 3.16.

Note also what does *not* change: the control is still 4.50 points behind GPTQ and the real arm is
still 1.33 behind it. The signal narrows the gap to error feedback at this budget; it does not
close it. The comparison that DynQuant wins outright is at 6.63 bits, not here.

### How much of the 3.16 is one decision?

The control and the real map differ on 87 modules, but one of those modules is 27% of the network.
`model.embed_tokens` carries a floor of **8 bits** — not from its own role, whose default is 4, but
inherited from the `lm_head` it is tied to, since `floor_for` takes the strictest floor across a
tie. Neither map can afford 8. The control breaches it to **2**; the signal breaches it to **3**.

If that single rescue carries the +3.16, then what the signal supplies at this budget is a *rule* —
"do not take a tied embedding below 3 bits" — that any implementation can hardcode for free. That
is a much weaker claim than "the training signal allocates well," and it is the first thing a
reviewer should ask.

So there is a third arm: **uniform scores, embedding pinned at 3 bits, same budget.** The pin is
applied through the allocator's own `floor_overrides` plus a structural pin, not by editing the
emitted map, so the knapsack finds the bits itself and the result is byte-exact against both of the
other two (+0 bits against each, verified directly). It pays for the rescue the way a size-aware
allocator would — **55 modules from 4 bits to 3**, ending at `{2b: 1, 3b: 150, 8b: 36}` and 102
breached floors — where the signal instead keeps 32 modules at 4 bits and pushes 24 down to 2,
ending at `{2b: 25, 3b: 94, 4b: 32, 8b: 36}` and 79 breaches. Same embedding decision, opposite
distribution of its cost: the control-plus-rule spreads the damage evenly, the signal concentrates
it.

| 3.25 b, all at 0.7118 GiB | embedding | other 186 modules | accuracy | correct |
|---|---:|---|---:|---:|
| DynQuant (signal) | 3 b | signal-ROI | **86.70%** | 4607/5314 |
| uniform-score control | 2 b | size-ROI | 83.53% | 4439 |
| uniform-score control + 3-bit pin | 3 b | size-ROI | **80.56%** | 4281 |

The reading rule was: land near 86.70% and the signal's contribution reduces to the embedding rule;
land near 83.53% and the +3.16 lives in the other 86 moves. It landed below both. **Pinning the
embedding to 3 bits made the control 2.97 points *worse*** — CI [−3.78, −2.17], flips 100/258,
p = 2.4e-12 — and 6.13 points behind the real map, p = 4.5e-45.

So the answer is neither branch of the rule. The embedding rescue is not a free hand-writable
improvement; **in isolation it is actively harmful.** At a fixed budget the rescue has to be paid
for, and *how* it is paid for dominates the rescue itself: spreading 55 modules from 4 bits to 3
costs more than the 4th embedding bit is worth, while the signal's schedule — hold 32 modules at 4
and drive 24 to 2 — earns it back and more. What the signal supplies at this budget is the payment
schedule, not the embedding decision.

Two caveats, both of which cut against reading this row too hard:

- **It measures a pair, not the pin.** The −2.97 is "embedding rescue *paid for by* size-ROI
  spreading." A different payment method could plausibly make the same pin helpful, and this arm
  cannot rule that out.
- **The 2×2 has three cells, not four.** The missing cell is signal scores with the embedding forced
  to 2 bits, which would separate the two factors properly. `AllocationPolicy` exposes floors and
  structural exemptions but no per-module *ceiling*, so there is no clean way to force a width down.
  A CPU-only probe found the nearest available construction: lowering the embedding's floor from the
  inherited 8 to 2 leaves the real map's embedding at 2 and lands at 3.2489 average bits, a
  ten-thousandth *under* the real map, so the comparison would be conservative. Its histogram is
  `{2b: 8, 3b: 65, 4b: 78, 8b: 36}` against the real map's `{2b: 25, 3b: 94, 4b: 32, 8b: 36}`.
  Same scores, same budget, radically different map — **the allocator is path-dependent**: it
  matters whether the knapsack starts at the role floor and downgrades or starts low and upgrades.
  That is a finding about the allocator rather than about the signal, and it is a lever worth
  pulling later; the arm is not run here.

### The signal's value tracks how many decisions it is allowed to make

Putting the two tiers together gives the cleanest statement of the method's operating regime in this
document, and it is not a statement about bit width — it is a statement about slack.

| target | decisions the signal moves | signal's contribution | p |
|---|---:|---:|---:|
| 4.25 b | 12 / 187 modules (6%) | +0.19 | 0.154 |
| 3.25 b | 87 / 187 modules (47%) | **+3.16** | 1.9e-15 |
| 3.01 b, **per row** | 196,071 / 571,968 rows (34%) | **+2.31** | **<0.0001** |

The third row was measured later, in [phase 2](#phase-2--the-3-bit-gap-reversed-on-qwencasehold),
and it is the sharpest form of this thesis the campaign produced: hand the *same* signal a
partition with 3,000× more decisions in it and its value goes from a 12% garnish to the whole
mechanism. The comparison is against a control with the identical width histogram and randomly
permuted row order, so it isolates the ordering and nothing else. Read the first two rows as
scoped to *module* granularity, not as the signal's ceiling.

At 4.25 bits the floor map is almost affordable — 1 breach for the control, 5 for the real map — so
the knapsack has nothing to decide and the signal has nowhere to act. At 3.25 the floors are
comprehensively unaffordable, 64 and 79 breaches, and the question becomes *which* floors to break
and how far. That is the question the signal answers, and it is the only regime in which consulting
it pays. The same ordering holds across the other two tiers by move count: 6.63 bits moves 90 of 187
widths and is the one tier where DynQuant beats GPTQ with separation; 7.36 moves 88 and beats RTN,
NF4 and AWQ.

**The practical rule this implies: DynQuant is worth running when the target is tight enough that
the role floors cannot be satisfied.** Above that point it is an expensive way to reproduce
round-to-nearest.

## Secondary panel — Qwen3.5-2B-Base on Banking77

The same model and the same allocator on the other task, from the earlier pass. 58.0% base →
93.41% fine-tuned, so it has genuine headroom; it is secondary only because the comparison pairs
each model with the task its own screen chose, not because the numbers are weaker.

It earns its place by isolating something the two main panels structurally cannot. Panels 1–3
vary model *and* task together, so a difference between them has two candidate causes. This panel
holds the model fixed and varies only the task, which is the sole place in the document where the
question "is the bit map task-specific or is it just the architecture?" can be asked at all.

### How task-specific is the bit map? It depends entirely on the denominator

Both Qwen runs use the identical regime — LoRA r=32, lr 1e-4, 2 epochs — so the *only* input that
differs is the task, and `bitmap_diff.py` compares the two allocations module by module:

| target | modules assigned a different width | modules off the modal width | same modules off-modal in both | task-invariant share of the off-modal decisions |
|---|---:|---:|---:|---:|
| 4.25 b | **10 / 187 (5.3%)** | 47 and 47 | 42 | **89%** |
| 3.25 b | **26 / 187 (13.9%)** | 93 and 93 | 80 | **86%** |

That table was originally read as settling the question against the method: 86–89% of the
allocator's off-modal decisions are task-invariant, so "the signal is doing the minority of the
work." **That reading is wrong, and the error is the denominator.** "Off the modal width" counts
every module the *role floors* raised to 8 bits — and the floors are a fixed architectural prior
that has nothing to do with the signal, so it would be identical for any pair of tasks. It inflates
the apparent freedom by about 4×.

The correct denominator is the uniform-score control described above: run the same knapsack with
every score replaced by a constant, and count only the modules the *signal* actually moved.
Restricted to those, and computed by scratchpad `signal_slots.py`:

| target | modules the signal moves | unconditional agreement | **disagreement given the signal moved it** |
|---|---:|---:|---:|
| 4.25 b | 17 / 187 (9%) | 177/187 = 95% | **10 / 17 = 59%** |
| 3.25 b | 103 / 187 (55%) | 161/187 = 86% | **26 / 103 = 25%** |

The move counts here are the **union over the two tasks**, since a module either task moved is a
module the signal was free to act on. Per task the counts are 12 (CaseHOLD) and 12 (Banking77) at
4.25 bits and 87 / 93 at 3.25 — so the 4.25-bit sets overlap on only **7 of 17**, while the 3.25-bit
sets overlap on **77 of 103**. The tier where the signal has the least room to act is also the tier
where the two tasks least agree about how to use it.

**Where the allocator is free to choose at 4.25 bits, the two tasks disagree more often than they
agree.** The 95% agreement figure was measuring how tightly the knapsack was constrained, not how
task-invariant the signal is. So the honest summary inverts: the signal is genuinely task-specific;
what is architecture-driven is the *number of decisions the signal is allowed to make*, and at
4.25 bits that number is 12 of 187.

One trap to avoid restating. Every cross-task width difference necessarily falls inside the
signal-driven set — a module the signal moved in neither task equals the control in both maps and
therefore equals itself — so that containment is arithmetic and carries no information. Only the
ratio does. `signal_slots.py` asserts the containment rather than reporting it.

This also revises the observation flagged earlier as unexplained, that two tasks produced 177 of 187
identical widths at 4.25 bits. The count replicates on a second task pair, so it is a property of
the allocator; but it is now explained rather than merely confirmed, and the explanation is the
budget, not the signal. A confounded earlier attempt, which compared this LoRA run against a
60-step *full* fine-tune, reported 11.8% and 43.3% disagreement; with the regime held constant those
drop to 5.3% and 13.9% unconditional, so most of the apparent task-sensitivity in that first look
was fine-tuning regime, not task.

Two structural points about what the differences are:

- **Every move is paid for.** The disagreements come in exactly balanced pairs — 3 modules up and
  3 down at 4.25 bits, 8 up and 8 down at 3.25 — because the budget is a binding constraint. The
  two tasks are not disagreeing about how many bits the model needs; they are disagreeing about
  where to put a fixed number of them. The strongest form of this: the two tasks produce **identical
  width histograms** at both budgets — `{3b: 8, 4b: 140, 8b: 39}` at 4.25 and `{2b: 25, 3b: 94,
  4b: 32, 8b: 36}` at 3.25 — and the same count of breached floors (5 and 79). The signal changes
  which modules receive which width and never how many of each there are.
- **The SwiGLU gate is the most task-sensitive role.** `mlp.gate` accounts for 3 of the 10 moves
  at 4.25 bits and 10 of the 26 at 3.25, well above its share of module count. `linear_attn`
  projections are next. Attention `q/o_proj` and the embedding never move at either budget.

Task-specificity in absolute terms rises as the budget tightens, which is the expected direction: at
4.25 bits most modules sit comfortably at their floor and the knapsack is barely under pressure, so
the ranking hardly matters; at 3.25 bits it is choosing which tensors to starve. Note the *rate*
moves the other way — 59% down to 25% — because tightening the budget grants far more freedom (17
slots to 103) than it grants disagreement (10 moves to 26). Both statements are true and they answer
different questions: the rate says how task-specific each decision is, the count says how much
task-specific allocation ends up in the checkpoint.

### Do the map differences buy accuracy? Banking77 arms

Whether the two maps differ is a separate question from whether the difference is worth anything.
All ten arms are recorded. Shipped convention, so the baselines carry the tied head at fp16 and
measure 7.36 / 6.63 bits:

| arm | bits | GiB | accuracy | correct |
|---|---:|---:|---:|---:|
| fine-tuned bf16 | 16.0000 | 3.5050 | 93.41% | 2877/3080 |
| GPTQ 4b g128 | 7.3605 | 1.6125 | **93.08%** | 2867 |
| bnb NF4 | 7.3391 | 1.6078 | 92.89% | 2861 |
| **DynQuant 4.25b** | **4.2486** | **0.9307** | **92.66%** | 2854 |
| AWQ 4b g128 | 7.3605 | 1.6125 | 92.50% | 2849 |
| GPTQ 3b g128 | 6.6253 | 1.4512 | 92.14% | 2838 |
| RTN 4b g128 | 7.3605 | 1.6125 | 91.53% | 2819 |
| **DynQuant 3.25b** | **3.2494** | **0.7118** | **91.10%** | 2806 |
| AWQ 3b g128 | 6.6253 | 1.4512 | 88.64% | 2730 |
| RTN 3b g128 | 6.6253 | 1.4512 | 61.62% | 1898 |

The ordering reproduces Panel 2's almost exactly, and the same three statements hold. DynQuant's
4.25-bit arm is statistically indistinguishable from GPTQ (−0.42, p = 0.14), from NF4 (−0.23,
p = 0.48) and from AWQ (+0.16, p = 0.64) while storing **1.73× fewer bytes** than any of them; it
separates above RTN (+1.14, CI [+0.49, +1.78], flips 69/34, p = 7.3e-04); and at 3.25 bits GPTQ is
ahead by 1.04 points (p = 0.011) while spending 2.04× the bytes. The two DynQuant tiers separate
from each other (+1.56, p = 9.7e-06), so the budget is doing something measurable between them.

The two arms that landed last are the ones that make the 3-bit tier legible. **DynQuant at 3.25 bits
beats AWQ at 3 bits by 2.47 points** — CI [+1.52, +3.41], flips 148/72, p = 3.3e-07 — while storing
2.04× fewer bytes, and this replicates CaseHOLD's +3.33 (p = 7.5e-14) closely. That is the clearest
outright win in the secondary panel: same direction, same rough magnitude, separated on both tasks,
against the baseline whose mechanism is nearest to DynQuant's own (no error feedback, just a smarter
choice of what to preserve). NF4, meanwhile, lands between GPTQ and DynQuant on both tasks while
spending 4-bit bytes, so it changes no conclusion — it is a check that the 4-bit tier's tie is a
property of the tier and not of `llm-compressor`.

### Same model, same allocator, different task — what carried over

Every pairing run on both tasks, DynQuant on the left:

| comparison | CaseHOLD | Banking77 | carried over? |
|---|---:|---:|---|
| DQ 4.25 vs GPTQ 4b | −0.49 (p = 0.10) | −0.42 (p = 0.14) | yes — neither separated |
| DQ 4.25 vs AWQ 4b | +0.08 (p = 0.85) | +0.16 (p = 0.64) | yes — neither separated |
| DQ 4.25 vs RTN 4b | +0.32 (p = 0.34) | **+1.14 (p = 7.3e-04)** | **no** — separated on one task only |
| DQ 4.25 vs bf16 | −0.49 (p = 0.099) | −0.75 (p = 0.0027) | **no** — separated on one task only |
| DQ 4.25 vs NF4 | +0.24 (p = 0.45) | −0.23 (p = 0.48) | yes — neither separated |
| DQ 3.25 vs GPTQ 3b | −2.28 (p = 7.0e-10) | −1.04 (p = 0.011) | yes — GPTQ ahead on both |
| DQ 3.25 vs AWQ 3b | +3.33 (p = 7.5e-14) | +2.47 (p = 3.3e-07) | yes — DynQuant ahead on both |
| DQ 3.25 vs RTN 3b | +21.17 | +29.48 | yes — both overwhelming |
| DQ 3.25 vs bf16 | −3.05 (p = 4.9e-17) | −2.31 (p = 6.8e-10) | yes — separated on both |
| GPTQ 4b vs bf16 | +0.00 (p = 1.0) | −0.32 (p = 0.076) | yes — lossless on both |
| DQ 4.25 vs DQ 3.25 | +2.56 | +1.56 | yes — tiers separate on both |

**Eleven of thirteen conclusions carry over; the two that don't move in the same direction and have
one explanation.** Banking77 is harsher on quantization than CaseHOLD — but only for the arms that
lose little to begin with, which is a narrower statement than the earlier draft of this section made:

| degradation vs bf16 | CaseHOLD | Banking77 | ratio |
|---|---:|---:|---:|
| **DynQuant 4.25b** | **−0.49** | **−0.75** | **1.53×** |
| AWQ 4b | −0.56 | −0.91 | 1.63× |
| GPTQ 3b | −0.77 | −1.27 | 1.65× |
| RTN 4b | −0.81 | −1.88 | 2.32× |
| RTN 3b | −24.22 | −31.79 | 1.31× |
| — the ratio inverts below — | | | |
| bnb NF4 | −0.73 | −0.52 | 0.71× |
| AWQ 3b | −6.38 | −4.77 | 0.75× |
| **DynQuant 3.25b** | **−3.05** | **−2.31** | **0.76×** |

So DynQuant's 4.25-bit arm did not get better on Banking77 — it got worse, by 0.26 points. It
separated from RTN there because RTN got worse *faster*, losing 1.07 points where DynQuant lost
0.26. That is the mechanism to state: the gap opens when the task punishes uniform rounding, not
when the signal finds something extra. Of the arms measured on both tasks DynQuant's is the least
task-sensitive at 4.25 bits, which is what a method that spends its bits on structure rather than
on this dataset should look like.

**The reversal is not DynQuant's alone.** An earlier draft of this section said DynQuant at 3.25
bits was the only method to degrade less on Banking77; adding the last two arms falsified that. AWQ
3-bit inverts by the same ratio (0.75× against 0.76×) and NF4 inverts by more (0.71×). Three of nine
arms invert, and the split is not by method — it is by how much the arm loses in the first place.
Every arm that loses under ~2 points on CaseHOLD loses relatively *more* on Banking77; the three
that lose more than 2 lose relatively *less*. RTN 3-bit at −24 is the exception that keeps this from
being a rule, and NF4 at −0.73 is a second one, so it is a tendency in nine points, not a law.

Two of the three inversions rest on deltas too small to separate from each other. NF4's is a 0.21-
point difference between two different datasets — no test can be run across them. The one inversion
with room to be real is DynQuant's and AWQ's at 3 bits, both built on degradations that separate
overwhelmingly on both tasks. With one model and two tasks even that is a single observation.

## What replicated, and what did not

All arms are in: the Mistral repair chain finished `rc=0` with both corrected DynQuant arms and both
iso-byte arms, and the signal ablation is complete at both targets. The settled version of this
section, with the full tables, is §8 of
[REPORT-quantization-comparison.md](REPORT-quantization-comparison.md). What follows is the summary
plus the two things that only become visible once all three panels sit side by side.

### The scorecard

DynQuant on the left, shipped-convention baselines, so DynQuant is the smaller arm in every row.

| claim | Qwen/CaseHOLD | Qwen/Banking77 | Mistral/Banking77 | verdict |
|---|---:|---:|---:|---|
| beats GPTQ at 3 b | −2.28 → **reversed**¹ | −1.04 | −0.78 | **superseded on one panel** |
| beats AWQ at 3 b | +3.33 | +2.47 | −0.42 (ns) | two of three |
| beats RTN at 3 b | +21.17 | +29.48 | +0.26 (ns) → **+0.71** at matched bytes | direction on all three |
| ties GPTQ/AWQ/NF4 at 4 b | ties | ties | ties | **holds on all three** |
| lossless vs bf16 at 4.25 b | −0.49 (ns) | −0.75 (sig) | −0.19 (ns) | two of three |

The one row that was unambiguous across every panel, and was also the unfavourable one, was **GPTQ
wins the 3-bit tier** — described here as the most robust finding the campaign produced. It has
since been overturned on the panel where it was measured most sharply, and that sentence is left
standing because the overturning took a rebuilt allocator, a rebuilt clip objective, a rebuilt
embedding path and a change of allocation granularity. It was robust against everything short of
that.

> ¹ **Reversed on Qwen/CaseHOLD.** That row was true of the shipped recipe and was measured
> correctly — `dq_3p25` vs `gptq_3b_head` is −1.34, p = 0.0004, significant and unfavourable. A
> follow-up rebuilt the allocator, the clip objective and the tied-embedding handling and moved body
> allocation from per-module to per-row; the matched-byte comparison is now **+1.54 at p < 0.0001
> while being 4.5% smaller**, still ahead at 8.3% smaller (+1.05, p = 0.0026) and level at 16.4%
> smaller (−0.19, p = 0.64). It passed through an intermediate stage where it was +0.51 at p = 0.16
> — a tie, reported as a tie. The other two panels have *not* been re-run with the new recipe and
> those cells stand as written. See
> [Phase 2 — the 3-bit gap, reversed on Qwen/CaseHOLD](#phase-2--the-3-bit-gap-reversed-on-qwencasehold),
> including its list of reasons to expect less elsewhere.

### Reading 1: the "collapse" DynQuant avoids is a property of the model, not of DynQuant

The two Qwen panels make DynQuant look enormously better than RTN and AWQ at 3 bits. Mistral makes
it look identical. The temptation is to conclude DynQuant transfers badly. It does not — what changes
is the baseline:

| 3-bit arm, vs its own bf16 | Qwen/CaseHOLD | Qwen/Banking77 | Mistral/Banking77 |
|---|---:|---:|---:|
| RTN | −24.22 | −31.79 | −1.04 |
| AWQ | −6.38 | −4.77 | −0.36 |
| DynQuant | −3.05 | −2.31 | −0.78 |

RTN at 3 bits loses 24–32 points on the tied 1.88 B model and **1.04** on the untied 7.25 B one
(all six degradations above are measured pairs, not differences of rounded accuracies). A
method whose selling point is "avoids the collapse" has nothing to sell where nothing collapses.
DynQuant's own degradation is the most stable row in that table — 3.05 / 2.31 / 0.78 — which is the
better way to state the finding: **DynQuant's damage is consistent; its advantage is not, because
its advantage is measured against baselines whose damage is wildly inconsistent.**

### Reading 2: the shipped convention hides a size handicap, and it flipped one verdict

`ignore=["lm_head"]` is llm-compressor's default and it leaves the head in fp16. On a *tied* model
that is 27.05% of the parameters, so a "3-bit" GPTQ measures 6.63 bits. On an untied model it is
3.71%, so a "3-bit" RTN measures 3.62 bits against DynQuant's 3.25 — a 10% size handicap, small
enough to look like nothing and large enough to matter:

| Mistral, matched bytes | arm | bits | GiB | acc |
|---|---|---:|---:|---:|
| 3.06 GiB | `rtn_3b` | 3.6249 | 3.0586 | 93.34% |
| 3.06 GiB | `dq_iso3p62` | 3.6244 | 3.0581 | **94.06%** |

`dq_iso3p62 vs rtn_3b = +0.71, p = 7.2e-03`. The shipped-convention cell for the same pair is
`+0.26, p = 0.42` — *not significant*. **The same comparison changes verdict depending on whether
the arms are the same size**, and this was caught only because the iso-byte arms were run. Every
"does not replicate" claim in a quantization comparison should be checked against the byte accounting
before it is believed.

### The honest summary

Three panels agree on: GPTQ wins at 3 bits; everything ties at 4 bits; DynQuant beats naive rounding
at every budget once sizes are matched. They disagree about *magnitude* everywhere, and the
disagreement tracks the baselines' behaviour rather than DynQuant's. Two models is not enough to
separate the tie from the scale as the cause, and this document does not claim to.

**Addendum.** The first clause no longer holds on Qwen/CaseHOLD — a follow-up reversed that gap,
**+1.54 pts at 4.5% fewer bytes, p < 0.0001**, and lands 0.17 pts under the fp16 ceiling at 3.01
stored bits. Everything else in this summary stands, including the reading that DynQuant's
advantage tracks the baselines' inconsistency. The section below is a chronological addendum
rather than a rewrite: the analysis above was correct about the recipe it measured, and leaving it
intact is what makes the delta legible.

---

## Phase 2 — the 3-bit gap, reversed on Qwen/CaseHOLD

Everything above was written against the shipped recipe. The scorecard's one unambiguous
row — *"GPTQ wins the 3-bit tier, the most robust finding the campaign produced"* — was
correct and is now out of date on the panel where it was measured most sharply. This
section supersedes it for Qwen3.5-2B-Base / CaseHOLD and **only** for that panel; see
[What this does not overturn](#what-this-does-not-overturn).

Every number in this section is a McNemar paired test on the stored per-item `hits`
arrays. All arms score the same 5,314 items in the same order, so the arms are paired and
the unpaired binomial σ used earlier in this document (0.43–0.44 pts) is the wrong ruler —
it prices variance the pairing removes. The paired SE of a *difference* is 0.21–0.37 pts
here, roughly half `sqrt(2)·σ`. Nothing was re-run to get this; the sharper test came free
from data already on disk.

### The arc

| | arm | acc % | bytes | avg bits | vs `gptq_3b_head` | p |
|---|---|---:|---:|---:|---:|---:|
| before | `dq_3p25` (shipped) | 86.70 | 764,290,013 | 3.2494 | **−1.34** | **0.0004** |
| tie | `p2_wc_agg` (per-module body) | 88.54 | 708,087,808 | 3.0104 | +0.51 | 0.16 (ns) |
| **after** | `p2_rb_agg` (per-row body) | **89.57** | **708,087,808** | 3.0104 | **+1.54** | **<0.0001** |

`rb_agg vs dq_3p25 = +2.87 pts` end to end, at **7.4% fewer bytes** than the shipped arm.
The fp16 ceiling on this panel is 89.74%, so the final arm gives up **0.17 points against
full precision at 3.01 stored bits**.

The middle row is worth keeping in the table rather than deleting, because the discipline
it records was vindicated. At the time it was the best arm and it was +0.51 at p = 0.16 —
a tie — and the write-up refused to call that "beats." It never did become significant.
The win came from a *different mechanism* landing underneath it, not from re-reading the
same number more favourably. Had +0.51 been promoted then, the actual result would have
had nothing left to report.

Both endpoints are significant in their own right: GPTQ really did win before
(−1.34, p = 0.0004), and it really is beaten now (+1.54, 203/121, CI [+0.88, +2.21]).

### The frontier

The recipe was walked down in ~30 MB steps to find where it breaks. `gptq_3b_head` sits at
88.03% / 741,475,927 B throughout. Both descents are below, because the second one relocates
the cliff rather than merely lifting the curve, and that is only visible with the first one
next to it.

**Per-module body** (the recipe as it stood before row allocation):

| arm | acc % | bytes | avg bits | vs GPTQ | p | vs the rung above | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p2_wc_agg` | 88.54 | 708,087,808 (−4.5%) | 3.0104 | +0.51 | 0.16 | — | |
| `p2_wc_680` | 88.22 | 679,776,256 (−8.3%) | 2.8900 | +0.19 | 0.63 | −0.32 | 0.14 |
| `p2_wc_650` | 87.22 | 649,891,840 (−12.3%) | 2.7630 | **−0.81** | **0.034** | **−1.00** | **0.0004** |
| `p2_wc_620` | 86.41 | 619,876,352 (−16.4%) | 2.6354 | **−1.62** | **<0.0001** | **−0.81** | **0.0058** |

The floor here is 680 MB. 708 → 680 is free (−0.32, only 68/51 discordant, not separated);
680 → 650 is a cliff (−1.00, p = 0.0004), and 650 is the first rung that loses to GPTQ
significantly. The defensible statement was *GPTQ's 3-bit accuracy at 8.3% fewer bytes*,
with 12.3% measurably too far.

**Per-row body**, same targets, same pricer, same encoder:

| arm | acc % | bytes | avg bits | vs GPTQ | p | vs the rung above | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p2_rb_agg` | **89.57** | 708,087,808 (−4.5%) | 3.0104 | **+1.54** | **<0.0001** | — | |
| `p2_rb_680` | **89.09** | 680,000,000 (−8.3%) | 2.8910 | **+1.05** | **0.0026** | −0.49 | 0.014 |
| `p2_rb_650` | 88.61 | 649,999,872 (−12.3%) | 2.7634 | +0.58 | 0.11 | −0.47 | 0.027 |
| `p2_rb_620` | 87.84 | 619,999,744 (−16.4%) | 2.6358 | −0.19 | 0.64 | −0.77 | 0.0003 |
| `p2_rb_590` | 87.00 | 589,999,872 (−20.4%) | 2.5084 | **−1.04** | **0.0065** | −0.85 | 0.0042 |
| `p2_rb_560` | 83.74 | 560,000,000 (−24.5%) | 2.3809 | **−4.29** | **<0.0001** | **−3.26** | **<0.0001** |

Every rung improves, and by more as the budget tightens: **+0.87 at 680** (p = 0.0022),
**+1.39 at 650** (p < 0.0001), **+1.43 at 620** (p = 0.0001). That ordering is the same
mechanism as the barbell above — the tighter the budget, the more the module-uniform
constraint costs, because forcing a whole module to the mean of its rows is exactly the
wrong move when there are not enough bits to go around.

Three thresholds, all moved:

- **Beats GPTQ down to 680 MB** (−8.3%), significantly, where the module recipe managed
  +0.19 at p = 0.63.
- **Ties GPTQ down to 620 MB** (−16.4%, −0.19 at p = 0.64) — a budget at which the module
  recipe was losing by 1.62 with p < 0.0001.
- **The cliff moved from 650 to below 590.** The module descent broke by a full point at the
  first step past its floor; the row descent gives up 0.47–0.85 per 30 MB rung all the way
  down to 590, then falls 3.26 in one step at 560.

So the claim the frontier supports is **GPTQ's 3-bit accuracy at 16.4% fewer bytes, and
better accuracy at 8.3% fewer**. 560 MB is over the edge and is reported because a descent
that never finds its own floor has not located anything.

The allocator's own diagnostics locate the module cliff: floors breached climb 60 → 75 →
**92** of 187 modules across the first three rungs. The break arrives when about half the
model is pushed below its architectural floor — which is also the point at which the
soft-floor path, the fix for the original allocator's silent `return floor_map`, is doing
nearly all the work. That the curve degrades gracefully rather than collapsing is itself the
soft-floor design being vindicated: the shipped allocator would have returned the floor map
unchanged and reported nothing. The row descent has no equivalent diagnostic — role floors
are defined per module and the row allocator does not consult them — which is worth flagging
as a gap rather than a feature: the reason 560 MB collapses is not currently instrumented.

One caveat on the 650 rung specifically. `rb_650` inherits its clip metadata from the
`wc_650` map, which was priced with the weighted objective, but its *row* sensitivities were
priced unweighted — so that single rung mixes granularity with pricing objective and its
+1.39 is not a clean single-variable A/B. The 680 and 620 rungs are clean, and they bracket
it at +0.87 and +1.43.

### Where the margin actually comes from

Four levers, each isolated at **byte-identical** budgets. None of them is GPTQ's
mechanism: no error feedback, no inverse Hessian, no sequential column compensation, and
**no calibration set** — every input is a moment the fine-tuning hook already accumulates
online.

| lever | A/B | Δ | b/c | p |
|---|---|---:|---:|---:|
| Row-partitioned tied embedding vs uniform 2b | `tie_u2_ctl` → `rows_agg`, +7.00 MiB | **+2.90** | 245/91 | **<0.0001** |
| Gauss-Newton sensitivity allocation vs rank-product | `rows_agg` → `combo_agg` @ 708,087,808 B | **+1.07** | 141/84 | **0.0002** |
| **Per-row body allocation vs per-module** | `wc_agg` → `rb_agg` @ 708,087,808 B | **+1.04** | 122/67 | **0.0001** |
| `E[x²]`-weighted clip objective (encoder only) | `combo_iso` → `wclip_enc` @ 739,545,088 B | +0.70 | 147/110 | 0.025 |
| — replication on a second map | `combo_agg` → `wc_agg` @ 708,087,808 B | +0.41 | 135/113 | 0.18 |
| — **pooled over both replications** | | | 282/223 | **0.0098** |

The tie row is the largest single effect in the campaign and it costs 7 MiB — 0.41 pts/MiB
against the body's 0.011, roughly 37×. The rank-product body is held fixed on both sides of
it, so it isolates the tie and nothing else. The allocator swap is the cleanest: one
substitution, byte-identical output, p = 0.0002.

The two middle rows are the same idea applied twice — spend the signal at the finest
partition the format allows — and the second is the one that turned this section's headline
from a tie into a win. It also has the most misleading Δ in the table, for a reason the next
subsection is entirely about.

### Granularity is a multiplier on the signal, not a gain of its own

`rb_agg` allocates a width to each of the model's **571,968 body rows** instead of to each of
its 187 modules. Same solver, same sensitivity table, same 5,664,702,464-bit budget, same
encoder, same clip grid, same group size; only the decision variable changes. It is worth
+1.04 pts at identical bytes, and reporting that as "row allocation pays" would have been
wrong about which component paid.

The control that shows this permutes each module's width vector under a fixed seed. The width
*histogram* is preserved exactly — so the byte count is identical by construction, not by
tuning — and only the question of *which* rows get the wide ones changes. 196,071 of 571,968
rows (34.3%) end up at a different width.

| body allocation at 708,087,808 B | acc % | vs module body | b/c | p |
|---|---:|---:|---:|---:|
| per-module widths (`p2_wc_agg`) | 88.54 | — | | |
| per-row, **shuffled within module** (`p2_rb_shuf`) | 87.26 | **−1.28** | 125/193 | **0.0002** |
| per-row, ordered by sensitivity (`p2_rb_agg`) | **89.57** | **+1.04** | 122/67 | **0.0001** |

`rb_agg` vs `rb_shuf` — the signal's ordering, with everything else including the histogram
held fixed — is **+2.31, 228/105, p < 0.0001, CI [+1.64, +2.98]**. The decomposition is
additive to 0.01 pts: −1.28 (structure) + 2.31 (signal) = +1.03 against a measured +1.04.

**Row granularity on its own is negative.** Going from 187 decisions to 571,968 buys no
structural advantage — it buys *more decisions*, and more decisions made badly are worse than
fewer. This is the opposite of the natural intuition, which is that finer granularity must be
close to free because a module's rows are heterogeneous. They *are* heterogeneous: the width
the module allocator picks covers only **42.5% of that module's rows at the median**, and
under half the rows in 99 of 186 modules. The heterogeneity is real, and exploiting it still
requires being right about which rows are which.

What the row allocator does with the freedom is trade the middle for a **barbell**:

| width | per-row params | per-module params | delta |
|---|---:|---:|---:|
| 2b | 628.0M | 390.1M | +237.9M |
| 3b | 280.8M | 612.4M | **−331.6M** |
| 4b | 424.6M | 367.0M | +57.6M |
| 8b | 39.4M | 3.3M | **+36.1M (12×)** |

A per-module width is forced to the *mean* of its rows' sensitivities; a per-row width serves
the *distribution*. Loss is convex in width, so the optimum protects a small set of rows at 8
bits — twelve times as many parameters as the module map could afford there — and pays for it
by dumping the bulk from 3 to 2. Per role: `self_attn.k_proj` gains 0.87 average bits,
`o_proj` 0.74, `mlp.down_proj` 0.51, while `linear_attn.in_proj_qkv` gives up 0.59 and the
tiny `linear_attn.in_proj_a`/`in_proj_b` give up 4.06 and 2.50 — the module allocator had
those pinned at an 8-bit floor that most of their rows turn out not to need.

Two things this costs that the tables do not show. First, the byte accounting charges payload
plus one fp16 scale and offset per row per group — partition-invariant, so the A/Bs are
honest — but not the per-row **width code**: 4 distinct widths is 2 bits × 571,968 rows =
142,992 B, **0.0202% of the checkpoint, 0.0006 average bits**. Three orders of magnitude too
small to explain any result here, but it is a real debit and it is not in the numbers above.
Second, 89.57% sits 0.17 under the fp16 ceiling and 0.13 *above* the tie-only arm that leaves
the body in bf16 — which is exactly where a body that silently never got quantized would
land, and `nbytes` is computed analytically from the plan so it cannot rule that out. The arm
was therefore re-encoded and audited directly: 2-bit rows produce exactly 4 distinct values
inside a group, 3-bit exactly 8, 4-bit exactly 16, with relative reconstruction error from
0.0090 (an all-8b `out_proj`) to 0.3799 (`in_proj_qkv` L18). It is materialized.

The operating rule: **never ship a granularity change without the shuffled control.** It is
one extra eval, it preserves the byte count exactly by construction, and here it converted
"row allocation is worth +1.04" into "the signal is worth +2.31 and the structure costs 1.28"
— a different claim about a different component. The same control is the honest test for any
future partitioning, per-group or per-column or per-expert.

### The clip objective pays in the encoder — and the deep grid did not

Both clip knobs were decomposed into (priced-old, encoded-new) vs (priced-new,
encoded-new). They came out **opposite**, which is the useful part:

| knob | encoder-only | + re-pricing |
|---|---:|---:|
| deep grid (floor 0.80 → 0.40) | +0.09 | **+0.58** |
| `E[x²]`-weighted objective | **+0.41** | −0.09 (p = 0.53) |

Re-pricing on top of the weighted encoder is *nothing* — 18 vs 23 discordant items, the
smallest disagreement of any pair measured. Diffing the two maps says why: **4 of 187
module widths changed.** The weighted objective cuts sensitivity 38.6% at 2b and 44.7% at
3b, a near-common factor, and an allocator reads only *ratios* — between modules and across
widths — so a common factor cancels and the ROI ordering survives. The deep grid instead
un-clamped only the low widths, moving `mlp.gate` L0's 2b/3b ratio 4.69 → 3.47: roughly
three times the ratio distortion, which is exactly where its +0.58 came from.

So the question that predicts which half of the pipeline a clip change pays through is not
"grid or objective" but **does it distort sensitivity ratios, or merely scale them?**
Scaling pays in the encoder; distorting ratios pays in the allocator.

This also corrects a generalisation made earlier in the campaign. The deep grid's +0.09
encoder result was read as "the encoder is not where the points are." That was right about
that change and wrong as a rule — the second split was run anyway, out of routine rather
than expectation, and it returned +0.70. Run the decomposition every time, especially when
a previous one makes it look like a formality.

### The mechanism behind the size advantage: the tie

The clearest explanation of *why* there are bytes to save is visible in GPTQ's own two
configurations, same algorithm on both sides:

| GPTQ 3-bit | acc % | bytes | note |
|---|---:|---:|---|
| `ignore=["lm_head"]` | 88.97 | 1,558,371,533 | tie left fp16 = 65.3% of stored bits |
| `ignore=[]` | 88.03 | 741,475,927 | tie quantized uniformly at 3 bits |

`+0.94, 106/56, p = 0.0001, CI [+0.47, +1.41]`. Quantizing the tied embedding at 3 bits
costs GPTQ a full point. Compare the 4-bit RTN pair measured earlier in this document:
0.13 pts at p = 0.48, *not separated*, for a 43% byte saving. **The convention is one fact
at 4 bits and a different one at 3** — at 4 bits the tie survives quantization and
`ignore=["lm_head"]` is an unexamined default worth dropping for free; at 3 bits the tie is
where the damage concentrates.

GPTQ has no per-row width mechanism, so its only two options are a 2.2× larger checkpoint
or that full point. The row plan `8b:2048, 4b:8192, 2b:*` is a third option neither
reaches, and metadata is one fp16 scale+offset per row per group regardless of how the rows
are split — so the granularity is nearly free. That asymmetry is the size advantage. Note
that this is the *same* asymmetry the body exploits one subsection above: the format prices
metadata per row per group either way, so a per-row width vector is close to free wherever
it is applied, and the only question is whether the ordering handed to it is any good.

It also licenses a second statement about the configuration that is not in the headline:
`rb_agg` 89.57% @ 708,087,808 B against `gptq_3b` 88.97% @ 1,558,371,533 B is **+0.60,
161/129, p = 0.069, CI [−0.03, +1.23]**. That interval includes zero, so the correct phrasing
is *indistinguishable from the best GPTQ configuration available at 45% of its size* — never
"beats." The direction reversed when the row body landed (it was −0.43 with the module body)
and it is tempting to promote on that basis; it does not clear the threshold and it is
recorded here as a tie.

### What this does not overturn

Phase 2 was built and measured on **one panel**: Qwen3.5-2B-Base / CaseHOLD. The new recipe
has not been run on Mistral-7B / Banking77 or on Qwen / Banking77. The scorecard rows for
those panels stand as written, and so does the campaign's broader reading that DynQuant's
advantage tracks the *baselines'* inconsistency rather than its own.

Three specific reasons to expect less elsewhere, all established above:

- **The tie lever cannot transfer to an untied model.** On Mistral the head is 3.71% of the
  checkpoint, not 27.05%, so the row-partitioned embedding — the largest single lever here
  at +2.90 — has almost nothing to work with. Expect the Mistral gap to close by much less.
- **The allocator lever needs budget pressure.** At 4.25 bits the knapsack is barely
  constrained and the map is close to a fixed architectural prior; the signal was worth
  nothing measurable there. These gains are a 3-bit-tier phenomenon by construction.
- **The row-body lever is only as good as the signal on that panel.** Its shuffled control
  *loses* 1.28 pts, so it is not a structural improvement that travels — it is an amplifier,
  and on a panel where the sensitivity table is less informative it amplifies less, or
  amplifies the wrong thing. The control has to be re-run wherever the recipe is re-run.

The honest one-line update to the scorecard: **"beats GPTQ at 3 b" moves from *fails on all
three* to *reversed on Qwen/CaseHOLD — +1.54 at 4.5% fewer bytes (p < 0.0001), still ahead at
8.3% fewer, level at 16.4% fewer; untested with the new recipe on the other two.***

That is a stronger claim than the one this section originally carried, and it is worth being
explicit about how it was reached, because the sequence matters more than the endpoint. The
intermediate arm was +0.51 at p = 0.16 and was written up as a tie, over several days, while
it was the best result available. Had it been promoted to "beats" then, the actual result —
which came from a different mechanism landing underneath it, not from re-reading the same
number more favourably — would have had nothing left to report.
